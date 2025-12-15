# -*- coding: utf-8 -*-
"""
FireBase_LSTM_v2.py
KERAS 3 SAFE – sample_weight + 完整畫圖
"""

import os, json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import firebase_admin
from firebase_admin import credentials, firestore

# ================= Firebase 初始化 =================
key_dict = json.loads(os.environ.get("FIREBASE", "{}"))
db = None
if key_dict:
    cred = credentials.Certificate(key_dict)
    try:
        firebase_admin.get_app()
    except Exception:
        firebase_admin.initialize_app(cred)
    db = firestore.client()

# ================= Firestore 讀取 =================
def load_df_from_firestore(ticker, collection="NEW_stock_data_liteon", days=400):
    rows = []
    for doc in db.collection(collection).stream():
        p = doc.to_dict().get(ticker)
        if p:
            rows.append({"date": doc.id, **p})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("⚠️ Firestore 無資料")

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").tail(days).set_index("date")

def ensure_today_row(df):
    today = pd.Timestamp(datetime.now().date())
    if df.index.max() < today:
        df.loc[today] = df.iloc[-1]
        print(f"⚠️ 今日無資料，用 {df.index[-2].date()} 補")
    return df.sort_index()

# ================= Sequence =================
def create_sequences(df, features, steps, window):
    X, y = [], []
    data = df[features].values
    logret = np.log(df["Close"] / df["Close"].shift()).clip(-0.1, 0.1)

    for i in range(window, len(df) - steps):
        X.append(data[i-window:i])
        y.append(logret.iloc[i:i+steps].values)

    return np.array(X), np.array(y)

# ================= LSTM =================
def build_lstm(input_shape, steps):
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.1),
        LSTM(32),
        Dropout(0.1),
        Dense(steps)
    ])
    m.compile(
        optimizer="adam",
        loss=tf.keras.losses.Huber()
    )
    return m

# ================= 原預測圖 =================
def plot_and_save(df_hist, future_df):
    hist = df_hist.tail(10)

    hist_dates = hist.index.strftime("%Y-%m-%d").tolist()
    future_dates = future_df["date"].dt.strftime("%Y-%m-%d").tolist()
    all_dates = hist_dates + future_dates

    x_hist = np.arange(len(hist_dates))
    x_future = np.arange(len(hist_dates), len(all_dates))

    plt.figure(figsize=(18,8))
    ax = plt.gca()

    ax.plot(x_hist, hist["Close"], label="Close")
    ax.plot(x_hist, hist["SMA5"], label="SMA5")
    ax.plot(x_hist, hist["SMA10"], label="SMA10")

    ax.plot(
        np.concatenate([[x_hist[-1]], x_future]),
        [hist["Close"].iloc[-1]] + future_df["Pred_Close"].tolist(),
        "r:o", label="Pred Close"
    )

    ax.plot(
        np.concatenate([[x_hist[-1]], x_future]),
        [hist["SMA5"].iloc[-1]] + future_df["Pred_MA5"].tolist(),
        "g--o", label="Pred MA5"
    )

    ax.plot(
        np.concatenate([[x_hist[-1]], x_future]),
        [hist["SMA10"].iloc[-1]] + future_df["Pred_MA10"].tolist(),
        "b--o", label="Pred MA10"
    )

    ax.set_xticks(np.arange(len(all_dates)))
    ax.set_xticklabels(all_dates, rotation=45)
    ax.legend()
    ax.set_title("2301.TW LSTM 預測（KERAS3 SAFE）")

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{datetime.now():%Y-%m-%d}_pred.png",
                dpi=300, bbox_inches="tight")
    plt.close()

# ================= 回測誤差圖 =================
def plot_backtest_error(df, X_te_s, y_te, model, steps):
    X_last = X_te_s[-1:]
    y_true = y_te[-1]

    pred_ret = model.predict(X_last, verbose=0)[0]
    dates = df.index[-steps:]
    start_price = df.loc[dates[0] - BDay(1), "Close"]

    true_prices, pred_prices = [], []
    p_t = p_p = start_price

    for rt, rp in zip(y_true, pred_ret):
        p_t *= np.exp(rt)
        p_p *= np.exp(rp)
        true_prices.append(p_t)
        pred_prices.append(p_p)

    mae = np.mean(np.abs(np.array(true_prices) - np.array(pred_prices)))
    rmse = np.sqrt(np.mean((np.array(true_prices) - np.array(pred_prices)) ** 2))

    plt.figure(figsize=(12,6))
    plt.plot(dates, true_prices, label="Actual Close")
    plt.plot(dates, pred_prices, "--o", label="Pred Close")
    plt.title(f"Backtest | MAE={mae:.2f}, RMSE={rmse:.2f}")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{datetime.now():%Y-%m-%d}_backtest.png",
                dpi=300, bbox_inches="tight")
    plt.close()

# ================= Main =================
if __name__ == "__main__":
    TICKER = "2301.TW"
    LOOKBACK = 60
    STEPS = 10

    FEATURES = ["Close","Volume","RSI","MACD","K","D","ATR_14"]

    df = load_df_from_firestore(TICKER)
    df = ensure_today_row(df)

    df["Volume"] = np.log1p(df["Volume"])
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()
    df = df.dropna()

    X, y = create_sequences(df, FEATURES, STEPS, LOOKBACK)
    split = int(len(X) * 0.85)

    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    scaler = MinMaxScaler()
    scaler.fit(df[FEATURES].iloc[:split + LOOKBACK])

    def scale_X(X):
        n, t, f = X.shape
        return scaler.transform(X.reshape(-1, f)).reshape(n, t, f)

    X_tr_s = scale_X(X_tr)
    X_te_s = scale_X(X_te)

    step_weights = np.linspace(1.0, 0.3, STEPS)
    sw_tr = np.tile(step_weights, (len(y_tr), 1))

    model = build_lstm((LOOKBACK, len(FEATURES)), STEPS)
    model.fit(
        X_tr_s, y_tr,
        sample_weight=sw_tr,
        epochs=60,
        batch_size=16,
        verbose=2,
        callbacks=[EarlyStopping(patience=8, restore_best_weights=True)]
    )

    # ===== 未來預測 =====
    raw_returns = model.predict(X_te_s, verbose=0)[-1]

    last_trade_date = df.index[-1]
    last_close = df.loc[last_trade_date, "Close"]

    prices, p = [], last_close
    for r in raw_returns:
        p *= np.exp(r)
        prices.append(p)

    seq = df["Close"].iloc[-10:].tolist()
    future = []
    for p in prices:
        seq.append(p)
        future.append({
            "Pred_Close": p,
            "Pred_MA5": np.mean(seq[-5:]),
            "Pred_MA10": np.mean(seq[-10:])
        })

    future_df = pd.DataFrame(future)
    future_df["date"] = pd.bdate_range(
        start=last_trade_date + BDay(1),
        periods=STEPS
    )

    plot_and_save(df, future_df)
    plot_backtest_error(df, X_te_s, y_te, model, STEPS)
