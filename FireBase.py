# -*- coding: utf-8 -*-
"""
FireBase_Attention_LSTM_Direction.py
- Attention-LSTM
- Multi-task: Return path + Direction
- 小資料友善版
"""

import os, json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    Softmax, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping

# Firebase
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
def load_df_from_firestore(ticker, collection="NEW_stock_data_liteon", days=500):
    rows = []
    if db:
        for doc in db.collection(collection).stream():
            p = doc.to_dict().get(ticker)
            if p:
                rows.append({"date": doc.id, **p})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("⚠️ Firestore 無資料")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(days).set_index("date")
    return df

# ================= 假日補今天 =================
def ensure_today_row(df):
    today = pd.Timestamp(datetime.now().date())
    last_date = df.index.max()
    if last_date < today:
        df.loc[today] = df.loc[last_date]
        print(f"⚠️ 今日無資料，使用 {last_date.date()} 補今日")
    return df.sort_index()

# ================= Feature Engineering =================
def add_features(df):
    if "Volume" in df.columns:
        df["Volume"] = np.log1p(df["Volume"].astype(float))
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()
    return df

# ================= Sequence =================
def create_sequences(df, features, steps=5, window=40):
    X, y_ret, y_dir, idx = [], [], [], []

    close = df["Close"].astype(float)
    logret = np.log(close).diff()
    feat = df[features].values

    for i in range(window, len(df) - steps):
        x_seq = feat[i - window:i]
        future_ret = logret.iloc[i:i + steps].values
        if np.any(np.isnan(future_ret)) or np.any(np.isnan(x_seq)):
            continue
        X.append(x_seq)
        y_ret.append(future_ret)
        y_dir.append(1.0 if future_ret.sum() > 0 else 0.0)
        idx.append(df.index[i])

    return np.array(X), np.array(y_ret), np.array(y_dir), np.array(idx)

# ================= Attention-LSTM =================
def build_attention_lstm(input_shape, steps, max_daily_logret=0.06):
    inp = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.2)(x)

    score = Dense(1)(x)
    weights = Softmax(axis=1)(score)
    context = Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1))([x, weights])

    raw = Dense(steps, activation="tanh")(context)
    out_ret = Lambda(lambda t: t * max_daily_logret, name="return")(raw)
    out_dir = Dense(1, activation="sigmoid", name="direction")(context)

    model = Model(inp, [out_ret, out_dir])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(7e-4),
        loss={"return": tf.keras.losses.Huber(), "direction": "binary_crossentropy"},
        loss_weights={"return": 1.0, "direction": 0.4},
        metrics={"direction": ["accuracy"]}
    )
    return model

# ================= 預測圖 =================
def plot_and_save(df_hist, future_df):
    hist = df_hist.tail(10)
    hist_dates = hist.index.strftime("%Y-%m-%d").tolist()
    future_dates = future_df["date"].dt.strftime("%Y-%m-%d").tolist()

    x_hist = np.arange(len(hist_dates))
    x_future = np.arange(len(hist_dates), len(hist_dates) + len(future_dates))

    plt.figure(figsize=(18,8))
    ax = plt.gca()

    ax.plot(x_hist, hist["Close"], label="Close")
    ax.plot(x_hist, hist["SMA5"], label="SMA5")
    ax.plot(x_hist, hist["SMA10"], label="SMA10")

    today_x = x_hist[-1]
    today_y = hist["Close"].iloc[-1]
    ax.scatter([today_x], [today_y], marker="*", s=160)
    ax.text(today_x, today_y + 0.3, f"Today {today_y:.2f}", ha="center")

    ax.plot(
        np.concatenate([[today_x], x_future]),
        [today_y] + future_df["Pred_Close"].tolist(),
        "r:o", label="Pred Close"
    )

    ax.set_xticks(np.concatenate([x_hist, x_future]))
    ax.set_xticklabels(hist_dates + future_dates, rotation=45)
    ax.legend()

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{datetime.now():%Y-%m-%d}_pred.png", dpi=300)
    plt.close()

# ================= 真・回測（昨天預測 vs 今天實際） =================
def plot_yesterday_backtest(df):
    today = df.index.max()
    yesterday = today - BDay(1)

    forecast_path = f"results/{yesterday:%Y-%m-%d}_forecast.csv"
    if not os.path.exists(forecast_path):
        print("⚠️ 沒有昨天的 forecast.csv，跳過回測")
        return

    fc = pd.read_csv(forecast_path)
    fc["date"] = pd.to_datetime(fc["date"])

    row = fc.loc[fc["date"] == today]
    if row.empty:
        print("⚠️ 昨天沒有預測今天")
        return

    pred_price = float(row["Pred_Close"].iloc[0])
    actual_price = float(df.loc[today, "Close"])
    y_close = float(df.loc[yesterday, "Close"])

    bt_df = pd.DataFrame([{
        "date": today,
        "Yesterday_Close": y_close,
        "Pred_Close": pred_price,
        "Actual_Close": actual_price,
        "Error": actual_price - pred_price
    }])

    bt_df.to_csv(f"results/{today:%Y-%m-%d}_backtest.csv",
                 index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10,5))
    plt.plot([yesterday, today], [y_close, actual_price], "-o", label="Actual")
    plt.plot([yesterday, today], [y_close, pred_price], "--o", label="Forecast")
    plt.legend()
    plt.grid(True)
    plt.title("Yesterday Forecast Backtest")

    plt.savefig(f"results/{today:%Y-%m-%d}_backtest.png", dpi=300)
    plt.close()

# ================= Main =================
if __name__ == "__main__":
    TICKER = "2301.TW"
    LOOKBACK = 40
    STEPS = 5

    df = load_df_from_firestore(TICKER)
    df = ensure_today_row(df)
    df = add_features(df)
    FEATURES = ["Close", "Volume", "RSI", "MACD", "K", "D", "ATR_14"]
    df = df.dropna()

    X, y_ret, y_dir, idx = create_sequences(df, FEATURES, STEPS, LOOKBACK)
    split = int(len(X) * 0.85)

    X_tr, X_te = X[:split], X[split:]
    y_ret_tr, y_dir_tr = y_ret[:split], y_dir[:split]

    sx = MinMaxScaler()
    sx.fit(df.loc[:idx[split-1], FEATURES].values)

    def scale(X):
        n,t,f = X.shape
        return sx.transform(X.reshape(-1,f)).reshape(n,t,f)

    model = build_attention_lstm((LOOKBACK, len(FEATURES)), STEPS)
    model.fit(
        scale(X_tr),
        {"return": y_ret_tr, "direction": y_dir_tr},
        epochs=80,
        batch_size=16,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=2
    )

    pred_ret, _ = model.predict(scale(X_te[-1:]), verbose=0)
    price = df["Close"].iloc[-1]
    prices = []
    for r in pred_ret[0]:
        price *= np.exp(r)
        prices.append(price)

    future_df = pd.DataFrame({
        "date": pd.bdate_range(df.index.max() + BDay(1), periods=STEPS),
        "Pred_Close": prices
    })

    future_df.to_csv(f"results/{datetime.now():%Y-%m-%d}_forecast.csv",
                     index=False, encoding="utf-8-sig")

    plot_and_save(df, future_df)
    plot_yesterday_backtest(df)
