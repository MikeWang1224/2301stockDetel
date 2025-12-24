# -*- coding: utf-8 -*-
"""
FireBase_Attention_LSTM_Direction.py  (8110stock.py)
- Attention-LSTM
- Multi-task: Return path + Direction
（中略：你原本整段 docstring 完整保留）
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

from zoneinfo import ZoneInfo
now_tw = datetime.now(ZoneInfo("Asia/Taipei"))

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
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Volume" in df.columns:
        df["Volume"] = np.log1p(df["Volume"].astype(float))

    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()

    if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        df["HL_RANGE"] = (df["High"] - df["Low"]) / df["Close"]
        df["GAP"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    else:
        df["HL_RANGE"] = np.nan
        df["GAP"] = np.nan

    df["VOL_REL"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)

    close = df["Close"].astype(float)
    df["RET_STD_20"] = np.log(close).diff().rolling(20).std()

    return df

# ================= Sequence =================
def create_sequences(df, features, steps=5, window=40, eps=1e-9):
    X, y_ret, y_dir, idx = [], [], [], []

    close = df["Close"].astype(float)
    logret = np.log(close).diff()
    feat = df[features].values

    for i in range(window, len(df) - steps):
        x_seq = feat[i - window:i]
        future_raw = logret.iloc[i:i + steps].values

        if np.any(np.isnan(x_seq)) or np.any(np.isnan(future_raw)):
            continue

        scale = df["RET_STD_20"].iloc[i - 1]
        if pd.isna(scale) or scale < eps:
            continue

        X.append(x_seq)
        y_ret.append(future_raw / (scale + eps))
        y_dir.append(1.0 if future_raw.sum() > 0 else 0.0)
        idx.append(df.index[i])

    return np.array(X), np.array(y_ret), np.array(y_dir), np.array(idx)

# ================= Attention LSTM =================
def build_attention_lstm(input_shape, steps, max_daily_normret=3.0,
                         learning_rate=6e-4, lstm_units=64):

    inp = Input(shape=input_shape)
    x = LSTM(lstm_units, return_sequences=True)(inp)
    x = Dropout(0.2)(x)

    score = Dense(1)(x)
    weights = Softmax(axis=1)(score)
    context = Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1))([x, weights])

    raw = Dense(steps, activation="tanh")(context)
    out_ret = Lambda(lambda t: t * max_daily_normret, name="return")(raw)
    out_dir = Dense(1, activation="sigmoid", name="direction")(context)

    model = Model(inp, [out_ret, out_dir])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss={"return": tf.keras.losses.Huber(),
              "direction": "binary_crossentropy"},
        loss_weights={"return": 1.0, "direction": 0.8},
        metrics={"direction": ["accuracy"]}
    )
    return model

# ================= 🆕 半年 recursive rollout（日頻） =================
def forecast_half_year_daily(model, df, FEATURES, scaler,
                             lookback, steps, months=6):

    trading_days = months * 21
    prices, dates = [], []

    df_roll = df.copy()
    cur_date = df_roll.index.max()
    cur_price = float(df_roll.loc[cur_date, "Close"])

    while len(prices) < trading_days:
        feat = df_roll[FEATURES].iloc[-lookback:].values
        X = scaler.transform(feat).reshape(1, lookback, -1)

        pred_ret, _ = model.predict(X, verbose=0)
        norm_rets = pred_ret[0]

        scale = float(df_roll.loc[cur_date, "RET_STD_20"])
        scale = max(scale, 1e-6)

        for r_norm in norm_rets:
            r = float(r_norm) * scale
            cur_price *= np.exp(r)
            cur_date += BDay(1)

            prices.append(cur_price)
            dates.append(cur_date)

            new_row = df_roll.iloc[-1].copy()
            new_row["Close"] = cur_price
            df_roll.loc[cur_date] = new_row

            if len(prices) >= trading_days:
                break

    return pd.DataFrame({
        "date": dates,
        "Pred_Close": prices
    })

# ================= 🆕 日轉月（月底） =================
def daily_to_monthly(df_daily):
    return (
        df_daily
        .set_index("date")
        .resample("M")
        .last()
        .reset_index()
    )

# ================= Main =================
if __name__ == "__main__":

    TICKER = "8110.TW"
    COLLECTION = "NEW_stock_data_liteon"

    df = load_df_from_firestore(TICKER, COLLECTION)
    df = ensure_today_row(df)
    df = add_features(df)
    df = df.dropna()

    FEATURES = [
        "Close", "Open", "High", "Low",
        "Volume", "RSI", "MACD", "K", "D",
        "ATR_14", "HL_RANGE", "GAP", "VOL_REL"
    ]

    X, y_ret, y_dir, idx = create_sequences(df, FEATURES)

    split = int(len(X) * 0.85)
    X_tr, X_te = X[:split], X[split:]
    y_ret_tr, y_dir_tr = y_ret[:split], y_dir[:split]

    sx = MinMaxScaler()
    sx.fit(df.loc[:idx[split - 1], FEATURES])

    X_tr_s = sx.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)

    model = build_attention_lstm(
        (X_tr_s.shape[1], X_tr_s.shape[2]),
        steps=5
    )

    model.fit(
        X_tr_s,
        {"return": y_ret_tr, "direction": y_dir_tr},
        epochs=60,
        batch_size=16,
        verbose=2,
        callbacks=[EarlyStopping(patience=8, restore_best_weights=True)]
    )

    # ================= 🆕 半年預測輸出 =================
    half_daily = forecast_half_year_daily(
        model, df, FEATURES, sx,
        lookback=40, steps=5, months=6
    )

    os.makedirs("results", exist_ok=True)

    daily_csv = f"results/{datetime.now():%Y-%m-%d}_{TICKER}_6M_daily.csv"
    half_daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")

    half_monthly = daily_to_monthly(half_daily)
    monthly_csv = f"results/{datetime.now():%Y-%m-%d}_{TICKER}_6M_monthly.csv"
    half_monthly.to_csv(monthly_csv, index=False, encoding="utf-8-sig")

    print("✅ 半年預測（日 / 月）完成")
