# -*- coding: utf-8 -*-
import os, json, random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

from sklearn.preprocessing import MinMaxScaler
import joblib

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    Softmax, Dot, Reshape, Permute
)
from tensorflow.keras.callbacks import EarlyStopping

from zoneinfo import ZoneInfo
now_tw = datetime.now(ZoneInfo("Asia/Taipei"))

# ================= Seed =================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ================= Firestore =================
import firebase_admin
from firebase_admin import credentials, firestore

key_dict = json.loads(os.environ.get("FIREBASE", "{}"))
db = None

if key_dict:
    cred = credentials.Certificate(key_dict)
    try:
        firebase_admin.get_app()
    except Exception:
        firebase_admin.initialize_app(cred)
    db = firestore.client()

# ================= Load =================
def load_df_from_firestore(ticker, collection="NEW_stock_data_liteon", days=500):
    rows = []
    if db:
        for doc in db.collection(collection).stream():
            p = doc.to_dict().get(ticker)
            if p:
                rows.append({"date": doc.id, **p})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("⚠️ 無資料")

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").tail(days).set_index("date")

# ================= Feature =================
def add_features(df):
    df["Volume"] = np.log1p(df["Volume"].astype(float))
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()
    return df

# ================= Sequence（🔥已修） =================
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

        # ✅ FIX: shape (1,)
        y_dir.append([1.0 if future_ret.sum() > 0 else 0.0])

        idx.append(df.index[i])

    return (
        np.array(X),
        np.array(y_ret),
        np.array(y_dir).reshape(-1, 1),  # ✅ FIX
        np.array(idx)
    )

# ================= Model（🔥完全穩定版） =================
def build_model(input_shape, steps, cap=0.06):
    inp = Input(shape=input_shape)

    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.2)(x)

    # ✅ SAFE attention（不會炸）
    score = Dense(1)(x)
    weights = Softmax(axis=1)(score)

    weights = Permute((2, 1))(weights)
    context = Dot(axes=(2, 1))([weights, x])
    context = Reshape((64,))(context)

    # return
    raw = Dense(steps, activation="tanh")(context)
    out_ret = raw * cap

    # direction
    out_dir = Dense(1, activation="sigmoid", name="direction")(context)

    model = Model(inp, [out_ret, out_dir])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(7e-4),
        loss=[tf.keras.losses.Huber(), "binary_crossentropy"],
        loss_weights=[1.0, 0.8],
        metrics={
            "direction": [
                tf.keras.metrics.BinaryAccuracy(name="acc")  # ✅ 移除 AUC
            ]
        }
    )
    return model

# ================= Main =================
if __name__ == "__main__":
    TICKER = "2408.TW"
    LOOKBACK = 40
    STEPS = 5

    df = load_df_from_firestore(TICKER)
    df = add_features(df)

    FEATURES = ["Close", "Volume", "RSI", "MACD", "K", "D", "ATR_14"]
    df = df.dropna(subset=FEATURES)

    X, y_ret, y_dir, idx = create_sequences(df, FEATURES, STEPS, LOOKBACK)

    if len(X) < 40:
        raise ValueError("資料太少")

    split = int(len(X) * 0.85)

    X_tr, X_te = X[:split], X[split:]
    y_ret_tr, y_ret_te = y_ret[:split], y_ret[split:]
    y_dir_tr, y_dir_te = y_dir[:split], y_dir[split:]

    # scaler
    sx = MinMaxScaler()
    sx.fit(X_tr.reshape(-1, X_tr.shape[-1]))

    def scale(Xb):
        n, t, f = Xb.shape
        return sx.transform(Xb.reshape(-1, f)).reshape(n, t, f)

    X_tr = scale(X_tr)
    X_te = scale(X_te)

    model = build_model((LOOKBACK, len(FEATURES)), STEPS)

    model.fit(
        X_tr,
        [y_ret_tr, y_dir_tr],
        epochs=60,
        batch_size=16,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=8, restore_best_weights=True)],
        verbose=2
    )

    pred_ret, pred_dir = model.predict(X_te)

    print(f"📈 上漲機率: {pred_dir[-1][0]:.2%}")
