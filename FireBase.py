# -*- coding: utf-8 -*-
"""
最佳化版：2301.TW 多步 LSTM -> 預測未來 10 個交易日 Close，再計算 MA5/MA10
改動重點：
 - 精簡特徵 (Close, Volume, SMA_5, SMA_10, RSI, MACD)
 - window=30
 - model: LSTM(128, return_sequences=True) -> LSTM(64) -> Dense(32, relu) -> Dense(10)
 - scaler 僅以訓練集 fit，避免資料洩漏
 - 使用加權 MSE loss（可調權重，預設越遠期權重越大）
 - 保留 Firebase/Storage 與圖檔上傳、annotate
"""
import os, json
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------- Firebase 初始化（含 Storage） ----------------
key_dict = json.loads(os.environ.get("FIREBASE", "{}"))
db = None
bucket = None
storage_client = None

if key_dict:
    cred = credentials.Certificate(key_dict)
    try:
        firebase_admin.get_app()
    except Exception:
        firebase_admin.initialize_app(cred, {"storageBucket": f"{key_dict.get('project_id')}.appspot.com"})
    db = firestore.client()
    try:
        storage_client = storage.Client.from_service_account_info(key_dict)
        bucket = storage_client.bucket(f"{key_dict.get('project_id')}.appspot.com")
    except Exception as e:
        print("⚠️ Storage client 初始化失敗，Storage 功能停用:", e)
        bucket = None
else:
    print("⚠️ FIREBASE env 未設定 — 會略過上傳步驟")

# ---------------- 指標（SMA/RSI/MACD）與精簡特徵 ----------------
def add_basic_indicators(df):
    df = df.copy()
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()

    # RSI (14)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['SignalLine'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df = df.dropna()
    return df

# ---------------- 取得資料 ----------------
def fetch_and_prepare(ticker="2301.TW", period="18mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = add_basic_indicators(df)
    df = df.dropna()
    return df

# ---------------- 更新今天 Close 從 Firestore（若有） ----------------
def update_today_from_firestore(df):
    if db is None:
        return df
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        doc_ref = db.collection("NEW_stock_data_liteon").document(today_str)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict().get("2301.TW", {})
            if "Close" in data:
                try:
                    df.loc[pd.Timestamp(today_str), 'Close'] = float(data["Close"])
                except Exception:
                    pass
    except Exception:
        pass
    df = df.dropna()
    return df

# ---------------- 寫入股票資料回 Firestore（歷史資料） ----------------
def save_stock_data_to_firestore(df, ticker="2301.TW", collection_name="NEW_stock_data_liteon"):
    if db is None:
        print("⚠️ Firebase 未啟用，略過寫入股票資料")
        return
    batch = db.batch()
    count = 0
    try:
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d")
            try:
                payload = {
                    "Close": float(row["Close"]),
                    "Volume": float(row["Volume"]),
                    "MACD": float(row["MACD"]),
                    "RSI": float(row["RSI"])
                }
            except Exception:
                continue
            doc_ref = db.collection(collection_name).document(date_str)
            batch.set(doc_ref, {ticker: payload})
            count += 1
            if count >= 300:
                batch.commit()
                batch = db.batch()
                count = 0
        if count > 0:
            batch.commit()
        print(f"🔥 歷史股票資料已寫入 Firestore （collection: {collection_name}）")
    except Exception as e:
        print("❌ 寫入 Firestore 發生錯誤：", e)

# ---------------- 建資料集（精簡特徵） ----------------
def create_sequences(df, features, target_steps=10, window=30):
    X, y = [], []
    closes = df['Close'].values
    data = df[features].values
    for i in range(window, len(df) - target_steps + 1):
        X.append(data[i-window:i])
        y.append(closes[i:i+target_steps])
    return np.array(X), np.array(y)

# ---------------- 加權 loss（加權 MSE） ----------------
def weighted_mse_loss(weights):
    """
    weights: list/np.array of length output_steps, e.g. [0.1,0.1,...,0.2] (sum可不為1)
    返回一個可以給 model.compile 使用的 loss function
    """
    w = tf.constant(np.array(weights, dtype=np.float32))
    def loss(y_true, y_pred):
        # y_true/y_pred shape: (batch, steps)
        se = tf.square(y_true - y_pred)  # (batch, steps)
        weighted_se = se * w  # broadcast by steps
        return tf.reduce_mean(weighted_se)
    return loss

# ---------------- 建模型 ----------------
def build_lstm_multi_step(input_shape, output_steps=10):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_steps))
    return model

# ---------------- MA 計算 ----------------
def compute_pred_ma_from_pred_closes(last_known_closes, pred_closes):
    closes_seq = list(last_known_closes)
    results = []
    for pc in pred_closes:
        closes_seq.append(pc)
        ma5 = np.mean(closes_seq[-5:]) if len(closes_seq) >= 5 else np.mean(closes_seq)
        ma10 = np.mean(closes_seq[-10:]) if len(closes_seq) >= 10 else np.mean(closes_seq)
        results.append((pc, ma5, ma10))
    return results

# ---------------- 繪圖 + 上傳 Storage ----------------
def plot_and_upload_to_storage(df_real, df_future, bucket_obj=None):
    df_real_plot = df_real.copy().tail(10)
    if df_real_plot.empty:
        print("⚠️ df_real_plot 為空，無法繪圖")
        return None
    df_future = df_future.copy().reset_index(drop=True)
    last_hist_date = df_real_plot.index[-1]
    start_row = {
        "date": last_hist_date,
        "Pred_Close": df_real_plot['Close'].iloc[-1],
        "Pred_MA5": df_real_plot['SMA_5'].iloc[-1] if 'SMA_5' in df_real_plot.columns else df_real_plot['Close'].iloc[-1],
        "Pred_MA10": df_real_plot['SMA_10'].iloc[-1] if 'SMA_10' in df_real_plot.columns else df_real_plot['Close'].iloc[-1]
    }
    df_future_plot = pd.concat([pd.DataFrame([start_row]), df_future], ignore_index=True)

    plt.figure(figsize=(14,7))
    x_real = list(range(len(df_real_plot)))
    plt.plot(x_real, df_real_plot['Close'].values, label="Close")
    if 'SMA_5' in df_real_plot.columns:
        plt.plot(x_real, df_real_plot['SMA_5'].values, label="SMA5")
    if 'SMA_10' in df_real_plot.columns:
        plt.plot(x_real, df_real_plot['SMA_10'].values, label="SMA10")

    offset = len(df_real_plot) - 1
    x_future = [offset + i for i in range(len(df_future_plot))]
    plt.plot(x_future, df_future_plot['Pred_Close'].values, linestyle=':', marker='o', color='red', label="Pred Close")

    for xf, val in zip(x_future, df_future_plot['Pred_Close'].values):
        plt.annotate(f"{val:.2f}", xy=(xf, val), xytext=(6,6), textcoords='offset points', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    plt.plot(x_future, df_future_plot['Pred_MA5'].values, linestyle='--', label="Pred MA5")
    plt.plot(x_future, df_future_plot['Pred_MA10'].values, linestyle='--', label="Pred MA10")

    labels = [d.strftime('%m-%d') for d in df_real_plot.index[:-1]] + [d.strftime('%m-%d') for d in df_future_plot['date']]
    ticks = list(range(len(labels)))
    plt.xticks(ticks, labels, rotation=45)
    plt.grid(True, alpha=0.25)
    plt.title("2301.TW 歷史 + 預測（近 10 日 + 未來 10 日）")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")

    os.makedirs("results", exist_ok=True)
    file_name = f"{datetime.now().strftime('%Y-%m-%d')}_future_trade_days.png"
    file_path = os.path.join("results", file_name)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("📌 圖片已儲存：", file_path)

    if bucket_obj is not None:
        try:
            blob = bucket_obj.blob(f"LSTM_Pred_Images/{file_name}")
            blob.upload_from_filename(file_path)
            try:
                blob.make_public()
            except Exception:
                pass
            public_url = getattr(blob, 'public_url', None)
            print("🔥 圖片已上傳至 Storage：", public_url)
            return public_url
        except Exception as e:
            print("❌ 上傳 Storage 發生錯誤：", e)
            return None
    return None

# ---------------- Baseline/MA Helper ----------------
def compute_metrics(y_true, y_pred):
    maes = []
    rmses = []
    for step in range(y_true.shape[1]):
        maes.append(mean_absolute_error(y_true[:, step], y_pred[:, step]))
        rmses.append(math.sqrt(mean_squared_error(y_true[:, step], y_pred[:, step])))
    return np.array(maes), np.array(rmses)

def compute_ma_from_predictions(last_known_window_closes, y_pred_matrix, ma_period=5):
    n_samples, window = last_known_window_closes.shape
    steps = y_pred_matrix.shape[1]
    preds_ma = np.zeros((n_samples, steps))
    for i in range(n_samples):
        seq = list(last_known_window_closes[i])
        for t in range(steps):
            seq.append(y_pred_matrix[i, t])
            look = seq[-ma_period:] if len(seq) >= ma_period else seq
            preds_ma[i, t] = np.mean(look)
    return preds_ma

def compute_true_ma(last_window, y_true, ma_period=5):
    n_samples, window = last_window.shape
    steps = y_true.shape[1]
    true_ma = np.zeros((n_samples, steps))
    for i in range(n_samples):
        seq = list(last_window[i])
        for t in range(steps):
            seq.append(y_true[i, t])
            look = seq[-ma_period:] if len(seq) >= ma_period else seq
            true_ma[i, t] = np.mean(look)
    return true_ma

# ---------------- 主流程 ----------------
if __name__ == "__main__":
    TICKER = "2301.TW"
    LOOKBACK = 30
    PRED_STEPS = 10
    PERIOD = "18mo"
    TEST_RATIO = 0.15

    # 1) 讀資料
    df = fetch_and_prepare(ticker=TICKER, period=PERIOD)

    # 2) 若 Firestore 有今天 close，則更新
    df = update_today_from_firestore(df)

    # 3) 寫入歷史（可選）
    save_stock_data_to_firestore(df, ticker=TICKER)

    # 4) 選特徵：精簡版
    features = ['Close', 'Volume', 'SMA_5', 'SMA_10', 'RSI', 'MACD']
    df_features = df[features].dropna()

    # 5) 建序列
    X, y = create_sequences(df_features, features, target_steps=PRED_STEPS, window=LOOKBACK)
    print("X shape:", X.shape, "y shape:", y.shape)

    # 6) 時序切分（train / test）
    n = len(X)
    test_n = int(n * TEST_RATIO)
    split_idx = n - test_n
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 7) scaler：只在 train 上 fit（避免洩漏）
    nsamples, tw, nfeatures = X_train.shape
    scaler_x = MinMaxScaler()
    scaler_x.fit(X_train.reshape((nsamples*tw, nfeatures)))

    def scale_X(X_raw):
        s = X_raw.reshape((-1, X_raw.shape[-1]))
        return scaler_x.transform(s).reshape(X_raw.shape)

    X_train_s = scale_X(X_train)
    X_test_s = scale_X(X_test)

    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train)  # fit on train only
    y_train_s = scaler_y.transform(y_train)
    y_test_s = scaler_y.transform(y_test)  # 只是為了驗證時比較（注意：這是假設 scaler_y 是根據 train fit 的）

    # 8) 建模型與 compile（使用 weighted loss）
    model = build_lstm_multi_step(input_shape=(LOOKBACK, nfeatures), output_steps=PRED_STEPS)
    # 權重設計：預設把遠期權重略微提高（可調）
    steps = PRED_STEPS
    # example: 線性增加權重 (1,...,steps)
    raw_weights = np.arange(1, steps+1, dtype=np.float32)
    # normalize to sum=steps (或不正規化都可以)，這裡不做 normalize 也沒關係，只相對影響 loss scale
    loss_fn = weighted_mse_loss(raw_weights)
    model.compile(optimizer='adam', loss=loss_fn)
    model.summary()

    # callbacks
    os.makedirs("models", exist_ok=True)
    ckpt_path = f"models/{TICKER}_best.h5"
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    mc = ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)

    # 9) 訓練
    history = model.fit(X_train_s, y_train_s, validation_data=(X_test_s, y_test_s),
                        epochs=80, batch_size=32, callbacks=[es, mc], verbose=2)

    # 10) 預測並 inverse scale
    pred_s = model.predict(X_test_s)
    pred = scaler_y.inverse_transform(pred_s)

    # 評估（每步 MAE / RMSE）
    maes, rmses = [], []
    for step in range(PRED_STEPS):
        y_true_step = y_test[:, step]
        y_pred_step = pred[:, step]
        maes.append(mean_absolute_error(y_true_step, y_pred_step))
        rmses.append(math.sqrt(mean_squared_error(y_true_step, y_pred_step)))
    print("MAE per step (model):", np.round(maes, 4))
    print("RMSE per step (model):", np.round(rmses, 4))
    print("Avg MAE (model):", np.round(np.mean(maes), 4))

    # 11) 取最後一個測試 sample 的已知 closes 作為起始，計算未來 Pred MA
    last_known_window = X_test[-1]           # shape (LOOKBACK, nfeatures)
    last_known_closes = list(last_known_window[:, 0])  # Close 在 features index 0
    results = compute_pred_ma_from_pred_closes(last_known_closes, pred[-1])

    # 12) 建立未來交易日日期
    today = pd.Timestamp(datetime.now().date())
    first_bday = (today + BDay(1)).date()
    business_days = pd.bdate_range(start=first_bday, periods=PRED_STEPS)
    df_future = pd.DataFrame({
        "date": business_days,
        "Pred_Close": [r[0] for r in results],
        "Pred_MA5": [r[1] for r in results],
        "Pred_MA10": [r[2] for r in results]
    })

    # 13) 繪圖並上傳
    image_url = plot_and_upload_to_storage(df, df_future, bucket_obj=bucket)
    print("Image URL:", image_url)
    print(df_future)

    # 14) Baseline 評估（保留你原本流程）
    print("\n===== Baseline 評估開始 =====")
    last_known_closes_all = X_test[:, -1, 0]
    baselineA = np.vstack([last_known_closes_all for _ in range(pred.shape[1])]).T
    try:
        if 'SMA_5' in df.columns and not df['SMA_5'].dropna().empty:
            last_sma5_val = df['SMA_5'].dropna().iloc[-1]
            last_known_sma5_all = np.array([last_sma5_val] * X_test.shape[0])
            baselineB = np.vstack([last_known_sma5_all for _ in range(pred.shape[1])]).T
        else:
            baselineB = baselineA.copy()
    except Exception:
        baselineB = baselineA.copy()

    last_ret_1 = X_test[:, -1, features.index('Close')] if 'RET_1' not in features else None
    # baselineC fallback to A if no RET_1
    baselineC = baselineA.copy()

    maes_model, rmses_model = compute_metrics(y_test, pred)
    maes_bA, rmses_bA = compute_metrics(y_test, baselineA)
    maes_bB, rmses_bB = compute_metrics(y_test, baselineB)
    maes_bC, rmses_bC = compute_metrics(y_test, baselineC)

    print("=== Per-step MAE (model) ===\n", np.round(maes_model, 4))
    print("=== Per-step RMSE (model) ===\n", np.round(rmses_model, 4))
    print("=== Per-step MAE (Baseline A: last close) ===\n", np.round(maes_bA, 4))
    print("\nAvg MAE model:", np.round(maes_model.mean(), 4),
          "baselineA:", np.round(maes_bA.mean(), 4))

    # Evaluate MA5/MA10 errors
    last_closes_window = X_test[:, :, 0]
    model_MA5 = compute_ma_from_predictions(last_closes_window, pred, ma_period=5)
    model_MA10 = compute_ma_from_predictions(last_closes_window, pred, ma_period=10)
    true_MA5 = compute_true_ma(last_closes_window, y_test, ma_period=5)
    true_MA10 = compute_true_ma(last_closes_window, y_test, ma_period=10)

    mae_model_MA5 = np.mean(np.abs(model_MA5 - true_MA5))
    mae_model_MA10 = np.mean(np.abs(model_MA10 - true_MA10))
    print("\nMAE on derived MA5 -> model:", np.round(mae_model_MA5, 4))
    print("MAE on derived MA10 -> model:", np.round(mae_model_MA10, 4))
    print("===== Baseline 評估結束 =====\n")

    # 15) 寫入預測到 Firestore（如啟用）
    if db is not None:
        for i, row in df_future.iterrows():
            try:
                db.collection("NEW_stock_data_liteon_preds").document(row['date'].strftime("%Y-%m-%d")).set({
                    TICKER: {
                        "Pred_Close": float(row['Pred_Close']),
                        "Pred_MA5": float(row['Pred_MA5']),
                        "Pred_MA10": float(row['Pred_MA10'])
                    }
                })
            except Exception as e:
                print("寫入預測到 Firestore 發生錯誤：", e)
        try:
            pred_table_serialized = []
            for _, r in df_future.reset_index(drop=True).iterrows():
                rec = {
                    "date": pd.Timestamp(r['date']).strftime("%Y-%m-%d"),
                    "Pred_Close": float(r['Pred_Close']),
                    "Pred_MA5": float(r['Pred_MA5']),
                    "Pred_MA10": float(r['Pred_MA10'])
                }
                pred_table_serialized.append(rec)
            meta_doc = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "image_url": image_url,
                "pred_table": pred_table_serialized,
                "update_time": datetime.now().isoformat()
            }
            db.collection("NEW_stock_data_liteon_preds_meta").document(datetime.now().strftime("%Y-%m-%d")).set(meta_doc)
        except Exception as e:
            print("寫入預測 metadata 到 Firestore 發生錯誤：", e)
        print("🔥 預測寫入 Firestore 完成")
