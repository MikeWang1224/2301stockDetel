# === 只貼「需要改的 create_sequences」完整版本 ===
def create_sequences(
    df, features,
    steps=5, window=40,
    trend_h=20,
    k_flat=0.8,
    eps=1e-9
):
    X, y_ret, y_dir, y_trend3, idx = [], [], [], [], []

    close = df["Close"].astype(float)
    logret = np.log(close).diff()

    feat = df[features].values
    max_h = max(steps, trend_h)

    for i in range(window, len(df) - max_h):
        x_seq = feat[i - window:i]
        if np.any(np.isnan(x_seq)):
            continue

        scale = df["RET_STD_20"].iloc[i - 1]
        if pd.isna(scale) or scale < eps:
            continue
        scale = float(scale) + eps

        # === return ===
        future_ret_raw_5d = logret.iloc[i:i + steps].values
        if np.any(np.isnan(future_ret_raw_5d)):
            continue

        future_ret_norm_5d = future_ret_raw_5d / scale

        # === direction (FIX HERE) ===
        dir_5d = 1.0 if future_ret_raw_5d.sum() > 0 else 0.0

        # 🔥 修正：一定要 (1,) shape
        y_dir.append([dir_5d])

        # === trend3 ===
        future_ret_raw_tr = logret.iloc[i:i + trend_h].values
        if np.any(np.isnan(future_ret_raw_tr)):
            continue

        cum = float(future_ret_raw_tr.sum())
        thr = float(k_flat) * scale * np.sqrt(float(trend_h))

        if cum > thr:
            cls = 2
        elif cum < -thr:
            cls = 0
        else:
            cls = 1

        onehot = np.zeros(3, dtype=np.float32)
        onehot[cls] = 1.0

        X.append(x_seq)
        y_ret.append(future_ret_norm_5d)
        y_trend3.append(onehot)
        idx.append(df.index[i])

    # 🔥 FINAL FIX（避免 TF 爆）
    return (
        np.array(X, dtype=np.float32),
        np.array(y_ret, dtype=np.float32),
        np.array(y_dir, dtype=np.float32).reshape(-1, 1),   # ✅ 關鍵
        np.array(y_trend3, dtype=np.float32),
        np.array(idx)
    )
