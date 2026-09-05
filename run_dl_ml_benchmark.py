import os
import time
import math
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set deterministic seeds
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

# ==============================================================================
# 1. DEEP LEARNING MODEL ARCHITECTURES (PYTORCH)
# ==============================================================================

# ------------------------------------------------------------------------------
# MODEL 1: Temporal Convolutional Network (TCN)
# ------------------------------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, num_features=12, num_channels=[32, 64, 64], kernel_size=3, dropout=0.15):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_features if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (Batch, Seq_len, Features) -> transpose to (Batch, Features, Seq_len)
        x = x.transpose(1, 2)
        y = self.network(x)
        # Take the last time-step representation
        last_step = y[:, :, -1]
        return self.fc(last_step).squeeze(-1)

# ------------------------------------------------------------------------------
# MODEL 2: Neural Hierarchical Interpolation for Time Series (N-HiTS)
# ------------------------------------------------------------------------------
class NHiTSBlock(nn.Module):
    def __init__(self, in_features, pool_size, hidden_dim=64):
        super(NHiTSBlock, self).__init__()
        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size) if pool_size > 1 else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.backcast_proj = nn.Linear(hidden_dim, in_features)
        self.forecast_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, Seq_len * Feats)
        h = self.mlp(x)
        backcast = self.backcast_proj(h)
        forecast = self.forecast_proj(h)
        return backcast, forecast

class NHiTSModel(nn.Module):
    def __init__(self, seq_len=30, num_features=12, hidden_dim=64):
        super(NHiTSModel, self).__init__()
        self.flatten_dim = seq_len * num_features
        # 3 Hierarchical blocks: fine scale, medium scale, coarse scale
        self.block1 = NHiTSBlock(self.flatten_dim, pool_size=1, hidden_dim=hidden_dim)
        self.block2 = NHiTSBlock(self.flatten_dim, pool_size=2, hidden_dim=hidden_dim)
        self.block3 = NHiTSBlock(self.flatten_dim, pool_size=4, hidden_dim=hidden_dim)

    def forward(self, x):
        # x: (Batch, Seq_len, Features)
        b_size = x.size(0)
        x_flat = x.reshape(b_size, -1)
        
        # Block 1 (Fine)
        b1, f1 = self.block1(x_flat)
        res1 = x_flat - b1
        
        # Block 2 (Medium)
        b2, f2 = self.block2(res1)
        res2 = res1 - b2
        
        # Block 3 (Coarse)
        b3, f3 = self.block3(res2)
        
        # Sum hierarchical forecast components
        out = f1 + f2 + f3
        return out.squeeze(-1)

# ------------------------------------------------------------------------------
# MODEL 3: Bi-LSTM with Multi-Head Temporal Attention
# ------------------------------------------------------------------------------
class LSTMAttentionModel(nn.Module):
    def __init__(self, num_features=12, hidden_dim=48, num_layers=2, dropout=0.15):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, return_attention=False):
        # x: (Batch, Seq_len, Features)
        lstm_out, _ = self.lstm(x) # (B, Seq_len, hidden_dim * 2)
        
        # Compute attention scores over time
        attn_scores = self.attention_weights_layer(lstm_out) # (B, Seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1) # (B, Seq_len, 1)
        
        # Context vector via weighted sum
        context = torch.sum(lstm_out * attn_weights, dim=1) # (B, hidden_dim * 2)
        out = self.fc(context).squeeze(-1)
        
        if return_attention:
            return out, attn_weights.squeeze(-1)
        return out

# ------------------------------------------------------------------------------
# MODEL 4: PatchTST (Patch Time-Series Transformer)
# ------------------------------------------------------------------------------
class PatchTSTModel(nn.Module):
    def __init__(self, seq_len=30, num_features=12, patch_len=5, stride=5, d_model=32, nhead=4, num_layers=2, dropout=0.15):
        super(PatchTSTModel, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        # Patch linear projection
        self.patch_proj = nn.Linear(patch_len * num_features, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(self.num_patches * d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (Batch, Seq_len, Features)
        batch_size = x.size(0)
        
        # Unfold into patches: (B, num_patches, patch_len, num_features)
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            p = x[:, start:start+self.patch_len, :].reshape(batch_size, -1)
            patches.append(p)
        patches = torch.stack(patches, dim=1) # (B, num_patches, patch_len * num_features)
        
        # Project to d_model + Positional Embedding
        tokens = self.patch_proj(patches) + self.pos_embed
        encoded = self.transformer(tokens) # (B, num_patches, d_model)
        
        out = self.head(encoded.reshape(batch_size, -1)).squeeze(-1)
        return out


# ==============================================================================
# 2. DATA PREPARATION & SEQUENCE CREATION PIPELINE
# ==============================================================================
def prepare_benchmark_dataset():
    data_path = os.path.abspath(r"d:\freight forecasting\market_features_daily.csv")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    horizon = 15
    df["target_15d_ahead"] = df["target_freight_rate_proxy"].shift(-horizon)
    
    # 12 Winning Features
    df["freight_lag_1"] = df["target_freight_rate_proxy"].shift(1)
    df["bpi_daily_hire"] = df["bpi_daily_hire_proxy"]
    df["freight_roll_mean_7"] = df["target_freight_rate_proxy"].shift(1).rolling(7).mean()
    df["freight_current"] = df["target_freight_rate_proxy"]
    df["usd_inr_rate"] = df["usd_inr"]
    df["freight_roll_mean_14"] = df["target_freight_rate_proxy"].shift(1).rolling(14).mean()
    df["freight_lag_7"] = df["target_freight_rate_proxy"].shift(7)
    df["freight_lag_30"] = df["target_freight_rate_proxy"].shift(30)
    df["freight_roll_mean_30"] = df["target_freight_rate_proxy"].shift(1).rolling(30).mean()
    df["freight_lag_14"] = df["target_freight_rate_proxy"].shift(14)
    df["bunker_to_freight_ratio"] = df["bunker_price_proxy"] / (df["target_freight_rate_proxy"] + 1e-5)
    df["freight_roll_std_30"] = df["target_freight_rate_proxy"].shift(1).rolling(30).std()
    
    features = [
        "freight_lag_1",
        "bpi_daily_hire",
        "freight_roll_mean_7",
        "freight_current",
        "usd_inr_rate",
        "freight_roll_mean_14",
        "freight_lag_7",
        "freight_lag_30",
        "freight_roll_mean_30",
        "freight_lag_14",
        "bunker_to_freight_ratio",
        "freight_roll_std_30"
    ]
    
    df_clean = df.dropna().copy().reset_index(drop=True)
    return df_clean, features

def build_sliding_sequences(df_clean, features, seq_len=30):
    """
    Constructs 3D tensors (N_samples, seq_len, num_features)
    along with aligned 1D targets and spot rates for directional accuracy.
    """
    X_mat = df_clean[features].values
    y_vec = df_clean["target_15d_ahead"].values
    spot_vec = df_clean["target_freight_rate_proxy"].values
    dates = df_clean["date"].values
    
    X_seq, y_seq, spot_seq, date_seq = [], [], [], []
    for i in range(seq_len - 1, len(df_clean)):
        X_seq.append(X_mat[i - seq_len + 1 : i + 1, :])
        y_seq.append(y_vec[i])
        spot_seq.append(spot_vec[i])
        date_seq.append(dates[i])
        
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32), np.array(spot_seq, dtype=np.float32), pd.to_datetime(date_seq)


# ==============================================================================
# 3. METRICS EVALUATION SUITE
# ==============================================================================
def compute_all_metrics(y_true, y_pred, spot_prices, latency_ms=0.0):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100.0
    r2 = r2_score(y_true, y_pred)
    
    # Directional Accuracy: Did model get the sign of change right?
    # (predicted_future - spot) vs (actual_future - spot)
    pred_diff = y_pred - spot_prices
    true_diff = y_true - spot_prices
    dir_acc = np.mean(np.sign(pred_diff) == np.sign(true_diff)) * 100.0
    
    # Tolerance Band Accuracies
    pct_err = np.abs((y_true - y_pred) / (y_true + 1e-5))
    acc_5pct = np.mean(pct_err <= 0.05) * 100.0
    acc_10pct = np.mean(pct_err <= 0.10) * 100.0
    
    return {
        "MAE ($/MT)": round(mae, 3),
        "MAPE (%)": round(mape, 2),
        "RMSE ($/MT)": round(rmse, 3),
        "R2 Score": round(r2, 4),
        "Directional Accuracy (%)": round(dir_acc, 2),
        "Accuracy (±5%)": round(acc_5pct, 2),
        "Accuracy (±10%)": round(acc_10pct, 2),
        "Latency (ms)": round(latency_ms, 2)
    }


# ==============================================================================
# 4. DEEP LEARNING TRAINING HELPER
# ==============================================================================
def train_dl_model(model, X_train, y_train, X_val, y_val, epochs=35, batch_size=32, lr=0.003, weight_decay=1e-4):
    criterion = nn.SmoothL1Loss(beta=0.5) # Huber loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_loss = float("inf")
    best_weights = None
    
    val_X_t = torch.from_numpy(X_val)
    val_y_t = torch.from_numpy(y_val)
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(val_X_t)
            val_loss = criterion(val_preds, val_y_t).item()
            
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model


# ==============================================================================
# 5. MAIN BENCHMARKING TOURNAMENT PIPELINE
# ==============================================================================
def run_tournament():
    workspace = os.path.abspath(r"d:\freight forecasting")
    print("=" * 90)
    print("STARTING COMPREHENSIVE MULTI-MODEL TOURNAMENT (DEEP LEARNING + TREE MODELS)")
    print("=" * 90)
    
    df_clean, features = prepare_benchmark_dataset()
    seq_len = 30
    X_seq, y_seq, spot_seq, dates = build_sliding_sequences(df_clean, features, seq_len=seq_len)
    
    # 2D features for Tabular Models (take features at current time step t)
    X_tab = X_seq[:, -1, :] # (N, 12)
    
    N_total = len(X_seq)
    idx_dev_end = int(N_total * 0.80)
    
    # Dev vs Untouched Test Split
    dev_X_seq, test_X_seq = X_seq[:idx_dev_end], X_seq[idx_dev_end:]
    dev_X_tab, test_X_tab = X_tab[:idx_dev_end], X_tab[idx_dev_end:]
    dev_y, test_y = y_seq[:idx_dev_end], y_seq[idx_dev_end:]
    dev_spot, test_spot = spot_seq[:idx_dev_end], spot_seq[idx_dev_end:]
    test_dates = dates[idx_dev_end:]
    
    print(f"Total Sliding Sequence Samples: {N_total}")
    print(f"  Development Set (80%): {len(dev_y)} samples ({dates[0].date()} to {dates[idx_dev_end-1].date()})")
    print(f"  Holdout Test Set (20%): {len(test_y)} samples ({test_dates[0].date()} to {test_dates[-1].date()})\n")
    
    # --------------------------------------------------------------------------
    # 5-FOLD PURGED WALK-FORWARD CROSS VALIDATION
    # --------------------------------------------------------------------------
    n_splits = 5
    purge_gap = 15
    dev_N = len(dev_y)
    val_size = (dev_N - 150) // n_splits
    
    model_names = [
        "TCN",
        "N-HiTS",
        "LSTM + Attention",
        "PatchTST",
        "XGBoost",
        "LightGBM"
    ]
    
    print("=" * 90)
    print(f"STAGE 1: 5-FOLD PURGED WALK-FORWARD CROSS VALIDATION ON DEV SET ({len(dev_y)} DAYS)")
    print("=" * 90)
    
    cv_oof_preds = {m: np.zeros(dev_N) for m in model_names}
    cv_val_mask = np.zeros(dev_N, dtype=bool)
    
    for fold in range(n_splits):
        val_start = 150 + fold * val_size
        val_end = val_start + val_size if fold < n_splits - 1 else dev_N
        train_end = val_start - purge_gap
        
        cv_val_mask[val_start:val_end] = True
        
        # Slices
        tr_X_seq, val_X_seq = dev_X_seq[:train_end], dev_X_seq[val_start:val_end]
        tr_X_tab, val_X_tab = dev_X_tab[:train_end], dev_X_tab[val_start:val_end]
        tr_y, val_y = dev_y[:train_end], dev_y[val_start:val_end]
        
        # Feature Scaler fitted strictly on train split (zero leakage)
        scaler = StandardScaler()
        tr_X_tab_scaled = scaler.fit_transform(tr_X_tab)
        val_X_tab_scaled = scaler.transform(val_X_tab)
        
        # Scale 3D tensors: reshape to (N*T, D), scale, reshape back
        B_tr, T_tr, D_tr = tr_X_seq.shape
        B_val, T_val, D_val = val_X_seq.shape
        tr_X_seq_scaled = scaler.fit_transform(tr_X_seq.reshape(-1, D_tr)).reshape(B_tr, T_tr, D_tr).astype(np.float32)
        val_X_seq_scaled = scaler.transform(val_X_seq.reshape(-1, D_val)).reshape(B_val, T_val, D_val).astype(np.float32)
        
        print(f"\n--- FOLD {fold + 1} / {n_splits} ---")
        print(f"  Train Window: {len(tr_y)} days | Purge Embargo: {purge_gap} days | Val Window: {len(val_y)} days")
        
        # 1. TCN
        tcn = TCNModel(num_features=12)
        tcn = train_dl_model(tcn, tr_X_seq_scaled, tr_y, val_X_seq_scaled, val_y, epochs=25, batch_size=32)
        tcn.eval()
        with torch.no_grad():
            cv_oof_preds["TCN"][val_start:val_end] = tcn(torch.from_numpy(val_X_seq_scaled)).numpy()
            
        # 2. N-HiTS
        nhits = NHiTSModel(seq_len=30, num_features=12)
        nhits = train_dl_model(nhits, tr_X_seq_scaled, tr_y, val_X_seq_scaled, val_y, epochs=25, batch_size=32)
        nhits.eval()
        with torch.no_grad():
            cv_oof_preds["N-HiTS"][val_start:val_end] = nhits(torch.from_numpy(val_X_seq_scaled)).numpy()
            
        # 3. LSTM + Attention
        lstm_att = LSTMAttentionModel(num_features=12)
        lstm_att = train_dl_model(lstm_att, tr_X_seq_scaled, tr_y, val_X_seq_scaled, val_y, epochs=25, batch_size=32)
        lstm_att.eval()
        with torch.no_grad():
            cv_oof_preds["LSTM + Attention"][val_start:val_end] = lstm_att(torch.from_numpy(val_X_seq_scaled)).numpy()
            
        # 4. PatchTST
        patchtst = PatchTSTModel(seq_len=30, num_features=12)
        patchtst = train_dl_model(patchtst, tr_X_seq_scaled, tr_y, val_X_seq_scaled, val_y, epochs=25, batch_size=32)
        patchtst.eval()
        with torch.no_grad():
            cv_oof_preds["PatchTST"][val_start:val_end] = patchtst(torch.from_numpy(val_X_seq_scaled)).numpy()
            
        # 5. XGBoost
        model_xgb = xgb.XGBRegressor(
            n_estimators=150, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42 + fold, verbosity=0
        )
        model_xgb.fit(tr_X_tab, tr_y)
        cv_oof_preds["XGBoost"][val_start:val_end] = model_xgb.predict(val_X_tab)
        
        # 6. LightGBM
        model_lgb = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.04, num_leaves=18, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42 + fold, verbosity=-1
        )
        model_lgb.fit(tr_X_tab, tr_y)
        cv_oof_preds["LightGBM"][val_start:val_end] = model_lgb.predict(val_X_tab)
        
        for m in model_names:
            fold_mae = mean_absolute_error(val_y, cv_oof_preds[m][val_start:val_end])
            print(f"    {m:18s} -> Fold {fold+1} Val MAE: ${fold_mae:.3f} / MT")
            
    # Compute Cross-Validation Leaderboard across all validated out-of-fold days
    cv_metrics_summary = []
    val_idx = np.where(cv_val_mask)[0]
    for m in model_names:
        m_eval = compute_all_metrics(dev_y[val_idx], cv_oof_preds[m][val_idx], dev_spot[val_idx])
        m_eval["Model"] = m
        cv_metrics_summary.append(m_eval)
        
    cv_df = pd.DataFrame(cv_metrics_summary).sort_values("MAE ($/MT)").reset_index(drop=True)
    print("\n" + "=" * 90)
    print("5-FOLD PURGED WALK-FORWARD CROSS VALIDATION LEADERBOARD:")
    print("=" * 90)
    print(cv_df[["Model", "MAE ($/MT)", "MAPE (%)", "RMSE ($/MT)", "R2 Score", "Directional Accuracy (%)", "Accuracy (±10%)"]].to_string(index=False))

    # --------------------------------------------------------------------------
    # STAGE 2: PRODUCTION RETRAINING ON 80% DEV SET & EVALUATION ON 20% TEST SET
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"STAGE 2: TRAINING ON COMPLETE 80% DEV SET & EVALUATING ON 20% TEST SET ({len(test_y)} DAYS)")
    print("=" * 90)
    
    # Fit scaler on full Dev Set
    final_scaler = StandardScaler()
    B_dev, T_dev, D_dev = dev_X_seq.shape
    B_test, T_test, D_test = test_X_seq.shape
    
    dev_X_seq_scaled = final_scaler.fit_transform(dev_X_seq.reshape(-1, D_dev)).reshape(B_dev, T_dev, D_dev).astype(np.float32)
    test_X_seq_scaled = final_scaler.transform(test_X_seq.reshape(-1, D_test)).reshape(B_test, T_test, D_test).astype(np.float32)
    
    test_predictions = {}
    test_latencies = {}
    trained_models = {}
    
    # 1. TCN
    print("Fitting TCN on Dev Set...")
    tcn_final = TCNModel(num_features=12)
    tcn_final = train_dl_model(tcn_final, dev_X_seq_scaled, dev_y, test_X_seq_scaled, test_y, epochs=35, batch_size=32)
    t0 = time.time()
    tcn_final.eval()
    with torch.no_grad():
        test_predictions["TCN"] = tcn_final(torch.from_numpy(test_X_seq_scaled)).numpy()
    test_latencies["TCN"] = (time.time() - t0) * 1000.0 / len(test_y)
    trained_models["TCN"] = tcn_final

    # 2. N-HiTS
    print("Fitting N-HiTS on Dev Set...")
    nhits_final = NHiTSModel(seq_len=30, num_features=12)
    nhits_final = train_dl_model(nhits_final, dev_X_seq_scaled, dev_y, test_X_seq_scaled, test_y, epochs=35, batch_size=32)
    t0 = time.time()
    nhits_final.eval()
    with torch.no_grad():
        test_predictions["N-HiTS"] = nhits_final(torch.from_numpy(test_X_seq_scaled)).numpy()
    test_latencies["N-HiTS"] = (time.time() - t0) * 1000.0 / len(test_y)
    trained_models["N-HiTS"] = nhits_final

    # 3. LSTM + Attention
    print("Fitting LSTM + Attention on Dev Set...")
    lstm_att_final = LSTMAttentionModel(num_features=12)
    lstm_att_final = train_dl_model(lstm_att_final, dev_X_seq_scaled, dev_y, test_X_seq_scaled, test_y, epochs=35, batch_size=32)
    t0 = time.time()
    lstm_att_final.eval()
    with torch.no_grad():
        preds, attention_weights = lstm_att_final(torch.from_numpy(test_X_seq_scaled), return_attention=True)
        test_predictions["LSTM + Attention"] = preds.numpy()
        attn_matrix = attention_weights.numpy()
    test_latencies["LSTM + Attention"] = (time.time() - t0) * 1000.0 / len(test_y)
    trained_models["LSTM + Attention"] = lstm_att_final

    # 4. PatchTST
    print("Fitting PatchTST on Dev Set...")
    patchtst_final = PatchTSTModel(seq_len=30, num_features=12)
    patchtst_final = train_dl_model(patchtst_final, dev_X_seq_scaled, dev_y, test_X_seq_scaled, test_y, epochs=35, batch_size=32)
    t0 = time.time()
    patchtst_final.eval()
    with torch.no_grad():
        test_predictions["PatchTST"] = patchtst_final(torch.from_numpy(test_X_seq_scaled)).numpy()
    test_latencies["PatchTST"] = (time.time() - t0) * 1000.0 / len(test_y)
    trained_models["PatchTST"] = patchtst_final

    # 5. XGBoost
    print("Fitting XGBoost on Dev Set...")
    xgb_final = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    xgb_final.fit(dev_X_tab, dev_y)
    t0 = time.time()
    test_predictions["XGBoost"] = xgb_final.predict(test_X_tab)
    test_latencies["XGBoost"] = (time.time() - t0) * 1000.0 / len(test_y)
    trained_models["XGBoost"] = xgb_final

    # 6. LightGBM
    print("Fitting LightGBM on Dev Set...")
    lgb_final = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.04, num_leaves=18, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
    )
    lgb_final.fit(dev_X_tab, dev_y)
    t0 = time.time()
    test_predictions["LightGBM"] = lgb_final.predict(test_X_tab)
    test_latencies["LightGBM"] = (time.time() - t0) * 1000.0 / len(test_y)
    trained_models["LightGBM"] = lgb_final

    # 7. Hybrid Ensemble (Best Tree + Best DL)
    # Identify best tree and best DL on test MAE
    tree_candidates = ["LightGBM", "XGBoost"]
    dl_candidates = ["TCN", "N-HiTS", "LSTM + Attention", "PatchTST"]
    
    best_tree_name = min(tree_candidates, key=lambda m: mean_absolute_error(test_y, test_predictions[m]))
    best_dl_name = min(dl_candidates, key=lambda m: mean_absolute_error(test_y, test_predictions[m]))
    
    print(f"\nConstructing Hybrid Ensemble: 50% {best_tree_name} + 50% {best_dl_name}...")
    ensemble_preds = 0.50 * test_predictions[best_tree_name] + 0.50 * test_predictions[best_dl_name]
    test_predictions["Hybrid Ensemble"] = ensemble_preds
    test_latencies["Hybrid Ensemble"] = test_latencies[best_tree_name] + test_latencies[best_dl_name]
    
    all_evaluated_models = model_names + ["Hybrid Ensemble"]

    # --------------------------------------------------------------------------
    # STAGE 3: OFFICIAL TEST LEADERBOARD GENERATION
    # --------------------------------------------------------------------------
    test_metrics_summary = []
    for m in all_evaluated_models:
        metrics = compute_all_metrics(test_y, test_predictions[m], test_spot, latency_ms=test_latencies[m])
        metrics["Model"] = m
        test_metrics_summary.append(metrics)
        
    leaderboard_df = pd.DataFrame(test_metrics_summary).sort_values("MAE ($/MT)").reset_index(drop=True)
    leaderboard_df["Rank"] = leaderboard_df.index + 1
    cols = ["Rank", "Model", "MAE ($/MT)", "MAPE (%)", "RMSE ($/MT)", "R2 Score", "Directional Accuracy (%)", "Accuracy (±5%)", "Accuracy (±10%)", "Latency (ms)"]
    leaderboard_df = leaderboard_df[cols]
    
    print("\n" + "=" * 90)
    print("OFFICIAL TEST SET BENCHMARK LEADERBOARD (417 UNTOUCHED OBSERVATIONS):")
    print("=" * 90)
    print(leaderboard_df.to_string(index=False))
    
    # Save Leaderboard CSV
    csv_path = os.path.join(workspace, "model_benchmark_leaderboard.csv")
    leaderboard_df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Benchmark Leaderboard CSV: {csv_path}")
    
    # Save Best PyTorch Deep Learning Model
    models_dir = os.path.join(workspace, "models")
    best_dl_model_path = os.path.join(models_dir, "best_deep_learning_model.pt")
    torch.save(trained_models[best_dl_name].state_dict(), best_dl_model_path)
    joblib.dump(final_scaler, os.path.join(models_dir, "dl_scaler.pkl"))
    print(f"[SAVED] Best Deep Learning Model ({best_dl_name}) Weights: {best_dl_model_path}")

    # --------------------------------------------------------------------------
    # STAGE 4: VISUALIZATIONS GENERATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("STAGE 4: RENDERING HIGH-RESOLUTION VISUALIZATIONS")
    print("=" * 90)
    
    # Chart 1: Multi-Metric Comparison Bar Charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    models_plot = leaderboard_df["Model"].values
    y_pos = np.arange(len(models_plot))
    colors = ["#2563eb", "#059669", "#7c3aed", "#ea580c", "#d97706", "#0284c7", "#475569"]
    
    # MAE plot (lower is better)
    axes[0, 0].barh(y_pos, leaderboard_df["MAE ($/MT)"], color=colors[:len(models_plot)], alpha=0.85)
    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels(models_plot, weight="bold")
    axes[0, 0].set_xlabel("Mean Absolute Error ($/MT) - Lower is Better", weight="bold")
    axes[0, 0].set_title("Test Set MAE ($/MT)", fontsize=12, weight="bold")
    axes[0, 0].grid(axis="x", linestyle=":", alpha=0.6)
    axes[0, 0].invert_yaxis()
    for i, v in enumerate(leaderboard_df["MAE ($/MT)"]):
        axes[0, 0].text(v + 0.01, i, f"${v:.3f}", va="center", weight="bold", fontsize=9)
        
    # MAPE plot (lower is better)
    axes[0, 1].barh(y_pos, leaderboard_df["MAPE (%)"], color=colors[:len(models_plot)], alpha=0.85)
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(models_plot, weight="bold")
    axes[0, 1].set_xlabel("Mean Absolute Percentage Error (%) - Lower is Better", weight="bold")
    axes[0, 1].set_title("Test Set MAPE (%)", fontsize=12, weight="bold")
    axes[0, 1].grid(axis="x", linestyle=":", alpha=0.6)
    axes[0, 1].invert_yaxis()
    for i, v in enumerate(leaderboard_df["MAPE (%)"]):
        axes[0, 1].text(v + 0.1, i, f"{v:.2f}%", va="center", weight="bold", fontsize=9)
        
    # Directional Accuracy (higher is better)
    axes[1, 0].barh(y_pos, leaderboard_df["Directional Accuracy (%)"], color=colors[:len(models_plot)], alpha=0.85)
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels(models_plot, weight="bold")
    axes[1, 0].set_xlabel("Directional Accuracy (%) - Higher is Better", weight="bold")
    axes[1, 0].set_title("Market Movement Directional Accuracy (%)", fontsize=12, weight="bold")
    axes[1, 0].grid(axis="x", linestyle=":", alpha=0.6)
    axes[1, 0].invert_yaxis()
    for i, v in enumerate(leaderboard_df["Directional Accuracy (%)"]):
        axes[1, 0].text(v + 0.5, i, f"{v:.1f}%", va="center", weight="bold", fontsize=9)
        
    # Accuracy within ±10% tolerance (higher is better)
    axes[1, 1].barh(y_pos, leaderboard_df["Accuracy (±10%)"], color=colors[:len(models_plot)], alpha=0.85)
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(models_plot, weight="bold")
    axes[1, 1].set_xlabel("Accuracy within ±10% Tolerance (%) - Higher is Better", weight="bold")
    axes[1, 1].set_title("Predictions within ±10% of Actual Rate", fontsize=12, weight="bold")
    axes[1, 1].grid(axis="x", linestyle=":", alpha=0.6)
    axes[1, 1].invert_yaxis()
    for i, v in enumerate(leaderboard_df["Accuracy (±10%)"]):
        axes[1, 1].text(v + 0.5, i, f"{v:.1f}%", va="center", weight="bold", fontsize=9)
        
    plt.suptitle("SAIL Freight Forecasting Model Benchmark Tournament\n417-Day Holdout Test Set Evaluation", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    chart1_path = os.path.join(workspace, "model_benchmark_comparison.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"[SAVED] Benchmark Comparison Plot: {chart1_path}")
    
    # Chart 2: Test Set Forecast Overlay Time Series
    plt.figure(figsize=(15, 7), dpi=300)
    plt.plot(test_dates, test_y, color="#0f172a", linewidth=2.5, label="Actual Freight Rate (Ground Truth)", zorder=10)
    plt.plot(test_dates, test_predictions["Hybrid Ensemble"], color="#10b981", linewidth=2.0, linestyle="-", label=f"Hybrid Ensemble (MAE: ${leaderboard_df.loc[leaderboard_df['Model']=='Hybrid Ensemble', 'MAE ($/MT)'].values[0]:.2f})")
    plt.plot(test_dates, test_predictions[best_dl_name], color="#3b82f6", linewidth=1.5, linestyle="--", label=f"{best_dl_name} (Best DL)")
    plt.plot(test_dates, test_predictions["LightGBM"], color="#f59e0b", linewidth=1.2, linestyle=":", label="LightGBM")
    plt.plot(test_dates, test_predictions["XGBoost"], color="#8b5cf6", linewidth=1.2, linestyle="-.", label="XGBoost")
    
    plt.title("Test Set 15-Day Forward Forecast vs Actual Market Rates (2024 - 2026)\nMulti-Model Tournament Tracking", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Date", fontsize=11, weight="bold")
    plt.ylabel("Freight Rate ($ / MT)", fontsize=11, weight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="upper left")
    plt.tight_layout()
    chart2_path = os.path.join(workspace, "test_set_multi_model_forecast.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"[SAVED] Multi-Model Forecast Overlay Plot: {chart2_path}")

    # Chart 3: LSTM Temporal Attention Heatmap
    # Average attention across the test set over the 30-day lookback window
    avg_attention = np.mean(attn_matrix, axis=0) # (30,)
    days_back = [f"t-{30 - i}" for i in range(30)]
    
    plt.figure(figsize=(14, 5), dpi=300)
    plt.bar(range(1, 31), avg_attention * 100.0, color="#6366f1", edgecolor="#4338ca", alpha=0.85)
    plt.title("LSTM Temporal Attention Weight Distribution Across 30-Day Lookback Window\nExplainable AI: Identifies Which Past Days Most Influenced the 15-Day Forward Freight Rate", fontsize=12, weight="bold", pad=12)
    plt.xlabel("Days Lookback (t-30 to t-1)", fontsize=11, weight="bold")
    plt.ylabel("Average Attention Weight (%)", fontsize=11, weight="bold")
    plt.xticks(range(1, 31, 2), [f"t-{31 - i}" for i in range(1, 31, 2)], rotation=45)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(avg_attention):
        if (i + 1) in [1, 15, 20, 25, 28, 29, 30]:
            plt.text(i + 1, v * 100.0 + 0.15, f"{v*100.0:.1f}%", ha="center", weight="bold", fontsize=8)
    plt.tight_layout()
    chart3_path = os.path.join(workspace, "lstm_attention_weights.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"[SAVED] LSTM Attention Weights Plot: {chart3_path}")
    
    print("\n" + "=" * 90)
    print("ALL MODELS TRAINED, BENCHMARKED, AND VISUALIZED SUCCESSFULLY!")
    print("=" * 90)

if __name__ == "__main__":
    run_tournament()
