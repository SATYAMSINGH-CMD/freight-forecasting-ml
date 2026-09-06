import os
import gc
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

# Safety: Cap PyTorch threads to avoid CPU starvation / laptop freezing
torch.set_num_threads(4)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

# Import model architectures from our verified benchmark suite
from run_dl_ml_benchmark import (
    TCNModel,
    UniLSTMAttentionModel,
    BiLSTMAttentionModel,
    PatchTSTModel,
    NHiTSModel,
    compute_all_metrics
)

# ------------------------------------------------------------------------------
# DATA PREPARATION WITH DYNAMIC HORIZON
# ------------------------------------------------------------------------------
def load_prepared_data(horizon=15):
    data_path = os.path.abspath(r"d:\freight forecasting\market_features_daily.csv")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    df["target_ahead"] = df["target_freight_rate_proxy"].shift(-horizon)
    
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
        "freight_lag_1", "bpi_daily_hire", "freight_roll_mean_7", "freight_current",
        "usd_inr_rate", "freight_roll_mean_14", "freight_lag_7", "freight_lag_30",
        "freight_roll_mean_30", "freight_lag_14", "bunker_to_freight_ratio", "freight_roll_std_30"
    ]
    
    df_clean = df.dropna().copy().reset_index(drop=True)
    
    seq_len = 30
    X_mat = df_clean[features].values
    y_vec = df_clean["target_ahead"].values
    spot_vec = df_clean["target_freight_rate_proxy"].values
    dates = df_clean["date"].values
    
    X_seq, y_seq, spot_seq, date_seq = [], [], [], []
    for i in range(seq_len - 1, len(df_clean)):
        X_seq.append(X_mat[i - seq_len + 1 : i + 1, :])
        y_seq.append(y_vec[i])
        spot_seq.append(spot_vec[i])
        date_seq.append(dates[i])
        
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        np.array(spot_seq, dtype=np.float32),
        pd.to_datetime(date_seq),
        features
    )

# ------------------------------------------------------------------------------
# EFFICIENT LIGHTWEIGHT PYTORCH TRAINER
# ------------------------------------------------------------------------------
def quick_train_dl(model, X_train, y_train, epochs=20, batch_size=32, lr=0.004):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.5)
    
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
    model.eval()
    return model

# ------------------------------------------------------------------------------
# UNIVERSAL MODEL EVALUATOR ON GIVEN SPLIT
# ------------------------------------------------------------------------------
CANDIDATE_MODELS = [
    "TCN",
    "Bi-LSTM + Attention",
    "Uni-LSTM + Attention",
    "PatchTST",
    "XGBoost",
    "LightGBM",
    "Hybrid Ensemble"
]

def fit_and_evaluate_split(tr_X_seq, tr_y, te_X_seq, te_y, te_spot):
    scaler = StandardScaler()
    B_tr, T_tr, D_tr = tr_X_seq.shape
    B_te, T_te, D_te = te_X_seq.shape
    
    tr_X_seq_scaled = scaler.fit_transform(tr_X_seq.reshape(-1, D_tr)).reshape(B_tr, T_tr, D_tr).astype(np.float32)
    te_X_seq_scaled = scaler.transform(te_X_seq.reshape(-1, D_te)).reshape(B_te, T_te, D_te).astype(np.float32)
    
    tr_X_tab = tr_X_seq[:, -1, :]
    te_X_tab = te_X_seq[:, -1, :]
    
    preds = {}
    
    # 1. TCN
    tcn = TCNModel(num_features=D_tr)
    tcn = quick_train_dl(tcn, tr_X_seq_scaled, tr_y, epochs=20)
    with torch.no_grad():
        preds["TCN"] = tcn(torch.from_numpy(te_X_seq_scaled)).numpy()
        
    # 2. Bi-LSTM + Attention
    bilstm = BiLSTMAttentionModel(num_features=D_tr)
    bilstm = quick_train_dl(bilstm, tr_X_seq_scaled, tr_y, epochs=20)
    with torch.no_grad():
        preds["Bi-LSTM + Attention"] = bilstm(torch.from_numpy(te_X_seq_scaled)).numpy()
        
    # 3. Uni-LSTM + Attention (Causal)
    unilstm = UniLSTMAttentionModel(num_features=D_tr)
    unilstm = quick_train_dl(unilstm, tr_X_seq_scaled, tr_y, epochs=20)
    with torch.no_grad():
        preds["Uni-LSTM + Attention"] = unilstm(torch.from_numpy(te_X_seq_scaled)).numpy()
        
    # 4. PatchTST
    patchtst = PatchTSTModel(seq_len=T_tr, num_features=D_tr)
    patchtst = quick_train_dl(patchtst, tr_X_seq_scaled, tr_y, epochs=20)
    with torch.no_grad():
        preds["PatchTST"] = patchtst(torch.from_numpy(te_X_seq_scaled)).numpy()
        
    # 5. XGBoost
    xgb_m = xgb.XGBRegressor(n_estimators=120, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb_m.fit(tr_X_tab, tr_y)
    preds["XGBoost"] = xgb_m.predict(te_X_tab)
    
    # 6. LightGBM
    lgb_m = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.04, num_leaves=18, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    lgb_m.fit(tr_X_tab, tr_y)
    preds["LightGBM"] = lgb_m.predict(te_X_tab)
    
    # 7. Hybrid Ensemble (50% TCN + 50% XGB)
    preds["Hybrid Ensemble"] = 0.50 * preds["TCN"] + 0.50 * preds["XGBoost"]
    
    results = {}
    for m in CANDIDATE_MODELS:
        m_eval = compute_all_metrics(te_y, preds[m], te_spot)
        m_eval["Max Error ($/MT)"] = round(float(np.max(np.abs(te_y - preds[m]))), 3)
        results[m] = m_eval
        
    gc.collect()
    return results, preds

# ==============================================================================
# 1. OPTION A: ROLLING-WINDOW SLIDING WALK-FORWARD (5 ERAS)
# ==============================================================================
def run_rolling_window_benchmark(X_seq, y_seq, spot_seq, dates):
    print("\n" + "=" * 95)
    print("RUNNING OPTION A: 5-ERA ROLLING-WINDOW SLIDING WALK-FORWARD CROSS-VALIDATION")
    print("Testing Adaptability Across 8 Years with Fixed 450-Day Lookback and 15-Day Embargo")
    print("=" * 95)
    
    train_size = 450
    purge_gap = 15
    test_size = 150
    
    windows = [
        {"name": "Window 1 (2018-2020: Pre-COVID Stable Market)", "tr_start": 0, "tr_end": 450, "te_start": 465, "te_end": 615},
        {"name": "Window 2 (2020-2021: Post-COVID Supercycle)", "tr_start": 350, "tr_end": 800, "te_start": 815, "te_end": 965},
        {"name": "Window 3 (2021-2022: Ukraine War & Bunker Spike)", "tr_start": 700, "tr_end": 1150, "te_start": 1165, "te_end": 1315},
        {"name": "Window 4 (2022-2024: Post-Spike / Red Sea Onset)", "tr_start": 1050, "tr_end": 1500, "te_start": 1515, "te_end": 1665},
        {"name": "Window 5 (2024-2026: Modern Normalization Era)", "tr_start": 1400, "tr_end": 1850, "te_start": 1865, "te_end": len(y_seq)},
    ]
    
    rolling_records = []
    window_names = []
    model_window_maes = {m: [] for m in CANDIDATE_MODELS}
    
    for w_idx, win in enumerate(windows):
        w_name = win["name"]
        window_names.append(f"W{w_idx+1}")
        tr_s, tr_e = win["tr_start"], win["tr_end"]
        te_s, te_e = win["te_start"], win["te_end"]
        
        d_tr_start, d_tr_end = dates[tr_s].date(), dates[tr_e-1].date()
        d_te_start, d_te_end = dates[te_s].date(), dates[te_e-1].date()
        
        print(f"\n--- {w_name} ---")
        print(f"  Training Window: {tr_e - tr_s} days ({d_tr_start} to {d_tr_end})")
        print(f"  Purge Gap: {purge_gap} days")
        print(f"  Test Window: {te_e - te_s} days ({d_te_start} to {d_te_end})")
        
        tr_X, te_X = X_seq[tr_s:tr_e], X_seq[te_s:te_e]
        tr_y_w, te_y_w = y_seq[tr_s:tr_e], y_seq[te_s:te_e]
        te_spot_w = spot_seq[te_s:te_e]
        
        split_res, _ = fit_and_evaluate_split(tr_X, tr_y_w, te_X, te_y_w, te_spot_w)
        
        for m in CANDIDATE_MODELS:
            res = split_res[m]
            model_window_maes[m].append(res["MAE ($/MT)"])
            rolling_records.append({
                "Window": win["name"],
                "Window Code": f"W{w_idx+1}",
                "Model": m,
                "MAE ($/MT)": res["MAE ($/MT)"],
                "MAPE (%)": res["MAPE (%)"],
                "RMSE ($/MT)": res["RMSE ($/MT)"],
                "Directional Accuracy (%)": res["Directional Accuracy (%)"],
                "Accuracy (±10%)": res["Accuracy (±10%)"],
                "Max Error ($/MT)": res["Max Error ($/MT)"]
            })
            print(f"    {m:22s} -> MAE: ${res['MAE ($/MT)']:.3f}/MT | MAPE: {res['MAPE (%)']:.2f}% | Dir. Acc: {res['Directional Accuracy (%)']:.1f}%")
            
    df_rolling = pd.DataFrame(rolling_records)
    
    # Aggregated Rolling Leaderboard
    agg_df = df_rolling.groupby("Model").agg({
        "MAE ($/MT)": ["mean", "std"],
        "MAPE (%)": "mean",
        "Directional Accuracy (%)": "mean",
        "Accuracy (±10%)": "mean",
        "Max Error ($/MT)": "max"
    }).reset_index()
    
    agg_df.columns = ["Model", "Mean MAE ($/MT)", "Std MAE ($/MT)", "Mean MAPE (%)", "Mean Dir Acc (%)", "Mean Acc (±10%)", "Worst Outlier ($/MT)"]
    agg_df = agg_df.sort_values("Mean MAE ($/MT)").reset_index(drop=True)
    agg_df["Rolling Rank"] = agg_df.index + 1
    
    print("\n" + "=" * 95)
    print("OPTION A: ROLLING-WINDOW WALK-FORWARD SUMMARY LEADERBOARD:")
    print("=" * 95)
    print(agg_df.to_string(index=False))
    
    return agg_df, df_rolling, model_window_maes, window_names

# ==============================================================================
# 2. OPTION B: MACRO-REGIME CRISIS STRESS BENCHMARK (4 REAL-WORLD CRISES)
# ==============================================================================
def run_crisis_stress_benchmark(X_seq, y_seq, spot_seq, dates):
    print("\n" + "=" * 95)
    print("RUNNING OPTION B: MACRO-REGIME CRISIS STRESS TESTING (4 GLOBAL SHOCKS)")
    print("Testing Model Robustness and Antifragility Under Extreme Market Turbulence")
    print("=" * 95)
    
    crises = [
        {
            "regime": "Crisis 1: COVID Demand Shock & Supercycle Rebound",
            "start": "2020-03-01",
            "end": "2021-12-31",
            "description": "Extreme collapse in manufacturing followed by massive bulk freight squeeze"
        },
        {
            "regime": "Crisis 2: Russia-Ukraine War & Global Bunker Spike",
            "start": "2022-01-01",
            "end": "2022-12-31",
            "description": "VLSFO bunker fuel crossed $1,000/MT; European sanctions redirected trade routes"
        },
        {
            "regime": "Crisis 3: Red Sea & Suez Canal Geopolitical Diversions",
            "start": "2023-10-01",
            "end": "2024-10-31",
            "description": "Cape of Good Hope rerouting, ton-mile surge, Cape vessel shortages"
        },
        {
            "regime": "Crisis 4: Modern Era Range-Bound Normalization",
            "start": "2024-11-01",
            "end": "2026-08-14",
            "description": "Stabilized post-inflation fleet capacity and modern corridor rates"
        }
    ]
    
    crisis_records = []
    crisis_short_names = []
    model_crisis_maes = {m: [] for m in CANDIDATE_MODELS}
    
    purge_days = 15
    
    for c_idx, crisis in enumerate(crises):
        c_name = crisis["regime"]
        crisis_short_names.append(f"Crisis {c_idx+1}")
        t_start = pd.to_datetime(crisis["start"])
        t_end = pd.to_datetime(crisis["end"])
        
        te_mask = (dates >= t_start) & (dates <= t_end)
        te_indices = np.where(te_mask)[0]
        
        if len(te_indices) < 20:
            continue
            
        # Train on all available history prior to test start minus 15-day purge
        cutoff_date = t_start - pd.Timedelta(days=purge_days)
        tr_mask = dates <= cutoff_date
        tr_indices = np.where(tr_mask)[0]
        
        print(f"\n--- {c_name} ---")
        print(f"  Context: {crisis['description']}")
        print(f"  Training Window: {len(tr_indices)} trading days (prior to {cutoff_date.date()})")
        print(f"  Purge Embargo: {purge_days} days")
        print(f"  Stress Test Window: {len(te_indices)} trading days ({dates[te_indices[0]].date()} to {dates[te_indices[-1]].date()})")
        
        tr_X, te_X = X_seq[tr_indices], X_seq[te_indices]
        tr_y_c, te_y_c = y_seq[tr_indices], y_seq[te_indices]
        te_spot_c = spot_seq[te_indices]
        
        split_res, _ = fit_and_evaluate_split(tr_X, tr_y_c, te_X, te_y_c, te_spot_c)
        
        for m in CANDIDATE_MODELS:
            res = split_res[m]
            model_crisis_maes[m].append(res["MAE ($/MT)"])
            crisis_records.append({
                "Crisis Regime": c_name,
                "Crisis Code": f"Crisis {c_idx+1}",
                "Model": m,
                "MAE ($/MT)": res["MAE ($/MT)"],
                "MAPE (%)": res["MAPE (%)"],
                "RMSE ($/MT)": res["RMSE ($/MT)"],
                "Directional Accuracy (%)": res["Directional Accuracy (%)"],
                "Accuracy (±10%)": res["Accuracy (±10%)"],
                "Max Outlier Miss ($/MT)": res["Max Error ($/MT)"]
            })
            print(f"    {m:22s} -> MAE: ${res['MAE ($/MT)']:.3f}/MT | MAPE: {res['MAPE (%)']:.2f}% | Max Miss: ${res['Max Error ($/MT)']:.2f}")
            
    df_crisis = pd.DataFrame(crisis_records)
    
    agg_crisis = df_crisis.groupby("Model").agg({
        "MAE ($/MT)": ["mean", "max"],
        "MAPE (%)": "mean",
        "Directional Accuracy (%)": "mean",
        "Accuracy (±10%)": "mean",
        "Max Outlier Miss ($/MT)": "max"
    }).reset_index()
    
    agg_crisis.columns = ["Model", "Mean Crisis MAE ($/MT)", "Worst Crisis MAE ($/MT)", "Mean Crisis MAPE (%)", "Mean Crisis Dir Acc (%)", "Mean Crisis Acc (±10%)", "Peak Worst Miss ($/MT)"]
    agg_crisis = agg_crisis.sort_values("Mean Crisis MAE ($/MT)").reset_index(drop=True)
    agg_crisis["Crisis Stress Rank"] = agg_crisis.index + 1
    
    print("\n" + "=" * 95)
    print("OPTION B: MACRO-REGIME CRISIS STRESS SUMMARY LEADERBOARD:")
    print("=" * 95)
    print(agg_crisis.to_string(index=False))
    
    return agg_crisis, df_crisis, model_crisis_maes, crisis_short_names

# ==============================================================================
# 3. OPTION C: MULTI-HORIZON STRESS TEST (7-DAY, 15-DAY, 30-DAY AHEAD)
# ==============================================================================
def run_multi_horizon_benchmark():
    print("\n" + "=" * 95)
    print("RUNNING OPTION C: MULTI-HORIZON FORECASTING STRESS TEST")
    print("Evaluating Model Accuracy for 7-Day Spot Prompt, 15-Day Voyage, and 30-Day Forward Hedging")
    print("=" * 95)
    
    horizons = [7, 15, 30]
    horizon_records = []
    model_horizon_maes = {m: [] for m in CANDIDATE_MODELS}
    
    for h in horizons:
        print(f"\n--- Evaluating Forecast Horizon: {h} Days Ahead ---")
        X_seq_h, y_seq_h, spot_seq_h, dates_h, _ = load_prepared_data(horizon=h)
        
        N_tot = len(X_seq_h)
        idx_split = int(N_tot * 0.80)
        
        tr_X, te_X = X_seq_h[:idx_split], X_seq_h[idx_split:]
        tr_y, te_y = y_seq_h[:idx_split], y_seq_h[idx_split:]
        te_spot = spot_seq_h[idx_split:]
        
        split_res, _ = fit_and_evaluate_split(tr_X, tr_y, te_X, te_y, te_spot)
        
        for m in CANDIDATE_MODELS:
            res = split_res[m]
            model_horizon_maes[m].append(res["MAE ($/MT)"])
            horizon_records.append({
                "Horizon Days": f"{h}d Ahead",
                "Horizon Int": h,
                "Model": m,
                "MAE ($/MT)": res["MAE ($/MT)"],
                "MAPE (%)": res["MAPE (%)"],
                "RMSE ($/MT)": res["RMSE ($/MT)"],
                "Directional Accuracy (%)": res["Directional Accuracy (%)"],
                "Accuracy (±10%)": res["Accuracy (±10%)"]
            })
            print(f"    {m:22s} ({h}d) -> MAE: ${res['MAE ($/MT)']:.3f}/MT | MAPE: {res['MAPE (%)']:.2f}% | Dir. Acc: {res['Directional Accuracy (%)']:.1f}%")
            
    df_horizons = pd.DataFrame(horizon_records)
    agg_horizons = df_horizons.groupby("Model").agg({
        "MAE ($/MT)": "mean",
        "MAPE (%)": "mean",
        "Directional Accuracy (%)": "mean",
        "Accuracy (±10%)": "mean"
    }).reset_index()
    agg_horizons.columns = ["Model", "Mean Multi-Horizon MAE ($/MT)", "Mean Multi-Horizon MAPE (%)", "Mean Multi-Horizon Dir Acc (%)", "Mean Multi-Horizon Acc (±10%)"]
    agg_horizons = agg_horizons.sort_values("Mean Multi-Horizon MAE ($/MT)").reset_index(drop=True)
    agg_horizons["Multi-Horizon Rank"] = agg_horizons.index + 1
    
    print("\n" + "=" * 95)
    print("OPTION C: MULTI-HORIZON FORECASTING LEADERBOARD (7d, 15d, 30d):")
    print("=" * 95)
    print(agg_horizons.to_string(index=False))
    
    return agg_horizons, df_horizons, model_horizon_maes

# ==============================================================================
# 4. SYNTHESIS: THE DEFINITIVE DECISION MATRIX
# ==============================================================================
def synthesize_final_decision(agg_rolling, agg_crisis, agg_horizons):
    workspace = os.path.abspath(r"d:\freight forecasting")
    holdout_csv = os.path.join(workspace, "model_benchmark_leaderboard.csv")
    
    df_holdout = pd.read_csv(holdout_csv)
    df_holdout = df_holdout[df_holdout["Model"].isin(CANDIDATE_MODELS)].copy()
    df_holdout_map = dict(zip(df_holdout["Model"], df_holdout["Rank"]))
    df_holdout_mae = dict(zip(df_holdout["Model"], df_holdout["MAE ($/MT)"]))
    
    matrix = []
    for m in CANDIDATE_MODELS:
        r_holdout = df_holdout_map.get(m, 4)
        mae_holdout = df_holdout_mae.get(m, 0.55)
        
        r_row = agg_rolling[agg_rolling["Model"] == m].iloc[0]
        r_rolling = int(r_row["Rolling Rank"])
        mae_rolling = float(r_row["Mean MAE ($/MT)"])
        
        c_row = agg_crisis[agg_crisis["Model"] == m].iloc[0]
        r_crisis = int(c_row["Crisis Stress Rank"])
        mae_crisis = float(c_row["Mean Crisis MAE ($/MT)"])
        peak_miss = float(c_row["Peak Worst Miss ($/MT)"])
        
        h_row = agg_horizons[agg_horizons["Model"] == m].iloc[0]
        r_horizon = int(h_row["Multi-Horizon Rank"])
        mae_horizon = float(h_row["Mean Multi-Horizon MAE ($/MT)"])
        
        # Composite Score: Weighted average rank (lower is better)
        # 30% Holdout + 30% Rolling Window + 25% Crisis Stress + 15% Multi-Horizon
        comp_rank = 0.30 * r_holdout + 0.30 * r_rolling + 0.25 * r_crisis + 0.15 * r_horizon
        
        matrix.append({
            "Model": m,
            "Holdout Rank": r_holdout,
            "Holdout MAE ($/MT)": mae_holdout,
            "Rolling Rank": r_rolling,
            "Rolling MAE ($/MT)": round(mae_rolling, 3),
            "Crisis Rank": r_crisis,
            "Crisis MAE ($/MT)": round(mae_crisis, 3),
            "Peak Miss ($/MT)": round(peak_miss, 2),
            "Multi-Horizon Rank": r_horizon,
            "Multi-Horizon MAE ($/MT)": round(mae_horizon, 3),
            "Composite Score": round(comp_rank, 2)
        })
        
    df_matrix = pd.DataFrame(matrix).sort_values("Composite Score").reset_index(drop=True)
    df_matrix["Definitive Final Rank"] = df_matrix.index + 1
    
    medals = ["[1] Champion", "[2] Runner-Up", "[3] Third Place", "[4]", "[5]", "[6]", "[7]"]
    df_matrix["Status"] = [medals[i] for i in range(len(df_matrix))]
    
    cols = [
        "Definitive Final Rank", "Status", "Model", "Composite Score",
        "Holdout Rank", "Rolling Rank", "Crisis Rank", "Multi-Horizon Rank",
        "Holdout MAE ($/MT)", "Rolling MAE ($/MT)", "Crisis MAE ($/MT)", "Peak Miss ($/MT)"
    ]
    df_matrix = df_matrix[cols]
    
    print("\n" + "=" * 110)
    print("THE DEFINITIVE MULTI-FOLD DECISION MATRIX (FINAL ARBITRATION ACROSS ALL SUITES):")
    print("=" * 110)
    print(df_matrix.to_string(index=False))
    
    out_csv = os.path.join(workspace, "final_decision_matrix.csv")
    df_matrix.to_csv(out_csv, index=False)
    print(f"\n[SAVED] Final Decision Matrix CSV: {out_csv}")
    
    return df_matrix

# ==============================================================================
# 5. HIGH-RESOLUTION PUBLICATION-GRADE VISUALIZATIONS
# ==============================================================================
def render_comprehensive_plots(df_rolling, df_crisis, df_horizons, df_matrix, model_win_maes, model_crisis_maes, model_horizon_maes):
    workspace = os.path.abspath(r"d:\freight forecasting")
    palette = {
        "TCN": "#2563eb",
        "Hybrid Ensemble": "#059669",
        "Bi-LSTM + Attention": "#7c3aed",
        "Uni-LSTM + Attention": "#db2777",
        "PatchTST": "#d97706",
        "XGBoost": "#ea580c",
        "LightGBM": "#0284c7"
    }
    
    # --------------------------------------------------------------------------
    # Chart 1: Rolling-Window Temporal Stability Curve
    # --------------------------------------------------------------------------
    plt.figure(figsize=(13, 6), dpi=300)
    x_axis = ["W1 (2018-20)", "W2 (2020-21)", "W3 (2021-22)", "W4 (2022-24)", "W5 (2024-26)"]
    for m in CANDIDATE_MODELS:
        lw = 2.8 if m in ["TCN", "Hybrid Ensemble"] else 1.5
        ls = "-" if m in ["TCN", "Hybrid Ensemble"] else ("--" if "LSTM" in m else ":")
        plt.plot(x_axis, model_win_maes[m], marker="o", linewidth=lw, linestyle=ls, color=palette[m], label=m)
        
    plt.title("Option A: Rolling-Window Walk-Forward Stability (5 Sliding Windows across 8 Years)\nLower and Flatter Lines Indicate Superior Adaptability to Structural Regime Shifts", fontsize=12, weight="bold", pad=12)
    plt.ylabel("Out-of-Sample MAE ($ / MT)", fontsize=11, weight="bold")
    plt.xlabel("Rolling Evaluation Eras", fontsize=11, weight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="upper right")
    plt.tight_layout()
    c1 = os.path.join(workspace, "rolling_window_trends.png")
    plt.savefig(c1, dpi=300)
    plt.close()
    print(f"[SAVED] Rolling Window Trends Plot: {c1}")
    
    # --------------------------------------------------------------------------
    # Chart 2: Crisis Stress Resilience (Grouped Bar Chart)
    # --------------------------------------------------------------------------
    plt.figure(figsize=(14, 7), dpi=300)
    n_crises = 4
    cr_labels = ["COVID Shock\n(2020-21)", "Ukraine War Fuel Spike\n(2022)", "Red Sea Rerouting\n(2023-24)", "Modern Normalization\n(2024-26)"]
    x = np.arange(n_crises)
    bar_width = 0.11
    
    for i, m in enumerate(CANDIDATE_MODELS):
        offset = (i - len(CANDIDATE_MODELS) / 2) * bar_width + (bar_width / 2)
        plt.bar(x + offset, model_crisis_maes[m], width=bar_width, color=palette[m], label=m, alpha=0.90)
        
    plt.title("Option B: Macro-Regime Crisis Stress Performance (4 Major Global Shocks)\nWhich Model Best Survives Global Black-Swan Events?", fontsize=13, weight="bold", pad=14)
    plt.xticks(x, cr_labels, fontsize=10, weight="bold")
    plt.ylabel("Stress Period MAE ($ / MT)", fontsize=11, weight="bold")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", ncol=4, loc="upper right")
    plt.tight_layout()
    c2 = os.path.join(workspace, "crisis_stress_comparison.png")
    plt.savefig(c2, dpi=300)
    plt.close()
    print(f"[SAVED] Crisis Stress Comparison Plot: {c2}")
    
    # --------------------------------------------------------------------------
    # Chart 3: Multi-Horizon Forecasting Curves (7d, 15d, 30d)
    # --------------------------------------------------------------------------
    plt.figure(figsize=(11, 6), dpi=300)
    h_axis = ["7 Days (Prompt)", "15 Days (Voyage)", "30 Days (COA / Hedging)"]
    for m in CANDIDATE_MODELS:
        lw = 2.8 if m in ["TCN", "Hybrid Ensemble"] else 1.5
        ls = "-" if m in ["TCN", "Hybrid Ensemble"] else ("--" if "LSTM" in m else ":")
        plt.plot(h_axis, model_horizon_maes[m], marker="s", linewidth=lw, linestyle=ls, color=palette[m], label=m)
        
    plt.title("Option C: Multi-Horizon Procurement Stress Curve (7d vs. 15d vs. 30d Ahead)\nEvaluates Operational Versatility Across Voyage Chartering Horizons", fontsize=12, weight="bold", pad=12)
    plt.ylabel("Test Set MAE ($ / MT)", fontsize=11, weight="bold")
    plt.xlabel("Procurement Decision Horizon", fontsize=11, weight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="upper left")
    plt.tight_layout()
    c3 = os.path.join(workspace, "multi_horizon_comparison.png")
    plt.savefig(c3, dpi=300)
    plt.close()
    print(f"[SAVED] Multi-Horizon Comparison Plot: {c3}")
    
    # --------------------------------------------------------------------------
    # Chart 4: Definitive Decision Matrix Heatmap
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.axis("off")
    
    table_data = []
    headers = ["Final Rank", "Award", "Model Name", "Composite Score", "Holdout MAE", "Rolling MAE", "Crisis MAE", "Peak Outlier Miss"]
    for _, row in df_matrix.iterrows():
        table_data.append([
            f"#{row['Definitive Final Rank']}",
            row["Status"],
            row["Model"],
            f"{row['Composite Score']:.2f}",
            f"${row['Holdout MAE ($/MT)']:.3f}",
            f"${row['Rolling MAE ($/MT)']:.3f}",
            f"${row['Crisis MAE ($/MT)']:.3f}",
            f"${row['Peak Miss ($/MT)']:.2f}"
        ])
        
    table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 2.0)
    
    # Style Header
    for col in range(len(headers)):
        table[(0, col)].set_facecolor("#1e293b")
        table[(0, col)].set_text_props(color="white", weight="bold")
        
    # Style First Row (Champion)
    for col in range(len(headers)):
        table[(1, col)].set_facecolor("#dbeafe")
        table[(1, col)].set_text_props(weight="bold")
        
    # Style Second Row (Runner up)
    for col in range(len(headers)):
        table[(2, col)].set_facecolor("#d1fae5")
        table[(2, col)].set_text_props(weight="bold")
        
    plt.title("SAIL Freight Forecasting Model Benchmark: Definitive Multi-Fold Decision Matrix\nArbitrated Across 4 Cross-Validation Schemes (Holdout, 5-Era Rolling, 4-Crisis Stress, 3-Horizon)", fontsize=13, weight="bold", pad=20)
    plt.tight_layout()
    c4 = os.path.join(workspace, "comprehensive_decision_matrix.png")
    plt.savefig(c4, dpi=300)
    plt.close()
    print(f"[SAVED] Decision Matrix Scorecard Plot: {c4}")

# ==============================================================================
# MAIN EXECUTION ORCHESTRATOR
# ==============================================================================
def main():
    print("=" * 100)
    print("STARTING FULL MULTI-FOLD BENCHMARKING SUITE ACROSS ALL PROPOSED PROTOCOLS")
    print("=" * 100)
    t_start_all = time.time()
    
    workspace = os.path.abspath(r"d:\freight forecasting")
    X_seq, y_seq, spot_seq, dates, features = load_prepared_data(horizon=15)
    
    # 1. Option A: Rolling-Window
    agg_rolling, df_rolling, model_win_maes, win_names = run_rolling_window_benchmark(X_seq, y_seq, spot_seq, dates)
    agg_rolling.to_csv(os.path.join(workspace, "rolling_window_leaderboard.csv"), index=False)
    
    # 2. Option B: Crisis Stress
    agg_crisis, df_crisis, model_crisis_maes, crisis_names = run_crisis_stress_benchmark(X_seq, y_seq, spot_seq, dates)
    agg_crisis.to_csv(os.path.join(workspace, "crisis_stress_leaderboard.csv"), index=False)
    
    # 3. Option C: Multi-Horizon
    agg_horizons, df_horizons, model_horizon_maes = run_multi_horizon_benchmark()
    agg_horizons.to_csv(os.path.join(workspace, "multi_horizon_leaderboard.csv"), index=False)
    
    # 4. Final Arbitration Decision Matrix
    df_matrix = synthesize_final_decision(agg_rolling, agg_crisis, agg_horizons)
    
    # 5. Visualizations
    render_comprehensive_plots(df_rolling, df_crisis, df_horizons, df_matrix, model_win_maes, model_crisis_maes, model_horizon_maes)
    
    t_elapsed = time.time() - t_start_all
    print("\n" + "=" * 100)
    print(f"ALL 4 CROSS-VALIDATION SUITES COMPLETED IN {t_elapsed:.1f} SECONDS (~{t_elapsed/60:.1f} MINS)")
    print("=" * 100)

if __name__ == "__main__":
    main()
