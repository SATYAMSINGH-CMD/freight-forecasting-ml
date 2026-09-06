import os
import gc
import sys
import time
import math
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure console compatibility and thread safety (Prevents laptop freeze)
sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(4)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

# Import verified model architectures and utilities
from run_dl_ml_benchmark import (
    TCNModel,
    PatchTSTModel,
    UniLSTMAttentionModel,
    BiLSTMAttentionModel,
    prepare_benchmark_dataset,
    build_sliding_sequences,
    compute_all_metrics
)

def train_quick_dl(model, X_train, y_train, epochs=20, batch_size=32, lr=0.004):
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

EVALUATED_MODELS = [
    "LightGBM (Solo Baseline)",
    "Hybrid: LightGBM + TCN",
    "Hybrid: LightGBM + PatchTST",
    "Hybrid: LightGBM + Bi-LSTM",
    "Hybrid: LightGBM + Uni-LSTM",
    "TCN (Solo Causal)",
    "PatchTST (Solo Transformer)",
    "Bi-LSTM (Solo Attention)",
    "Uni-LSTM (Solo Attention)"
]

def evaluate_split_hybrids(tr_X_seq, tr_y, te_X_seq, te_y, te_spot):
    scaler = StandardScaler()
    B_tr, T_tr, D_tr = tr_X_seq.shape
    B_te, T_te, D_te = te_X_seq.shape
    
    tr_X_scaled = scaler.fit_transform(tr_X_seq.reshape(-1, D_tr)).reshape(B_tr, T_tr, D_tr).astype(np.float32)
    te_X_scaled = scaler.transform(te_X_seq.reshape(-1, D_te)).reshape(B_te, T_te, D_te).astype(np.float32)
    
    tr_X_tab = tr_X_seq[:, -1, :]
    te_X_tab = te_X_seq[:, -1, :]
    
    raw_preds = {}
    
    # 1. Base LightGBM
    lgb_m = lgb.LGBMRegressor(
        n_estimators=120,
        learning_rate=0.04,
        num_leaves=18,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    lgb_m.fit(tr_X_tab, tr_y)
    raw_preds["lgb"] = lgb_m.predict(te_X_tab)
    
    # 2. Base TCN
    tcn = TCNModel(num_features=D_tr)
    tcn = train_quick_dl(tcn, tr_X_scaled, tr_y, epochs=20)
    with torch.no_grad():
        raw_preds["tcn"] = tcn(torch.from_numpy(te_X_scaled)).numpy()
        
    # 3. Base PatchTST
    patchtst = PatchTSTModel(seq_len=T_tr, num_features=D_tr)
    patchtst = train_quick_dl(patchtst, tr_X_scaled, tr_y, epochs=20)
    with torch.no_grad():
        raw_preds["patchtst"] = patchtst(torch.from_numpy(te_X_scaled)).numpy()
        
    # 4. Base Uni-LSTM
    unilstm = UniLSTMAttentionModel(num_features=D_tr)
    unilstm = train_quick_dl(unilstm, tr_X_scaled, tr_y, epochs=20)
    with torch.no_grad():
        raw_preds["unilstm"] = unilstm(torch.from_numpy(te_X_scaled)).numpy()
        
    # 5. Base Bi-LSTM
    bilstm = BiLSTMAttentionModel(num_features=D_tr)
    bilstm = train_quick_dl(bilstm, tr_X_scaled, tr_y, epochs=20)
    with torch.no_grad():
        raw_preds["bilstm"] = bilstm(torch.from_numpy(te_X_scaled)).numpy()
        
    # Construct Predictions Dict
    preds = {
        "LightGBM (Solo Baseline)": raw_preds["lgb"],
        "Hybrid: LightGBM + TCN": 0.50 * raw_preds["lgb"] + 0.50 * raw_preds["tcn"],
        "Hybrid: LightGBM + PatchTST": 0.50 * raw_preds["lgb"] + 0.50 * raw_preds["patchtst"],
        "Hybrid: LightGBM + Bi-LSTM": 0.50 * raw_preds["lgb"] + 0.50 * raw_preds["bilstm"],
        "Hybrid: LightGBM + Uni-LSTM": 0.50 * raw_preds["lgb"] + 0.50 * raw_preds["unilstm"],
        "TCN (Solo Causal)": raw_preds["tcn"],
        "PatchTST (Solo Transformer)": raw_preds["patchtst"],
        "Bi-LSTM (Solo Attention)": raw_preds["bilstm"],
        "Uni-LSTM (Solo Attention)": raw_preds["unilstm"]
    }
    
    res = {}
    for m in EVALUATED_MODELS:
        m_eval = compute_all_metrics(te_y, preds[m], te_spot)
        m_eval["Max Outlier Miss ($/MT)"] = round(float(np.max(np.abs(te_y - preds[m]))), 3)
        res[m] = m_eval
        
    gc.collect()
    return res

# ==============================================================================
# SCHEME 1: 3-STAGE EXPANDING ANCHOR WALK-FORWARD
# ==============================================================================
def run_scheme_1(X_seq, y_seq, spot_seq, dates):
    print("\n" + "=" * 95)
    print("SCHEME 1: 3-STAGE EXPANDING ANCHOR WALK-FORWARD")
    print("=" * 95)
    
    splits = [
        {"name": "Fold 1", "tr_end": 1026, "purge": 15, "te_start": 1041, "te_end": 1378},
        {"name": "Fold 2", "tr_end": 1378, "purge": 15, "te_start": 1393, "te_end": 1715},
        {"name": "Fold 3", "tr_end": 1715, "purge": 15, "te_start": 1730, "te_end": len(y_seq)},
    ]
    
    fold_records = []
    for s in splits:
        tr_X, tr_y = X_seq[:s["tr_end"]], y_seq[:s["tr_end"]]
        te_X, te_y = X_seq[s["te_start"]:s["te_end"]], y_seq[s["te_start"]:s["te_end"]]
        te_spot = spot_seq[s["te_start"]:s["te_end"]]
        
        split_res = evaluate_split_hybrids(tr_X, tr_y, te_X, te_y, te_spot)
        for m, met in split_res.items():
            fold_records.append({
                "Scheme": "Scheme 1 (Anchor Walk-Forward)",
                "Fold": s["name"],
                "Model": m,
                "MAE ($/MT)": met["MAE ($/MT)"],
                "MAPE (%)": met["MAPE (%)"],
                "RMSE ($/MT)": met["RMSE ($/MT)"],
                "Directional Accuracy (%)": met["Directional Accuracy (%)"],
                "Accuracy (±10%)": met["Accuracy (±10%)"],
                "Max Outlier Miss ($/MT)": met["Max Outlier Miss ($/MT)"]
            })
            
    df_s1 = pd.DataFrame(fold_records)
    summary_s1 = df_s1.groupby("Model")["MAE ($/MT)"].mean().to_dict()
    print("Scheme 1 Finished.")
    return summary_s1, df_s1

# ==============================================================================
# SCHEME 2: 3 STRUCTURAL MACRO-REGIME SHOCK FOLDS
# ==============================================================================
def run_scheme_2(X_seq, y_seq, spot_seq, dates):
    print("\n" + "=" * 95)
    print("SCHEME 2: 3 STRUCTURAL MACRO-REGIME SHOCK FOLDS")
    print("=" * 95)
    
    splits = [
        {"name": "Regime 1 (Commodity Supercycle)", "tr_end": 800,  "purge": 15, "te_start": 815,  "te_end": 1050},
        {"name": "Regime 2 (Ukraine War Shock)",    "tr_end": 1100, "purge": 15, "te_start": 1115, "te_end": 1475},
        {"name": "Regime 3 (Red Sea Crisis)",       "tr_end": 1600, "purge": 15, "te_start": 1615, "te_end": len(y_seq)},
    ]
    
    fold_records = []
    for s in splits:
        tr_X, tr_y = X_seq[:s["tr_end"]], y_seq[:s["tr_end"]]
        te_X, te_y = X_seq[s["te_start"]:s["te_end"]], y_seq[s["te_start"]:s["te_end"]]
        te_spot = spot_seq[s["te_start"]:s["te_end"]]
        
        split_res = evaluate_split_hybrids(tr_X, tr_y, te_X, te_y, te_spot)
        for m, met in split_res.items():
            fold_records.append({
                "Scheme": "Scheme 2 (Macro-Regime Folds)",
                "Fold": s["name"],
                "Model": m,
                "MAE ($/MT)": met["MAE ($/MT)"],
                "MAPE (%)": met["MAPE (%)"],
                "RMSE ($/MT)": met["RMSE ($/MT)"],
                "Directional Accuracy (%)": met["Directional Accuracy (%)"],
                "Accuracy (±10%)": met["Accuracy (±10%)"],
                "Max Outlier Miss ($/MT)": met["Max Outlier Miss ($/MT)"]
            })
            
    df_s2 = pd.DataFrame(fold_records)
    summary_s2 = df_s2.groupby("Model")["MAE ($/MT)"].mean().to_dict()
    print("Scheme 2 Finished.")
    return summary_s2, df_s2

# ==============================================================================
# SCHEME 3: TRI-SPLIT TEMPORAL BLOCK TRANSFER VALIDATION
# ==============================================================================
def run_scheme_3(X_seq, y_seq, spot_seq, dates):
    print("\n" + "=" * 95)
    print("SCHEME 3: TRI-SPLIT TEMPORAL BLOCK TRANSFER VALIDATION")
    print("=" * 95)
    
    N = len(y_seq)
    block_len = N // 3
    b1_start, b1_end = 0, block_len
    b2_start, b2_end = block_len, 2 * block_len
    b3_start, b3_end = 2 * block_len, N
    
    splits = [
        {"name": "Transition 1 (Block 1 -> Block 2)", "tr_start": b1_start, "tr_end": b1_end - 15, "te_start": b2_start, "te_end": b2_end},
        {"name": "Transition 2 (Block 2 -> Block 3)", "tr_start": b2_start, "tr_end": b2_end - 15, "te_start": b3_start, "te_end": b3_end},
        {"name": "Transition 3 (Blocks 1+2 -> Block 3)", "tr_start": b1_start, "tr_end": b2_end - 15, "te_start": b3_start, "te_end": b3_end},
    ]
    
    fold_records = []
    for s in splits:
        tr_X, tr_y = X_seq[s["tr_start"]:s["tr_end"]], y_seq[s["tr_start"]:s["tr_end"]]
        te_X, te_y = X_seq[s["te_start"]:s["te_end"]], y_seq[s["te_start"]:s["te_end"]]
        te_spot = spot_seq[s["te_start"]:s["te_end"]]
        
        split_res = evaluate_split_hybrids(tr_X, tr_y, te_X, te_y, te_spot)
        for m, met in split_res.items():
            fold_records.append({
                "Scheme": "Scheme 3 (Tri-Split Block Validation)",
                "Fold": s["name"],
                "Model": m,
                "MAE ($/MT)": met["MAE ($/MT)"],
                "MAPE (%)": met["MAPE (%)"],
                "RMSE ($/MT)": met["RMSE ($/MT)"],
                "Directional Accuracy (%)": met["Directional Accuracy (%)"],
                "Accuracy (±10%)": met["Accuracy (±10%)"],
                "Max Outlier Miss ($/MT)": met["Max Outlier Miss ($/MT)"]
            })
            
    df_s3 = pd.DataFrame(fold_records)
    summary_s3 = df_s3.groupby("Model")["MAE ($/MT)"].mean().to_dict()
    print("Scheme 3 Finished.")
    return summary_s3, df_s3

# ==============================================================================
# SYNTHESIZE HYBRID LEADERBOARD & COMPARISON CHARTS
# ==============================================================================
def synthesize_hybrid_results(s1_dict, s2_dict, s3_dict, df_all):
    workspace = r"d:\freight forecasting"
    art_dir = r"C:\Users\ASUS-PC\.gemini\antigravity-ide\brain\8f94b5e6-8036-4de3-8a81-dc5c95309725"
    
    # Calculate overarching metrics across all 9 splits
    agg_all = df_all.groupby("Model").agg({
        "MAE ($/MT)": "mean",
        "MAPE (%)": "mean",
        "RMSE ($/MT)": "mean",
        "Directional Accuracy (%)": "mean",
        "Accuracy (±10%)": "mean",
        "Max Outlier Miss ($/MT)": "mean"
    }).reset_index()
    
    records = []
    for m in EVALUATED_MODELS:
        s1_mae = s1_dict[m]
        s2_mae = s2_dict[m]
        s3_mae = s3_dict[m]
        ov_row = agg_all[agg_all["Model"] == m].iloc[0]
        
        records.append({
            "Model": m,
            "Overall Avg MAE ($/MT)": round(float(ov_row["MAE ($/MT)"]), 3),
            "Overall MAPE (%)": round(float(ov_row["MAPE (%)"]), 2),
            "Overall RMSE ($/MT)": round(float(ov_row["RMSE ($/MT)"]), 3),
            "Directional Acc (%)": round(float(ov_row["Directional Accuracy (%)"]), 2),
            "Accuracy (±10%)": round(float(ov_row["Accuracy (±10%)"]), 2),
            "Max Miss ($/MT)": round(float(ov_row["Max Outlier Miss ($/MT)"]), 3),
            "S1 MAE ($/MT)": round(float(s1_mae), 3),
            "S2 MAE ($/MT)": round(float(s2_mae), 3),
            "S3 MAE ($/MT)": round(float(s3_mae), 3),
        })
        
    df_res = pd.DataFrame(records)
    
    # Compute rank scores
    df_res["S1 Rank"] = df_res["S1 MAE ($/MT)"].rank().astype(int)
    df_res["S2 Rank"] = df_res["S2 MAE ($/MT)"].rank().astype(int)
    df_res["S3 Rank"] = df_res["S3 MAE ($/MT)"].rank().astype(int)
    df_res["Composite Rank Score"] = (
        0.30 * df_res["S1 Rank"] +
        0.40 * df_res["S2 Rank"] +
        0.30 * df_res["S3 Rank"]
    ).round(2)
    
    df_res = df_res.sort_values(by=["Composite Rank Score", "Overall Avg MAE ($/MT)"]).reset_index(drop=True)
    df_res["Final Rank"] = df_res.index + 1
    
    # Save CSVs
    csv_lead = os.path.join(workspace, "hybrid_ensemble_leaderboard.csv")
    csv_all = os.path.join(workspace, "hybrid_ensemble_all_fold_records.csv")
    df_res.to_csv(csv_lead, index=False)
    df_all.to_csv(csv_all, index=False)
    print(f"\n[SAVED] Hybrid Leaderboard: {csv_lead}")
    
    # --------------------------------------------------------------------------
    # RENDER PLOT
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    # Custom color palette: highlight Hybrids vs Solo models
    colors = []
    for m in df_res["Model"]:
        if "Hybrid: LightGBM + TCN" in m:
            colors.append("#059669") # Emerald / Champion
        elif "Hybrid: LightGBM + PatchTST" in m:
            colors.append("#0284c7") # Sky Blue
        elif "Hybrid" in m:
            colors.append("#10b981") # Green
        elif "LightGBM" in m:
            colors.append("#3b82f6") # Blue
        else:
            colors.append("#94a3b8") # Slate
            
    # Chart 1: Overall Avg MAE & Directional Accuracy
    y_pos = np.arange(len(df_res))
    bars = axes[0].barh(y_pos, df_res["Overall Avg MAE ($/MT)"], color=colors, alpha=0.9, edgecolor="#0f172a", linewidth=0.5)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(df_res["Model"], weight="bold", fontsize=9.5)
    axes[0].set_xlabel("Overall Avg MAE ($/MT) across All 9 Splits (Lower is Better)", weight="bold", fontsize=10)
    axes[0].set_title("Overall Hybrid & Solo Model Accuracy", fontsize=11, weight="bold")
    axes[0].grid(axis="x", linestyle=":", alpha=0.6)
    axes[0].invert_yaxis()
    
    for i, row in df_res.iterrows():
        mae_v = row["Overall Avg MAE ($/MT)"]
        dir_v = row["Directional Acc (%)"]
        axes[0].text(mae_v + 0.02, i, f"${mae_v:.3f} | Dir: {dir_v:.1f}%", va="center", weight="bold", fontsize=8.5)
        
    # Chart 2: Scheme by scheme grouped bar
    x = np.arange(3)
    bar_width = 0.09
    scheme_labels = ["Scheme 1\n(Anchor WF)", "Scheme 2\n(Regime Shock Folds)", "Scheme 3\n(Block Transfer)"]
    
    for i, row in df_res.iterrows():
        m = row["Model"]
        maes = [row["S1 MAE ($/MT)"], row["S2 MAE ($/MT)"], row["S3 MAE ($/MT)"]]
        offset = (i - len(df_res) / 2) * bar_width + (bar_width / 2)
        axes[1].bar(x + offset, maes, width=bar_width, color=colors[i], label=m if i < 5 else None, alpha=0.9)
        
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scheme_labels, weight="bold", fontsize=10)
    axes[1].set_ylabel("MAE ($ / MT) - Lower is Better", weight="bold")
    axes[1].set_title("Cross-Scheme Performance Comparison", fontsize=11, weight="bold")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8, loc="upper right")
    
    plt.suptitle("Systematic Hybrid Ensembling: LightGBM + Proven Deep Learning Architectures (TCN, PatchTST, LSTM)\nAcross 3 Rigorous Time-Series Fold Schemes (9 Evaluation Splits)", fontsize=12, weight="bold", y=0.98)
    plt.tight_layout()
    
    c_ws = os.path.join(workspace, "hybrid_ensemble_comparison.png")
    c_art = os.path.join(art_dir, "hybrid_ensemble_comparison.png")
    plt.savefig(c_ws, dpi=300)
    plt.savefig(c_art, dpi=300)
    plt.close()
    print(f"[SAVED] Comparison Plot: {c_ws}")
    
    return df_res

def main():
    print("=" * 100)
    print("STARTING SYSTEMATIC HYBRID ENSEMBLE BENCHMARK (LIGHTGBM + 4 PROVEN DL ARCHITECTURES)")
    print("=" * 100)
    t0 = time.time()
    
    df_clean, features = prepare_benchmark_dataset()
    X_seq, y_seq, spot_seq, dates = build_sliding_sequences(df_clean, features, seq_len=30)
    
    s1_dict, df_s1 = run_scheme_1(X_seq, y_seq, spot_seq, dates)
    s2_dict, df_s2 = run_scheme_2(X_seq, y_seq, spot_seq, dates)
    s3_dict, df_s3 = run_scheme_3(X_seq, y_seq, spot_seq, dates)
    
    df_all = pd.concat([df_s1, df_s2, df_s3], ignore_index=True)
    df_res = synthesize_hybrid_results(s1_dict, s2_dict, s3_dict, df_all)
    
    elapsed = time.time() - t0
    print(f"\n[DONE] Systematic Hybrid Ensemble Benchmark Completed in {elapsed:.1f}s!")

if __name__ == "__main__":
    main()
