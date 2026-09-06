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

# Import verified model architectures
from run_dl_ml_benchmark import (
    TCNModel,
    PatchTSTModel,
    UniLSTMAttentionModel,
    BiLSTMAttentionModel,
    prepare_benchmark_dataset,
    build_sliding_sequences,
    compute_all_metrics
)

# ------------------------------------------------------------------------------
# EFFICIENT LIGHTWEIGHT TRAINER
# ------------------------------------------------------------------------------
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

DL_MODELS = [
    "TCN (Causal)",
    "PatchTST (Transformer)",
    "Uni-LSTM + Attention (Causal)",
    "Bi-LSTM + Attention (Retrospective)",
    "LightGBM (Baseline Reference)"
]

def evaluate_split_models(tr_X_seq, tr_y, te_X_seq, te_y, te_spot):
    scaler = StandardScaler()
    B_tr, T_tr, D_tr = tr_X_seq.shape
    B_te, T_te, D_te = te_X_seq.shape
    
    tr_X_scaled = scaler.fit_transform(tr_X_seq.reshape(-1, D_tr)).reshape(B_tr, T_tr, D_tr).astype(np.float32)
    te_X_scaled = scaler.transform(te_X_seq.reshape(-1, D_te)).reshape(B_te, T_te, D_te).astype(np.float32)
    
    tr_X_tab = tr_X_seq[:, -1, :]
    te_X_tab = te_X_seq[:, -1, :]
    
    preds = {}
    
    # 1. TCN (Causal)
    tcn = TCNModel(num_features=D_tr)
    tcn = train_quick_dl(tcn, tr_X_scaled, tr_y, epochs=20)
    with torch.no_grad():
        preds["TCN (Causal)"] = tcn(torch.from_numpy(te_X_scaled)).numpy()
        
    # 2. PatchTST (Transformer)
    patchtst = PatchTSTModel(seq_len=T_tr, num_features=D_tr)
    patchtst = train_quick_dl(patchtst, tr_X_scaled, tr_y, epochs=20)
    with torch.no_grad():
        preds["PatchTST (Transformer)"] = patchtst(torch.from_numpy(te_X_scaled)).numpy()
        
    # 3. Uni-LSTM + Attention (Causal)
    unilstm = UniLSTMAttentionModel(num_features=D_tr)
    unilstm = train_quick_dl(unilstm, tr_X_scaled, tr_y, epochs=20)
    with torch.no_grad():
        preds["Uni-LSTM + Attention (Causal)"] = unilstm(torch.from_numpy(te_X_scaled)).numpy()
        
    # 4. Bi-LSTM + Attention (Retrospective)
    bilstm = BiLSTMAttentionModel(num_features=D_tr)
    bilstm = train_quick_dl(bilstm, tr_X_scaled, tr_y, epochs=20)
    with torch.no_grad():
        preds["Bi-LSTM + Attention (Retrospective)"] = bilstm(torch.from_numpy(te_X_scaled)).numpy()
        
    # 5. LightGBM (Reference Baseline)
    lgb_m = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.04, num_leaves=18, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    lgb_m.fit(tr_X_tab, tr_y)
    preds["LightGBM (Baseline Reference)"] = lgb_m.predict(te_X_tab)
    
    res = {}
    for m in DL_MODELS:
        m_eval = compute_all_metrics(te_y, preds[m], te_spot)
        m_eval["Max Outlier Miss ($/MT)"] = round(float(np.max(np.abs(te_y - preds[m]))), 3)
        res[m] = m_eval
        
    gc.collect()
    return res

# ==============================================================================
# SCHEME 1: 3-STAGE EXPANDING ANCHOR WALK-FORWARD (AQR / HEDGE FUND STANDARD)
# ==============================================================================
def run_scheme_1(X_seq, y_seq, spot_seq, dates):
    print("\n" + "=" * 95)
    print("SCHEME 1: 3-STAGE EXPANDING ANCHOR WALK-FORWARD (HEDGE FUND STANDARD)")
    print("Guarantees >= 1,000 Training Days in Every Fold (No Deep Learning Sample Starvation)")
    print("=" * 95)
    
    splits = [
        {"name": "Fold 1 (Early Modern Era: 2022-2023)", "tr_end": 1026, "purge": 15, "te_start": 1041, "te_end": 1378},
        {"name": "Fold 2 (Disruption Era: 2023-2024)",   "tr_end": 1378, "purge": 15, "te_start": 1393, "te_end": 1715},
        {"name": "Fold 3 (Current Operational: 2024-2026)", "tr_end": 1715, "purge": 15, "te_start": 1730, "te_end": len(y_seq)},
    ]
    
    records = []
    for s_idx, sp in enumerate(splits):
        tr_end = sp["tr_end"]
        te_s, te_e = sp["te_start"], sp["te_end"]
        
        d_tr_start, d_tr_end = dates[0].date(), dates[tr_end-1].date()
        d_te_start, d_te_end = dates[te_s].date(), dates[te_e-1].date()
        
        print(f"\n--- {sp['name']} ---")
        print(f"  Training Window: {tr_end} days ({d_tr_start} to {d_tr_end})")
        print(f"  15-Day Embargo Purged")
        print(f"  Test Window: {te_e - te_s} days ({d_te_start} to {d_te_end})")
        
        tr_X, te_X = X_seq[:tr_end], X_seq[te_s:te_e]
        tr_y, te_y = y_seq[:tr_end], y_seq[te_s:te_e]
        te_spot = spot_seq[te_s:te_e]
        
        eval_res = evaluate_split_models(tr_X, tr_y, te_X, te_y, te_spot)
        
        for m in DL_MODELS:
            r = eval_res[m]
            records.append({
                "Scheme": "Scheme 1 (Expanding Anchor Walk-Forward)",
                "Fold": f"Fold {s_idx+1}",
                "Model": m,
                "MAE ($/MT)": r["MAE ($/MT)"],
                "MAPE (%)": r["MAPE (%)"],
                "RMSE ($/MT)": r["RMSE ($/MT)"],
                "Directional Accuracy (%)": r["Directional Accuracy (%)"],
                "Accuracy (±10%)": r["Accuracy (±10%)"],
                "Max Outlier Miss ($/MT)": r["Max Outlier Miss ($/MT)"]
            })
            print(f"    {m:35s} -> MAE: ${r['MAE ($/MT)']:.3f}/MT | MAPE: {r['MAPE (%)']:.2f}% | Dir Acc: {r['Directional Accuracy (%)']:.1f}%")
            
    df_s1 = pd.DataFrame(records)
    summary_s1 = df_s1.groupby("Model").agg({
        "MAE ($/MT)": ["mean", "max"],
        "MAPE (%)": "mean",
        "Directional Accuracy (%)": "mean",
        "Accuracy (±10%)": "mean"
    }).reset_index()
    summary_s1.columns = ["Model", "Mean MAE ($/MT)", "Worst MAE ($/MT)", "Mean MAPE (%)", "Mean Dir Acc (%)", "Mean Acc (±10%)"]
    summary_s1 = summary_s1.sort_values("Mean MAE ($/MT)").reset_index(drop=True)
    summary_s1["Rank S1"] = summary_s1.index + 1
    
    print("\n" + "-" * 90)
    print("SCHEME 1 SUMMARY LEADERBOARD:")
    print("-" * 90)
    print(summary_s1.to_string(index=False))
    return summary_s1, df_s1

# ==============================================================================
# SCHEME 2: 3 STRUCTURAL MACRO-REGIME SHOCK FOLDS
# ==============================================================================
def run_scheme_2(X_seq, y_seq, spot_seq, dates):
    print("\n" + "=" * 95)
    print("SCHEME 2: 3 STRUCTURAL MACRO-REGIME FOLDS (CRISIS & EXPANSION)")
    print("Testing Real Historical Shocks: Supercycle Surge -> Ukraine Fuel Spike -> Red Sea Disruption")
    print("=" * 95)
    
    regimes = [
        {"name": "Regime 1: Post-COVID Supercycle Surge", "start": "2021-09-01", "end": "2022-05-31"},
        {"name": "Regime 2: Ukraine War Energy & Fuel Shock", "start": "2022-06-01", "end": "2023-10-31"},
        {"name": "Regime 3: Red Sea Diversions & Modern Era", "start": "2023-11-01", "end": "2026-08-14"},
    ]
    
    purge_gap = 15
    records = []
    
    for r_idx, reg in enumerate(regimes):
        t_start = pd.to_datetime(reg["start"])
        t_end = pd.to_datetime(reg["end"])
        
        te_mask = (dates >= t_start) & (dates <= t_end)
        te_indices = np.where(te_mask)[0]
        
        cutoff_date = t_start - pd.Timedelta(days=purge_gap)
        tr_mask = dates <= cutoff_date
        tr_indices = np.where(tr_mask)[0]
        
        print(f"\n--- {reg['name']} ---")
        print(f"  Training Window: {len(tr_indices)} days (prior to {cutoff_date.date()})")
        print(f"  Test Window: {len(te_indices)} days ({dates[te_indices[0]].date()} to {dates[te_indices[-1]].date()})")
        
        tr_X, te_X = X_seq[tr_indices], X_seq[te_indices]
        tr_y, te_y = y_seq[tr_indices], y_seq[te_indices]
        te_spot = spot_seq[te_indices]
        
        eval_res = evaluate_split_models(tr_X, tr_y, te_X, te_y, te_spot)
        
        for m in DL_MODELS:
            r = eval_res[m]
            records.append({
                "Scheme": "Scheme 2 (Macro-Regime Folds)",
                "Fold": f"Regime {r_idx+1}",
                "Model": m,
                "MAE ($/MT)": r["MAE ($/MT)"],
                "MAPE (%)": r["MAPE (%)"],
                "RMSE ($/MT)": r["RMSE ($/MT)"],
                "Directional Accuracy (%)": r["Directional Accuracy (%)"],
                "Accuracy (±10%)": r["Accuracy (±10%)"],
                "Max Outlier Miss ($/MT)": r["Max Outlier Miss ($/MT)"]
            })
            print(f"    {m:35s} -> MAE: ${r['MAE ($/MT)']:.3f}/MT | MAPE: {r['MAPE (%)']:.2f}% | Dir Acc: {r['Directional Accuracy (%)']:.1f}%")
            
    df_s2 = pd.DataFrame(records)
    summary_s2 = df_s2.groupby("Model").agg({
        "MAE ($/MT)": ["mean", "max"],
        "MAPE (%)": "mean",
        "Directional Accuracy (%)": "mean",
        "Accuracy (±10%)": "mean"
    }).reset_index()
    summary_s2.columns = ["Model", "Mean MAE ($/MT)", "Worst MAE ($/MT)", "Mean MAPE (%)", "Mean Dir Acc (%)", "Mean Acc (±10%)"]
    summary_s2 = summary_s2.sort_values("Mean MAE ($/MT)").reset_index(drop=True)
    summary_s2["Rank S2"] = summary_s2.index + 1
    
    print("\n" + "-" * 90)
    print("SCHEME 2 SUMMARY LEADERBOARD:")
    print("-" * 90)
    print(summary_s2.to_string(index=False))
    return summary_s2, df_s2

# ==============================================================================
# SCHEME 3: TRI-SPLIT TEMPORAL BLOCK VALIDATION (ACADEMIC BENCHMARK)
# ==============================================================================
def run_scheme_3(X_seq, y_seq, spot_seq, dates):
    print("\n" + "=" * 95)
    print("SCHEME 3: TRI-SPLIT TEMPORAL BLOCK VALIDATION (CROSS-ERA GENERALIZATION)")
    print("Testing Era-to-Era Transferability Across 3 Distinct Historical Blocks (~684 Days Each)")
    print("=" * 95)
    
    # Block A: 0..684 (2018-2020)
    # Block B: 684..1368 (2021-2023)
    # Block C: 1368..2052 (2023-2026)
    
    transitions = [
        {"name": "Transition 1 (Train on Block A -> Test on Block B)", "tr_start": 0, "tr_end": 669, "te_start": 684, "te_end": 1368},
        {"name": "Transition 2 (Train on Block B -> Test on Block C)", "tr_start": 684, "tr_end": 1353, "te_start": 1368, "te_end": len(y_seq)},
        {"name": "Transition 3 (Train on Block A+B -> Test on Block C)", "tr_start": 0, "tr_end": 1353, "te_start": 1368, "te_end": len(y_seq)},
    ]
    
    records = []
    for t_idx, tr in enumerate(transitions):
        tr_s, tr_e = tr["tr_start"], tr["tr_end"]
        te_s, te_e = tr["te_start"], tr["te_end"]
        
        print(f"\n--- {tr['name']} ---")
        print(f"  Training Window: {tr_e - tr_s} days ({dates[tr_s].date()} to {dates[tr_e-1].date()})")
        print(f"  Test Window: {te_e - te_s} days ({dates[te_s].date()} to {dates[te_e-1].date()})")
        
        tr_X, te_X = X_seq[tr_s:tr_e], X_seq[te_s:te_e]
        tr_y, te_y = y_seq[tr_s:tr_e], y_seq[te_s:te_e]
        te_spot = spot_seq[te_s:te_e]
        
        eval_res = evaluate_split_models(tr_X, tr_y, te_X, te_y, te_spot)
        
        for m in DL_MODELS:
            r = eval_res[m]
            records.append({
                "Scheme": "Scheme 3 (Tri-Split Block Validation)",
                "Fold": f"Transition {t_idx+1}",
                "Model": m,
                "MAE ($/MT)": r["MAE ($/MT)"],
                "MAPE (%)": r["MAPE (%)"],
                "RMSE ($/MT)": r["RMSE ($/MT)"],
                "Directional Accuracy (%)": r["Directional Accuracy (%)"],
                "Accuracy (±10%)": r["Accuracy (±10%)"],
                "Max Outlier Miss ($/MT)": r["Max Outlier Miss ($/MT)"]
            })
            print(f"    {m:35s} -> MAE: ${r['MAE ($/MT)']:.3f}/MT | MAPE: {r['MAPE (%)']:.2f}% | Dir Acc: {r['Directional Accuracy (%)']:.1f}%")
            
    df_s3 = pd.DataFrame(records)
    summary_s3 = df_s3.groupby("Model").agg({
        "MAE ($/MT)": ["mean", "max"],
        "MAPE (%)": "mean",
        "Directional Accuracy (%)": "mean",
        "Accuracy (±10%)": "mean"
    }).reset_index()
    summary_s3.columns = ["Model", "Mean MAE ($/MT)", "Worst MAE ($/MT)", "Mean MAPE (%)", "Mean Dir Acc (%)", "Mean Acc (±10%)"]
    summary_s3 = summary_s3.sort_values("Mean MAE ($/MT)").reset_index(drop=True)
    summary_s3["Rank S3"] = summary_s3.index + 1
    
    print("\n" + "-" * 90)
    print("SCHEME 3 SUMMARY LEADERBOARD:")
    print("-" * 90)
    print(summary_s3.to_string(index=False))
    return summary_s3, df_s3

# ==============================================================================
# SYNTHESIS & DECISION MATRIX
# ==============================================================================
def synthesize_dl_results(s1, s2, s3, df_all):
    workspace = os.path.abspath(r"d:\freight forecasting")
    art_dir = os.path.abspath(r"C:\Users\ASUS-PC\.gemini\antigravity-ide\brain\8f94b5e6-8036-4de3-8a81-dc5c95309725")
    
    matrix = []
    for m in DL_MODELS:
        r1 = s1[s1["Model"] == m].iloc[0]
        r2 = s2[s2["Model"] == m].iloc[0]
        r3 = s3[s3["Model"] == m].iloc[0]
        
        mae1, rank1 = float(r1["Mean MAE ($/MT)"]), int(r1["Rank S1"])
        mae2, rank2 = float(r2["Mean MAE ($/MT)"]), int(r2["Rank S2"])
        mae3, rank3 = float(r3["Mean MAE ($/MT)"]), int(r3["Rank S3"])
        
        overall_avg_mae = (mae1 + mae2 + mae3) / 3.0
        comp_rank = 0.40 * rank1 + 0.35 * rank2 + 0.25 * rank3
        
        matrix.append({
            "Model": m,
            "S1 Rank (Anchor WF)": rank1,
            "S1 MAE ($/MT)": round(mae1, 3),
            "S2 Rank (Regime Folds)": rank2,
            "S2 MAE ($/MT)": round(mae2, 3),
            "S3 Rank (Block Trans)": rank3,
            "S3 MAE ($/MT)": round(mae3, 3),
            "Overall Avg MAE ($/MT)": round(overall_avg_mae, 3),
            "Composite Rank Score": round(comp_rank, 2)
        })
        
    df_matrix = pd.DataFrame(matrix).sort_values("Composite Rank Score").reset_index(drop=True)
    df_matrix["Final DL Rank"] = df_matrix.index + 1
    
    status_labels = ["[1] Champion DL", "[2] Runner-Up DL", "[3] Third Place DL", "[4] Fourth", "[5] Fifth"]
    df_matrix["Status"] = [status_labels[i] for i in range(len(df_matrix))]
    
    cols = [
        "Final DL Rank", "Status", "Model", "Overall Avg MAE ($/MT)", "Composite Rank Score",
        "S1 Rank (Anchor WF)", "S1 MAE ($/MT)", "S2 Rank (Regime Folds)", "S2 MAE ($/MT)",
        "S3 Rank (Block Trans)", "S3 MAE ($/MT)"
    ]
    df_matrix = df_matrix[cols]
    
    print("\n" + "=" * 110)
    print("THE DEFINITIVE 3-FOLD DEEP LEARNING BENCHMARK DECISION MATRIX:")
    print("=" * 110)
    print(df_matrix.to_string(index=False))
    
    csv_out = os.path.join(workspace, "proven_dl_benchmark_leaderboard.csv")
    df_matrix.to_csv(csv_out, index=False)
    df_all.to_csv(os.path.join(workspace, "proven_dl_all_fold_records.csv"), index=False)
    print(f"\n[SAVED] Benchmark CSV: {csv_out}")
    
    # --------------------------------------------------------------------------
    # RENDER PLOT
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    palette = ["#2563eb", "#d97706", "#db2777", "#7c3aed", "#0284c7"]
    
    # Chart 1: Average MAE across the 3 fold schemes
    models_plot = df_matrix["Model"].values
    y_pos = np.arange(len(models_plot))
    axes[0].barh(y_pos, df_matrix["Overall Avg MAE ($/MT)"], color=palette, alpha=0.85)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(models_plot, weight="bold", fontsize=10)
    axes[0].set_xlabel("Average MAE ($/MT) Across All 3 Fold Schemes (Lower is Better)", weight="bold")
    axes[0].set_title("Overall Deep Learning Accuracy Across 3 Reliable Fold Schemes", fontsize=11, weight="bold")
    axes[0].grid(axis="x", linestyle=":", alpha=0.6)
    axes[0].invert_yaxis()
    for i, v in enumerate(df_matrix["Overall Avg MAE ($/MT)"]):
        axes[0].text(v + 0.02, i, f"${v:.3f}", va="center", weight="bold", fontsize=9)
        
    # Chart 2: Scheme by scheme comparison (grouped bar)
    x = np.arange(3)
    bar_width = 0.15
    scheme_labels = ["Scheme 1\n(Anchor WF)", "Scheme 2\n(Regime Folds)", "Scheme 3\n(Block Transfer)"]
    
    for i, m in enumerate(models_plot):
        r = df_matrix[df_matrix["Model"] == m].iloc[0]
        maes = [r["S1 MAE ($/MT)"], r["S2 MAE ($/MT)"], r["S3 MAE ($/MT)"]]
        offset = (i - len(models_plot) / 2) * bar_width + (bar_width / 2)
        axes[1].bar(x + offset, maes, width=bar_width, color=palette[i], label=m, alpha=0.85)
        
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scheme_labels, weight="bold", fontsize=10)
    axes[1].set_ylabel("MAE ($ / MT) - Lower is Better", weight="bold")
    axes[1].set_title("Scheme-by-Scheme Stability Comparison", fontsize=11, weight="bold")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8, loc="upper right")
    
    plt.suptitle("Proven Time-Series Deep Learning Architectures on Small Shipping Dataset (~2,052 Days)\nRigorous 3-Scheme Cross-Validation (Anchor Walk-Forward, Macro-Regimes, Block Transfer)", fontsize=12, weight="bold", y=0.98)
    plt.tight_layout()
    
    c_ws = os.path.join(workspace, "proven_dl_3fold_comparison.png")
    c_art = os.path.join(art_dir, "proven_dl_3fold_comparison.png")
    plt.savefig(c_ws, dpi=300)
    plt.savefig(c_art, dpi=300)
    plt.close()
    print(f"[SAVED] Comparison Plot: {c_ws}")
    
    return df_matrix

# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def main():
    print("=" * 100)
    print("STARTING 3-SCHEME RIGOROUS BENCHMARK FOR PROVEN DEEP LEARNING TIME-SERIES MODELS")
    print("Dataset: 2,052 Days (2018-2026) | 12 Winning Maritime Features")
    print("=" * 100)
    t0 = time.time()
    
    df_clean, features = prepare_benchmark_dataset()
    X_seq, y_seq, spot_seq, dates = build_sliding_sequences(df_clean, features, seq_len=30)
    
    # 1. Scheme 1: Anchor Walk-Forward
    s1_sum, df_s1 = run_scheme_1(X_seq, y_seq, spot_seq, dates)
    
    # 2. Scheme 2: Macro-Regimes
    s2_sum, df_s2 = run_scheme_2(X_seq, y_seq, spot_seq, dates)
    
    # 3. Scheme 3: Block Validation
    s3_sum, df_s3 = run_scheme_3(X_seq, y_seq, spot_seq, dates)
    
    df_all = pd.concat([df_s1, df_s2, df_s3], ignore_index=True)
    df_matrix = synthesize_dl_results(s1_sum, s2_sum, s3_sum, df_all)
    
    elapsed = time.time() - t0
    print(f"\nCompleted all 3 Deep Learning validation schemes in {elapsed:.1f}s (~{elapsed/60:.1f} mins)!")

if __name__ == "__main__":
    main()
