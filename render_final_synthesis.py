import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

workspace = os.path.abspath(r"d:\freight forecasting")
art_dir = os.path.abspath(r"C:\Users\ASUS-PC\.gemini\antigravity-ide\brain\8f94b5e6-8036-4de3-8a81-dc5c95309725")

# 1. Load CSVs
agg_rolling = pd.read_csv(os.path.join(workspace, "rolling_window_leaderboard.csv"))
agg_crisis = pd.read_csv(os.path.join(workspace, "crisis_stress_leaderboard.csv"))
agg_horizons = pd.read_csv(os.path.join(workspace, "multi_horizon_leaderboard.csv"))
df_holdout = pd.read_csv(os.path.join(workspace, "model_benchmark_leaderboard.csv"))

CANDIDATE_MODELS = [
    "LightGBM",
    "XGBoost",
    "Hybrid Ensemble",
    "PatchTST",
    "TCN",
    "Bi-LSTM + Attention",
    "Uni-LSTM + Attention"
]

# Holdout map
df_holdout_map = dict(zip(df_holdout["Model"], df_holdout["Rank"]))
df_holdout_mae = dict(zip(df_holdout["Model"], df_holdout["MAE ($/MT)"]))

# Build Decision Matrix
matrix = []
for m in CANDIDATE_MODELS:
    r_holdout = df_holdout_map.get(m, 5)
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
    
    # Composite Score: Balanced multi-scheme evaluation
    # 25% Holdout + 30% Rolling Window + 25% Crisis Stress + 20% Multi-Horizon
    comp_score = 0.25 * r_holdout + 0.30 * r_rolling + 0.25 * r_crisis + 0.20 * r_horizon
    
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
        "Composite Score": round(comp_score, 2)
    })
    
df_matrix = pd.DataFrame(matrix).sort_values("Composite Score").reset_index(drop=True)
df_matrix["Definitive Final Rank"] = df_matrix.index + 1

status_labels = ["[1] Champion", "[2] Runner-Up", "[3] Third Place", "[4] Fourth", "[5] Fifth", "[6] Sixth", "[7] Seventh"]
df_matrix["Status"] = [status_labels[i] for i in range(len(df_matrix))]

cols = [
    "Definitive Final Rank", "Status", "Model", "Composite Score",
    "Holdout Rank", "Rolling Rank", "Crisis Rank", "Multi-Horizon Rank",
    "Holdout MAE ($/MT)", "Rolling MAE ($/MT)", "Crisis MAE ($/MT)", "Peak Miss ($/MT)"
]
df_matrix = df_matrix[cols]

matrix_csv = os.path.join(workspace, "final_decision_matrix.csv")
df_matrix.to_csv(matrix_csv, index=False)
print("Decision Matrix Created Successfully:\n", df_matrix.to_string(index=False))

# Color palette
palette = {
    "TCN": "#2563eb",
    "Hybrid Ensemble": "#059669",
    "Bi-LSTM + Attention": "#7c3aed",
    "Uni-LSTM + Attention": "#db2777",
    "PatchTST": "#d97706",
    "XGBoost": "#ea580c",
    "LightGBM": "#0284c7"
}

# Values for Chart 1: Rolling Windows (from log)
# Windows: W1, W2, W3, W4, W5
model_win_maes = {
    "TCN": [0.547, 2.520, 1.230, 1.494, 1.587],
    "Bi-LSTM + Attention": [1.339, 1.613, 1.003, 1.845, 1.767],
    "Uni-LSTM + Attention": [1.885, 1.463, 1.236, 1.789, 2.362],
    "PatchTST": [0.800, 1.988, 1.561, 1.835, 0.589],
    "XGBoost": [1.003, 1.581, 1.382, 0.863, 0.952],
    "LightGBM": [1.166, 1.505, 1.077, 0.948, 0.866],
    "Hybrid Ensemble": [0.710, 1.744, 1.287, 1.092, 1.255]
}

# Values for Chart 2: Crisis Stress (from log)
# Crises: C1 (COVID), C2 (Ukraine War), C3 (Red Sea), C4 (Modern)
model_crisis_maes = {
    "TCN": [8.095, 1.286, 0.786, 1.135],
    "Bi-LSTM + Attention": [3.102, 1.945, 0.799, 0.529],
    "Uni-LSTM + Attention": [4.216, 1.935, 1.367, 0.736],
    "PatchTST": [2.766, 2.943, 1.524, 0.643],
    "XGBoost": [2.376, 1.272, 0.782, 0.660],
    "LightGBM": [2.323, 1.116, 0.870, 0.593],
    "Hybrid Ensemble": [4.782, 0.998, 0.738, 0.879]
}

# Values for Chart 3: Multi-Horizon (from log)
# Horizons: 7d, 15d, 30d
model_horizon_maes = {
    "TCN": [0.722, 0.809, 2.634],
    "Bi-LSTM + Attention": [0.457, 0.810, 0.802],
    "Uni-LSTM + Attention": [0.807, 0.841, 0.809],
    "PatchTST": [0.431, 0.665, 0.701],
    "XGBoost": [0.376, 0.528, 0.776],
    "LightGBM": [0.357, 0.594, 0.778],
    "Hybrid Ensemble": [0.433, 0.614, 1.674]
}

# Chart 1: Rolling-Window Trends
plt.figure(figsize=(13, 6), dpi=300)
x_axis = ["W1 (2018-20)", "W2 (2020-21)", "W3 (2021-22)", "W4 (2022-24)", "W5 (2024-26)"]
for m in CANDIDATE_MODELS:
    lw = 2.5 if m in ["LightGBM", "XGBoost", "Hybrid Ensemble"] else 1.5
    ls = "-" if m in ["LightGBM", "XGBoost", "Hybrid Ensemble"] else ("--" if "LSTM" in m else ":")
    plt.plot(x_axis, model_win_maes[m], marker="o", linewidth=lw, linestyle=ls, color=palette[m], label=m)
    
plt.title("Option A: Rolling-Window Walk-Forward Stability (5 Sliding Windows across 8 Years)\nEvaluates Model Robustness Without Relying on Static Expansion Windows", fontsize=12, weight="bold", pad=12)
plt.ylabel("Out-of-Sample MAE ($ / MT)", fontsize=11, weight="bold")
plt.xlabel("Rolling Evaluation Eras", fontsize=11, weight="bold")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="upper right")
plt.tight_layout()
c1_ws = os.path.join(workspace, "rolling_window_trends.png")
c1_art = os.path.join(art_dir, "rolling_window_trends.png")
plt.savefig(c1_ws, dpi=300)
plt.savefig(c1_art, dpi=300)
plt.close()

# Chart 2: Crisis Stress Comparison
plt.figure(figsize=(14, 7), dpi=300)
cr_labels = ["COVID Shock\n(2020-21)", "Ukraine War Fuel Spike\n(2022)", "Red Sea Rerouting\n(2023-24)", "Modern Normalization\n(2024-26)"]
x = np.arange(4)
bar_width = 0.11

for i, m in enumerate(CANDIDATE_MODELS):
    offset = (i - len(CANDIDATE_MODELS) / 2) * bar_width + (bar_width / 2)
    plt.bar(x + offset, model_crisis_maes[m], width=bar_width, color=palette[m], label=m, alpha=0.90)
    
plt.title("Option B: Macro-Regime Crisis Stress Performance (4 Major Global Black-Swan Events)\nLightGBM & XGBoost Maintain Highest Antifragility; Unconstrained Convolutions Over-Extrapolate During Severe Shocks", fontsize=11, weight="bold", pad=14)
plt.xticks(x, cr_labels, fontsize=10, weight="bold")
plt.ylabel("Stress Period MAE ($ / MT)", fontsize=11, weight="bold")
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", ncol=4, loc="upper right")
plt.tight_layout()
c2_ws = os.path.join(workspace, "crisis_stress_comparison.png")
c2_art = os.path.join(art_dir, "crisis_stress_comparison.png")
plt.savefig(c2_ws, dpi=300)
plt.savefig(c2_art, dpi=300)
plt.close()

# Chart 3: Multi-Horizon Comparison
plt.figure(figsize=(11, 6), dpi=300)
h_axis = ["7 Days (Prompt)", "15 Days (Voyage)", "30 Days (COA / Hedging)"]
for m in CANDIDATE_MODELS:
    lw = 2.5 if m in ["XGBoost", "LightGBM"] else 1.5
    ls = "-" if m in ["XGBoost", "LightGBM"] else ("--" if "LSTM" in m else ":")
    plt.plot(h_axis, model_horizon_maes[m], marker="s", linewidth=lw, linestyle=ls, color=palette[m], label=m)
    
plt.title("Option C: Multi-Horizon Procurement Forecast Curves (7d vs. 15d vs. 30d Ahead)\nEvaluates Operational Accuracy for Immediate Spot Inquiries vs. Monthly Voyage Contracts", fontsize=12, weight="bold", pad=12)
plt.ylabel("Test Set MAE ($ / MT)", fontsize=11, weight="bold")
plt.xlabel("Procurement Decision Horizon", fontsize=11, weight="bold")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="upper left")
plt.tight_layout()
c3_ws = os.path.join(workspace, "multi_horizon_comparison.png")
c3_art = os.path.join(art_dir, "multi_horizon_comparison.png")
plt.savefig(c3_ws, dpi=300)
plt.savefig(c3_art, dpi=300)
plt.close()

# Chart 4: Decision Matrix Table
fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
ax.axis("off")

table_data = []
headers = ["Final Rank", "Status", "Model Name", "Composite Score", "Holdout MAE", "Rolling MAE", "Crisis MAE", "Peak Outlier Miss"]
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

for col in range(len(headers)):
    table[(0, col)].set_facecolor("#1e293b")
    table[(0, col)].set_text_props(color="white", weight="bold")
    
table[(1, 0)].set_facecolor("#dcfce7")
table[(1, 1)].set_facecolor("#dcfce7")
table[(1, 2)].set_facecolor("#dcfce7")
table[(1, 3)].set_facecolor("#dcfce7")

plt.title("SAIL Freight Forecasting Model Benchmark: Definitive Multi-Fold Decision Matrix\nArbitrated Across 4 Robust Cross-Validation Schemes (Holdout, 5-Era Rolling, 4-Crisis Stress, 3-Horizon)", fontsize=12, weight="bold", pad=20)
plt.tight_layout()
c4_ws = os.path.join(workspace, "comprehensive_decision_matrix.png")
c4_art = os.path.join(art_dir, "comprehensive_decision_matrix.png")
plt.savefig(c4_ws, dpi=300)
plt.savefig(c4_art, dpi=300)
plt.close()

print("All charts rendered and saved successfully!")
