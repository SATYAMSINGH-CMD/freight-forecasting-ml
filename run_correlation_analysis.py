import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_correlation_analysis():
    data_path = os.path.abspath(r"d:\freight forecasting\market_features_daily.csv")
    output_dir = os.path.dirname(data_path)
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    print("Engineering candidate features...")
    
    # 1. Target: Future freight rate 15 days ahead
    horizon = 15
    df["target_15d_ahead"] = df["target_freight_rate_proxy"].shift(-horizon)
    
    # 2. Lags (Memory)
    df["freight_lag_1"] = df["target_freight_rate_proxy"].shift(1)
    df["freight_lag_7"] = df["target_freight_rate_proxy"].shift(7)
    df["freight_lag_14"] = df["target_freight_rate_proxy"].shift(14)
    df["freight_lag_30"] = df["target_freight_rate_proxy"].shift(30)
    
    df["bpi_lag_7"] = df["bpi_daily_hire_proxy"].shift(7)
    df["bunker_lag_7"] = df["bunker_price_proxy"].shift(7)
    
    # 3. Rolling window stats (Trends & Volatilities)
    df["freight_roll_mean_7"] = df["target_freight_rate_proxy"].rolling(7).mean()
    df["freight_roll_mean_14"] = df["target_freight_rate_proxy"].rolling(14).mean()
    df["freight_roll_mean_30"] = df["target_freight_rate_proxy"].rolling(30).mean()
    
    df["freight_roll_std_7"] = df["target_freight_rate_proxy"].rolling(7).std()
    df["freight_roll_std_30"] = df["target_freight_rate_proxy"].rolling(30).std()
    
    # 4. Momentum & % Changes
    df["bpi_pct_change_7d"] = df["bpi_daily_hire_proxy"].pct_change(7) * 100
    df["bunker_pct_change_7d"] = df["bunker_price_proxy"].pct_change(7) * 100
    df["bunker_pct_change_14d"] = df["bunker_price_proxy"].pct_change(14) * 100
    df["freight_pct_change_7d"] = df["target_freight_rate_proxy"].pct_change(7) * 100
    
    # 5. Cross-feature ratios & spreads
    df["bunker_to_freight_ratio"] = df["bunker_price_proxy"] / df["target_freight_rate_proxy"]
    df["freight_spread_to_ma14"] = df["target_freight_rate_proxy"] - df["freight_roll_mean_14"]
    
    # 6. Seasonality cyclics
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Drop rows with NaNs caused by rolling/shifting
    df_clean = df.dropna().copy()
    print(f"Dataset after lagging and shifting: {len(df_clean)} rows ready for analysis.\n")
    
    # Select feature columns to analyze
    feature_cols = [
        "target_freight_rate_proxy",
        "freight_lag_1",
        "freight_lag_7",
        "freight_lag_14",
        "freight_lag_30",
        "freight_roll_mean_7",
        "freight_roll_mean_14",
        "freight_roll_mean_30",
        "freight_roll_std_7",
        "freight_roll_std_30",
        "bdi_proxy",
        "bpi_daily_hire_proxy",
        "bpi_lag_7",
        "bpi_pct_change_7d",
        "bunker_price_proxy",
        "bunker_lag_7",
        "bunker_pct_change_7d",
        "bunker_pct_change_14d",
        "bunker_to_freight_ratio",
        "freight_spread_to_ma14",
        "coal_price_aus",
        "usd_inr",
        "sin_month",
        "cos_month"
    ]
    
    # Compute Pearson (Linear) and Spearman (Rank/Non-linear) correlations with the 15-day target
    pearson_corr = df_clean[feature_cols].apply(lambda s: s.corr(df_clean["target_15d_ahead"], method="pearson"))
    spearman_corr = df_clean[feature_cols].apply(lambda s: s.corr(df_clean["target_15d_ahead"], method="spearman"))
    
    corr_summary = pd.DataFrame({
        "Pearson_Corr": pearson_corr,
        "Spearman_Rank_Corr": spearman_corr,
        "Abs_Pearson": pearson_corr.abs()
    }).sort_values("Abs_Pearson", ascending=False)
    
    print("=" * 80)
    print("CORRELATION RANKING WITH TARGET (target_15d_ahead):")
    print("=" * 80)
    for idx, row in corr_summary.iterrows():
        bar = "#" * int(row["Abs_Pearson"] * 25)
        print(f"{idx:<25} | Pearson: {row['Pearson_Corr']:>+6.3f} | Spearman: {row['Spearman_Rank_Corr']:>+6.3f} | {bar}")
        
    # Generate Heatmap Visualization
    print("\nGenerating Correlation Heatmap...")
    top_features = corr_summary.head(14).index.tolist()
    sub_cols = ["target_15d_ahead"] + top_features
    corr_matrix = df_clean[sub_cols].corr()
    
    plt.figure(figsize=(14, 11), dpi=300)
    sns.set_theme(style="white")
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8, "weight": "bold"},
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"}
    )
    
    plt.title("Correlation Analysis: Engineered Features vs 15-Day Future Freight Rate", fontsize=14, weight="bold", pad=20)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"[SAVED] Correlation Heatmap saved to: {plot_file}")
    
    # Save the correlation summary table to CSV for reference
    corr_csv = os.path.join(output_dir, "feature_correlations.csv")
    corr_summary.to_csv(corr_csv)
    print(f"[SAVED] Numerical summary saved to: {corr_csv}")

if __name__ == "__main__":
    run_correlation_analysis()
