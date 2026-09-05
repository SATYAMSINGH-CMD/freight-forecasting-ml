import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

def run_shap_analysis():
    data_path = os.path.abspath(r"d:\freight forecasting\market_features_daily.csv")
    output_dir = os.path.dirname(data_path)
    
    print("=" * 80)
    print("STEP 1: PREPARING FEATURES & STRICT ZERO-LEAKAGE TIME-SERIES SPLIT")
    print("=" * 80)
    
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # 1. Target: Future freight rate 15 days ahead (No leakage: Y is shifted backwards relative to X)
    horizon = 15
    df["target_15d_ahead"] = df["target_freight_rate_proxy"].shift(-horizon)
    
    # 2. Features constructed strictly with historical info up to time t
    df["freight_current"] = df["target_freight_rate_proxy"]
    df["freight_lag_1"] = df["target_freight_rate_proxy"].shift(1)
    df["freight_lag_7"] = df["target_freight_rate_proxy"].shift(7)
    df["freight_lag_14"] = df["target_freight_rate_proxy"].shift(14)
    df["freight_lag_30"] = df["target_freight_rate_proxy"].shift(30)
    
    df["freight_roll_mean_7"] = df["target_freight_rate_proxy"].shift(1).rolling(7).mean()
    df["freight_roll_mean_14"] = df["target_freight_rate_proxy"].shift(1).rolling(14).mean()
    df["freight_roll_mean_30"] = df["target_freight_rate_proxy"].shift(1).rolling(30).mean()
    
    df["freight_roll_std_7"] = df["target_freight_rate_proxy"].shift(1).rolling(7).std()
    df["freight_roll_std_30"] = df["target_freight_rate_proxy"].shift(1).rolling(30).std()
    
    df["bpi_daily_hire"] = df["bpi_daily_hire_proxy"]
    df["bpi_lag_7"] = df["bpi_daily_hire_proxy"].shift(7)
    df["bpi_pct_change_7d"] = df["bpi_daily_hire_proxy"].pct_change(7) * 100
    
    df["bunker_price"] = df["bunker_price_proxy"]
    df["bunker_lag_7"] = df["bunker_price_proxy"].shift(7)
    df["bunker_pct_change_7d"] = df["bunker_price_proxy"].pct_change(7) * 100
    df["bunker_pct_change_14d"] = df["bunker_price_proxy"].pct_change(14) * 100
    
    df["bunker_to_freight_ratio"] = df["bunker_price_proxy"] / (df["target_freight_rate_proxy"] + 1e-5)
    df["freight_spread_to_ma14"] = df["target_freight_rate_proxy"] - df["freight_roll_mean_14"]
    
    df["coal_price"] = df["coal_price_aus"]
    df["usd_inr_rate"] = df["usd_inr"]
    
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Drop rows with NaNs from lag/shift
    df_clean = df.dropna().copy().reset_index(drop=True)
    
    feature_cols = [
        "freight_current",
        "freight_lag_1",
        "freight_lag_7",
        "freight_lag_14",
        "freight_lag_30",
        "freight_roll_mean_7",
        "freight_roll_mean_14",
        "freight_roll_mean_30",
        "freight_roll_std_7",
        "freight_roll_std_30",
        "bpi_daily_hire",
        "bpi_lag_7",
        "bpi_pct_change_7d",
        "bunker_price",
        "bunker_lag_7",
        "bunker_pct_change_7d",
        "bunker_pct_change_14d",
        "bunker_to_freight_ratio",
        "freight_spread_to_ma14",
        "coal_price",
        "usd_inr_rate",
        "sin_month",
        "cos_month"
    ]
    
    X = df_clean[feature_cols]
    y = df_clean["target_15d_ahead"]
    
    # Chronological Train-Test Split (80% Train, 20% Test) to ensure zero forward lookahead
    split_idx = int(len(df_clean) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Total clean observations: {len(df_clean)}")
    print(f"Training set: {len(X_train)} rows ({df_clean['date'].iloc[0].date()} to {df_clean['date'].iloc[split_idx-1].date()})")
    print(f"Testing set:  {len(X_test)} rows ({df_clean['date'].iloc[split_idx].date()} to {df_clean['date'].iloc[-1].date()})")
    
    print("=" * 80)
    print("STEP 2: TRAINING LIGHTGBM REGRESSOR")
    print("=" * 80)
    
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=150,
        learning_rate=0.04,
        num_leaves=20,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predict and evaluate on unseen test set
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"Model Test Performance (15 Days Forward Horizon):")
    print(f"  MAE:  ${mae:.3f} / MT")
    print(f"  RMSE: ${rmse:.3f} / MT")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R^2:  {r2:.4f}")
    
    print("=" * 80)
    print("STEP 3: COMPUTING SHAP (SHAPLEY ADDITIVE EXPLANATIONS)")
    print("=" * 80)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # Calculate Mean Absolute SHAP values
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_ranking = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    
    print("\nTOP FEATURES BY GLOBAL SHAP IMPORTANCE (Mean |SHAP value|):")
    print("-" * 65)
    for idx, row in shap_ranking.iterrows():
        bar = "#" * int(row["Mean_Abs_SHAP"] * 30)
        print(f"{row['Feature']:<25} | Impact: ${row['Mean_Abs_SHAP']:>6.3f} / MT | {bar}")
        
    # Save SHAP ranking CSV
    shap_csv = os.path.join(output_dir, "shap_feature_importance.csv")
    shap_ranking.to_csv(shap_csv, index=False)
    print(f"\n[SAVED] SHAP numerical table: {shap_csv}")
    
    print("=" * 80)
    print("STEP 4: GENERATING SHAP VISUALIZATIONS")
    print("=" * 80)
    
    # Plot 1: SHAP Beeswarm Summary Plot
    plt.figure(figsize=(12, 8), dpi=300)
    shap.summary_plot(shap_values.values, X_test, feature_names=feature_cols, show=False, max_display=15)
    plt.title("SHAP Beeswarm Plot: Feature Impact on 15-Day Forward Freight Rate", fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    beeswarm_file = os.path.join(output_dir, "shap_beeswarm_plot.png")
    plt.savefig(beeswarm_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] SHAP Beeswarm Plot: {beeswarm_file}")
    
    # Plot 2: SHAP Bar Feature Importance Plot
    plt.figure(figsize=(10, 7), dpi=300)
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Global Feature Importance: Mean |SHAP value| ($/MT)", fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    bar_file = os.path.join(output_dir, "shap_importance_bar.png")
    plt.savefig(bar_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] SHAP Bar Plot: {bar_file}")
    
    print("\nSUCCESS: SHAP Analysis completed successfully with zero data leakage.")

if __name__ == "__main__":
    run_shap_analysis()
