import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_method2_search():
    data_path = os.path.abspath(r"d:\freight forecasting\market_features_daily.csv")
    output_dir = os.path.dirname(data_path)
    
    print("=" * 80)
    print("STEP 1: PREPARING RIGOROUS 3-WAY TIME-SERIES SPLIT (60% TRAIN / 20% VAL / 20% TEST)")
    print("=" * 80)
    
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # 1. Target: Future freight rate 15 days ahead (strictly shifted backwards)
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
    
    df_clean = df.dropna().copy().reset_index(drop=True)
    N = len(df_clean)
    
    # 60% Train, 20% Validation, 20% Untouched Test
    idx_train_end = int(N * 0.60)
    idx_val_end = int(N * 0.80)
    
    train_df = df_clean.iloc[:idx_train_end]
    val_df = df_clean.iloc[idx_train_end:idx_val_end]
    test_df = df_clean.iloc[idx_val_end:]
    
    print(f"Total Observations: {N}")
    print(f"  [1] Train Split (60%):      {len(train_df)} rows ({train_df['date'].iloc[0].date()} to {train_df['date'].iloc[-1].date()})")
    print(f"  [2] Validation Split (20%): {len(val_df)} rows ({val_df['date'].iloc[0].date()} to {val_df['date'].iloc[-1].date()})")
    print(f"  [3] Untouched Test (20%):   {len(test_df)} rows ({test_df['date'].iloc[0].date()} to {test_df['date'].iloc[-1].date()})")
    
    # TOP 10 FIXED BASE FEATURES (Identified by SHAP)
    base_features = [
        "freight_lag_1",
        "bpi_daily_hire",
        "freight_roll_mean_7",
        "freight_current",
        "usd_inr_rate",
        "freight_roll_mean_14",
        "freight_lag_7",
        "freight_lag_30",
        "freight_roll_mean_30",
        "freight_lag_14"
    ]
    
    # 6 CANDIDATE FEATURES TO PERMUTE (2^6 = 64 Combinations)
    candidate_features = [
        "bunker_to_freight_ratio",
        "freight_roll_std_30",
        "freight_spread_to_ma14",
        "bunker_price",
        "coal_price",
        "sin_month"
    ]
    
    print("\n" + "=" * 80)
    print("STEP 2: EVALUATING ALL 64 COMBINATIONS ON 20% VALIDATION SPLIT")
    print("=" * 80)
    
    all_candidate_subsets = []
    for r in range(len(candidate_features) + 1):
        for combo in itertools.combinations(candidate_features, r):
            all_candidate_subsets.append(list(combo))
            
    print(f"Total Combinations to train and evaluate: {len(all_candidate_subsets)}")
    
    results = []
    
    for i, subset in enumerate(all_candidate_subsets):
        current_features = base_features + subset
        
        # Train strictly on the 60% train set
        X_tr = train_df[current_features]
        y_tr = train_df["target_15d_ahead"]
        
        # Evaluate on the 20% validation set
        X_v = val_df[current_features]
        y_v = val_df["target_15d_ahead"]
        
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=120,
            learning_rate=0.04,
            num_leaves=18,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        model.fit(X_tr, y_tr)
        
        y_v_pred = model.predict(X_v)
        val_mae = mean_absolute_error(y_v, y_v_pred)
        val_rmse = np.sqrt(mean_squared_error(y_v, y_v_pred))
        val_mape = np.mean(np.abs((y_v - y_v_pred) / y_v)) * 100
        
        results.append({
            "combo_id": i + 1,
            "num_features": len(current_features),
            "added_candidates": "+".join(subset) if subset else "None (Base Only)",
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_mape": val_mape,
            "features_list": current_features
        })
        
    results_df = pd.DataFrame(results).sort_values("val_mae").reset_index(drop=True)
    
    print("\nTOP 5 BEST FEATURE COMBINATIONS ON VALIDATION SPLIT:")
    print("-" * 80)
    for idx in range(min(5, len(results_df))):
        row = results_df.iloc[idx]
        print(f"Rank #{idx+1} [Combo ID {row['combo_id']}] ({row['num_features']} features):")
        print(f"  Added Features: {row['added_candidates']}")
        print(f"  Validation MAE:  ${row['val_mae']:.4f} / MT | RMSE: ${row['val_rmse']:.4f} | MAPE: {row['val_mape']:.2f}%")
        print()
        
    best_combo = results_df.iloc[0]
    best_features = best_combo["features_list"]
    
    print("=" * 80)
    print("STEP 3: TESTING WINNING COMBINATION ON UNTOUCHED 20% TEST SET")
    print("=" * 80)
    print(f"Winner: Combo ID #{best_combo['combo_id']} ({len(best_features)} features)")
    print(f"Features: {best_features}\n")
    
    # Train winning model on Train + Validation (80%), then test on the untouched 20% Test
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    
    X_train_val = train_val_df[best_features]
    y_train_val = train_val_df["target_15d_ahead"]
    
    X_final_test = test_df[best_features]
    y_final_test = test_df["target_15d_ahead"]
    
    final_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=150,
        learning_rate=0.04,
        num_leaves=18,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    final_model.fit(X_train_val, y_train_val)
    
    y_test_pred = final_model.predict(X_final_test)
    
    test_mae = mean_absolute_error(y_final_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_final_test, y_test_pred))
    test_mape = np.mean(np.abs((y_final_test - y_test_pred) / y_final_test)) * 100
    test_r2 = r2_score(y_final_test, y_test_pred)
    
    print(f"FINAL UNTOUCHED TEST SET RESULTS (Zero Leakage):")
    print(f"  Test MAE:   ${test_mae:.3f} / MT")
    print(f"  Test RMSE:  ${test_rmse:.3f} / MT")
    print(f"  Test MAPE:  {test_mape:.2f}%")
    print(f"  Test R^2:   {test_r2:.4f}")
    
    # Save search table to CSV
    search_csv = os.path.join(output_dir, "feature_combination_search_results.csv")
    results_df.drop(columns=["features_list"]).to_csv(search_csv, index=False)
    print(f"\n[SAVED] All 64 combination results: {search_csv}")
    
    # Plot Actual vs Predicted on Untouched Test Set
    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(test_df["date"], y_final_test.values, label="Actual Freight Rate ($/MT)", color="#38bdf8", linewidth=2.2)
    plt.plot(test_df["date"], y_test_pred, label=f"Predicted 15d-Ahead (MAE: ${test_mae:.2f})", color="#f43f5e", linestyle="--", linewidth=2.0)
    plt.fill_between(test_df["date"], y_test_pred - test_mae, y_test_pred + test_mae, color="#f43f5e", alpha=0.15, label="±MAE Error Band")
    
    plt.title("Untouched 20% Test Set: Actual vs Predicted 15-Day Forward Freight Rate", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Date", fontsize=11, weight="bold")
    plt.ylabel("Freight Rate ($/MT)", fontsize=11, weight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    
    test_plot_file = os.path.join(output_dir, "test_set_forecast_vs_actual.png")
    plt.savefig(test_plot_file, dpi=300)
    plt.close()
    print(f"[SAVED] Test Set Verification Plot: {test_plot_file}")
    
    print("\nSUCCESS: Method 2 completed in seconds with zero data leakage.")

if __name__ == "__main__":
    run_method2_search()
