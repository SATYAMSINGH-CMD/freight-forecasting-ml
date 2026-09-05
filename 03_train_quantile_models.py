import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def pinball_loss(y_true, y_pred, alpha):
    """Pinball loss for quantile regression."""
    diff = y_true - y_pred
    return np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff))

def run_walk_forward_quantile_pipeline():
    data_path = os.path.abspath(r"d:\freight forecasting\market_features_daily.csv")
    output_dir = os.path.dirname(data_path)
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 85)
    print("STEP 1: PREPARING TIME-SERIES DATASET WITH WINNING 12 FEATURES")
    print("=" * 85)
    
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Target: 15-day forward freight rate proxy
    horizon = 15
    df["target_15d_ahead"] = df["target_freight_rate_proxy"].shift(-horizon)
    
    # 12 Optimal Features from Method 2 Validation
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
    N = len(df_clean)
    
    # Strict 80% Dev vs 20% Untouched Holdout Test
    idx_dev_end = int(N * 0.80)
    dev_df = df_clean.iloc[:idx_dev_end].copy().reset_index(drop=True)
    test_df = df_clean.iloc[idx_dev_end:].copy().reset_index(drop=True)
    
    print(f"Total Observations: {N}")
    print(f"  Development Set (80%): {len(dev_df)} rows ({dev_df['date'].iloc[0].date()} to {dev_df['date'].iloc[-1].date()})")
    print(f"  Untouched Test Set (20%): {len(test_df)} rows ({test_df['date'].iloc[0].date()} to {test_df['date'].iloc[-1].date()})\n")
    
    print("=" * 85)
    print("STEP 2: 5-FOLD PURGED WALK-FORWARD CROSS VALIDATION ON DEV SET")
    print("=" * 85)
    
    n_splits = 5
    purge_gap = 15  # 15 days safety gap between train end and val start
    dev_N = len(dev_df)
    val_size = (dev_N - 150) // n_splits
    
    fold_metrics = []
    all_oof_residuals = []
    
    for fold in range(n_splits):
        val_start = 150 + fold * val_size
        val_end = val_start + val_size if fold < n_splits - 1 else dev_N
        train_end = val_start - purge_gap
        
        fold_train = dev_df.iloc[:train_end]
        fold_val = dev_df.iloc[val_start:val_end]
        
        X_tr = fold_train[features]
        y_tr = fold_train["target_15d_ahead"]
        
        X_val = fold_val[features]
        y_val = fold_val["target_15d_ahead"]
        
        fold_model = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.04,
            num_leaves=18,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbosity=-1
        )
        fold_model.fit(X_tr, y_tr)
        val_pred = fold_model.predict(X_val)
        
        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mape = np.mean(np.abs((y_val - val_pred) / y_val)) * 100
        
        # Out-of-fold residuals for conformal quantile calibration
        residuals = y_val.values - val_pred
        all_oof_residuals.extend(residuals)
        
        print(f"--- FOLD {fold + 1} / {n_splits} ---")
        print(f"  Train Window: {len(fold_train)} days ({fold_train['date'].iloc[0].date()} to {fold_train['date'].iloc[-1].date()})")
        print(f"  [15-Day Purge Embargo: {dev_df['date'].iloc[train_end].date()} to {dev_df['date'].iloc[val_start-1].date()}]")
        print(f"  Validation:   {len(fold_val)} days ({fold_val['date'].iloc[0].date()} to {fold_val['date'].iloc[-1].date()})")
        print(f"  OOF Metrics:  MAE: ${val_mae:.3f}/MT | RMSE: ${val_rmse:.3f} | MAPE: {val_mape:.2f}%\n")
        
        fold_metrics.append({
            "fold": fold + 1,
            "train_days": len(fold_train),
            "val_days": len(fold_val),
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_mape": val_mape
        })
        
    fold_df = pd.DataFrame(fold_metrics)
    print("=" * 85)
    print("WALK-FORWARD CROSS VALIDATION SUMMARY:")
    print(fold_df.to_string(index=False))
    print(f"\nAverage Walk-Forward Out-of-Fold MAE:  ${fold_df['val_mae'].mean():.3f} / MT")
    print(f"Average Walk-Forward Out-of-Fold MAPE: {fold_df['val_mape'].mean():.2f}%")
    print("=" * 85)
    
    # Compute centered calibrated quantiles from walk-forward residuals
    all_oof_residuals = np.array(all_oof_residuals)
    median_res = np.median(all_oof_residuals)
    centered_residuals = all_oof_residuals - median_res
    residual_q10 = np.percentile(centered_residuals, 10)
    residual_q90 = np.percentile(centered_residuals, 90)
    
    print(f"\nEmpirical Walk-Forward Residual Quantiles (Centered):")
    print(f"  P10 Optimistic Dip Delta:    {residual_q10:+.3f} $/MT below expected")
    print(f"  P90 Pessimistic Surge Delta: {residual_q90:+.3f} $/MT above expected")
    
    # Save fold summary to CSV
    fold_csv = os.path.join(output_dir, "walk_forward_kfold_summary.csv")
    fold_df.to_csv(fold_csv, index=False)
    
    print("\n" + "=" * 85)
    print("STEP 3: TRAINING PRODUCTION MODEL ON FULL 80% DEV SET")
    print("=" * 85)
    
    X_dev = dev_df[features]
    y_dev = dev_df["target_15d_ahead"]
    
    prod_model_p50 = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.04,
        num_leaves=18,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    prod_model_p50.fit(X_dev, y_dev)
    print("Production Model successfully fitted on complete 80% Development History.")
    
    print("\n" + "=" * 85)
    print("STEP 4: EVALUATING QUANTILE PIPELINE ON UNTOUCHED 20% TEST SET (ZERO LEAKAGE)")
    print("=" * 85)
    
    X_test = test_df[features]
    y_test = test_df["target_15d_ahead"].values
    
    # 1. Point Forecast (P50 Median)
    p50_test = prod_model_p50.predict(X_test)
    
    # 2. Conformalized Quantiles from Walk-Forward Residuals
    p10_test = p50_test + residual_q10
    p90_test = p50_test + residual_q90
    
    # Strictly maintain ordering: P10 <= P50 <= P90
    p50_test = np.maximum(p10_test, p50_test)
    p90_test = np.maximum(p50_test, p90_test)
    
    test_mae = mean_absolute_error(y_test, p50_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, p50_test))
    test_mape = np.mean(np.abs((y_test - p50_test) / y_test)) * 100
    test_r2 = r2_score(y_test, p50_test)
    
    # Band Coverage: Percentage of actual future rates falling inside [P10, P90] (Expected ~80%)
    coverage = np.mean((y_test >= p10_test) & (y_test <= p90_test)) * 100
    
    pinball_10 = pinball_loss(y_test, p10_test, 0.10)
    pinball_50 = pinball_loss(y_test, p50_test, 0.50)
    pinball_90 = pinball_loss(y_test, p90_test, 0.90)
    
    print(f"FINAL UNTOUCHED TEST SET METRICS:")
    print(f"  P50 Forecast MAE:           ${test_mae:.3f} / MT")
    print(f"  P50 Forecast RMSE:          ${test_rmse:.3f} / MT")
    print(f"  P50 Forecast MAPE:          {test_mape:.2f}%")
    print(f"  P50 Forecast R^2:           {test_r2:.4f}")
    print(f"  P10 - P90 Band Coverage:    {coverage:.1f}% (Actual rates enclosed in uncertainty cone!)")
    print(f"  Pinball Loss (P10):         {pinball_10:.4f}")
    print(f"  Pinball Loss (P50):         {pinball_50:.4f}")
    print(f"  Pinball Loss (P90):         {pinball_90:.4f}")
    print(f"  P10 Mean: ${p10_test.mean():.2f} | P50 Mean: ${p50_test.mean():.2f} | P90 Mean: ${p90_test.mean():.2f}")
    
    # Save predictions to CSV
    test_preds_df = pd.DataFrame({
        "date": test_df["date"],
        "actual_freight_rate": y_test,
        "p10_optimistic_dip": np.round(p10_test, 2),
        "p50_expected_median": np.round(p50_test, 2),
        "p90_pessimistic_surge": np.round(p90_test, 2)
    })
    preds_csv = os.path.join(output_dir, "test_set_quantile_predictions.csv")
    test_preds_df.to_csv(preds_csv, index=False)
    print(f"\n[SAVED] Test predictions table: {preds_csv}")
    
    # Save Model Bundle for the Optimizer
    production_bundle = {
        "features": features,
        "model_p50": prod_model_p50,
        "residual_q10": residual_q10,
        "residual_q90": residual_q90,
        "metrics": {
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_mape": test_mape,
            "test_r2": test_r2,
            "coverage_pct": coverage
        }
    }
    bundle_path = os.path.join(models_dir, "quantile_production_bundle.pkl")
    joblib.dump(production_bundle, bundle_path)
    print(f"[SAVED] Production Quantile Bundle: {bundle_path}")
    
    # Plot Quantile Forecast Cone
    print("\nRendering high-resolution Quantile Forecast Cone...")
    plt.figure(figsize=(13, 6.5), dpi=300)
    
    # Actual test freight rate
    plt.plot(test_df["date"], y_test, label="Actual Freight Rate ($/MT)", color="#0284c7", linewidth=2.5, zorder=4)
    
    # P50 Expected Forecast
    plt.plot(test_df["date"], p50_test, label=f"P50 Expected Forecast (MAE: ${test_mae:.2f}, MAPE: {test_mape:.1f}%)", color="#6366f1", linewidth=2.2, linestyle="-", zorder=3)
    
    # P10 Optimistic bound
    plt.plot(test_df["date"], p10_test, label=f"P10 Optimistic Dip (Shift: {residual_q10:+.2f}$)", color="#10b981", linewidth=1.6, linestyle="--", alpha=0.9, zorder=2)
    
    # P90 Pessimistic bound
    plt.plot(test_df["date"], p90_test, label=f"P90 Pessimistic Surge (Shift: {residual_q90:+.2f}$)", color="#f43f5e", linewidth=1.6, linestyle="--", alpha=0.9, zorder=2)
    
    # Shaded Quantile Cone
    plt.fill_between(
        test_df["date"],
        p10_test,
        p90_test,
        color="#818cf8",
        alpha=0.18,
        label=f"P10 - P90 Uncertainty Cone ({coverage:.1f}% Enclosed)",
        zorder=1
    )
    
    plt.title("SAIL Freight Forecasting: 15-Day Forward Quantile Forecast Cone (P10 - P50 - P90)\nPurged Walk-Forward Validated • Evaluated on Untouched 20% Holdout Test Set", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Forecasting Date", fontsize=11, weight="bold")
    plt.ylabel("15-Day Ahead Freight Rate ($/MT)", fontsize=11, weight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, "quantile_forecast_cone.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"[SAVED] Quantile Forecast Cone Visualization: {plot_file}")
    
    print("\n" + "=" * 85)
    print("SUCCESS: Full Walk-Forward Quantile Training & Evaluation Completed.")
    print("=" * 85)

if __name__ == "__main__":
    run_walk_forward_quantile_pipeline()
