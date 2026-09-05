import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint

class FreightOptimizer:
    def __init__(self, workspace_dir=r"d:\freight forecasting"):
        self.workspace = os.path.abspath(workspace_dir)
        
        # 1. Load Master Tables
        self.vessels_df = pd.read_csv(os.path.join(self.workspace, "vessel_fleet_master.csv"))
        self.ports_df = pd.read_csv(os.path.join(self.workspace, "port_constraints_master.csv"))
        self.routes_df = pd.read_csv(os.path.join(self.workspace, "trade_routes_master.csv"))
        self.market_df = pd.read_csv(os.path.join(self.workspace, "market_features_daily.csv"))
        self.market_df["date"] = pd.to_datetime(self.market_df["date"])
        self.market_df = self.market_df.sort_values("date").reset_index(drop=True)
        
        # 2. Load Production Quantile Model Bundle
        bundle_path = os.path.join(self.workspace, "models", "quantile_production_bundle.pkl")
        self.bundle = joblib.load(bundle_path)
        self.features = self.bundle["features"]
        self.model_p50 = self.bundle["model_p50"]
        self.residual_q10 = self.bundle["residual_q10"]
        self.residual_q90 = self.bundle["residual_q90"]
        
        # Benchmark base route: Gladstone -> East Coast India (5,250 NM)
        self.base_distance = 5250.0

    def get_route_distance(self, origin, destination):
        """Lookup nautical distance from trade routes table or estimate."""
        match = self.routes_df[
            (self.routes_df["origin_port"].str.lower() == origin.lower()) &
            (self.routes_df["destination_port"].str.lower() == destination.lower())
        ]
        if len(match) > 0:
            return float(match.iloc[0]["distance_nm"])
        
        # Fallback dictionary for common pairs
        lookup = {
            ("gladstone", "haldia"): 5250.0,
            ("gladstone", "paradip"): 5100.0,
            ("gladstone", "vizag"): 4950.0,
            ("gladstone", "dhamra"): 5150.0,
            ("taboneo", "haldia"): 2150.0,
            ("taboneo", "paradip"): 2000.0,
            ("taboneo", "vizag"): 1900.0,
            ("taboneo", "dhamra"): 2050.0,
            ("norfolk", "haldia"): 9800.0,
            ("norfolk", "paradip"): 9650.0,
            ("norfolk", "vizag"): 9500.0,
            ("richards bay", "paradip"): 4500.0,
            ("richards bay", "haldia"): 4650.0,
        }
        pair = (origin.lower(), destination.lower())
        return lookup.get(pair, 5200.0)

    def predict_freight_quantiles(self, origin, destination, as_of_date=None):
        """Predict 15-day forward freight rate (P10, P50, P90) scaled to route distance."""
        if as_of_date is None:
            row_idx = len(self.market_df) - 1
        else:
            dt = pd.to_datetime(as_of_date)
            matches = self.market_df[self.market_df["date"] <= dt]
            row_idx = matches.index[-1] if len(matches) > 0 else 0

        # Construct input features for as_of_date
        window_df = self.market_df.iloc[:row_idx + 1].copy()
        
        # Re-compute lag and rolling features up to today t
        freight_series = window_df["target_freight_rate_proxy"]
        current_spot = freight_series.iloc[-1]
        
        row_feat = {
            "freight_lag_1": freight_series.iloc[-2] if len(freight_series) >= 2 else current_spot,
            "bpi_daily_hire": window_df["bpi_daily_hire_proxy"].iloc[-1],
            "freight_roll_mean_7": freight_series.iloc[:-1].tail(7).mean() if len(freight_series) >= 8 else current_spot,
            "freight_current": current_spot,
            "usd_inr_rate": window_df["usd_inr"].iloc[-1],
            "freight_roll_mean_14": freight_series.iloc[:-1].tail(14).mean() if len(freight_series) >= 15 else current_spot,
            "freight_lag_7": freight_series.iloc[-8] if len(freight_series) >= 8 else current_spot,
            "freight_lag_30": freight_series.iloc[-31] if len(freight_series) >= 31 else current_spot,
            "freight_roll_mean_30": freight_series.iloc[:-1].tail(30).mean() if len(freight_series) >= 31 else current_spot,
            "freight_lag_14": freight_series.iloc[-15] if len(freight_series) >= 15 else current_spot,
            "bunker_to_freight_ratio": window_df["bunker_price_proxy"].iloc[-1] / (current_spot + 1e-5),
            "freight_roll_std_30": freight_series.iloc[:-1].tail(30).std() if len(freight_series) >= 31 else 0.5,
        }
        X_now = pd.DataFrame([row_feat])[self.features]
        
        # Predict base benchmark 15d ahead freight rate
        base_p50 = float(self.model_p50.predict(X_now)[0])
        base_p10 = float(base_p50 + self.residual_q10)
        base_p90 = float(base_p50 + self.residual_q90)
        
        # Scale to route nautical distance
        dist = self.get_route_distance(origin, destination)
        route_factor = dist / self.base_distance
        
        route_spot = round(current_spot * route_factor, 2)
        route_p10 = round(base_p10 * route_factor, 2)
        route_p50 = round(base_p50 * route_factor, 2)
        route_p90 = round(base_p90 * route_factor, 2)
        
        current_bunker = float(window_df["bunker_price_proxy"].iloc[-1])
        current_coal = float(window_df["coal_price_aus"].iloc[-1])
        date_str = window_df["date"].iloc[-1].strftime("%Y-%m-%d")
        
        return {
            "date": date_str,
            "origin": origin,
            "destination": destination,
            "distance_nm": dist,
            "current_bunker_usd_mt": current_bunker,
            "current_coal_usd_mt": current_coal,
            "spot_freight_usd_mt": route_spot,
            "forward_p10_usd_mt": route_p10,
            "forward_p50_usd_mt": route_p50,
            "forward_p90_usd_mt": route_p90,
        }

    def calculate_single_voyage_cost(self, vessel_class, origin, destination, timing="NOW", scenario="P50", as_of_date=None):
        """Compute the 4 landed cost heads for a single voyage."""
        preds = self.predict_freight_quantiles(origin, destination, as_of_date)
        dist = preds["distance_nm"]
        bunker_price = preds["current_bunker_usd_mt"]
        
        # Select active freight rate based on timing
        if timing.upper() == "NOW":
            freight_rate = preds["spot_freight_usd_mt"]
            timing_label = "FIX NOW (Spot Rate)"
        else:
            if scenario.upper() == "P10":
                freight_rate = preds["forward_p10_usd_mt"]
            elif scenario.upper() == "P90":
                freight_rate = preds["forward_p90_usd_mt"]
            else:
                freight_rate = preds["forward_p50_usd_mt"]
            timing_label = f"HOLD 15 DAYS ({scenario} Forecast)"
            
        # 1. Vessel Parameters
        vessel_row = self.vessels_df[self.vessels_df["vessel_class"].str.lower() == vessel_class.lower()].iloc[0]
        capacity_mt = float(vessel_row["capacity_mt"])
        laden_draft = float(vessel_row["laden_draft_m"])
        speed_knots = float(vessel_row["speed_knots"])
        daily_fuel_burn = float(vessel_row["daily_fuel_burn_mt"])
        demurrage_rate = float(vessel_row["demurrage_rate_usd_day"])
        
        # Scale freight rate for vessel size (Capesize is cheaper/MT, Handysize is more expensive/MT)
        scale_multipliers = {
            "handysize": 1.25,
            "supramax": 1.10,
            "panamax": 1.00,
            "capesize": 0.85
        }
        effective_freight_rate = freight_rate * scale_multipliers.get(vessel_class.lower(), 1.0)
        
        # 2. Port Parameters
        port_matches = self.ports_df[self.ports_df["port_name"].str.lower() == destination.lower()]
        if len(port_matches) > 0:
            port_row = port_matches.iloc[0]
            port_draft = float(port_row["max_draft_m"])
            discharge_rate = float(port_row["discharge_rate_mt_day"])
            avg_wait_days = float(port_row["avg_waiting_days"])
            port_lightering_rate = float(port_row["lightering_cost_per_mt"])
        else:
            port_draft = 15.5
            discharge_rate = 25000.0
            avg_wait_days = 3.0
            port_lightering_rate = 0.0
            
        # COST HEAD 1: Base Ocean Freight
        base_freight_cost = capacity_mt * effective_freight_rate
        
        # COST HEAD 2: Voyage Bunker Fuel
        sea_days = dist / (speed_knots * 24.0)
        bunker_fuel_cost = sea_days * daily_fuel_burn * bunker_price
        
        # COST HEAD 3: Port Congestion Demurrage
        laytime_allowed = capacity_mt / discharge_rate
        demurrage_days = max(0.0, avg_wait_days - laytime_allowed)
        demurrage_cost = demurrage_days * demurrage_rate
        
        # COST HEAD 4: Sandheads Lightering (Mandatory at Haldia for draft > 8.5m)
        lightering_mt = 0.0
        lightering_cost = 0.0
        is_lightered = False
        
        if destination.lower() == "haldia" and laden_draft > port_draft:
            is_lightered = True
            # Cargo needed to offload to reduce draft to 8.5m
            draft_excess = laden_draft - port_draft
            lightering_mt = min(capacity_mt * 0.50, capacity_mt * (draft_excess / laden_draft) * 1.15)
            lightering_cost = lightering_mt * port_lightering_rate
            
        total_landed_cost = base_freight_cost + bunker_fuel_cost + demurrage_cost + lightering_cost
        landed_cost_per_mt = total_landed_cost / capacity_mt
        
        return {
            "vessel_class": vessel_class,
            "timing": timing_label,
            "cargo_volume_mt": capacity_mt,
            "sea_transit_days": round(sea_days, 1),
            "effective_freight_usd_mt": round(effective_freight_rate, 2),
            "head1_ocean_freight_usd": round(base_freight_cost, 0),
            "head2_bunker_fuel_usd": round(bunker_fuel_cost, 0),
            "head3_demurrage_usd": round(demurrage_cost, 0),
            "head4_lightering_usd": round(lightering_cost, 0),
            "is_lightered": is_lightered,
            "lightered_mt": round(lightering_mt, 0),
            "total_landed_cost_usd": round(total_landed_cost, 0),
            "landed_cost_per_mt": round(landed_cost_per_mt, 2)
        }

    def solve_single_shipment_optimization(self, origin, destination, as_of_date=None):
        """Evaluate Handysize, Supramax, and Panamax under FIX NOW vs HOLD."""
        preds = self.predict_freight_quantiles(origin, destination, as_of_date)
        vessel_options = ["Handysize", "Supramax", "Panamax"]
        
        # Capesize only allowed if port draft >= 17.5m
        if destination.lower() in ["vizag", "dhamra"]:
            vessel_options.append("Capesize")
            
        eval_rows = []
        for v in vessel_options:
            cost_now = self.calculate_single_voyage_cost(v, origin, destination, timing="NOW", as_of_date=as_of_date)
            cost_hold_p50 = self.calculate_single_voyage_cost(v, origin, destination, timing="HOLD", scenario="P50", as_of_date=as_of_date)
            cost_hold_p10 = self.calculate_single_voyage_cost(v, origin, destination, timing="HOLD", scenario="P10", as_of_date=as_of_date)
            cost_hold_p90 = self.calculate_single_voyage_cost(v, origin, destination, timing="HOLD", scenario="P90", as_of_date=as_of_date)
            
            # Decision rule for this vessel:
            savings_per_mt = cost_now["landed_cost_per_mt"] - cost_hold_p50["landed_cost_per_mt"]
            total_savings = savings_per_mt * cost_now["cargo_volume_mt"]
            
            if savings_per_mt > 0.15:
                rec = "HOLD (Wait 15d)"
            elif savings_per_mt < -0.15:
                rec = "FIX NOW (Spot)"
            else:
                rec = "NEUTRAL (Fix Now)"
                
            eval_rows.append({
                "vessel": v,
                "capacity_mt": cost_now["cargo_volume_mt"],
                "cost_now_per_mt": cost_now["landed_cost_per_mt"],
                "cost_hold_p50_per_mt": cost_hold_p50["landed_cost_per_mt"],
                "cost_hold_p10_per_mt": cost_hold_p10["landed_cost_per_mt"],
                "cost_hold_p90_per_mt": cost_hold_p90["landed_cost_per_mt"],
                "expected_savings_total_usd": total_savings,
                "recommendation": rec,
                "cost_now_detail": cost_now,
                "cost_hold_detail": cost_hold_p50
            })
            
        summary_df = pd.DataFrame(eval_rows).sort_values("cost_now_per_mt").reset_index(drop=True)
        best_vessel_now = summary_df.iloc[0]
        
        # Best overall option across all combinations
        best_overall = summary_df.sort_values("cost_hold_p50_per_mt").iloc[0]
        
        return {
            "as_of_date": preds["date"],
            "corridor": f"{origin} -> {destination} ({preds['distance_nm']} NM)",
            "market_rates": preds,
            "comparison_table": summary_df,
            "best_vessel_class": best_vessel_now["vessel"],
            "best_timing_recommendation": best_vessel_now["recommendation"],
            "expected_savings_usd": best_vessel_now["expected_savings_total_usd"]
        }

    def solve_monthly_fleet_milp(self, total_demand_mt=200000, origin="Gladstone", destination_options=None, as_of_date=None):
        """Mixed Integer Linear Program (MILP) allocating vessels to meet total cargo demand at minimum landed cost."""
        if destination_options is None:
            destination_options = ["Haldia", "Paradip"]
            
        vessel_classes = ["Handysize", "Supramax", "Panamax"]
        timing_options = ["NOW", "HOLD"]
        
        # Build candidate options list
        options = []
        for dest in destination_options:
            for v in vessel_classes:
                for t in timing_options:
                    cost_info = self.calculate_single_voyage_cost(v, origin, dest, timing=t, scenario="P50", as_of_date=as_of_date)
                    options.append({
                        "destination": dest,
                        "vessel_class": v,
                        "timing": t,
                        "capacity_mt": cost_info["cargo_volume_mt"],
                        "total_cost_usd": cost_info["total_landed_cost_usd"],
                        "cost_per_mt": cost_info["landed_cost_per_mt"],
                        "detail": cost_info
                    })
                    
        opt_df = pd.DataFrame(options)
        num_options = len(opt_df)
        
        # Cost vector c (Objective: minimize total dollars)
        c = opt_df["total_cost_usd"].values
        
        # Integrality: all variables must be non-negative integers
        integrality = np.ones(num_options)
        
        # Constraint 1: Total Delivered Cargo >= total_demand_mt
        # sum(capacity_i * x_i) >= total_demand_mt
        A_row1 = opt_df["capacity_mt"].values.reshape(1, -1)
        b_l_row1 = [float(total_demand_mt)]
        b_u_row1 = [np.inf]
        
        # Constraint 2: Plant Operational Safety (At least 1 vessel must FIX NOW to maintain stock)
        # sum_{i with timing==NOW}(x_i) >= 1
        A_row2 = (opt_df["timing"] == "NOW").astype(float).values.reshape(1, -1)
        b_l_row2 = [1.0]
        b_u_row2 = [np.inf]
        
        # Combine linear constraints
        A = np.vstack([A_row1, A_row2])
        b_l = np.array(b_l_row1 + b_l_row2)
        b_u = np.array(b_u_row1 + b_u_row2)
        constraints = LinearConstraint(A, b_l, b_u)
        
        # Solve with SciPy HiGHS MILP Solver
        res = milp(c=c, integrality=integrality, constraints=constraints)
        
        if not res.success:
            return {"success": False, "message": res.status_message}
            
        opt_df["selected_vessels"] = np.round(res.x).astype(int)
        allocated = opt_df[opt_df["selected_vessels"] > 0].copy().reset_index(drop=True)
        
        total_delivered_mt = (allocated["selected_vessels"] * allocated["capacity_mt"]).sum()
        total_procurement_cost = res.fun
        avg_cost_per_mt = total_procurement_cost / total_delivered_mt
        
        # Calculate benchmark: What would this cost if blindly fixing all on SPOT today?
        blind_spot_cost = 0.0
        for _, row in allocated.iterrows():
            spot_cost = self.calculate_single_voyage_cost(row["vessel_class"], origin, row["destination"], timing="NOW", as_of_date=as_of_date)
            blind_spot_cost += row["selected_vessels"] * spot_cost["total_landed_cost_usd"]
            
        net_savings_usd = blind_spot_cost - total_procurement_cost
        
        return {
            "success": True,
            "total_demand_required_mt": total_demand_mt,
            "total_delivered_mt": total_delivered_mt,
            "total_procurement_cost_usd": round(total_procurement_cost, 0),
            "average_landed_cost_per_mt": round(avg_cost_per_mt, 2),
            "blind_spot_cost_usd": round(blind_spot_cost, 0),
            "net_optimizer_savings_usd": round(net_savings_usd, 0),
            "allocation_schedule": allocated
        }

    def run_historical_backtest(self):
        """Simulate the FIX NOW vs HOLD decision over the untouched 20% holdout test set."""
        print("=" * 80)
        print("RUNNING HISTORICAL BACKTEST SIMULATION OVER UNTOUCHED 20% TEST SET")
        print("=" * 80)
        
        test_preds_file = os.path.join(self.workspace, "test_set_quantile_predictions.csv")
        test_df = pd.read_csv(test_preds_file)
        test_df["date"] = pd.to_datetime(test_df["date"])
        
        cargo_mt = 75000.0  # Standard Panamax Cargo
        
        dates = []
        actual_spot_costs = []
        realized_decision_costs = []
        cumulative_savings = []
        decisions = []
        
        running_savings = 0.0
        
        for idx in range(len(test_df)):
            row = test_df.iloc[idx]
            dt = row["date"]
            
            # 1. Spot freight rate on date t
            spot_rate = row["actual_freight_rate"] # Today's actual market rate
            pred_15d_p50 = row["p50_expected_median"] # Model's 15d forward forecast
            
            # Actual rate 15 days later (the ground truth outcome if you held)
            # In our setup, actual_freight_rate in row is the target_15d_ahead
            # Let's get the contemporaneous spot from market_df
            m_row = self.market_df[self.market_df["date"] == dt]
            if len(m_row) == 0:
                continue
                
            true_spot_today = float(m_row.iloc[0]["target_freight_rate_proxy"])
            true_rate_15d_later = float(row["actual_freight_rate"])
            
            # Decision Rule:
            # If Model expects rate to drop by more than $0.20/MT -> HOLD; else FIX NOW
            expected_change = pred_15d_p50 - true_spot_today
            
            if expected_change < -0.20:
                decision = "HOLD"
                # If we hold, we pay the rate 15 days later
                cost_paid = true_rate_15d_later * cargo_mt
            else:
                decision = "FIX NOW"
                # If we fix now, we pay today's spot
                cost_paid = true_spot_today * cargo_mt
                
            # Naive baseline: Always fix immediately today
            naive_cost = true_spot_today * cargo_mt
            
            # Dollar savings of following model decision
            dollar_saving = naive_cost - cost_paid
            running_savings += dollar_saving
            
            dates.append(dt)
            actual_spot_costs.append(naive_cost)
            realized_decision_costs.append(cost_paid)
            cumulative_savings.append(running_savings)
            decisions.append(decision)
            
        backtest_results = pd.DataFrame({
            "date": dates,
            "decision": decisions,
            "cumulative_savings_usd": cumulative_savings
        })
        
        total_saved = cumulative_savings[-1]
        print(f"Backtest Completed across {len(dates)} trading days:")
        print(f"  Total Cumulative Savings for SAIL: ${total_saved:,.2f}")
        print(f"  Fix Decisions: {decisions.count('FIX NOW')}")
        print(f"  Hold Decisions: {decisions.count('HOLD')}")
        
        # Save backtest plot
        plt.figure(figsize=(12, 6), dpi=300)
        plt.plot(dates, np.array(cumulative_savings) / 1000.0, color="#10b981", linewidth=2.5, label="Cumulative Savings vs Blind Spot Fixing ($k)")
        plt.axhline(0, color="#64748b", linestyle="--", alpha=0.7)
        plt.fill_between(dates, 0, np.array(cumulative_savings) / 1000.0, color="#10b981", alpha=0.15)
        
        plt.title(f"SAIL Freight Decision Engine: Cumulative Net Savings Backtest\nTotal Verified Savings: ${total_saved:,.0f} over Holdout Test Set", fontsize=13, weight="bold", pad=15)
        plt.xlabel("Date", fontsize=11, weight="bold")
        plt.ylabel("Cumulative Savings ($ Thousand USD)", fontsize=11, weight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
        plt.tight_layout()
        
        plot_path = os.path.join(self.workspace, "backtest_cumulative_savings.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[SAVED] Backtest Savings Plot: {plot_path}")
        
        csv_path = os.path.join(self.workspace, "backtest_results.csv")
        backtest_results.to_csv(csv_path, index=False)
        print(f"[SAVED] Backtest Results CSV: {csv_path}")
        
        return backtest_results

if __name__ == "__main__":
    optimizer = FreightOptimizer()
    
    print("\n" + "=" * 80)
    print("DEMO 1: SINGLE VOYAGE CHARTERING DECISION (Gladstone -> Haldia)")
    print("=" * 80)
    single_res = optimizer.solve_single_shipment_optimization("Gladstone", "Haldia")
    print(f"Corridor: {single_res['corridor']}")
    print(f"As of Date: {single_res['as_of_date']}")
    print(f"Optimal Vessel Choice: {single_res['best_vessel_class']}")
    print(f"Timing Recommendation: {single_res['best_timing_recommendation']}")
    print("\nVessel Comparison Table:")
    print(single_res["comparison_table"][["vessel", "capacity_mt", "cost_now_per_mt", "cost_hold_p50_per_mt", "recommendation"]].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("DEMO 2: ENTERPRISE FLEET ALLOCATION MILP (200,000 MT Demand)")
    print("=" * 80)
    milp_res = optimizer.solve_monthly_fleet_milp(200000, origin="Gladstone", destination_options=["Haldia", "Paradip"])
    if milp_res["success"]:
        print(f"Total Cargo Required: {milp_res['total_demand_required_mt']:,} MT")
        print(f"Total Cargo Delivered: {milp_res['total_delivered_mt']:,} MT")
        print(f"Total Landed Cost: ${milp_res['total_procurement_cost_usd']:,.0f} (Avg: ${milp_res['average_landed_cost_per_mt']:.2f}/MT)")
        print(f"Net Savings vs Blind Spot Fixing: ${milp_res['net_optimizer_savings_usd']:,.0f}")
        print("\nOptimal Fleet Schedule:")
        print(milp_res["allocation_schedule"][["selected_vessels", "vessel_class", "destination", "timing", "cost_per_mt"]].to_string(index=False))
        
    print("\n" + "=" * 80)
    print("DEMO 3: RUNNING HISTORICAL BACKTEST SIMULATION")
    print("=" * 80)
    optimizer.run_historical_backtest()
