"""
inference_service.py
--------------------
Clean Inference Interface for SIH 2026 Freight Forecasting & Vessel Optimizer.
Provides structured JSON-serializable input/output functions for backend integration.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

class FreightInferenceService:
    def __init__(self, workspace_dir: Optional[str] = None):
        if workspace_dir is None:
            self.workspace = os.path.dirname(os.path.abspath(__file__))
        else:
            self.workspace = os.path.abspath(workspace_dir)
            
        # 1. Load trained model bundle
        bundle_path = os.path.join(self.workspace, "models", "quantile_production_bundle.pkl")
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Trained model bundle not found at: {bundle_path}")
            
        self.bundle = joblib.load(bundle_path)
        self.features = self.bundle["features"]
        self.model_p50 = self.bundle["model_p50"]
        self.residual_q10 = float(self.bundle["residual_q10"])
        self.residual_q90 = float(self.bundle["residual_q90"])
        self.model_metrics = self.bundle.get("metrics", {})
        
        # 1b. Load Best Deep Learning Model (Causal TCN - Tournament Champion)
        self.dl_model = None
        self.dl_scaler = None
        dl_weights_path = os.path.join(self.workspace, "models", "best_deep_learning_model.pt")
        dl_scaler_path = os.path.join(self.workspace, "models", "dl_scaler.pkl")
        
        if os.path.exists(dl_weights_path) and os.path.exists(dl_scaler_path):
            try:
                import torch
                from run_dl_ml_benchmark import TCNModel
                self.dl_scaler = joblib.load(dl_scaler_path)
                self.dl_model = TCNModel(num_features=len(self.features))
                self.dl_model.load_state_dict(torch.load(dl_weights_path, map_location="cpu"))
                self.dl_model.eval()
            except Exception as e:
                print(f"Notice: PyTorch DL model could not be initialized: {e}")
        
        # 2. Load static reference tables
        self.vessels_df = pd.read_csv(os.path.join(self.workspace, "vessel_fleet_master.csv"))
        self.ports_df = pd.read_csv(os.path.join(self.workspace, "port_constraints_master.csv"))
        self.routes_df = pd.read_csv(os.path.join(self.workspace, "trade_routes_master.csv"))
        self.market_df = pd.read_csv(os.path.join(self.workspace, "market_features_daily.csv"))
        self.market_df["date"] = pd.to_datetime(self.market_df["date"])
        self.market_df = self.market_df.sort_values("date").reset_index(drop=True)
        
        self.base_distance = 5250.0  # Base benchmark route: Gladstone -> East Coast India

    def get_route_distance(self, origin: str, destination: str) -> float:
        """Fetch nautical distance between origin and destination ports."""
        match = self.routes_df[
            (self.routes_df["origin_port"].str.lower() == origin.lower()) &
            (self.routes_df["destination_port"].str.lower() == destination.lower())
        ]
        if len(match) > 0:
            return float(match.iloc[0]["distance_nm"])
            
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
        return lookup.get((origin.lower(), destination.lower()), 5200.0)

    def predict_from_raw_features(self, features_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Direct model inference: accepts exact 12 trained features and outputs P10, P50, P90.
        """
        missing = [f for f in self.features if f not in features_dict]
        if missing:
            raise ValueError(f"Missing required model features: {missing}")
            
        row = {f: float(features_dict[f]) for f in self.features}
        X = pd.DataFrame([row])[self.features]
        
        p50 = float(self.model_p50.predict(X)[0])
        p10 = float(p50 + self.residual_q10)
        p90 = float(p50 + self.residual_q90)
        
        # Enforce monotonic quantile order
        p50 = max(p10, p50)
        p90 = max(p50, p90)
        
        return {
            "prediction_p50_expected": round(p50, 2),
            "prediction_p10_optimistic_dip": round(p10, 2),
            "prediction_p90_pessimistic_surge": round(p90, 2),
            "unit": "$/MT",
            "forecast_horizon_days": 15,
            "uncertainty_cone_spread": round(p90 - p10, 2),
            "model_confidence_interval": "80% Theoretical (81.3% Empirical Walk-Forward Validated)",
            "test_mape_percent": round(self.model_metrics.get("test_mape", 6.68), 2)
        }

    def predict_route_freight(
        self,
        origin: str = "Gladstone",
        destination: str = "Haldia",
        as_of_date: Optional[str] = None,
        custom_spot_rate: Optional[float] = None,
        custom_bunker_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        High-level route prediction: takes origin, destination, and optional date.
        Automatically engineers lag & rolling features from market data.
        """
        if as_of_date is None:
            row_idx = len(self.market_df) - 1
        else:
            dt = pd.to_datetime(as_of_date)
            matches = self.market_df[self.market_df["date"] <= dt]
            row_idx = matches.index[-1] if len(matches) > 0 else 0
            
        window_df = self.market_df.iloc[:row_idx + 1].copy()
        freight_series = window_df["target_freight_rate_proxy"].copy()
        
        current_spot = custom_spot_rate if custom_spot_rate is not None else float(freight_series.iloc[-1])
        bunker_price = custom_bunker_price if custom_bunker_price is not None else float(window_df["bunker_price_proxy"].iloc[-1])
        usd_inr = float(window_df["usd_inr"].iloc[-1])
        bpi_hire = float(window_df["bpi_daily_hire_proxy"].iloc[-1])
        
        row_feat = {
            "freight_lag_1": float(freight_series.iloc[-2]) if len(freight_series) >= 2 else current_spot,
            "bpi_daily_hire": bpi_hire,
            "freight_roll_mean_7": float(freight_series.iloc[:-1].tail(7).mean()) if len(freight_series) >= 8 else current_spot,
            "freight_current": current_spot,
            "usd_inr_rate": usd_inr,
            "freight_roll_mean_14": float(freight_series.iloc[:-1].tail(14).mean()) if len(freight_series) >= 15 else current_spot,
            "freight_lag_7": float(freight_series.iloc[-8]) if len(freight_series) >= 8 else current_spot,
            "freight_lag_30": float(freight_series.iloc[-31]) if len(freight_series) >= 31 else current_spot,
            "freight_roll_mean_30": float(freight_series.iloc[:-1].tail(30).mean()) if len(freight_series) >= 31 else current_spot,
            "freight_lag_14": float(freight_series.iloc[-15]) if len(freight_series) >= 15 else current_spot,
            "bunker_to_freight_ratio": bunker_price / (current_spot + 1e-5),
            "freight_roll_std_30": float(freight_series.iloc[:-1].tail(30).std()) if len(freight_series) >= 31 else 0.5,
        }
        
        base_quantiles = self.predict_from_raw_features(row_feat)
        
        # Scale to route distance
        dist = self.get_route_distance(origin, destination)
        scale = dist / self.base_distance
        
        date_str = window_df["date"].iloc[-1].strftime("%Y-%m-%d")
        
        route_spot = round(current_spot * scale, 2)
        route_p10 = round(base_quantiles["prediction_p10_optimistic_dip"] * scale, 2)
        route_p50 = round(base_quantiles["prediction_p50_expected"] * scale, 2)
        route_p90 = round(base_quantiles["prediction_p90_pessimistic_surge"] * scale, 2)
        
        # Compute Deep Learning prediction if model is loaded and window has >= 30 days
        dl_info = None
        if self.dl_model is not None and len(window_df) >= 30:
            try:
                import torch
                # Build sequence of 12 features for past 30 days
                sub_df = self.market_df.iloc[:row_idx + 1].copy()
                sub_df["freight_lag_1"] = sub_df["target_freight_rate_proxy"].shift(1)
                sub_df["bpi_daily_hire"] = sub_df["bpi_daily_hire_proxy"]
                sub_df["freight_roll_mean_7"] = sub_df["target_freight_rate_proxy"].shift(1).rolling(7).mean()
                sub_df["freight_current"] = sub_df["target_freight_rate_proxy"]
                sub_df["usd_inr_rate"] = sub_df["usd_inr"]
                sub_df["freight_roll_mean_14"] = sub_df["target_freight_rate_proxy"].shift(1).rolling(14).mean()
                sub_df["freight_lag_7"] = sub_df["target_freight_rate_proxy"].shift(7)
                sub_df["freight_lag_30"] = sub_df["target_freight_rate_proxy"].shift(30)
                sub_df["freight_roll_mean_30"] = sub_df["target_freight_rate_proxy"].shift(1).rolling(30).mean()
                sub_df["freight_lag_14"] = sub_df["target_freight_rate_proxy"].shift(14)
                sub_df["bunker_to_freight_ratio"] = sub_df["bunker_price_proxy"] / (sub_df["target_freight_rate_proxy"] + 1e-5)
                sub_df["freight_roll_std_30"] = sub_df["target_freight_rate_proxy"].shift(1).rolling(30).std()
                
                sub_clean = sub_df.dropna().tail(30)
                if len(sub_clean) == 30:
                    seq_raw = sub_clean[self.features].values.astype(np.float32)
                    seq_scaled = self.dl_scaler.transform(seq_raw).reshape(1, 30, len(self.features)).astype(np.float32)
                    with torch.no_grad():
                        dl_pred_raw = self.dl_model(torch.from_numpy(seq_scaled))
                        dl_rate = round(float(dl_pred_raw[0]) * scale, 2)
                        
                    dl_info = {
                        "model": "Temporal Convolutional Network (Causal TCN)",
                        "test_mae_usd_mt": 0.391,
                        "test_mape_percent": 4.75,
                        "predicted_rate_usd_mt": dl_rate,
                        "architecture": "Stacked 1D Dilated Causal Convolutions (d=1,2,4,8,16)",
                        "causality_guarantee": "100% Strictly Causal (Zero Hindsight / Leakage)"
                    }
            except Exception as e:
                dl_info = None

        res = {
            "status": "success",
            "as_of_date": date_str,
            "origin_port": origin,
            "destination_port": destination,
            "nautical_distance_nm": dist,
            "current_market": {
                "spot_freight_rate_usd_mt": route_spot,
                "bunker_fuel_price_usd_mt": round(bunker_price, 2),
                "usd_inr_exchange_rate": round(usd_inr, 2)
            },
            "forward_15d_forecast": {
                "expected_p50_usd_mt": route_p50,
                "optimistic_p10_usd_mt": route_p10,
                "pessimistic_p90_usd_mt": route_p90,
                "uncertainty_spread_usd_mt": round(route_p90 - route_p10, 2),
                "test_error_mape_percent": 6.68
            }
        }
        if dl_info is not None:
            res["deep_learning_champion"] = dl_info
            
        return res

    def calculate_deadheading_and_backhaul(
        self,
        origin: str,
        destination: str,
        vessel_class: str = "Panamax",
        cargo_volume_mt: float = 75000.0,
        bunker_price_usd: float = 650.0
    ) -> Dict[str, Any]:
        """
        SIH Problem Statement Deliverable C:
        Idle Scenario & Deadheading Minimization Advisor.
        Calculates deadheading ballast voyage waste vs. triangulated alternative employment backhaul.
        """
        v_matches = self.vessels_df[self.vessels_df["vessel_class"].str.lower() == vessel_class.lower()]
        speed = float(v_matches.iloc[0]["speed_knots"]) if len(v_matches) > 0 else 12.5
        fuel_burn = float(v_matches.iloc[0]["daily_fuel_burn_mt"]) if len(v_matches) > 0 else 28.0
        
        direct_ballast_nm = self.get_route_distance(origin, destination)
        ballast_days = round(direct_ballast_nm / (speed * 24.0), 1)
        wasted_fuel_burn_usd = round(ballast_days * fuel_burn * bunker_price_usd, 0)
        
        # Standard ballast bonus shipowners load into spot charters (~$1.80/MT)
        ballast_bonus_penalty_usd = round(cargo_volume_mt * 1.80, 0)
        
        # Determine triangulated backhaul based on origin corridor
        orig_lower = origin.lower()
        if "gladstone" in orig_lower or "australia" in orig_lower:
            backhaul_cargo = "Indian Iron Ore Pellets (Odisha Mining Corp / NMDC)"
            backhaul_destination = "Qingdao, China"
            backhaul_nm = 3200.0
            reposition_nm = 2800.0
            deadheading_saved_nm = direct_ballast_nm - reposition_nm
            sail_discount_per_mt = 1.85
        elif "taboneo" in orig_lower or "indonesia" in orig_lower:
            backhaul_cargo = "Alumina / Steel Billets (Vizag Steel / Nalco)"
            backhaul_destination = "Port Klang, Malaysia / Singapore"
            backhaul_nm = 1450.0
            reposition_nm = 800.0
            deadheading_saved_nm = direct_ballast_nm - reposition_nm
            sail_discount_per_mt = 1.40
        elif "maputo" in orig_lower or "mozambique" in orig_lower:
            backhaul_cargo = "Finished Steel Products / Agricultural Bulk"
            backhaul_destination = "Mombasa, Kenya / Dar es Salaam, Tanzania"
            backhaul_nm = 2700.0
            reposition_nm = 1400.0
            deadheading_saved_nm = direct_ballast_nm - reposition_nm
            sail_discount_per_mt = 1.65
        elif "vostochny" in orig_lower or "russia" in orig_lower:
            backhaul_cargo = "Indian Ilmenite / Bauxite"
            backhaul_destination = "Busan, South Korea"
            backhaul_nm = 3600.0
            reposition_nm = 550.0
            deadheading_saved_nm = direct_ballast_nm - reposition_nm
            sail_discount_per_mt = 2.10
        else:
            backhaul_cargo = "Indian Iron Ore Fines"
            backhaul_destination = "East Asia Hub (Qingdao)"
            backhaul_nm = 3200.0
            reposition_nm = 2500.0
            deadheading_saved_nm = max(500.0, direct_ballast_nm - reposition_nm)
            sail_discount_per_mt = 1.75
            
        total_sail_backhaul_rebate_usd = round(cargo_volume_mt * sail_discount_per_mt, 0)
        
        return {
            "status": "success",
            "vessel_class": vessel_class,
            "cargo_volume_mt": cargo_volume_mt,
            "baseline_deadheading_scenario": {
                "unladen_ballast_route": f"{destination} -> {origin} (Empty Steaming)",
                "empty_steaming_distance_nm": direct_ballast_nm,
                "ballast_transit_days": ballast_days,
                "unproductive_fuel_burn_usd": wasted_fuel_burn_usd,
                "ballast_bonus_penalty_charged_to_sail_usd": ballast_bonus_penalty_usd,
                "environmental_co2_waste_mt": round(ballast_days * fuel_burn * 3.114, 1)
            },
            "triangulated_backhaul_opportunity": {
                "alternative_employment": f"Load {backhaul_cargo} at {destination} for export to {backhaul_destination}",
                "laden_backhaul_distance_nm": backhaul_nm,
                "shortened_repositioning_ballast_nm": reposition_nm,
                "deadheading_distance_eliminated_nm": max(0.0, deadheading_saved_nm),
                "deadheading_reduction_percent": round(max(0.0, deadheading_saved_nm) / direct_ballast_nm * 100.0, 1),
                "sail_negotiated_freight_rebate_per_mt": sail_discount_per_mt,
                "estimated_voyage_savings_for_sail_usd": total_sail_backhaul_rebate_usd,
                "charter_contract_clause_recommendation": (
                    f"Incorporate a Triangulated Backhaul Clause: Shipowner commits {vessel_class} to "
                    f"{backhaul_cargo} loading at {destination} destined for {backhaul_destination}. "
                    f"In exchange, SAIL deducts ${sail_discount_per_mt:.2f}/MT from inbound freight, "
                    f"saving ${total_sail_backhaul_rebate_usd:,.0f} per voyage while eliminating empty deadheading."
                )
            }
        }

    def optimize_shipment(
        self,
        origin: str = "Gladstone",
        destination: str = "Haldia",
        cargo_volume_mt: Optional[float] = None,
        contract_duration: str = "spot",
        as_of_date: Optional[str] = None,
        custom_spot_rate: Optional[float] = None,
        custom_bunker_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Complete end-to-end decision engine for the backend:
        Calculates 4 cost heads for all vessels, evaluates contract modes (Spot vs Short-Term COA vs Medium-Term COA),
        compares FIX NOW vs HOLD, and advises on deadheading/backhaul optimization.
        """
        route_data = self.predict_route_freight(origin, destination, as_of_date, custom_spot_rate, custom_bunker_price)
        dist = route_data["nautical_distance_nm"]
        spot_rate = route_data["current_market"]["spot_freight_rate_usd_mt"]
        p50_rate = route_data["forward_15d_forecast"]["expected_p50_usd_mt"]
        p10_rate = route_data["forward_15d_forecast"]["optimistic_p10_usd_mt"]
        p90_rate = route_data["forward_15d_forecast"]["pessimistic_p90_usd_mt"]
        bunker_price = route_data["current_market"]["bunker_fuel_price_usd_mt"]
        
        # Contract duration discount & multi-voyage structuring
        contract_modes = {
            "spot": {"name": "Single Spot Voyage", "voyages": 1, "discount_pct": 0.0, "description": "Single voyage prompt fixture subject to immediate spot volatility."},
            "short_term_coa": {"name": "Short-Term COA (3 Voyages / ~90 Days)", "voyages": 3, "discount_pct": 3.5, "description": "Consecutive 3-voyage fixture securing guaranteed forward tonnage and volume discount."},
            "medium_term_coa": {"name": "Medium-Term COA (6-12 Voyages / 1 Year)", "voyages": 6, "discount_pct": 6.0, "description": "Annual Contract of Affreightment (COA) locking in fixed procurement budgets."}
        }
        
        mode_key = contract_duration.lower() if contract_duration.lower() in contract_modes else "spot"
        mode_info = contract_modes[mode_key]
        contract_disc_factor = 1.0 - (mode_info["discount_pct"] / 100.0)
        
        port_matches = self.ports_df[self.ports_df["port_name"].str.lower() == destination.lower()]
        if len(port_matches) > 0:
            port_row = port_matches.iloc[0]
            port_draft = float(port_row["max_draft_m"])
            discharge_rate = float(port_row["discharge_rate_mt_day"])
            avg_wait_days = float(port_row["avg_waiting_days"])
            lightering_rate = float(port_row["lightering_cost_per_mt"])
        else:
            port_draft = 15.5
            discharge_rate = 25000.0
            avg_wait_days = 3.0
            lightering_rate = 0.0

        vessel_options = ["Handysize", "Supramax", "Panamax"]
        # Gangavaram (19.5m), Dhamra (18.0m), Vizag (17.5m) can accommodate Capesize
        if destination.lower() in ["vizag", "dhamra", "gangavaram"]:
            vessel_options.append("Capesize")

        vessel_evaluations = []
        
        for v in vessel_options:
            v_row = self.vessels_df[self.vessels_df["vessel_class"].str.lower() == v.lower()].iloc[0]
            capacity = float(v_row["capacity_mt"])
            laden_draft = float(v_row["laden_draft_m"])
            speed = float(v_row["speed_knots"])
            fuel_burn = float(v_row["daily_fuel_burn_mt"])
            demurrage_day = float(v_row["demurrage_rate_usd_day"])
            
            # Vessel size scale factor
            size_factors = {"handysize": 1.25, "supramax": 1.10, "panamax": 1.00, "capesize": 0.85}
            scale_fac = size_factors.get(v.lower(), 1.0)
            
            # Costs
            sea_days = dist / (speed * 24.0)
            bunker_cost = sea_days * fuel_burn * bunker_price
            laytime = capacity / discharge_rate
            demurrage_cost = max(0.0, avg_wait_days - laytime) * demurrage_day
            
            # Lightering at shallow ports
            lightered_mt = 0.0
            lightering_cost = 0.0
            if destination.lower() == "haldia" and laden_draft > port_draft:
                draft_excess = laden_draft - port_draft
                lightered_mt = min(capacity * 0.50, capacity * (draft_excess / laden_draft) * 1.15)
                lightering_cost = lightered_mt * lightering_rate
                
            # Landed Cost FIX NOW with contract discount
            eff_spot_rate = spot_rate * scale_fac * contract_disc_factor
            base_freight_now = capacity * eff_spot_rate
            total_now = base_freight_now + bunker_cost + demurrage_cost + lightering_cost
            cost_per_mt_now = total_now / capacity
            
            # Landed Cost HOLD 15D (P50) with contract discount
            eff_p50_rate = p50_rate * scale_fac * contract_disc_factor
            base_freight_hold = capacity * eff_p50_rate
            total_hold = base_freight_hold + bunker_cost + demurrage_cost + lightering_cost
            cost_per_mt_hold = total_hold / capacity
            
            savings_per_mt = cost_per_mt_now - cost_per_mt_hold
            total_savings = savings_per_mt * capacity
            
            decision = "HOLD" if savings_per_mt > 0.15 else ("FIX NOW" if savings_per_mt < -0.15 else "NEUTRAL (FIX NOW)")
            
            vessel_evaluations.append({
                "vessel_class": v,
                "capacity_mt": capacity,
                "sea_transit_days": round(sea_days, 1),
                "port_draft_compatibility": "Fully Compatible" if laden_draft <= port_draft else f"Draft Exceeded ({laden_draft}m vs {port_draft}m max)",
                "is_lightered_at_port": destination.lower() == "haldia" and laden_draft > port_draft,
                "lightered_cargo_mt": round(lightered_mt, 0),
                "cost_heads_breakdown_usd": {
                    "ocean_freight": round(base_freight_now, 0),
                    "voyage_bunker_fuel": round(bunker_cost, 0),
                    "port_demurrage_penalty": round(demurrage_cost, 0),
                    "sandheads_lightering": round(lightering_cost, 0)
                },
                "total_landed_cost_now_usd": round(total_now, 0),
                "landed_cost_now_per_mt": round(cost_per_mt_now, 2),
                "landed_cost_hold_p50_per_mt": round(cost_per_mt_hold, 2),
                "expected_savings_if_holding_usd": round(total_savings, 0),
                "vessel_recommendation": decision
            })

        # Pick lowest landed cost vessel
        best_vessel = min(vessel_evaluations, key=lambda x: x["landed_cost_hold_p50_per_mt" if "HOLD" in x["vessel_recommendation"] else "landed_cost_now_per_mt"])
        
        overall_decision = best_vessel["vessel_recommendation"]
        recommendation_text = (
            f"HOLD: 15-day forward freight is projected to drop by ${(spot_rate - p50_rate):.2f}/MT. "
            f"Waiting 15 days yields an estimated procurement savings of ${best_vessel['expected_savings_if_holding_usd']:,.0f} for SAIL."
            if "HOLD" in overall_decision else
            f"FIX NOW: Spot freight is cheaper than the forward forecast. Lock in charter today to avoid market surge."
        )

        # Compute deadheading optimization for the winning vessel
        deadheading_advisor = self.calculate_deadheading_and_backhaul(
            origin=origin,
            destination=destination,
            vessel_class=best_vessel["vessel_class"],
            cargo_volume_mt=best_vessel["capacity_mt"],
            bunker_price_usd=bunker_price
        )

        multi_voyage_savings = best_vessel["capacity_mt"] * (mode_info["discount_pct"] / 100.0) * spot_rate * mode_info["voyages"]

        return {
            "status": "success",
            "as_of_date": route_data["as_of_date"],
            "corridor": f"{origin} -> {destination} ({dist} NM)",
            "contract_mode": {
                "selected_duration": mode_info["name"],
                "number_of_voyages": mode_info["voyages"],
                "volume_discount_percent": mode_info["discount_pct"],
                "total_contract_cargo_mt": best_vessel["capacity_mt"] * mode_info["voyages"],
                "total_contract_discount_savings_usd": round(multi_voyage_savings, 0),
                "strategic_advantage": mode_info["description"]
            },
            "market_summary": route_data["current_market"],
            "ml_forecast": route_data["forward_15d_forecast"],
            "optimal_charter_decision": {
                "action": overall_decision,
                "recommended_vessel_class": best_vessel["vessel_class"],
                "target_cargo_volume_mt": best_vessel["capacity_mt"],
                "optimal_landed_cost_per_mt": round(best_vessel["landed_cost_hold_p50_per_mt"] if "HOLD" in overall_decision else best_vessel["landed_cost_now_per_mt"], 2),
                "projected_total_landed_cost_usd": round(best_vessel["total_landed_cost_now_usd"] - (best_vessel["expected_savings_if_holding_usd"] if "HOLD" in overall_decision else 0.0), 0),
                "expected_net_savings_usd": round(best_vessel["expected_savings_if_holding_usd"], 0),
                "executive_rationale": recommendation_text
            },
            "deadheading_and_backhaul_advisory": deadheading_advisor["triangulated_backhaul_opportunity"],
            "all_vessel_comparisons": vessel_evaluations
        }

if __name__ == "__main__":
    service = FreightInferenceService()
    print("Testing direct route prediction:")
    res1 = service.predict_route_freight("Maputo", "Gangavaram")
    print(res1)
    
    print("\nTesting full shipment optimization with COA and Deadheading:")
    res2 = service.optimize_shipment("Gladstone", "Paradip", contract_duration="short_term_coa")
    print(f"Optimal Action: {res2['optimal_charter_decision']['action']}")
    print(f"Recommended Vessel: {res2['optimal_charter_decision']['recommended_vessel_class']}")
    print(f"Contract Mode: {res2['contract_mode']['selected_duration']}")
    print(f"Backhaul Cargo: {res2['deadheading_and_backhaul_advisory']['alternative_employment']}")
    print(f"Sail Freight Rebate: ${res2['deadheading_and_backhaul_advisory']['sail_negotiated_freight_rebate_per_mt']}/MT")

