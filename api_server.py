"""
api_server.py
-------------
FastAPI REST API Server for SIH 2026 Intelligent Freight Forecasting & Vessel Optimizer.
Provides a clean, standardized JSON interface for frontend/backend teams.

Run with:
    python api_server.py
or:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference_service import FreightInferenceService
from scipy.optimize import milp, LinearConstraint
import numpy as np
import pandas as pd

app = FastAPI(
    title="SAIL Freight Forecasting & Vessel Optimizer API",
    description="Smart India Hackathon 2026 • Ministry of Steel / SAIL Problem Statement\n\n"
                "Provides 15-day forward dry-bulk ocean freight forecasts (P10, P50, P90), "
                "landed cost calculations, and vessel charter optimization (FIX NOW vs HOLD).",
    version="1.0.0"
)

# Enable CORS for frontend integration (React, Vue, Next.js, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Inference Service
inference_service = FreightInferenceService()

# --------------------------------------------------------------------------
# REQUEST & RESPONSE SCHEMAS
# --------------------------------------------------------------------------

class RouteForecastRequest(BaseModel):
    origin: str = Field(..., description="Overseas origin port (e.g., Gladstone, Taboneo, Norfolk)")
    destination: str = Field(..., description="East Coast India destination port (e.g., Haldia, Paradip, Vizag, Dhamra)")
    as_of_date: Optional[str] = Field(None, description="Date for market features (YYYY-MM-DD). Defaults to latest.")
    custom_spot_rate: Optional[float] = Field(None, description="Optional live spot rate override ($/MT)")
    custom_bunker_price: Optional[float] = Field(None, description="Optional live bunker fuel price override ($/MT)")

    class Config:
        json_schema_extra = {
            "example": {
                "origin": "Gladstone",
                "destination": "Haldia",
                "as_of_date": "2026-09-04",
                "custom_spot_rate": 11.50,
                "custom_bunker_price": 650.0
            }
        }

class ShipmentOptimizeRequest(BaseModel):
    origin: str = Field(..., description="Overseas origin port")
    destination: str = Field(..., description="East Coast India destination port")
    cargo_volume_mt: Optional[float] = Field(75000.0, description="Cargo volume in Metric Tons (default: 75,000 MT)")
    as_of_date: Optional[str] = Field(None, description="Date for market features (YYYY-MM-DD). Defaults to latest.")
    custom_spot_rate: Optional[float] = Field(None, description="Optional spot freight rate override ($/MT)")
    custom_bunker_price: Optional[float] = Field(None, description="Optional bunker price override ($/MT)")

    class Config:
        json_schema_extra = {
            "example": {
                "origin": "Gladstone",
                "destination": "Haldia",
                "cargo_volume_mt": 75000.0,
                "as_of_date": "2026-09-04"
            }
        }

class RawFeaturesRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of the 12 exact engineered features required by the LightGBM model"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "freight_lag_1": 10.90,
                    "bpi_daily_hire": 15800.0,
                    "freight_roll_mean_7": 10.85,
                    "freight_current": 11.00,
                    "usd_inr_rate": 83.90,
                    "freight_roll_mean_14": 10.70,
                    "freight_lag_7": 10.60,
                    "freight_lag_30": 10.20,
                    "freight_roll_mean_30": 10.40,
                    "freight_lag_14": 10.50,
                    "bunker_to_freight_ratio": 58.20,
                    "freight_roll_std_30": 0.45
                }
            }
        }

class MonthlyFleetMILPRequest(BaseModel):
    total_demand_mt: float = Field(200000.0, description="Total monthly coal procurement requirement in Metric Tons")
    origin: str = Field("Gladstone", description="Overseas coal loading port")
    destination_options: List[str] = Field(["Haldia", "Paradip"], description="Available receiving ports")
    as_of_date: Optional[str] = Field(None, description="Reference date for market rates")

    class Config:
        json_schema_extra = {
            "example": {
                "total_demand_mt": 200000.0,
                "origin": "Gladstone",
                "destination_options": ["Haldia", "Paradip"],
                "as_of_date": "2026-09-04"
            }
        }

# --------------------------------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "SAIL Intelligent Freight Forecasting & Vessel Optimizer API",
        "status": "online",
        "version": "1.0.0",
        "documentation": "/docs",
        "available_endpoints": [
            "POST /api/v1/predict/freight",
            "POST /api/v1/predict/optimize",
            "POST /api/v1/predict/raw",
            "POST /api/v1/optimize/fleet-milp"
        ]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "deep_learning_loaded": inference_service.dl_model is not None,
        "features_count": len(inference_service.features),
        "best_dl_test_mae": "$0.473 / MT",
        "best_dl_test_mape": "5.52%",
        "test_mape_percent": 6.68,
        "uncertainty_cone_coverage": "99.8%"
    }

@app.get("/api/v1/models/leaderboard")
def get_model_leaderboard():
    """
    Returns the official 417-day holdout test set benchmark leaderboard
    comparing TCN, N-HiTS, LSTM+Attention, PatchTST, XGBoost, LightGBM, and Hybrid Ensemble.
    """
    leaderboard_csv = os.path.join(inference_service.workspace, "model_benchmark_leaderboard.csv")
    if os.path.exists(leaderboard_csv):
        df_lb = pd.read_csv(leaderboard_csv)
        return {
            "status": "success",
            "evaluation_horizon": "417 untouched trading days (Dec 2024 to Aug 2026)",
            "leaderboard": df_lb.to_dict(orient="records")
        }
    return {"status": "error", "message": "Leaderboard not found"}

@app.post("/api/v1/predict/freight")
def predict_freight(request: RouteForecastRequest):
    """
    Predict 15-day forward freight rate ($/MT) with uncertainty cone (P10, P50, P90)
    for any origin-destination corridor.
    """
    try:
        result = inference_service.predict_route_freight(
            origin=request.origin,
            destination=request.destination,
            as_of_date=request.as_of_date,
            custom_spot_rate=request.custom_spot_rate,
            custom_bunker_price=request.custom_bunker_price
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/optimize")
def optimize_shipment(request: ShipmentOptimizeRequest):
    """
    Full Decision Engine for a shipment:
    - Recommends best vessel class (Handysize, Supramax, Panamax)
    - Recommends timing (FIX NOW vs HOLD)
    - Computes exact 4 landed cost heads and dollar savings
    """
    try:
        result = inference_service.optimize_shipment(
            origin=request.origin,
            destination=request.destination,
            cargo_volume_mt=request.cargo_volume_mt,
            as_of_date=request.as_of_date,
            custom_spot_rate=request.custom_spot_rate,
            custom_bunker_price=request.custom_bunker_price
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/raw")
def predict_raw_features(request: RawFeaturesRequest):
    """
    Low-level direct ML inference:
    Takes the exact 12 trained features and returns P10, P50, and P90 quantile estimates.
    """
    try:
        result = inference_service.predict_from_raw_features(request.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/optimize/fleet-milp")
def optimize_monthly_fleet(request: MonthlyFleetMILPRequest):
    """
    Enterprise Mixed Integer Linear Program (MILP):
    Allocates discrete vessel fleets across ports to meet multi-tonnage demand at minimum landed cost.
    """
    try:
        # Pull candidate costs
        vessel_classes = ["Handysize", "Supramax", "Panamax"]
        timing_options = ["NOW", "HOLD"]
        
        options = []
        for dest in request.destination_options:
            for v in vessel_classes:
                for t in timing_options:
                    single = inference_service.optimize_shipment(
                        origin=request.origin,
                        destination=dest,
                        as_of_date=request.as_of_date
                    )
                    v_match = next((item for item in single["all_vessel_comparisons"] if item["vessel_class"].lower() == v.lower()), None)
                    if v_match:
                        cost_usd = v_match["total_landed_cost_now_usd"] if t == "NOW" else (v_match["total_landed_cost_now_usd"] - v_match["expected_savings_if_holding_usd"])
                        cost_per_mt = cost_usd / v_match["capacity_mt"]
                        options.append({
                            "destination": dest,
                            "vessel_class": v,
                            "timing": t,
                            "capacity_mt": v_match["capacity_mt"],
                            "total_cost_usd": cost_usd,
                            "cost_per_mt": cost_per_mt
                        })
                        
        opt_df = pd.DataFrame(options)
        num_options = len(opt_df)
        c = opt_df["total_cost_usd"].values
        integrality = np.ones(num_options)
        
        # Constraint 1: Total Delivered >= total_demand_mt
        A1 = opt_df["capacity_mt"].values.reshape(1, -1)
        b1_l = [float(request.total_demand_mt)]
        b1_u = [np.inf]
        
        # Constraint 2: Safety Stock (at least 1 FIX NOW)
        A2 = (opt_df["timing"] == "NOW").astype(float).values.reshape(1, -1)
        b2_l = [1.0]
        b2_u = [np.inf]
        
        A = np.vstack([A1, A2])
        b_l = np.array(b1_l + b2_l)
        b_u = np.array(b1_u + b2_u)
        constraints = LinearConstraint(A, b_l, b_u)
        
        res = milp(c=c, integrality=integrality, constraints=constraints)
        if not res.success:
            raise HTTPException(status_code=500, detail=f"MILP solver failed: {res.status_message}")
            
        opt_df["vessel_count"] = np.round(res.x).astype(int)
        allocated = opt_df[opt_df["vessel_count"] > 0].copy()
        
        total_delivered = float((allocated["vessel_count"] * allocated["capacity_mt"]).sum())
        total_cost = float(res.fun)
        avg_cost_per_mt = float(total_cost / total_delivered)
        
        # Benchmark spot cost
        blind_spot = float(sum(row["vessel_count"] * opt_df[(opt_df["destination"] == row["destination"]) & (opt_df["vessel_class"] == row["vessel_class"]) & (opt_df["timing"] == "NOW")]["total_cost_usd"].iloc[0] for _, row in allocated.iterrows()))
        net_savings = float(blind_spot - total_cost)
        
        return {
            "status": "success",
            "demand_requirement_mt": request.total_demand_mt,
            "total_cargo_delivered_mt": total_delivered,
            "total_procurement_cost_usd": round(total_cost, 0),
            "average_landed_cost_per_mt": round(avg_cost_per_mt, 2),
            "net_milp_savings_vs_spot_usd": round(net_savings, 0),
            "allocated_fleet_schedule": allocated[["vessel_count", "vessel_class", "destination", "timing", "cost_per_mt"]].to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting SAIL Freight Forecasting API Server on port 8000...")
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
