# Backend Integration Guide: SAIL Freight Forecasting & Optimizer

> **For Backend Engineers**: This guide explains how to connect your backend (Node.js, Express, Django, Spring Boot, Go, FastAPI, etc.) to the Machine Learning & Optimization Engine for the SIH 2026 Freight Forecasting system.

---

## 1. How the Backend Can Connect (Choose Option A or Option B)

### 🌐 Option A: REST API (Recommended)
Run the lightweight FastAPI service (`api_server.py`). Your backend communicates via standard HTTP `POST` requests and receives JSON responses.
* **Interactive Swagger UI**: `http://localhost:8000/docs`
* **Health Check**: `GET http://localhost:8000/health`

### 📦 Option B: Direct Python In-Process Import
If your backend is written in Python (Django, Flask, FastAPI), you can import the service directly without running a separate HTTP server:
```python
from inference_service import FreightInferenceService

service = FreightInferenceService()
result = service.optimize_shipment(origin="Gladstone", destination="Haldia")
```

---

## 2. The 3 Available Endpoints

### 🚢 Endpoint 1: Full Shipment Optimization (The Main Decision Engine)
* **URL**: `POST /api/v1/predict/optimize`
* **What it does**: Computes the 15-day forward freight rate, evaluates Handysize vs Supramax vs Panamax, calculates all 4 Landed Cost heads, and outputs the **`FIX NOW` vs `HOLD`** recommendation with projected dollar savings.

#### Request JSON:
```json
{
  "origin": "Gladstone",
  "destination": "Haldia",
  "cargo_volume_mt": 75000.0,
  "as_of_date": "2026-09-04",
  "custom_spot_rate": null,
  "custom_bunker_price": null
}
```
* `origin`: Overseas coal port (`Gladstone`, `Taboneo`, `Norfolk`, `Richards Bay`).
* `destination`: East Coast India port (`Haldia`, `Paradip`, `Vizag`, `Dhamra`).
* `cargo_volume_mt` *(optional, default 75000)*: Metric tons of cargo.
* `as_of_date` *(optional)*: Historical date for market features (defaults to latest available trading day).
* `custom_spot_rate` *(optional)*: Live override for today's spot freight rate ($/MT).
* `custom_bunker_price` *(optional)*: Live override for marine fuel ($/MT).

#### Response JSON:
```json
{
  "status": "success",
  "as_of_date": "2026-09-04",
  "corridor": "Gladstone -> Haldia (5250.0 NM)",
  "market_summary": {
    "spot_freight_rate_usd_mt": 12.44,
    "bunker_fuel_price_usd_mt": 800.73,
    "usd_inr_exchange_rate": 94.49
  },
  "ml_forecast": {
    "expected_p50_usd_mt": 10.19,
    "optimistic_p10_usd_mt": 9.49,
    "pessimistic_p90_usd_mt": 11.80,
    "uncertainty_spread_usd_mt": 2.31,
    "test_error_mape_percent": 6.68
  },
  "optimal_charter_decision": {
    "action": "HOLD",
    "recommended_vessel_class": "Panamax",
    "target_cargo_volume_mt": 75000.0,
    "optimal_landed_cost_per_mt": 18.75,
    "projected_total_landed_cost_usd": 1406435.0,
    "expected_net_savings_usd": 168750.0,
    "executive_rationale": "HOLD: 15-day forward freight is projected to drop by $2.25/MT. Waiting 15 days yields an estimated procurement savings of $168,750 for SAIL."
  },
  "all_vessel_comparisons": [
    {
      "vessel_class": "Handysize",
      "capacity_mt": 35000.0,
      "sea_transit_days": 17.5,
      "is_lightered_at_port": true,
      "lightered_cargo_mt": 6037.0,
      "cost_heads_breakdown_usd": {
        "ocean_freight": 544250.0,
        "voyage_bunker_fuel": 252230.0,
        "port_demurrage_penalty": 25333.0,
        "sandheads_lightering": 42262.0
      },
      "total_landed_cost_now_usd": 864076.0,
      "landed_cost_now_per_mt": 24.69,
      "landed_cost_hold_p50_per_mt": 21.88,
      "expected_savings_if_holding_usd": 98438.0,
      "vessel_recommendation": "HOLD"
    },
    {
      "vessel_class": "Supramax",
      "capacity_mt": 55000.0,
      "sea_transit_days": 17.5,
      "is_lightered_at_port": true,
      "lightered_cargo_mt": 21248.0,
      "cost_heads_breakdown_usd": {
        "ocean_freight": 752620.0,
        "voyage_bunker_fuel": 336307.0,
        "port_demurrage_penalty": 0.0,
        "sandheads_lightering": 148736.0
      },
      "total_landed_cost_now_usd": 1237663.0,
      "landed_cost_now_per_mt": 22.50,
      "landed_cost_hold_p50_per_mt": 20.03,
      "expected_savings_if_holding_usd": 136125.0,
      "vessel_recommendation": "HOLD"
    },
    {
      "vessel_class": "Panamax",
      "capacity_mt": 75000.0,
      "sea_transit_days": 17.5,
      "is_lightered_at_port": true,
      "lightered_cargo_mt": 35690.0,
      "cost_heads_breakdown_usd": {
        "ocean_freight": 933000.0,
        "voyage_bunker_fuel": 392358.0,
        "port_demurrage_penalty": 0.0,
        "sandheads_lightering": 249828.0
      },
      "total_landed_cost_now_usd": 1575185.0,
      "landed_cost_now_per_mt": 21.00,
      "landed_cost_hold_p50_per_mt": 18.75,
      "expected_savings_if_holding_usd": 168750.0,
      "vessel_recommendation": "HOLD"
    }
  ]
}
```

---

### 📈 Endpoint 2: Route Freight Forecast Only ($P10, P50, P90$)
* **URL**: `POST /api/v1/predict/freight`
* **What it does**: Predicts only the forward ocean charter rate for any route without running vessel optimization.

#### Request JSON:
```json
{
  "origin": "Gladstone",
  "destination": "Paradip"
}
```

#### Response JSON:
```json
{
  "status": "success",
  "as_of_date": "2026-09-04",
  "origin_port": "Gladstone",
  "destination_port": "Paradip",
  "nautical_distance_nm": 5100.0,
  "current_market": {
    "spot_freight_rate_usd_mt": 12.08,
    "bunker_fuel_price_usd_mt": 800.73,
    "usd_inr_exchange_rate": 94.49
  },
  "forward_15d_forecast": {
    "expected_p50_usd_mt": 9.90,
    "optimistic_p10_usd_mt": 9.22,
    "pessimistic_p90_usd_mt": 11.46,
    "uncertainty_spread_usd_mt": 2.24,
    "test_error_mape_percent": 6.68
  }
}
```

---

### 🔢 Endpoint 3: Direct Low-Level Feature ML Inference
* **URL**: `POST /api/v1/predict/raw`
* **What it does**: Direct model inference for ML engineers. Accepts the exact 12 trained features and returns $P10, P50, P90$.

#### Request JSON:
```json
{
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
```

#### Response JSON:
```json
{
  "prediction_p50_expected": 10.36,
  "prediction_p10_optimistic_dip": 9.66,
  "prediction_p90_pessimistic_surge": 11.97,
  "unit": "$/MT",
  "forecast_horizon_days": 15,
  "uncertainty_cone_spread": 2.31,
  "model_confidence_interval": "80% Theoretical (81.3% Empirical Walk-Forward Validated)",
  "test_mape_percent": 6.68
}
```

---

### 🏭 Endpoint 4: Enterprise Monthly Fleet Allocation (MILP)
* **URL**: `POST /api/v1/optimize/fleet-milp`
* **What it does**: Solves the branch-and-cut Mixed Integer Linear Program for a multi-vessel monthly quota.

#### Request JSON:
```json
{
  "total_demand_mt": 200000.0,
  "origin": "Gladstone",
  "destination_options": ["Haldia", "Paradip"]
}
```

#### Response JSON:
```json
{
  "status": "success",
  "demand_requirement_mt": 200000.0,
  "total_cargo_delivered_mt": 205000.0,
  "total_procurement_cost_usd": 3320832.0,
  "average_landed_cost_per_mt": 16.20,
  "net_milp_savings_vs_spot_usd": 327000.0,
  "allocated_fleet_schedule": [
    {
      "vessel_count": 1,
      "vessel_class": "Supramax",
      "destination": "Paradip",
      "timing": "NOW",
      "cost_per_mt": 19.52
    },
    {
      "vessel_count": 2,
      "vessel_class": "Panamax",
      "destination": "Paradip",
      "timing": "HOLD",
      "cost_per_mt": 14.98
    }
  ]
}
```

---

## 3. How to Call from Node.js (Express / Axios)

```javascript
const axios = require('axios');

async function getFreightOptimization() {
  try {
    const response = await axios.post('http://127.0.0.1:8000/api/v1/predict/optimize', {
      origin: 'Gladstone',
      destination: 'Haldia',
      cargo_volume_mt: 75000
    });
    
    const decision = response.data.optimal_charter_decision;
    console.log(`Recommendation: ${decision.action}`);
    console.log(`Vessel Class: ${decision.recommended_vessel_class}`);
    console.log(`Landed Cost: $${decision.optimal_landed_cost_per_mt}/MT`);
    console.log(`Savings: $${decision.expected_net_savings_usd}`);
  } catch (error) {
    console.error('API Error:', error.response?.data || error.message);
  }
}

getFreightOptimization();
```

---

## 4. How to Run the Server

```bash
# 1. Install requirements
pip install fastapi uvicorn pydantic pandas numpy scipy lightgbm joblib

# 2. Start server
python api_server.py
```
Server runs on `http://127.0.0.1:8000` with Swagger UI at `http://127.0.0.1:8000/docs`.
