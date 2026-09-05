# Exactly What We Are Predicting (Executive Cheat Sheet)

> **Quick Reference Guide**: Whenever in doubt about what the ML model predicts versus what input features are and how costs are calculated, read this document or open [what_are_we_predicting.pdf](file:///d:/freight%20forecasting/what_are_we_predicting.pdf).

---

## 1. The Core ML Target ($Y$)

Our Machine Learning model predicts **EXACTLY ONE NUMBER**:

$$Y = \text{Future Ocean Freight Rate 15 Trading Days Ahead (in \$/MT)}$$

* **Specific Target Column**: `target_freight_rate_proxy` shifted 15 days ahead (`shift(-15)`).
* **Specific Route**: **Gladstone (Australia) $\to$ Haldia / Paradip (East Coast India)** carrying bulk coking coal.
* **Unit of Measurement**: **US Dollars per Metric Ton of Cargo ($\$/\text{MT}$)** (typical range: $\$9.00$ to $\$14.50/\text{MT}$).
* **Current Test Accuracy**: $\pm \$0.716/\text{MT}$ MAE ($6.68\%$ MAPE on untouched holdout test data).

---

## 2. The 3 Quantile Scenarios ($P10, P50, P90$)

Because freight rates fluctuate with global shipping markets, we don't output a single risky point forecast. We output a **3-scenario range**:

| Scenario | Percentile | What It Represents for SAIL Procurement |
| :--- | :--- | :--- |
| **$P10$ (Optimistic / Dip)** | $10^{\text{th}}$ Percentile | **Best-case low market:** Only 10% chance rates will be lower than this. Used as a lower bound for budgeting. |
| **$P50$ (Base Case / Median)** | $50^{\text{th}}$ Percentile | **Most likely expected rate:** The primary number compared against today's spot rate to decide `FIX NOW` vs `HOLD`. |
| **$P90$ (Pessimistic / Spike)** | $90^{\text{th}}$ Percentile | **Worst-case surge:** 90% chance market rates will remain below this. Critical for risk hedging against sudden charter spikes. |

---

## 3. What We Predict vs. What Are Inputs vs. What Is Calculated by Math

| Category | Variable / Item | Role in System | Reason / Logic |
| :--- | :--- | :--- | :--- |
| **ML Target ($Y$)** | Ocean Freight Rate ($/MT) | **PREDICTED by ML** | Volatile market determined by global vessel supply and demand. |
| **Inputs ($X$)** | Baltic Index (`BDRY`), Coal Price (`PCOALAUUSDM`), Bunker Fuel ($/MT), USD/INR Exchange Rate, Lags, Moving Averages | **Historical Input Features** | Exogenous market signals feeding the model up to today ($t$). |
| **Math Formula** | Voyage Transit Days | **Calculated by Physics** | $\text{Distance (NM)} / (\text{Vessel Speed} \times 24)$ |
| **Math Formula** | Voyage Bunker Fuel Cost ($) | **Calculated by Physics** | $\text{Voyage Days} \times \text{Daily Fuel Burn (MT/day)} \times \text{Bunker Price (\$/MT)}$ |
| **Math Formula** | Port Demurrage Fine ($) | **Calculated by Contract** | $\max(0, \text{Port Wait Days} - \text{Laytime Allowed}) \times \text{Daily Penalty (\$/day)}$ |
| **Math Formula** | Sandheads Lightering Cost ($) | **Calculated by Port Tariff** | $\text{Offloaded Cargo MT} \times \$7.00/\text{MT}$ (Mandatory for Haldia's shallow $8.5\text{m}$ draft) |
| **Math Formula** | Total Landed Cost per MT ($/MT) | **Deterministic Sum** | $(\text{Base Ocean Freight} + \text{Bunker Fuel} + \text{Demurrage} + \text{Lightering}) / \text{Cargo MT}$ |

---

## 4. The 4 Landed Cost Heads Paid by SAIL

When a ship delivers coal to SAIL, the final bill has 4 items:

$$\text{Total Landed Cost (\$) } = \underbrace{\text{Base Ocean Freight}}_{\text{Uses ML Forecast}} + \underbrace{\text{Voyage Bunker Fuel}}_{\text{Physics Math}} + \underbrace{\text{Port Demurrage}}_{\text{Contract Math}} + \underbrace{\text{Sandheads Lightering}}_{\text{Port Tariff Math}}$$

1. **Head 1: Base Ocean Freight Cost ($)**: $\text{Cargo MT} \times \mathbf{\text{ML Predicted Freight Rate (\$/MT)}}$
2. **Head 2: Voyage Bunker Fuel Cost ($)**: $\text{Voyage Days} \times \text{Daily Fuel Burn (MT/day)} \times \text{Bunker Price (\$/MT)}$
3. **Head 3: Port Demurrage Penalty Cost ($)**: $\text{Excess Waiting Days} \times \text{Daily Demurrage Rate (\$/day)}$
4. **Head 4: Sandheads Lightering Cost ($)**: $\text{Lightered MT} \times \$7.00/\text{MT}$ (Only applies to Haldia; $\$0$ for deep-water Paradip).

---

## 5. The Two Final Decisions We Give to SAIL

Our optimization engine uses the predicted freight rate to make two operational recommendations:

### Decision 1: Optimal Vessel Size
Calculates Total Landed Cost per MT for:
* **Handysize** ($35,000\text{ MT}$): Can berth anywhere without lightering, but higher freight cost per ton.
* **Supramax** ($55,000\text{ MT}$): Medium capacity compromise.
* **Panamax** ($75,000\text{ MT}$): Lowest base freight per ton due to bulk scale, but incurs lightering fee if calling at Haldia.

The optimizer outputs the vessel class that delivers the **absolute lowest Total Landed Cost per MT**.

### Decision 2: Charter Timing (`FIX NOW` vs `HOLD`)
* **`FIX NOW`**: If Today's Spot Freight Rate is **lower** than the 15-day Forward Forecast ($P50$), charter the vessel today before the market spikes.
* **`HOLD`**: If Today's Spot Freight Rate is **higher** than the 15-day Forward Forecast ($P50$), wait 15 days because freight rates are dropping, saving SAIL procurement costs.
