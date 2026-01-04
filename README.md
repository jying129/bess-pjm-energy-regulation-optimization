# Battery Energy Storage Participation in PJM Energy & Regulation Markets

**Repository:** `bess-pjm-energy-regulation-optimization`

This repository contains a **Python implementation** for modeling and optimizing the participation of a **Battery Energy Storage System (BESS)** in **PJM electricity markets**, to maximize the revenue from **joint energy arbitrage and frequency regulation (RegD)**.

The code implements a **scenario-based optimization framework** that captures:
- day-ahead energy market prices,
- frequency regulation revenues,
- battery operational constraints (SOC, power limits, efficiency),
- uncertainty in regulation signal response.

---

## Context

Battery energy storage systems can earn revenue from multiple markets:
- **Energy markets**: perform energy arbitrage (buy low, sell high),
- **Frequency regulation markets**: continuously balance short-term supply–demand mismatches and maintain system frequency.

PJM operates two regulation products:

- **RegA**  
  - Slower, lower mileage  
  - Typically provided by conventional generators  

- **RegD**  
  - Fast, high-magnitude signal  
  - Designed specifically for energy storage  
  - Issued every **2 seconds**  
  - Roughly energy-neutral *on average*, but **not within each hour**

This project focuses on **RegD**. Although RegD is designed to be energy-neutral over long periods, **actual deployment within an hour is uncertain**. 

This uncertainty affects the battery’s **state of charge (SOC)** and can cause:
- infeasible SOC trajectories,
- unexpected saturation at SOC bounds,
- violations of power constraints.

A battery that commits regulation capacity must ensure feasibility **under all plausible regulation deployments**, not just the average case.

This creates a key challenge:

> **Regulation deployment uncertainty directly affects battery state of charge (SOC).**

Ignoring this uncertainty can result in:
- infeasible SOC trajectories,
- overly optimistic revenue estimates,
- schedules that cannot be executed in real time.

This repository addresses the research question:

> *How should a battery optimally schedule energy and regulation capacity while accounting for uncertainty in regulation deployment and SOC dynamics?*

---

## Modeling Overview

The framework operates at **two time scales**: day-ahead and hour-ahead (close to real-time).

### 1. Day-Ahead (Two-stage Stochastic Optimization)
- The battery commits:
  - energy charging/discharging schedule,
  - regulation capacity offers.
- Regulation deployment uncertainty is modeled using **historical RegD signals**.
- Multiple regulation scenarios are enforced simultaneously to guarantee SOC feasibility.

The model enforces:
- SOC bounds,
- power limits,
- energy balance constraints,
for every regulation scenario.

This guarantees that the day-ahead schedule is robust to regulation uncertainty.

### 2. Hour-Ahead / Real-Time (Rolling Window)
In real-time operation:
- prices and SOC are observed,
- previously committed regulation capacity must be honored,
- decisions are updated using a short prediction horizon.

The hour-ahead model:
- re-optimizes energy dispatch,
- respects day-ahead commitments,
- reacts to realized regulation energy.

This rolling-window approach mimics how batteries are actually operated.

---

## Regulation Signal Representation

PJM RegD signals are issued every **2 seconds**, which is too granular for day-ahead optimization.

This project uses an **aggregate hourly signal representation**, where each hour is summarized by four parameters:

- `s_up` — average magnitude of up-regulation signal  
- `s_dn` — average magnitude of down-regulation signal  
- `δ_up` — fraction of the hour spent in up-regulation  
- `δ_dn` — fraction of the hour spent in down-regulation  

This representation preserves:
- net regulation energy,
- mileage characteristics,

while allowing hourly optimization.

---

## Repository Structure

```text
.
├── input/                       
│   ├── reg_market_results_new.csv      # Regulation market results
│   ├── reg_signal.csv.xz               # 2‑sec RegD signal (compressed)
│   ├── reserve_market_results_new.csv  # Reserve market results with regulation capability clearing prices
                                          and regulation performance clearing prices
│   └── rt_hrl_lmps-2.csv               # Real‑time hourly LMPs
├── output/                      
│   ├── bess_opt_dispatch.png           # Dispatch plot
│   ├── bess_parameters_table.png       # Parameter summary table
│   ├── rolling_horizon_profit.png      # Rolling‑horizon profit plot
│   └── rolling_horizon_sweep.csv       # Hourly-ahead rolling window results
├── src/                         
│   ├── BESS_constantParam.py           # BESS parameters & market inputs
│   ├── BESS_dayahead.py                # Stochastic optimization models for day-ahead time scale
│   ├── BESS_hourahead_rolling.py       # Rolling‑horizon control for hour-ahead time scale
│   ├── data_process.py                 # Market data processing & scenario generation
│   ├── main_dayahead.py                # Day‑ahead optimization
│   └── main_hourahead.py               # Hour‑ahead simulation
└── README.md

```
---


## Key Modules

### BESS_constantParam.py
Defines:
- battery capacity, power limits, SOC bounds,
- charging/discharging efficiency,
- degradation and SOC penalty parameters,
- processed hourly energy prices,
- regulation prices (capacity + performance + mileage).

### data_process.py
Implements:
- hourly aggregation of PJM data,
- construction of aggregate RegD signal parameters,
- stochastic scenario generation,
- scenario reduction via clustering,
- visualization utilities.

### BESS_model2s.py
Defines the mathematical optimization models, including:
- decision variables (energy, regulation, SOC),
- SOC dynamics under uncertain regulation deployment,
- power and energy constraints,
- objective function combining:
  - energy arbitrage revenue,
  - regulation revenue,
  - degradation costs,
  - SOC deviation penalties.
Both day-ahead stochastic and hour-ahead deterministic models are implemented here.

### BESS_hourahead_rolling.py
Implements a rolling-horizon control policy:
- updates decisions hourly,
- enforces day-ahead commitments,
- tracks SOC evolution realistically.

---

## Data and Dependencies

### Data
The code requires:
- PJM energy market prices,
- PJM regulation prices and mileage,
- historical RegD AGC signals.

### Dependencies
- Python 3.9+
- NumPy
- Pandas
- Matplotlib
- scikit-learn (optional)
- Optimization solver (e.g., Gurobi, CPLEX, CBC)

---
