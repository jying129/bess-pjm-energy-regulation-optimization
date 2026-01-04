"""
Parameter definitions for the Battery Energy Storage System (BESS).

This module:
- Defines simulation horizon settings.
- Sets BESS physical and operational parameters.
- Loads market price inputs for energy and regulation.
"""
import numpy as np
import pandas as pd
from src.data_process import *

# =============================================================================
# 1. Simulation horizon settings
# =============================================================================

T = 24  # Number of hourly time steps in the day-ahead model (24 hrs in a day)
H = 4  # Rolling-horizon length for hour-ahead execution (hours)

# =============================================================================
# 2. BESS parameters
# =============================================================================

# BESS physical parameters
BESS_capacity_kWh = 5000         # Total energy capacity (kWh)
BESS_max_power_kW = 1250         # Maximum charge/discharge power (kW)
SoC_min = 0.1                    # Minimum state of charge
SoC_max = 0.9                    # Maximum state of charge
SoC_initial = 0.5                # Starting state of charge

# Efficiency assumptions
eta_c = 0.90                     # Charging efficiency (η_c)
eta_d = 0.93                     # Discharging efficiency (η_d)

# Degradation and SOC penalty
degradation_cost = 0.01          # Cost in $/kWh of throughput (charge + discharge) to reflect battery wear (= $10/MWh)
soc_penalty = 20.0               # Terminal SOC deviation penalty ($/kWh), encouraging ending the day near the target SOC
soc_penalty_hourly = 30.0        # Hourly SOC deviation penalty ($/kWh), discouraging SOC drift during rolling execution

# Tight SOC band to keep SOC near mid-point (applied to all hours)
SoC_band_low = 0.48
SoC_band_high = 0.52

# Regulation feasibility and correction penalties
reg_energy_safety_factor = 1.5   # Safety factor applied to regulation energy requirements, providing conservative SOC headroom for RegD deployment
c_corr_adder = 0.005             # Additional cost adder ($/kWh) on correction energy, encouraging better baseline energy scheduling

# =============================================================================
# 3. Market price inputs (PJM, Summer 2025, May-Aug 2025 averages)
# =============================================================================

# ---------- Real-time energy prices ($/kWh): hourly averages over May-Aug 2025 ----------

market_start_date = '2025-05-01'
market_end_date = '2025-08-31'
df_energy_data = pd.read_csv('input/da_hrl_lmps.csv', sep=',')
pr_e_rt = transfer(
    df_energy_data,
    target_cols=['total_lmp_rt'],
    datetime_col='datetime_beginning_ept',
    start_date=market_start_date,
    end_date=market_end_date,
    rotate=False
).flatten() / 1000  # Real-time energy price vector ($/kWh)

# ---------- Frequency regulation prices ($/MWh): capacity prices, performance prices, and mileage ratios ----------

df_reserve = pd.read_csv('input/reserve_market_results_new.csv', sep=',')
df_reg = pd.read_csv('input/reg_market_results.csv', sep=',')

# Regulation capacity price ($/MWh)
c_rcap = transfer(
    df_reserve,
    target_cols=['reg_ccp'],
    datetime_col='datetime_beginning_ept',
    start_date=market_start_date,
    end_date=market_end_date,
    rotate=False
).flatten()

# Regulation performance price ($/MWh)
c_rper = transfer(
    df_reserve,
    target_cols=['reg_pcp'],
    datetime_col='datetime_beginning_ept',
    start_date=market_start_date,
    end_date=market_end_date,
    rotate=False
).flatten()

# Mileage ratio = RegD mileage / RegA mileage (dimensionless)
regd_mileage = transfer(
    df_reg,
    target_cols=['regd_mileage'],  # mileage value, not a ratio
    datetime_col='datetime_beginning_ept',
    start_date=market_start_date,
    end_date=market_end_date,
    rotate=False
).flatten()

rega_mileage = transfer(
    df_reg,
    target_cols=['rega_mileage'],
    datetime_col='datetime_beginning_ept',
    start_date=market_start_date,
    end_date=market_end_date,
    rotate=False
).flatten()

rega_floor = np.maximum(rega_mileage, 1e-3)
mileage_ratio = np.array(regd_mileage) / rega_floor

# Final regulation price ($/MWh)
# = capacity price + performance price x mileage ratio
pr_fre = c_rcap + np.array(c_rper) * mileage_ratio  # revenue will divide by 1000 for kW offers
