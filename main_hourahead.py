"""
Sweep script for rolling-horizon sensitivity analysis.

This script:
- Processes historical RegD signals into hourly aggregates.
- Generates stochastic regulation scenarios.
- Solves the day-ahead optimization model.
- Sweeps rolling-horizon lengths and exports comparison outputs.
"""

from src.BESS_constantParam import *
from src.data_process import process_historical_signals, generate_regulation_scenarios
from src.BESS_dayahead import BESS_model2s
from src.BESS_hourahead_rolling import sweep_rolling_horizon


# =============================================================================
# 1. Rolling-horizon sweep
# =============================================================================

def run_sweep(seed=42):
    """
    Run the day-ahead solve and the rolling-horizon sweep.

    Outputs:
      - output/rolling_horizon_sweep.csv
      - output/rolling_horizon_profit.png
      - output/rolling_horizon_profit_comparison.png
    """
    print("--- Step 1: Processing Historical Data ---")
    historical_signals_processed = process_historical_signals(
        filepath="input/reg_signal.csv",
        time_col_name="timestamp",
        signal_col_name="signal_value",
    )
    if historical_signals_processed is None:
        print("Failed to process historical data. Exiting.")
        return

    print("\n--- Step 2: Generating Scenarios ---")
    scenarios = generate_regulation_scenarios(
        historical_df=historical_signals_processed,
        num_scenarios=50,
        enforce_neutral=True,
        seed=seed,
    )
    print(f"Successfully created {len(scenarios)} scenarios.")

    print("\n--- Step 3: Running Stochastic Optimization (DA) ---")
    bess_params = {
        "T": T,
        "H": H,
        "BESS_capacity_kWh": BESS_capacity_kWh,
        "BESS_max_power_kW": BESS_max_power_kW,
        "SoC_min": SoC_min,
        "SoC_max": SoC_max,
        "SoC_initial": SoC_initial,
        "eta_c": eta_c,
        "eta_d": eta_d,
        "degradation_cost": degradation_cost,
        "reg_energy_safety_factor": reg_energy_safety_factor,
        "c_corr_adder": c_corr_adder,
    }
    market_data = {
        "pr_e_rt": pr_e_rt.flatten(),
        "pr_fre": pr_fre.flatten(),
    }

    my_bess = BESS_model2s(
        bess_params=bess_params,
        market_data=market_data,
        scenarios=scenarios,
        scenario_probs=None,
    )
    solution = my_bess.build_stochastic_model(terminal_soc_target=0.5)

    if not solution:
        print("Optimization failed to find a solution.")
        return

    da_plan = {
        "P_DA_net": [c - d for c, d in zip(solution["DA_charge"], solution["DA_discharge"])],
        "R_DA": solution.get("expected_regulation", [0] * len(solution["DA_charge"])),
    }
    ha_params = {
        "T": T,
        "BESS_capacity_kWh": BESS_capacity_kWh,
        "BESS_max_power_kW": BESS_max_power_kW,
        "SoC_min": SoC_min,
        "SoC_max": SoC_max,
        "SoC_initial": SoC_initial,
        "eta_c": eta_c,
        "eta_d": eta_d,
        "degradation_cost": degradation_cost,
        "reg_energy_safety_factor": reg_energy_safety_factor,
        "pr_e_rt": pr_e_rt.flatten(),
        "pr_fre": pr_fre.flatten(),
        "reg_price_is_per_mwh": True,
    }

    H_list = [1, 2, 3, 4, 6, 8, 12, 24]
    sweep_rolling_horizon(
        da_plan,
        scenarios,
        ha_params,
        H_list=H_list,
        realized_scenario_idx=0,
        save_csv="output/rolling_horizon_sweep.csv",
        save_plot="output/rolling_horizon_profit.png",
    )
    print("Sweep complete. See output/rolling_horizon_sweep.csv and output/rolling_horizon_profit.png")


# =============================================================================
# 2. Script entry point
# =============================================================================

if __name__ == "__main__":
    run_sweep(seed=42)
