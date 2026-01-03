"""
End-to-end run script for the day-ahead BESS optimization.

This script:
- Processes historical RegD signals into hourly aggregates.
- Generates stochastic regulation scenarios.
- Solves the day-ahead optimization model.
- Saves a dispatch plot for quick inspection.
"""

import numpy as np

from src.BESS_constantParam import *
from src.data_process import process_historical_signals, generate_regulation_scenarios, reduce_scenarios_kmeans
from src.BESS_dayahead import BESS_model2s


# =============================================================================
# 1. Full simulation
# =============================================================================

def run_full_simulation():
    """
    Run the day-ahead pipeline from data processing to optimization.
    """
    print("--- Step 1: Processing Historical Data ---")
    csv_filepath = 'input/reg_signal.csv'
    historical_signals_processed = process_historical_signals(
        filepath=csv_filepath,
        time_col_name='timestamp',
        signal_col_name='signal_value'
    )
    if historical_signals_processed is None:
        print("Failed to process historical data. Exiting.")
        return

    print("\n--- Step 2: Generating Scenarios ---")
    scenarios = generate_regulation_scenarios(
        historical_df=historical_signals_processed,
        num_scenarios=50,  # align with Monte Carlo sweep settings
        enforce_neutral=True,
        seed=42
    )
    print(f"Successfully created {len(scenarios)} scenarios.")

    print("\n--- Step 3: Running Stochastic Optimization ---")
    bess_params = {
        'T': T, 'H': H,
        'BESS_capacity_kWh': BESS_capacity_kWh,
        'BESS_max_power_kW': BESS_max_power_kW,
        'SoC_min': SoC_min, 'SoC_max': SoC_max, 'SoC_initial': SoC_initial,
        'eta_c': eta_c, 'eta_d': eta_d,
        'degradation_cost': degradation_cost,
        # Optional tunables for conservative headroom and correction adder
        'reg_energy_safety_factor': reg_energy_safety_factor,
        'c_corr_adder': c_corr_adder,
    }
    market_data = {
        # Flatten price arrays so Gurobi sees scalar prices per hour
        'pr_e_rt': pr_e_rt.flatten(),
        'pr_fre': pr_fre.flatten()
    }

    my_bess = BESS_model2s(
        bess_params=bess_params,
        market_data=market_data,
        scenarios=scenarios,
        scenario_probs=None  # equal weights
    )
    # Solve the day-ahead model
    solution = my_bess.build_stochastic_model(terminal_soc_target=0.5)

    if solution:
        print(f"\n--- FINAL RESULTS ---")
        print(f"Total EXPECTED profit for the day: ${solution['total_profit']:.2f}")
        my_bess.plot_results(solution)
        # Optional: run hour-ahead rolling using DA outputs (disabled for faster run)
        # try:
        #     from src.BESS_hourahead_rolling import run_hourahead_rolling
        #     da_plan = {
        #         'P_DA_net': [c - d for c, d in zip(solution['DA_charge'], solution['DA_discharge'])],
        #         'R_DA': solution.get('expected_regulation', [0]*len(solution['DA_charge'])),
        #     }
        #     ha_params = {
        #         'T': T,
        #         'BESS_capacity_kWh': BESS_capacity_kWh,
        #         'BESS_max_power_kW': BESS_max_power_kW,
        #         'SoC_min': SoC_min,
        #         'SoC_max': SoC_max,
        #         'SoC_initial': SoC_initial,
        #         'eta_c': eta_c,
        #         'eta_d': eta_d,
        #         'degradation_cost': degradation_cost,
        #         'reg_energy_safety_factor': reg_energy_safety_factor,
        #         'pr_e_rt': pr_e_rt.flatten(),
        #         'pr_fre': pr_fre.flatten(),
        #         'reg_price_is_per_mwh': True,
        #         'c_dev': 10.0,
        #     }
        #     run_hourahead_rolling(da_plan, scenarios, ha_params, H=4, save_plot="output/bess_hourahead_roll.png")
        #     print("Hour-ahead rolling run complete. See output/bess_hourahead_roll.png")
        # except Exception as e:
        #     print(f"Hour-ahead rolling run failed: {e}")
    else:
        print("Optimization failed to find a solution.")

# =============================================================================
# 2. Script entry point
# =============================================================================

if __name__ == "__main__":
    run_full_simulation()
