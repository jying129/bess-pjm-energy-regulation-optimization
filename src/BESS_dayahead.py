"""
Day-ahead stochastic model for BESS participation in PJM energy + RegD.

This module:
- Uses aggregated RegD signals (up/down means and durations).
- Builds stochastic scenarios for regulation feasibility.
- Enforces SOC and power constraints for a single BESS.
"""
from pathlib import Path
from gurobipy import *
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for CLI runs
import matplotlib.pyplot as plt


class BESS_model2s:

    def __init__(self, bess_params, market_data, scenarios, scenario_probs=None) -> None:
        self.T = bess_params['T']  # Optimization horizon (hours)
        self.capacity = bess_params['BESS_capacity_kWh']  # Energy capacity (kWh)
        self.max_power = bess_params['BESS_max_power_kW']  # Power limit (kW)
        self.eta_c, self.eta_d = bess_params['eta_c'], bess_params['eta_d']  # Charge/discharge efficiencies
        self.degradation_cost = bess_params['degradation_cost']  # $/kWh throughput
        self.soc_target = self.capacity * bess_params.get('SoC_target', 0.5)
        # Conservative reg energy requirement scaling
        self.reg_energy_safety_factor = bess_params.get('reg_energy_safety_factor', 1.2)  # Headroom conservatism
        self.reg_price_is_per_mwh = bess_params.get('reg_price_is_per_mwh', True)  # Price unit flag
        self.c_corr_adder = bess_params.get('c_corr_adder', 0.0)  # Extra adder for correction energy
        # Regulation net energy neutrality band (hours of full power equivalent)
        self.reg_energy_eps = bess_params.get('reg_energy_eps', 0.02)  # Hours of neutrality band
        self.pr_e = market_data['pr_e_rt']  # $/kWh RT energy price
        self.pr_f = market_data['pr_fre']  # $/MW-h reg price
        self.scenarios = scenarios
        self.num_scenarios = len(scenarios)
        if scenario_probs is None:
            self.scenario_probs = [1 / self.num_scenarios] * self.num_scenarios
        else:
            self.scenario_probs = scenario_probs
        self.E_max = self.capacity * bess_params['SoC_max']  # Max energy (kWh)
        self.E_min = self.capacity * bess_params['SoC_min']  # Min energy (kWh)
        self.E_initial = self.capacity * bess_params['SoC_initial']  # Initial SOC (kWh)
        self.P_max = self.max_power
        self.P_min = -self.max_power
        # Precompute conservative reg energy requirements across scenarios (per hour)
        self.delta_t_up_eff = []
        self.delta_t_dn_eff = []
        for t in range(self.T):
            ups = [scen.loc[t, 'delta_t_up'] for scen in scenarios]
            dns = [scen.loc[t, 'delta_t_dn'] for scen in scenarios]
            self.delta_t_up_eff.append(self.reg_energy_safety_factor * max(ups))
            self.delta_t_dn_eff.append(self.reg_energy_safety_factor * max(dns))
        print("BESS stochastic model initialized successfully.")

    def build_stochastic_model(self, terminal_soc_target=0.5):
        """
        Build and solve the stochastic day-ahead model.

        Args:
            terminal_soc_target: kept for backward compatibility; not enforced.
        """
        m = Model('BESS_Stochastic_Optimization')
        # Keep solves bounded in time for interactive runs
        m.Params.TimeLimit = 10

        # =============================================================================
        # 1. Decision variables
        # =============================================================================
        # DA market decisions (single schedule for all scenarios)
        P_charge_da = m.addVars(self.T, vtype=GRB.CONTINUOUS, lb=0, name="P_charge_da")
        P_discharge_da = m.addVars(self.T, vtype=GRB.CONTINUOUS, lb=0, name="P_discharge_da")
        # Scenario-specific regulation capacity (expected revenue)
        R_reg = m.addVars(self.num_scenarios, self.T, vtype=GRB.CONTINUOUS, lb=0, name="R_reg")
        # RT feasibility deployment (scenario-based, no RT revenue)
        E_soc = m.addVars(self.num_scenarios, self.T + 1, vtype=GRB.CONTINUOUS, name="E_soc")
        P_charge_up = m.addVars(self.num_scenarios, self.T, vtype=GRB.CONTINUOUS, lb=0, name="P_charge_up")
        P_discharge_up = m.addVars(self.num_scenarios, self.T, vtype=GRB.CONTINUOUS, lb=0, name="P_discharge_up")
        P_charge_dn = m.addVars(self.num_scenarios, self.T, vtype=GRB.CONTINUOUS, lb=0, name="P_charge_dn")
        P_discharge_dn = m.addVars(self.num_scenarios, self.T, vtype=GRB.CONTINUOUS, lb=0, name="P_discharge_dn")
        # Real-time correction energy to pull SOC back toward band (charged/discharged at RT prices)
        P_corr_chg = m.addVars(self.num_scenarios, self.T, vtype=GRB.CONTINUOUS, lb=0, name="P_corr_chg")
        P_corr_dis = m.addVars(self.num_scenarios, self.T, vtype=GRB.CONTINUOUS, lb=0, name="P_corr_dis")
        # Binary for preventing simultaneous charge/discharge in DA baseline
        z_da = m.addVars(self.T, vtype=GRB.BINARY, name="z_da")

        # =============================================================================
        # 2. Objective function
        # =============================================================================
        # DA-only objective: DA revenue/cost only for baseline and DA regulation payment
        # Day-ahead energy arbitrage on the baseline schedule
        total_energy_revenue_da = quicksum(self.pr_e[t] * P_discharge_da[t] for t in range(self.T))
        total_energy_cost_da = quicksum(self.pr_e[t] * P_charge_da[t] for t in range(self.T))
        reg_price_divisor = 1000 if self.reg_price_is_per_mwh else 1
        total_reg_revenue = quicksum(self.scenario_probs[s] * (R_reg[s,t] / reg_price_divisor) * self.pr_f[t] for s in range(self.num_scenarios) for t in range(self.T))
        # Degradation cost for throughput (charge + discharge) baseline + regulation
        total_deg_cost = (
            quicksum(self.scenario_probs[s] * ((P_charge_up[s,t] + P_discharge_up[s,t]) * self.scenarios[s].loc[t, 'delta_t_up'] + (P_charge_dn[s,t] + P_discharge_dn[s,t]) * self.scenarios[s].loc[t, 'delta_t_dn']) * self.degradation_cost for s in range(self.num_scenarios) for t in range(self.T))
            + quicksum((P_charge_da[t] + P_discharge_da[t]) * self.degradation_cost for t in range(self.T))
        )

        m.setObjective(
            total_energy_revenue_da - total_energy_cost_da
            + total_reg_revenue - total_deg_cost
            ,
            sense=GRB.MAXIMIZE
        )

        # =============================================================================
        # 3. Constraints
        # =============================================================================
        for s in range(self.num_scenarios):
            scenario_data = self.scenarios[s]
            m.addConstr(E_soc[s, 0] == self.E_initial, f"initial_soc_s{s}")
            for t in range(self.T):
                s_up = scenario_data.loc[t, 's_up']
                s_dn = scenario_data.loc[t, 's_dn']
                delta_t_up = scenario_data.loc[t, 'delta_t_up']
                delta_t_dn = scenario_data.loc[t, 'delta_t_dn']
                # Baseline power: +charge / -discharge
                baseline_power = P_charge_da[t] - P_discharge_da[t]
                # Regulation adjustments around the baseline
                m.addConstr(baseline_power - s_up * R_reg[s,t] == P_charge_up[s,t] - P_discharge_up[s,t], f"power_balance_up_s{s}_t{t}")
                m.addConstr(baseline_power - s_dn * R_reg[s,t] == P_charge_dn[s,t] - P_discharge_dn[s,t], f"power_balance_dn_s{s}_t{t}")
                # Enforce inverter bounds on realized reg movements
                m.addConstr(P_charge_up[s,t] <= self.P_max, f"reg_charge_up_max_s{s}_t{t}")
                m.addConstr(P_discharge_up[s,t] <= self.P_max, f"reg_discharge_up_max_s{s}_t{t}")
                m.addConstr(P_charge_dn[s,t] <= self.P_max, f"reg_charge_dn_max_s{s}_t{t}")
                m.addConstr(P_discharge_dn[s,t] <= self.P_max, f"reg_discharge_dn_max_s{s}_t{t}")
                # Disable correction energy: force to zero (DA must manage SOC)
                m.addConstr(P_corr_chg[s,t] == 0, f"corr_chg_zero_s{s}_t{t}")
                m.addConstr(P_corr_dis[s,t] == 0, f"corr_dis_zero_s{s}_t{t}")
                # SOC update: use actual up/down power (already includes baseline through the balance constraints)
                energy_change = (
                    delta_t_up * (P_charge_up[s,t]*self.eta_c - P_discharge_up[s,t]/self.eta_d)
                    + delta_t_dn * (P_charge_dn[s,t]*self.eta_c - P_discharge_dn[s,t]/self.eta_d)
                )
                m.addConstr(E_soc[s, t+1] == E_soc[s, t] + energy_change, f"soc_update_s{s}_t{t}")
                m.addConstr(E_soc[s, t+1] <= self.E_max, f"soc_max_s{s}_t{t}")
                m.addConstr(E_soc[s, t+1] >= self.E_min, f"soc_min_s{s}_t{t}")
                # DA baseline power limits and mutual exclusivity
                m.addConstr(P_charge_da[t] <= self.P_max * z_da[t], f"da_charge_max_t{t}")
                m.addConstr(P_discharge_da[t] <= self.P_max * (1 - z_da[t]), f"da_discharge_max_t{t}")
                m.addConstr(P_charge_da[t] + R_reg[s,t] + P_corr_chg[s,t] <= self.P_max, f"reg_headroom_charge_s{s}_t{t}")
                m.addConstr(P_discharge_da[t] + R_reg[s,t] + P_corr_dis[s,t] <= self.P_max, f"reg_headroom_discharge_s{s}_t{t}")
                # SOC headroom-based reg capacity (conservative, scenario-agnostic using effective deltas)
                m.addConstr(E_soc[s, t] - self.E_min >= R_reg[s,t] * (self.delta_t_up_eff[t] / self.eta_d), f"soc_headroom_dis_s{s}_t{t}")
                m.addConstr(self.E_max - E_soc[s, t] >= R_reg[s,t] * (self.delta_t_dn_eff[t] * self.eta_c), f"soc_headroom_chg_s{s}_t{t}")

        
        # =============================================================================
        # 4. Solve the model
        # =============================================================================
        print("Solving stochastic optimization model...")
        m.optimize()

        if m.status == GRB.Status.OPTIMAL:
            print("Optimal solution found!")
            # Diagnostics: compute economic components
            reg_revenue_val = 0
            da_energy_val = 0
            da_cost_val = 0
            deg_cost_val = 0
            for s in range(self.num_scenarios):
                for t in range(self.T):
                    delta_t_up = self.scenarios[s].loc[t, 'delta_t_up']
                    delta_t_dn = self.scenarios[s].loc[t, 'delta_t_dn']
                    reg_revenue_val += self.scenario_probs[s] * (R_reg[s,t].x / reg_price_divisor) * self.pr_f[t]
                    deg_cost_val += self.scenario_probs[s] * ((P_charge_up[s,t].x + P_discharge_up[s,t].x) * delta_t_up + (P_charge_dn[s,t].x + P_discharge_dn[s,t].x) * delta_t_dn) * self.degradation_cost
            for t in range(self.T):
                da_energy_val += self.pr_e[t] * P_discharge_da[t].x
                da_cost_val += self.pr_e[t] * P_charge_da[t].x
                deg_cost_val += (P_charge_da[t].x + P_discharge_da[t].x) * self.degradation_cost

            print(f" Reg capacity revenue: ${reg_revenue_val:.2f}")
            print(f" DA energy revenue: ${da_energy_val:.2f} | DA energy cost: ${da_cost_val:.2f}")
            print(f" Degradation cost: ${deg_cost_val:.2f}")
            # Price diagnostics
            print(f" Reg price stats (pr_f): mean={np.mean(self.pr_f):.4f}, median={np.median(self.pr_f):.4f}, assumed units={'$/MWh' if self.reg_price_is_per_mwh else '$/kWh'}")
            if np.median(self.pr_f) > 50 or np.median(self.pr_f) < 0:
                print(" WARNING: pr_f median is outside [0,50]; units may be wrong.")
            # Hourly diagnostics (scenario 0 as representative)
            print("Hour | R_reg | DA_charge | DA_discharge | E_soc_exp | headroom_dis(bind?) | headroom_chg(bind?) | inv_charge(bind?) | inv_dis(bind?) | delta_up_eff | delta_dn_eff")
            for t in range(self.T):
                r_val = R_reg[0, t].x
                c_da = P_charge_da[t].x
                d_da = P_discharge_da[t].x
                soc_exp = np.sum([self.scenario_probs[s] * E_soc[s, t].x for s in range(self.num_scenarios)])
                headroom_dis_slack = E_soc[0, t].x - self.E_min - r_val * (self.delta_t_up_eff[t] / self.eta_d)
                headroom_chg_slack = self.E_max - E_soc[0, t].x - r_val * (self.delta_t_dn_eff[t] * self.eta_c)
                inv_charge_slack = self.P_max - (c_da + r_val)
                inv_dis_slack = self.P_max - (d_da + r_val)
                print(f"{t:>4} | {r_val:6.1f} | {c_da:9.1f} | {d_da:12.1f} | {soc_exp:8.1f} | {headroom_dis_slack<=1e-6} | {headroom_chg_slack<=1e-6} | {inv_charge_slack<=1e-6} | {inv_dis_slack<=1e-6} | {self.delta_t_up_eff[t]:.3f} | {self.delta_t_dn_eff[t]:.3f}")
            # Headroom sample checks
            sample_hours = [0, min(8, self.T-1), min(16, self.T-1)]
            for t in sample_hours:
                lhs_dis = E_soc[0, t].x - self.E_min
                rhs_dis = R_reg[0, t].x * (self.delta_t_up_eff[t] / self.eta_d)
                lhs_chg = self.E_max - E_soc[0, t].x
                rhs_chg = R_reg[0, t].x * (self.delta_t_dn_eff[t] * self.eta_c)
                print(f"Headroom check t={t}: dis LHS {lhs_dis:.1f} kWh vs RHS {rhs_dis:.1f} kWh | chg LHS {lhs_chg:.1f} kWh vs RHS {rhs_chg:.1f} kWh")

            solution = {
                'DA_charge': [P_charge_da[t].x for t in range(self.T)],
                'DA_discharge': [P_discharge_da[t].x for t in range(self.T)],
                'expected_regulation': [np.sum([self.scenario_probs[s] * R_reg[s, t].x for s in range(self.num_scenarios)]) for t in range(self.T)],
                'expected_soc': [np.sum([self.scenario_probs[s] * E_soc[s, t].x for s in range(self.num_scenarios)]) for t in range(self.T + 1)],
                'total_profit': m.objVal,
                'debug': {
                    'reg_revenue': reg_revenue_val,
                    'da_energy_revenue': da_energy_val,
                    'da_energy_cost': da_cost_val,
                    'degradation_cost': deg_cost_val,
                }
            }
            return solution
        else:
            print(f"Optimization failed with status: {m.status}")
            return None
    
    def plot_results(self, solution, save_path="output/bess_opt_dispatch.png"):
        if not solution:
            print("No solution to plot.")
            return
        # Ensure output directory exists before saving
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(self.T)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(t, solution['DA_charge'], width=0.8, label='DA Charge Schedule (kW)', color='blue', alpha=0.7)
        ax1.bar(t, [-d for d in solution['DA_discharge']], width=0.8, label='DA Discharge Schedule (kW)', color='red', alpha=0.7)
        # Show expected regulation capacity commitment
        ax1.bar(t, solution.get('expected_regulation', [0]*len(t)), width=0.4, label='Reg Capacity (kW)', color='#F1B656', alpha=0.5)
        ax1.set_xlabel('Time (hour)')
        ax1.set_ylabel('Power (kW)')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        ax2 = ax1.twinx()
        ax2.plot(np.arange(self.T + 1), solution['expected_soc'], label='Expected State of Charge (kWh)', color='green', marker='o')
        ax2.set_ylabel('Energy (kWh)')
        ax2.legend(loc='upper right')
        plt.title('BESS Optimal Day-Ahead Stochastic Dispatch')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved dispatch plot to {save_path}")
