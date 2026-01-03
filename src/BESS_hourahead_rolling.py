"""
Hour-ahead rolling execution model for the BESS.

This module:
- Executes a receding-horizon dispatch each hour.
- Adjusts the baseline around the day-ahead plan and regulation offer.
- Enforces SOC and inverter limits across scenarios.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum

# =============================================================================
# 1. Rolling optimization
# =============================================================================

def run_hourahead_rolling(
    da_plan,
    scenarios,
    params,
    H=4,
    save_plot="output/bess_hourahead_roll.png",
    save_csv="output/results_hourahead_roll.csv",
):
    T = params["T"]  # Total hours in the day-ahead horizon
    num_scen = len(scenarios)  # Number of RegD scenarios for feasibility
    scenario_probs = [1 / num_scen] * num_scen  # Equal scenario weights

    # Parameters
    cap = params["BESS_capacity_kWh"]  # Energy capacity (kWh)
    p_max = params["BESS_max_power_kW"]  # Power limit (kW)
    eta_c, eta_d = params["eta_c"], params["eta_d"]  # Charge/discharge efficiencies
    soc_min = cap * params["SoC_min"]  # Min SOC bound (kWh)
    soc_max = cap * params["SoC_max"]  # Max SOC bound (kWh)
    soc_start = cap * params["SoC_initial"]  # Initial SOC (kWh)
    reg_price_divisor = 1000 if params.get("reg_price_is_per_mwh", True) else 1  # $/MW-h vs $/kWh
    pr_e = params["pr_e_rt"]  # $/kWh RT energy price
    pr_f = params["pr_fre"]  # $/MW-h regulation price
    degr_cost = params["degradation_cost"]  # $/kWh throughput
    reg_energy_safety_factor = params.get("reg_energy_safety_factor", 1.2)  # Headroom conservatism

    # Inputs from DA
    P_DA_net = np.array(da_plan["P_DA_net"])  # DA baseline net power (+charge / -discharge)
    R_DA = np.array(da_plan["R_DA"])  # DA regulation offer (kW)

    # Effective reg durations (conservative max across scenarios per hour)
    delta_up_eff = []  # Effective up duration per hour (conservative)
    delta_dn_eff = []  # Effective down duration per hour (conservative)
    for t in range(T):
        ups = [scen.loc[t, "delta_t_up"] for scen in scenarios]
        dns = [scen.loc[t, "delta_t_dn"] for scen in scenarios]
        delta_up_eff.append(reg_energy_safety_factor * max(ups))
        delta_dn_eff.append(reg_energy_safety_factor * max(dns))

    # Storage for executed schedules
    exec_R = []  # Executed regulation offers (hour by hour)
    exec_P_net = []  # Executed baseline power (hour by hour)
    exec_soc = [soc_start]  # Expected SOC trajectory (kWh)
    exec_soc_min = []  # Scenario min SOC per hour (kWh)
    exec_soc_max = []  # Scenario max SOC per hour (kWh)
    rows = []  # Per-hour reporting rows for CSV
    throughput_kwh = 0.0  # Accumulated throughput for degradation reporting
    total_degradation_cost = 0.0  # Accumulated degradation cost ($)
    inverter_binds = 0  # Count of inverter-binding hours
    headroom_binds = 0  # Count of headroom-binding hours

    for t0 in range(T):
        horizon = min(H, T - t0)  # Rolling horizon length
        m = Model("BESS_HourAhead")  # Hour-ahead subproblem
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = 60

        # Variables over horizon
        P_ch = m.addVars(horizon, lb=0, name="P_ch")  # Baseline charging power
        P_dis = m.addVars(horizon, lb=0, name="P_dis")  # Baseline discharging power
        z_base = m.addVars(horizon, vtype=GRB.BINARY, name="z_base")  # Charge/discharge switch
        R_HA = m.addVars(horizon, lb=0, name="R_HA")  # Hour-ahead reg offer
        # Scenario variables
        E_soc = m.addVars(num_scen, horizon + 1, name="E_soc")  # Scenario SOC paths
        P_ch_up = m.addVars(num_scen, horizon, lb=0, name="P_ch_up")  # Up-period charging
        P_dis_up = m.addVars(num_scen, horizon, lb=0, name="P_dis_up")  # Up-period discharging
        P_ch_dn = m.addVars(num_scen, horizon, lb=0, name="P_ch_dn")  # Down-period charging
        P_dis_dn = m.addVars(num_scen, horizon, lb=0, name="P_dis_dn")  # Down-period discharging

        # Init SOC
        for s in range(num_scen):
            m.addConstr(E_soc[s, 0] == soc_start)  # Fix initial SOC for all scenarios

        # Constraints over horizon
        for h in range(horizon):
            t = t0 + h  # Absolute hour index
            # DA deviation (signed)
            net_base = P_ch[h] - P_dis[h]  # Baseline net power (+charge / -discharge)
            delta_p = net_base - P_DA_net[t]  # Deviation from DA baseline
            # Reg offer cap
            m.addConstr(R_HA[h] <= R_DA[t], name=f"reg_cap_{t}")
            # Deviation bound (approx.)
            m.addConstr(delta_p <= R_DA[t])
            m.addConstr(delta_p >= -R_DA[t])
            # No simultaneous charge/discharge baseline
            m.addConstr(P_ch[h] <= p_max * z_base[h])
            m.addConstr(P_dis[h] <= p_max * (1 - z_base[h]))
            for s in range(num_scen):
                s_up = scenarios[s].loc[t, "s_up"]  # Avg up signal (normalized)
                s_dn = scenarios[s].loc[t, "s_dn"]  # Avg down signal (normalized)
                du = scenarios[s].loc[t, "delta_t_up"]  # Up duration fraction
                dd = scenarios[s].loc[t, "delta_t_dn"]  # Down duration fraction
                # Balance around baseline
                m.addConstr(net_base - s_up * R_HA[h] == P_ch_up[s, h] - P_dis_up[s, h])
                m.addConstr(net_base - s_dn * R_HA[h] == P_ch_dn[s, h] - P_dis_dn[s, h])
                # Inverter limits
                m.addConstr(P_ch[h] + R_HA[h] <= p_max)
                m.addConstr(P_dis[h] + R_HA[h] <= p_max)
                m.addConstr(P_ch_up[s, h] <= p_max)
                m.addConstr(P_dis_up[s, h] <= p_max)
                m.addConstr(P_ch_dn[s, h] <= p_max)
                m.addConstr(P_dis_dn[s, h] <= p_max)
                # SOC update: baseline + reg tracking with efficiencies (1-hour step)
                energy_change = (
                    du * (P_ch_up[s, h] * eta_c - P_dis_up[s, h] / eta_d)
                    + dd * (P_ch_dn[s, h] * eta_c - P_dis_dn[s, h] / eta_d)
                    + P_ch[h] * eta_c * 1.0
                    - P_dis[h] / eta_d * 1.0
                )
                m.addConstr(E_soc[s, h + 1] == E_soc[s, h] + energy_change)
                m.addConstr(E_soc[s, h + 1] <= soc_max)
                m.addConstr(E_soc[s, h + 1] >= soc_min)
                # Reg headroom (conservative)
                m.addConstr(E_soc[s, h] - soc_min >= R_HA[h] * (delta_up_eff[t] / eta_d))
                m.addConstr(soc_max - E_soc[s, h] >= R_HA[h] * (delta_dn_eff[t] * eta_c))

        # Objective components (units: $; power kW * hours -> kWh)
        reg_revenue = quicksum((R_HA[h] / reg_price_divisor) * pr_f[t0 + h] for h in range(horizon))  # Reg payments

        deviation_cost = quicksum(
            pr_e[t0 + h] * (P_ch[h] - P_dis[h] - P_DA_net[t0 + h])
            for h in range(horizon)
        )  # RT settlement for DA deviation
        deg_cost = quicksum(
            scenario_probs[s]
            * (
                (P_ch_up[s, h] + P_dis_up[s, h]) * scenarios[s].loc[t0 + h, "delta_t_up"]
                + (P_ch_dn[s, h] + P_dis_dn[s, h]) * scenarios[s].loc[t0 + h, "delta_t_dn"]
            )
            * degr_cost
            for s in range(num_scen)
            for h in range(horizon)
        ) + quicksum((P_ch[h] + P_dis[h]) * 1.0 * degr_cost for h in range(horizon))  # Throughput cost

        m.setObjective(reg_revenue - deviation_cost - deg_cost, GRB.MAXIMIZE)
        m.optimize()

        if m.status != GRB.OPTIMAL:
            raise RuntimeError(f"HA model infeasible at t0={t0}, status={m.status}")

        # Execute first-hour decisions
        exec_R.append(R_HA[0].x)  # Implement only the first hour
        exec_P_net.append(P_ch[0].x - P_dis[0].x)  # Implement only the first hour
        soc_end_exp = sum(scenario_probs[s] * E_soc[s, 1].x for s in range(num_scen))
        soc_end_min = min(E_soc[s, 1].x for s in range(num_scen))
        soc_end_max = max(E_soc[s, 1].x for s in range(num_scen))
        exec_soc.append(soc_end_exp)
        exec_soc_min.append(soc_end_min)
        exec_soc_max.append(soc_end_max)
        soc_start = soc_end_exp  # Receding horizon advance

        # Per-hour metrics for executed hour (h=0)
        t = t0
        r_da = R_DA[t]  # DA reg offer at hour t
        r_ha = R_HA[0].x  # Executed HA reg offer at hour t
        reg_retention = (r_ha / r_da) if r_da != 0 else 0.0  # Reg retention ratio
        p_da = P_DA_net[t]  # DA baseline
        p_ha = exec_P_net[-1]  # Executed baseline
        delta_p = p_ha - p_da  # Signed deviation (kW)

        # Objective components for executed hour
        reg_revenue = (r_ha / reg_price_divisor) * pr_f[t]  # Reg revenue for hour t
        dev_cost = pr_e[t] * (p_ha - p_da)  # RT deviation settlement for hour t
        deg_hour = sum(
            scenario_probs[s]
            * (
                (P_ch_up[s, 0].x + P_dis_up[s, 0].x) * scenarios[s].loc[t, "delta_t_up"]
                + (P_ch_dn[s, 0].x + P_dis_dn[s, 0].x) * scenarios[s].loc[t, "delta_t_dn"]
            )
            for s in range(num_scen)
        ) + (P_ch[0].x + P_dis[0].x) * 1.0  # Hourly throughput (kWh)
        degradation_cost = deg_hour * degr_cost
        da_energy = 0.0  # Not part of HA objective; kept for reporting

        throughput_kwh += deg_hour
        total_degradation_cost += degradation_cost

        tol = 1e-6
        inv_bind = (P_ch[0].x + r_ha >= p_max - tol) or (P_dis[0].x + r_ha >= p_max - tol)
        head_bind = (
            (exec_soc[-2] - soc_min <= r_ha * (delta_up_eff[t] / eta_d) + tol)
            or (soc_max - exec_soc[-2] <= r_ha * (delta_dn_eff[t] * eta_c) + tol)
        )
        inverter_binds += 1 if inv_bind else 0
        headroom_binds += 1 if head_bind else 0

        rows.append(
            {
                "hour": t,
                "R_DA_kW": r_da,
                "R_HA_exec_kW": r_ha,
                "reg_retention": reg_retention,
                "P_DA_base_kW": p_da,
                "P_HA_exec_kW": p_ha,
                "deltaP_kW": delta_p,
                "SOC_mean_kWh": exec_soc[-1],
                "SOC_min_kWh": exec_soc_min[-1],
                "SOC_max_kWh": exec_soc_max[-1],
                "inverter_binding_flag": bool(inv_bind),
                "headroom_binding_flag": bool(head_bind),
                "reg_revenue_$": reg_revenue,
                "da_energy_$": da_energy,
                "deviation_cost_$": dev_cost,
                "degradation_$": degradation_cost,
            }
        )

    results = {
        "R_HA": exec_R,
        "P_base_net": exec_P_net,
        "SOC_expected": exec_soc,
        "SOC_min": exec_soc_min,
        "SOC_max": exec_soc_max,
    }

    # Plot outputs
    Path(save_plot).parent.mkdir(parents=True, exist_ok=True)
    Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(T)
    fig, (ax1, ax_ret) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )
    ax1.bar(t, P_DA_net, width=0.8, label="DA Baseline (kW)", color="gray", alpha=0.4)
    ax1.bar(t, exec_P_net, width=0.5, label="HA Executed Baseline (kW)", color="blue", alpha=0.7)
    ax1.step(t, da_plan["R_DA"], where="mid", label="R_DA (kW)", color="#F1B656", alpha=0.5)
    ax1.step(t, exec_R, where="mid", label="R_HA (kW)", color="#C27BA0")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Power (kW)")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(np.arange(T + 1), exec_soc, label="SOC Expected (kWh)", color="green", marker="o")
    ax2.fill_between(
        np.arange(1, T + 1),
        exec_soc_min,
        exec_soc_max,
        color="green",
        alpha=0.3,
        label="SOC min/max (scenarios)",
    )
    ax2.axhline(soc_min, color="green", linestyle="--", alpha=0.5, label="SOC min")
    ax2.axhline(soc_max, color="green", linestyle="--", alpha=0.5, label="SOC max")
    ax2.set_ylabel("Energy (kWh)")
    ax2.legend(loc="upper right")
    ax_ret.plot(
        t,
        [r["reg_retention"] * 100.0 for r in rows],
        color="#444444",
        marker="o",
        linewidth=1.5,
    )
    ax_ret.set_ylabel("Reg Retention (%)")
    ax_ret.set_ylim(0, 100)
    ax_ret.grid(True, axis="y", alpha=0.4)
    ax_ret.set_xlabel("Hour")
    fig.suptitle("Hour-Ahead Rolling Execution")
    plt.tight_layout()
    plt.savefig(save_plot, bbox_inches="tight")
    plt.close(fig)

    # Write CSV
    with open(save_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # KPI block
    min_soc_min = min(exec_soc_min) if exec_soc_min else None
    max_soc_max = max(exec_soc_max) if exec_soc_max else None
    total_dev_mwh = sum(abs(r["deltaP_kW"]) for r in rows) / 1000.0
    sum_r_da = sum(r["R_DA_kW"] for r in rows)
    sum_r_ha = sum(r["R_HA_exec_kW"] for r in rows)
    reg_retention_ratio = (sum_r_ha / sum_r_da) if sum_r_da != 0 else 0.0
    implied_cycles = throughput_kwh / cap if cap != 0 else 0.0
    print("\n--- HOUR-AHEAD KPI ---")
    print(f"Min SOC_min_kWh over day: {min_soc_min:.2f}")
    print(f"Max SOC_max_kWh over day: {max_soc_max:.2f}")
    print(f"Total deviation energy (MWh): {total_dev_mwh:.4f}")
    print(f"Reg retention ratio (%): {reg_retention_ratio * 100:.2f}")
    print(f"Hours inverter binds: {inverter_binds}")
    print(f"Hours headroom binds: {headroom_binds}")
    print(f"Total degradation cost ($): {total_degradation_cost:.2f}")
    print(f"Implied cycles/day: {implied_cycles:.4f}")

    return results


def run_hourahead_rolling_metrics(
    da_plan,
    scenarios,
    params,
    H=4,
    realized_scenario_idx=0,
):
    T = params["T"]
    num_scen = len(scenarios)
    scenario_probs = [1 / num_scen] * num_scen

    cap = params["BESS_capacity_kWh"]
    p_max = params["BESS_max_power_kW"]
    eta_c, eta_d = params["eta_c"], params["eta_d"]
    soc_min = cap * params["SoC_min"]
    soc_max = cap * params["SoC_max"]
    soc_start = cap * params["SoC_initial"]
    reg_price_divisor = 1000 if params.get("reg_price_is_per_mwh", True) else 1
    pr_e = params["pr_e_rt"]
    pr_f = params["pr_fre"]
    degr_cost = params["degradation_cost"]
    reg_energy_safety_factor = params.get("reg_energy_safety_factor", 1.2)

    P_DA_net = np.array(da_plan["P_DA_net"])
    R_DA = np.array(da_plan["R_DA"])
    realized_scen = scenarios[realized_scenario_idx]

    delta_up_eff = []
    delta_dn_eff = []
    for t in range(T):
        ups = [scen.loc[t, "delta_t_up"] for scen in scenarios]
        dns = [scen.loc[t, "delta_t_dn"] for scen in scenarios]
        delta_up_eff.append(reg_energy_safety_factor * max(ups))
        delta_dn_eff.append(reg_energy_safety_factor * max(dns))

    totals = {
        "reg_revenue_total": 0.0,
        "da_energy_total": float(np.sum(-P_DA_net * pr_e)),
        "degradation_total": 0.0,
        "deviation_cost_total": 0.0,
        "num_soc_violations": 0,
        "total_baseline_deviation_MWh": 0.0,
    }
    tol = 1e-6

    for t0 in range(T):
        horizon = min(H, T - t0)
        m = Model("BESS_HourAhead_Rolling")
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = 60

        P_ch = m.addVars(horizon, lb=0, name="P_ch")
        P_dis = m.addVars(horizon, lb=0, name="P_dis")
        z_base = m.addVars(horizon, vtype=GRB.BINARY, name="z_base")
        R_HA = m.addVars(horizon, lb=0, name="R_HA")
        E_soc = m.addVars(num_scen, horizon + 1, name="E_soc")
        P_ch_up = m.addVars(num_scen, horizon, lb=0, name="P_ch_up")
        P_dis_up = m.addVars(num_scen, horizon, lb=0, name="P_dis_up")
        P_ch_dn = m.addVars(num_scen, horizon, lb=0, name="P_ch_dn")
        P_dis_dn = m.addVars(num_scen, horizon, lb=0, name="P_dis_dn")

        for s in range(num_scen):
            m.addConstr(E_soc[s, 0] == soc_start)

        for h in range(horizon):
            t = t0 + h
            net_base = P_ch[h] - P_dis[h]
            delta_p = net_base - P_DA_net[t]
            m.addConstr(R_HA[h] <= R_DA[t])
            m.addConstr(delta_p <= R_DA[t])
            m.addConstr(delta_p >= -R_DA[t])
            m.addConstr(P_ch[h] <= p_max * z_base[h])
            m.addConstr(P_dis[h] <= p_max * (1 - z_base[h]))
            for s in range(num_scen):
                s_up = scenarios[s].loc[t, "s_up"]
                s_dn = scenarios[s].loc[t, "s_dn"]
                du = scenarios[s].loc[t, "delta_t_up"]
                dd = scenarios[s].loc[t, "delta_t_dn"]
                m.addConstr(net_base - s_up * R_HA[h] == P_ch_up[s, h] - P_dis_up[s, h])
                m.addConstr(net_base - s_dn * R_HA[h] == P_ch_dn[s, h] - P_dis_dn[s, h])
                m.addConstr(P_ch[h] + R_HA[h] <= p_max)
                m.addConstr(P_dis[h] + R_HA[h] <= p_max)
                m.addConstr(P_ch_up[s, h] <= p_max)
                m.addConstr(P_dis_up[s, h] <= p_max)
                m.addConstr(P_ch_dn[s, h] <= p_max)
                m.addConstr(P_dis_dn[s, h] <= p_max)
                energy_change = (
                    du * (P_ch_up[s, h] * eta_c - P_dis_up[s, h] / eta_d)
                    + dd * (P_ch_dn[s, h] * eta_c - P_dis_dn[s, h] / eta_d)
                    + P_ch[h] * eta_c * 1.0
                    - P_dis[h] / eta_d * 1.0
                )
                m.addConstr(E_soc[s, h + 1] == E_soc[s, h] + energy_change)
                m.addConstr(E_soc[s, h + 1] <= soc_max)
                m.addConstr(E_soc[s, h + 1] >= soc_min)
                m.addConstr(E_soc[s, h] - soc_min >= R_HA[h] * (delta_up_eff[t] / eta_d))
                m.addConstr(soc_max - E_soc[s, h] >= R_HA[h] * (delta_dn_eff[t] * eta_c))

        reg_revenue = quicksum((R_HA[h] / reg_price_divisor) * pr_f[t0 + h] for h in range(horizon))
        deviation_cost = quicksum(pr_e[t0 + h] * (P_ch[h] - P_dis[h] - P_DA_net[t0 + h]) for h in range(horizon))
        deg_cost = quicksum(
            scenario_probs[s]
            * (
                (P_ch_up[s, h] + P_dis_up[s, h]) * scenarios[s].loc[t0 + h, "delta_t_up"]
                + (P_ch_dn[s, h] + P_dis_dn[s, h]) * scenarios[s].loc[t0 + h, "delta_t_dn"]
            )
            * degr_cost
            for s in range(num_scen)
            for h in range(horizon)
        ) + quicksum((P_ch[h] + P_dis[h]) * 1.0 * degr_cost for h in range(horizon))
        m.setObjective(reg_revenue - deviation_cost - deg_cost, GRB.MAXIMIZE)
        m.optimize()

        if m.status != GRB.OPTIMAL:
            raise RuntimeError(f"HA model infeasible at t0={t0}, status={m.status}")

        s = realized_scenario_idx
        t = t0
        r_ha = R_HA[0].x
        dev_kwh = abs(P_ch[0].x - P_dis[0].x - P_DA_net[t]) * 1.0
        totals["reg_revenue_total"] += (r_ha / reg_price_divisor) * pr_f[t]
        totals["deviation_cost_total"] += pr_e[t] * (P_ch[0].x - P_dis[0].x - P_DA_net[t])
        realized_deg = (
            (P_ch_up[s, 0].x + P_dis_up[s, 0].x) * realized_scen.loc[t, "delta_t_up"]
            + (P_ch_dn[s, 0].x + P_dis_dn[s, 0].x) * realized_scen.loc[t, "delta_t_dn"]
            + (P_ch[0].x + P_dis[0].x) * 1.0
        )
        totals["degradation_total"] += realized_deg * degr_cost
        totals["total_baseline_deviation_MWh"] += abs(dev_kwh) / 1000.0

        soc_next = E_soc[s, 1].x
        if soc_next < soc_min - tol or soc_next > soc_max + tol:
            totals["num_soc_violations"] += 1
        soc_start = soc_next

    totals["market_profit"] = (
        totals["reg_revenue_total"]
        + totals["da_energy_total"]
        - totals["degradation_total"]
    )
    totals["profit_with_deviation_cost"] = (
        totals["market_profit"] - totals["deviation_cost_total"]
    )
    return totals


def run_hourahead_perfect_information(da_plan, realized_scenario, params):
    T = params["T"]
    cap = params["BESS_capacity_kWh"]
    p_max = params["BESS_max_power_kW"]
    eta_c, eta_d = params["eta_c"], params["eta_d"]
    soc_min = cap * params["SoC_min"]
    soc_max = cap * params["SoC_max"]
    soc_start = cap * params["SoC_initial"]
    reg_price_divisor = 1000 if params.get("reg_price_is_per_mwh", True) else 1
    pr_e = params["pr_e_rt"]
    pr_f = params["pr_fre"]
    degr_cost = params["degradation_cost"]
    reg_energy_safety_factor = params.get("reg_energy_safety_factor", 1.2)

    P_DA_net = np.array(da_plan["P_DA_net"])
    R_DA = np.array(da_plan["R_DA"])

    delta_up_eff = []
    delta_dn_eff = []
    for t in range(T):
        delta_up_eff.append(reg_energy_safety_factor * realized_scenario.loc[t, "delta_t_up"])
        delta_dn_eff.append(reg_energy_safety_factor * realized_scenario.loc[t, "delta_t_dn"])

    m = Model("BESS_HourAhead_PerfectInfo")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = 60

    P_ch = m.addVars(T, lb=0, name="P_ch")
    P_dis = m.addVars(T, lb=0, name="P_dis")
    z_base = m.addVars(T, vtype=GRB.BINARY, name="z_base")
    R_HA = m.addVars(T, lb=0, name="R_HA")
    E_soc = m.addVars(T + 1, name="E_soc")
    P_ch_up = m.addVars(T, lb=0, name="P_ch_up")
    P_dis_up = m.addVars(T, lb=0, name="P_dis_up")
    P_ch_dn = m.addVars(T, lb=0, name="P_ch_dn")
    P_dis_dn = m.addVars(T, lb=0, name="P_dis_dn")

    m.addConstr(E_soc[0] == soc_start)

    for t in range(T):
        net_base = P_ch[t] - P_dis[t]
        delta_p = net_base - P_DA_net[t]
        m.addConstr(R_HA[t] <= R_DA[t])
        m.addConstr(delta_p <= R_DA[t])
        m.addConstr(delta_p >= -R_DA[t])
        m.addConstr(P_ch[t] <= p_max * z_base[t])
        m.addConstr(P_dis[t] <= p_max * (1 - z_base[t]))
        s_up = realized_scenario.loc[t, "s_up"]
        s_dn = realized_scenario.loc[t, "s_dn"]
        du = realized_scenario.loc[t, "delta_t_up"]
        dd = realized_scenario.loc[t, "delta_t_dn"]
        m.addConstr(net_base - s_up * R_HA[t] == P_ch_up[t] - P_dis_up[t])
        m.addConstr(net_base - s_dn * R_HA[t] == P_ch_dn[t] - P_dis_dn[t])
        m.addConstr(P_ch[t] + R_HA[t] <= p_max)
        m.addConstr(P_dis[t] + R_HA[t] <= p_max)
        m.addConstr(P_ch_up[t] <= p_max)
        m.addConstr(P_dis_up[t] <= p_max)
        m.addConstr(P_ch_dn[t] <= p_max)
        m.addConstr(P_dis_dn[t] <= p_max)
        energy_change = (
            du * (P_ch_up[t] * eta_c - P_dis_up[t] / eta_d)
            + dd * (P_ch_dn[t] * eta_c - P_dis_dn[t] / eta_d)
            + P_ch[t] * eta_c * 1.0
            - P_dis[t] / eta_d * 1.0
        )
        m.addConstr(E_soc[t + 1] == E_soc[t] + energy_change)
        m.addConstr(E_soc[t + 1] <= soc_max)
        m.addConstr(E_soc[t + 1] >= soc_min)
        m.addConstr(E_soc[t] - soc_min >= R_HA[t] * (delta_up_eff[t] / eta_d))
        m.addConstr(soc_max - E_soc[t] >= R_HA[t] * (delta_dn_eff[t] * eta_c))

    reg_revenue = quicksum((R_HA[t] / reg_price_divisor) * pr_f[t] for t in range(T))
    deviation_cost = quicksum(pr_e[t] * (P_ch[t] - P_dis[t] - P_DA_net[t]) for t in range(T))
    deg_cost = quicksum(
        (
            (P_ch_up[t] + P_dis_up[t]) * realized_scenario.loc[t, "delta_t_up"]
            + (P_ch_dn[t] + P_dis_dn[t]) * realized_scenario.loc[t, "delta_t_dn"]
        )
        * degr_cost
        for t in range(T)
    ) + quicksum((P_ch[t] + P_dis[t]) * 1.0 * degr_cost for t in range(T))
    m.setObjective(reg_revenue - deviation_cost - deg_cost, GRB.MAXIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"Perfect-info model infeasible, status={m.status}")

    totals = {
        "reg_revenue_total": sum((R_HA[t].x / reg_price_divisor) * pr_f[t] for t in range(T)),
        "da_energy_total": float(np.sum(-P_DA_net * pr_e)),
        "degradation_total": 0.0,
        "deviation_cost_total": sum(pr_e[t] * (P_ch[t].x - P_dis[t].x - P_DA_net[t]) for t in range(T)),
        "num_soc_violations": 0,
        "total_baseline_deviation_MWh": 0.0,
    }
    for t in range(T):
        realized_deg = (
            (P_ch_up[t].x + P_dis_up[t].x) * realized_scenario.loc[t, "delta_t_up"]
            + (P_ch_dn[t].x + P_dis_dn[t].x) * realized_scenario.loc[t, "delta_t_dn"]
            + (P_ch[t].x + P_dis[t].x) * 1.0
        )
        totals["degradation_total"] += realized_deg * degr_cost
        totals["total_baseline_deviation_MWh"] += abs(P_ch[t].x - P_dis[t].x - P_DA_net[t]) / 1000.0
        if E_soc[t + 1].x < soc_min - 1e-6 or E_soc[t + 1].x > soc_max + 1e-6:
            totals["num_soc_violations"] += 1

    totals["market_profit"] = (
        totals["reg_revenue_total"]
        + totals["da_energy_total"]
        - totals["degradation_total"]
    )
    totals["profit_with_deviation_cost"] = (
        totals["market_profit"] - totals["deviation_cost_total"]
    )
    return totals


def sweep_rolling_horizon(
    da_plan,
    scenarios,
    params,
    H_list,
    realized_scenario_idx=0,
    save_csv="output/rolling_horizon_sweep.csv",
    save_plot="output/rolling_horizon_profit.png",
    save_plot_comparison="output/rolling_horizon_profit_comparison.png",
):
    rows = []
    realized_scen = scenarios[realized_scenario_idx]
    for H in H_list:
        rolling = run_hourahead_rolling_metrics(
            da_plan,
            scenarios,
            params,
            H=H,
            realized_scenario_idx=realized_scenario_idx,
        )
        perfect = run_hourahead_perfect_information(
            da_plan,
            realized_scen,
            params,
        )
        if H <= 4 and abs(rolling["deviation_cost_total"]) > rolling["reg_revenue_total"]:
            print(
                f"WARNING: H={H} deviation cost ${rolling['deviation_cost_total']:.2f} "
                f"exceeds reg revenue ${rolling['reg_revenue_total']:.2f}; "
                "check deviation cost units."
            )
        rows.append(
            {
                "H": H,
                "rolling_reg_revenue_total": rolling["reg_revenue_total"],
                "rolling_da_energy_total": rolling["da_energy_total"],
                "rolling_degradation_total": rolling["degradation_total"],
                "rolling_deviation_cost_total": rolling["deviation_cost_total"],
                "rolling_market_profit": rolling["market_profit"],
                "rolling_profit_with_deviation_cost": rolling["profit_with_deviation_cost"],
                "rolling_num_soc_violations": rolling["num_soc_violations"],
                "rolling_total_baseline_deviation_MWh": rolling["total_baseline_deviation_MWh"],
                "perfect_reg_revenue_total": perfect["reg_revenue_total"],
                "perfect_da_energy_total": perfect["da_energy_total"],
                "perfect_degradation_total": perfect["degradation_total"],
                "perfect_deviation_cost_total": perfect["deviation_cost_total"],
                "perfect_market_profit": perfect["market_profit"],
                "perfect_profit_with_deviation_cost": perfect["profit_with_deviation_cost"],
                "perfect_num_soc_violations": perfect["num_soc_violations"],
                "perfect_total_baseline_deviation_MWh": perfect["total_baseline_deviation_MWh"],
            }
        )

    Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(save_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    Path(save_plot).parent.mkdir(parents=True, exist_ok=True)
    Path(save_plot_comparison).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    H_vals = [r["H"] for r in rows]
    ax.plot(
        H_vals,
        [r["rolling_profit_with_deviation_cost"] for r in rows],
        marker="o",
        label="Rolling hour-ahead",
    )
    ax.plot(
        H_vals,
        [r["perfect_profit_with_deviation_cost"] for r in rows],
        marker="^",
        label="Perfect information",
    )
    ax.set_xlabel("Rolling horizon H (hours)")
    ax.set_ylabel("Profit ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_plot, bbox_inches="tight")
    plt.savefig(save_plot_comparison, bbox_inches="tight")
    plt.close(fig)

    return rows
