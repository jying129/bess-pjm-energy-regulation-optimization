"""
Data processing and signal construction utilities for the BESS model.

This module:
- Aggregates high-frequency PJM data into hourly inputs.
- Builds aggregate RegD signal representations.
- Generates stochastic regulation scenarios.
- Reduces scenarios for tractable optimization.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Time-series aggregation
# =============================================================================

def transfer(data, datetime_col='datetime_beginning_ept', group_col='hour', target_cols=None, start_date=None, end_date=None, rotate=True):
    """
    Aggregate PJM time-series data into hourly averages, with optional date filtering and 12-hour rotation.

    This function is used to convert raw PJM market data (energy prices, regulation prices, mileage values)
    into hourly inputs compatible with the day-ahead and hour-ahead optimization models.

    Parameters
    ----------
    data : pd.DataFrame
        Input PJM market data.
    datetime_col : str
        Column containing timestamps.
    group_col : str
        Column name for grouping (hour of day).
    target_cols : list or None
        Columns to average. If None, all numeric columns are used.
    start_date : str or None
        Optional start date filter (YYYY-MM-DD).
    end_date : str or None
        Optional end date filter (YYYY-MM-DD).
    rotate : bool
        If True, apply a 12-hour rotation to align day structure.

    Returns
    -------
    np.ndarray
        Hourly-averaged values as a NumPy array.
    """
    # Ensure datetime format
    data[datetime_col] = pd.to_datetime(data[datetime_col])

    # Optional date filtering
    if start_date:
        data = data[data[datetime_col] >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data[datetime_col] <= pd.to_datetime(end_date)]
    
    # Extract hour of day
    data[group_col] = data[datetime_col].dt.hour
    
    # Compute hourly averages
    if target_cols is not None:
        hourly_avg = data.groupby(group_col)[target_cols].mean()
    else:
        hourly_avg = data.groupby(group_col).mean()
    
    arr = hourly_avg.values

    # Optional 12-hour rotation
    if rotate and len(arr) >= 24:
        arr[:12], arr[12:] = arr[12:], arr[:12].copy()
    return arr

# =============================================================================
# 2. Aggregate RegD signal processing
# =============================================================================

def process_historical_signals(filepath, time_col_name, signal_col_name):
    """
    Process high-frequency historical RegD signals into hourly aggregates.

    This function implements the signal aggregation,
    condensing the 2-second AGC signal into four hourly parameters:
        - average up-regulation signal (s_up)
        - average down-regulation signal (s_dn)
        - fraction of hour in up-regulation (delta_t_up)
        - fraction of hour in down-regulation (delta_t_dn)

    Parameters
    ----------
    filepath : str
        Path to CSV file containing high-frequency regulation signals.
    time_col_name : str
        Name of timestamp column.
    signal_col_name : str
        Name of regulation signal column.

    Returns
    -------
    pd.DataFrame
        Hourly aggregate RegD parameters.
    """
    
    df = pd.read_csv(filepath)
    df[time_col_name] = pd.to_datetime(df[time_col_name])
    df.set_index(time_col_name, inplace=True)

    hourly_groups = df.groupby(pd.Grouper(freq='H')) # Group data into hourly windows
    
    results = []
    for hour, group_df in hourly_groups:
        if group_df.empty:
            continue
        # Extract regulation signal values
        signals = group_df[signal_col_name]
        
        # Separate regulation-up and -down signals
        up_reg_signals = signals[signals >= 0]
        down_reg_signals = signals[signals < 0]
        
        total_points = len(signals)
        
        # Average magnitudes
        s_up = up_reg_signals.mean() if not up_reg_signals.empty else 0
        s_dn = down_reg_signals.mean() if not down_reg_signals.empty else 0
        
        # Time fractions
        delta_t_up = len(up_reg_signals) / total_points if total_points > 0 else 0
        delta_t_dn = 1 - delta_t_up
        
        results.append({
            'hour': hour, 's_up': s_up, 's_dn': s_dn,
            'delta_t_up': delta_t_up, 'delta_t_dn': delta_t_dn
        })
        
    return pd.DataFrame(results)

# =============================================================================
# 3. Stochastic scenario generation
# =============================================================================


def generate_regulation_scenarios(historical_df, num_scenarios, num_hours=24, enforce_neutral=False, seed=None):
    """
    Generates future scenarios for regulation signals using bootstrapping.

    This function supports the scenario-based stochastic optimization.
    Each scenario represents one possible realization of hourly aggregate RegD behavior.

    Parameters
    ----------
    historical_df : pd.DataFrame
        Historical hourly aggregate RegD data.
    num_scenarios : int
        Number of scenarios to generate.
    num_hours : int
        Hours per scenario (default: 24).
    enforce_neutral : bool
        If True, enforce approximate energy neutrality per hour.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    list[pd.DataFrame]
        List of scenario DataFrames.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    scenarios = []
    for i in range(num_scenarios):
        # Bootstrapping: Randomly sample 24 hours WITH REPLACEMENT from the historical data
        scenario_df = historical_df.sample(n=num_hours, replace=True, random_state=seed)
        
        # Reset the index to be a simple 0-23 hour sequence for the new day
        scenario_df = scenario_df.reset_index(drop=True)

        # Optional energy-neutral adjustment
        if enforce_neutral:
            scenario_df = scenario_df.copy()
            for idx, row in scenario_df.iterrows():
                net = row['delta_t_up'] * row['s_up'] + row['delta_t_dn'] * row['s_dn']
                if abs(net) > 1e-9 and row['delta_t_dn'] > 0:
                    adjust = net / row['delta_t_dn']
                    s_dn_new = row['s_dn'] - adjust
                    scenario_df.at[idx, 's_dn'] = min(s_dn_new, -1e-6)  # keep negative

        scenarios.append(scenario_df)
        
    return scenarios

# =============================================================================
# 4. Scenario reduction via k-means clustering
# =============================================================================

def reduce_scenarios_kmeans(scenarios, num_clusters):
    """
    Reduces a list of scenarios using k-means clustering and returns representative scenarios with weights.

    This function improves computational tractability of the day-ahead stochastic optimization 
    by selecting representative scenarios and associated probabilities.

    Each scenario is flattened into a feature vector consisting of:
        [s_up, s_dn, delta_t_up, delta_t_dn] over 24 hours.

    Parameters
    ----------
    scenarios : list[pd.DataFrame]
        Full scenario set.
    num_clusters : int
        Desired number of representative scenarios.

    Returns
    -------
    reduced_scenarios : list[pd.DataFrame]
        Representative (medoid) scenarios.
    scenario_probs : list[float]
        Scenario probabilities (weights summing to 1).
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("scikit-learn not installed; skipping k-means reduction and keeping all scenarios with equal weights.")
        probs = [1/len(scenarios)] * len(scenarios)
        return scenarios, probs

    if num_clusters >= len(scenarios):
        # Nothing to reduce; equal weights
        probs = [1/len(scenarios)] * len(scenarios)
        return scenarios, probs
    
    feature_matrix = []
    for scen in scenarios:
        # flatten columns in a consistent order
        vec = np.concatenate([
            scen['s_up'].values,
            scen['s_dn'].values,
            scen['delta_t_up'].values,
            scen['delta_t_dn'].values,
        ])
        feature_matrix.append(vec)
    feature_matrix = np.vstack(feature_matrix)

    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
    labels = kmeans.fit_predict(feature_matrix)
    centroids = kmeans.cluster_centers_

    reduced_scenarios = []
    scenario_probs = []

    for k in range(num_clusters):
        cluster_indices = np.where(labels == k)[0]
        if len(cluster_indices) == 0:
            continue

        cluster_feats = feature_matrix[cluster_indices]
        centroid = centroids[k]

        # choose medoid (closest to centroid)
        distances = np.linalg.norm(cluster_feats - centroid, axis=1)
        medoid_idx = cluster_indices[np.argmin(distances)]
        
        reduced_scenarios.append(scenarios[medoid_idx].reset_index(drop=True))
        scenario_probs.append(len(cluster_indices) / len(scenarios))

    # Normalize probabilities in case of any numerical drift
    total = sum(scenario_probs)
    scenario_probs = [p / total for p in scenario_probs]

    return reduced_scenarios, scenario_probs

# =============================================================================
# 5. Daily energy price helper function
# =============================================================================

def daily_energy_metrics(filepath, datetime_col, price_col, start_date=None, end_date=None, n_top=5):
    """
    Compute daily energy price spread and volatility to find lucrative days.
    
    This helper function is for identifying representative high-volatility or 
    high-arbitrage days used in sensitivity analysis and case study selection.

    Returns
    -------
    pd.DataFrame
        Top days ranked by energy price spread.
    """
    
    df = pd.read_csv(filepath)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    if start_date:
        df = df[df[datetime_col] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[datetime_col] <= pd.to_datetime(end_date)]
    
    df['date'] = df[datetime_col].dt.date
    df['hour'] = df[datetime_col].dt.hour
    
    grouped = df.groupby('date')
    metrics = grouped[price_col].agg(['max', 'min', 'std', 'mean']).reset_index()
    metrics['spread'] = metrics['max'] - metrics['min']
    metrics = metrics.sort_values('spread', ascending=False)

    return metrics.head(n_top)
