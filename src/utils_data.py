# src/utils_data.py
import pandas as pd
import numpy as np
import holidays
from sklearn.cluster import MiniBatchKMeans



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NYC_BOUNDS = {
    'lat_min': 40.5,
    'lat_max': 40.92,
    'lon_min': -74.05,
    'lon_max': -73.7
}

AIRPORTS = {
    'JFK': (-73.7781, 40.6413),
    'LGA': (-73.8740, 40.7769),
    'EWR': (-74.1745, 40.6895)
}

US_HOLIDAYS = holidays.US(years=list(range(2016, 2050)), state='NY')

# Hard filters: physics/logic bounds, not statistical (EDA-derived)
DURATION_MIN_SECONDS   = 30      # < 30 s  -> phantom / system record
DURATION_MAX_SECONDS   = 86_400  # > 24 h  -> system error
ZERO_DIST_DURATION_MAX = 120     # zero-distance + duration > 2 min -> GPS/meter fault


# ---------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------

def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance in kilometres between two coordinate pairs."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 6371  # earth radius in km


def bearing(lon1, lat1, lon2, lat2):
    """Compass bearing (0–360°) from point 1 to point 2."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def is_near_airport(lon, lat, airports=AIRPORTS, threshold=1):
    """True if (lon, lat) is within *threshold* km of any known airport."""
    for coords in airports.values():
        if haversine(lon, lat, *coords) <= threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Private low-level helpers (not pipeline steps)
# ---------------------------------------------------------------------------

def _encode_zone(lat_col, lon_col):
    """Coarse ~1 km grid-cell label from latitude/longitude columns."""
    return (
        (lat_col // 0.01).astype(int).astype(str)
        + "_"
        + (lon_col // 0.01).astype(int).astype(str)
    )


def _build_spatial_clusters(data, lat_col, lon_col,
                             n_clusters=20, sample_size=200_000, random_state=42):
    """
    Fit MiniBatchKMeans on a coordinate sample and return (label Series, model).
    Called only during training; the fitted model is stored in train_stats for
    inference reuse.
    """
    coords = data[[lat_col, lon_col]].dropna()
    sample = coords.sample(min(sample_size, len(coords)), random_state=random_state)
    model  = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10_000, random_state=random_state)
    model.fit(sample)
    labels = pd.Series(-1, index=data.index, dtype=int)
    labels.loc[coords.index] = model.predict(coords)
    return labels, model


# ---------------------------------------------------------------------------
# Pipeline step functions
#
# Convention: every step receives (df, train_stats, is_training) and returns df.
# train_stats is mutated in-place during training and read-only during inference.
# Steps that need extra scalar parameters (e.g. iqr_factor) take them as
# additional keyword arguments.
# ---------------------------------------------------------------------------

def _step_filter_nyc_bounds(df, train_stats, is_training):
    """Drop rows whose pickup or dropoff coordinates fall outside NYC."""
    mask = (
        df['pickup_latitude'].between(NYC_BOUNDS['lat_min'],  NYC_BOUNDS['lat_max'])
        & df['pickup_longitude'].between(NYC_BOUNDS['lon_min'], NYC_BOUNDS['lon_max'])
        & df['dropoff_latitude'].between(NYC_BOUNDS['lat_min'],  NYC_BOUNDS['lat_max'])
        & df['dropoff_longitude'].between(NYC_BOUNDS['lon_min'], NYC_BOUNDS['lon_max'])
    )
    return df.loc[mask].copy()


def _step_haversine_distance(df, train_stats, is_training):
    """Compute haversine distance — must run before the hard-filter step."""
    df['haversine_distance'] = haversine(
        df['pickup_longitude'], df['pickup_latitude'],
        df['dropoff_longitude'], df['dropoff_latitude']
    )
    return df


def _step_filter_hard_bounds(df, train_stats, is_training):
    """
    Remove rows that are physically impossible or system errors:
      1. trip_duration < 30 s  -> phantom record
      2. trip_duration > 24 h  -> meter / system error
      3. haversine_distance == 0 AND duration > 2 min -> GPS fault
    Applied identically at training and inference (logic-based, not statistical).
    """
    n_before = len(df)

    df = df[df['trip_duration'].between(DURATION_MIN_SECONDS, DURATION_MAX_SECONDS)].copy()

    zero_dist_noise = (df['haversine_distance'] == 0) & (df['trip_duration'] > ZERO_DIST_DURATION_MAX)
    df = df[~zero_dist_noise].copy()

    n_removed = n_before - len(df)
    if n_removed:
        print(f"  Hard-filter removed {n_removed} rows ({n_removed / n_before:.1%})")

    return df


def _step_encode_flag(df, train_stats, is_training):
    """Encode store_and_fwd_flag from Y/N string to 0/1 integer."""
    df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)
    return df


def _step_temporal_features(df, train_stats, is_training):
    """
    Extract all datetime-derived features:
      - calendar fields  : hour, dayofweek, month, day
      - binary flags     : rush_hours, night_flag, is_weekend, weekend_rush,
                           night_weekday_flag, is_holiday
      - ordinal          : week_hour
      - OHE-ready string : hour_weekday_interaction
    """
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['pickup_date']     = df['pickup_datetime'].dt.floor('D')

    df['is_holiday']  = df['pickup_date'].isin(US_HOLIDAYS).astype(int)
    df['dayofweek']   = df['pickup_datetime'].dt.dayofweek
    df['month']       = df['pickup_datetime'].dt.month
    df['hour']        = df['pickup_datetime'].dt.hour
    df['day']         = df['pickup_datetime'].dt.day

    # EDA: peak congestion windows are 8–10 AM and 4–7 PM
    df['rush_hours'] = (
        df['hour'].between(8, 10) | df['hour'].between(16, 19)
    ).astype(int)

    df['night_flag']         = ((df['hour'] >= 21) | (df['hour'] < 5)).astype(int)
    df['is_weekend']         = df['dayofweek'].isin([5, 6]).astype(int)
    df['weekend_rush']       = df['is_weekend'] * df['rush_hours']
    df['night_weekday_flag'] = ((df['is_weekend'] == 0) & (df['night_flag'] == 1)).astype(int)

    # Kept numeric for group-level aggregations in later steps
    df['week_hour'] = df['dayofweek'] * 24 + df['hour']

    # EDA: the same clock-hour means very different things on Mon vs Sun
    df['hour_weekday_interaction'] = (
        df['hour'].astype(str) + '_' + df['dayofweek'].astype(str)
    )
    return df


def _step_spatial_distance_features(df, train_stats, is_training):
    """
    Compute spatial features from raw coordinates:
      bearing, manhattan_distance, distance_per_passenger, log_distance.
    (haversine_distance is already present from _step_haversine_distance.)
    """
    df['bearing'] = bearing(
        df['pickup_longitude'], df['pickup_latitude'],
        df['dropoff_longitude'], df['dropoff_latitude']
    )
    df['manhattan_distance'] = (
        np.abs(df['dropoff_longitude'] - df['pickup_longitude'])
        + np.abs(df['dropoff_latitude'] - df['pickup_latitude'])
    )
    df['distance_per_passenger'] = df['haversine_distance'] / df['passenger_count'].clip(lower=1)

    # EDA: log-transform matches scale of log-transformed target and
    # reduces leverage of long-haul outliers.
    df['log_distance'] = np.log1p(df['haversine_distance'])
    return df


def _step_speed(df, train_stats, is_training):
    """Derive trip speed in km/h from haversine distance and trip_duration."""
    duration_hours   = df['trip_duration'].clip(lower=1) / 3600
    df['speed_kmph'] = df['haversine_distance'] / duration_hours
    df['speed_kmph'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['speed_kmph'].fillna(df['speed_kmph'].median(), inplace=True)
    return df


def _step_spatial_clusters(df, train_stats, is_training):
    """
    Assign pickup/dropoff cluster IDs via MiniBatchKMeans, then derive:
      - pickup_cluster_median_trip     : median trip duration per pickup cluster
      - pickup_cluster_median_speed    : median speed per pickup cluster
      - dropoff_cluster_median_trip    : median trip duration per dropoff cluster
      - dropoff_cluster_median_speed   : median speed per dropoff cluster
      - corridor_median_duration       : median duration per (pickup × dropoff) pair
      - vendor_hour_speed_median       : median speed per (vendor, hour) slot
      - speed_minus_cluster_median     : individual speed minus pickup cluster baseline

    Training  : fits two KMeans models and all aggregation tables; stores in train_stats.
    Inference : loads stored models/tables and applies them.
    """
    # --- Cluster assignment ---
    if is_training:
        print("Fitting MiniBatchKMeans clusters ...")
        df['pickup_cluster'],  pickup_km  = _build_spatial_clusters(
            df, 'pickup_latitude',  'pickup_longitude')
        df['dropoff_cluster'], dropoff_km = _build_spatial_clusters(
            df, 'dropoff_latitude', 'dropoff_longitude')
        train_stats['pickup_cluster_model']  = pickup_km
        train_stats['dropoff_cluster_model'] = dropoff_km
    else:
        pickup_km  = train_stats.get('pickup_cluster_model')
        dropoff_km = train_stats.get('dropoff_cluster_model')
        if pickup_km is None or dropoff_km is None:
            raise KeyError("Missing cluster models in train_stats. Rerun training feature engineering first.")
        pickup_coords  = df[['pickup_latitude',  'pickup_longitude']].dropna()
        dropoff_coords = df[['dropoff_latitude', 'dropoff_longitude']].dropna()
        df['pickup_cluster']  = pd.Series(-1, index=df.index, dtype=int)
        df['dropoff_cluster'] = pd.Series(-1, index=df.index, dtype=int)
        df.loc[pickup_coords.index,  'pickup_cluster']  = pickup_km.predict(pickup_coords)
        df.loc[dropoff_coords.index, 'dropoff_cluster'] = dropoff_km.predict(dropoff_coords)

    # --- Pickup cluster intensity stats ---
    if is_training:
        pickup_cluster_intensity = (
            df.groupby('pickup_cluster')
              .agg(
                  pickup_cluster_trip_ct     =('trip_duration', 'size'),
                  pickup_cluster_median_trip =('trip_duration', 'median'),
                  pickup_cluster_median_speed=('speed_kmph',    'median'),
              )
              .reset_index()
        )
        train_stats['pickup_cluster_intensity'] = pickup_cluster_intensity
    else:
        pickup_cluster_intensity = train_stats.get('pickup_cluster_intensity')
        if pickup_cluster_intensity is None:
            raise KeyError("Missing pickup_cluster_intensity in train_stats.")
    df = df.merge(pickup_cluster_intensity, on='pickup_cluster', how='left')
    df['pickup_cluster_median_speed'].fillna(df['pickup_cluster_median_speed'].median(), inplace=True)
    df['pickup_cluster_trip_ct'].fillna(0, inplace=True)
    df['pickup_cluster_median_trip'].fillna(df['pickup_cluster_median_trip'].median(), inplace=True)

    # --- Dropoff cluster intensity stats ---
    if is_training:
        dropoff_cluster_intensity = (
            df.groupby('dropoff_cluster')
              .agg(
                  dropoff_cluster_trip_ct     =('trip_duration', 'size'),
                  dropoff_cluster_median_trip =('trip_duration', 'median'),
                  dropoff_cluster_median_speed=('speed_kmph',    'median'),
              )
              .reset_index()
        )
        train_stats['dropoff_cluster_intensity'] = dropoff_cluster_intensity
    else:
        dropoff_cluster_intensity = train_stats.get('dropoff_cluster_intensity')
        if dropoff_cluster_intensity is None:
            raise KeyError("Missing dropoff_cluster_intensity in train_stats.")
    df = df.merge(dropoff_cluster_intensity, on='dropoff_cluster', how='left')
    df['dropoff_cluster_median_speed'].fillna(df['dropoff_cluster_median_speed'].median(), inplace=True)
    df['dropoff_cluster_trip_ct'].fillna(0, inplace=True)
    df['dropoff_cluster_median_trip'].fillna(df['dropoff_cluster_median_trip'].median(), inplace=True)

    # --- Corridor stats ---
    if is_training:
        corridor_stats = (
            df.groupby(['pickup_cluster', 'dropoff_cluster'])['trip_duration']
              .median()
              .rename('corridor_median_duration')
              .reset_index()
        )
        train_stats['corridor_stats'] = corridor_stats
    else:
        corridor_stats = train_stats.get('corridor_stats')
        if corridor_stats is None:
            raise KeyError("Missing corridor_stats in train_stats.")
    df = df.merge(corridor_stats, on=['pickup_cluster', 'dropoff_cluster'], how='left')
    df['corridor_median_duration'].fillna(df['corridor_median_duration'].median(), inplace=True)

    # --- Per-vendor hourly speed median ---
    if is_training:
        vendor_hour_speed = (
            df.groupby(['vendor_id', 'hour'])['speed_kmph']
              .median()
              .rename('vendor_hour_speed_median')
              .reset_index()
        )
        train_stats['vendor_hour_speed'] = vendor_hour_speed
    else:
        vendor_hour_speed = train_stats.get('vendor_hour_speed')
        if vendor_hour_speed is None:
            raise KeyError("Missing vendor_hour_speed in train_stats.")
    df = df.merge(vendor_hour_speed, on=['vendor_id', 'hour'], how='left')
    df['vendor_hour_speed_median'].fillna(df['vendor_hour_speed_median'].median(), inplace=True)

    # --- Speed deviation from cluster baseline ---
    df['speed_minus_cluster_median'] = df['speed_kmph'] - df['pickup_cluster_median_speed']

    return df


def _step_anomaly_detection(df, train_stats, is_training):
    """
    Flag days whose trip volume deviates > 2 std from the training mean
    (captures blizzards, major events, data outages).
    Training  : computes and stores mean/std of daily trip counts.
    Inference : reuses stored stats.
    """
    daily_trip_count = df.groupby('pickup_date').size()

    if is_training:
        train_stats['daily_trip_mean'] = daily_trip_count.mean()
        train_stats['daily_trip_std']  = daily_trip_count.std(ddof=0) or 1.0
    else:
        if 'daily_trip_mean' not in train_stats or 'daily_trip_std' not in train_stats:
            raise KeyError("Missing daily trip stats. Refit feature engineering on training data first.")

    mean_daily     = train_stats['daily_trip_mean']
    std_daily      = train_stats['daily_trip_std'] or 1.0
    z_scores       = (daily_trip_count - mean_daily) / std_daily
    anomalous_days = z_scores[abs(z_scores) > 2].index
    df['is_anomaly'] = df['pickup_date'].isin(anomalous_days).astype(int)
    return df


def _step_duration_stats(df, train_stats, is_training):
    """
    Merge per-day and per-day-of-week median/p95 trip duration statistics.
    Training  : computes and stores both tables.
    Inference : loads stored tables; falls back to dow stats for unseen dates.
    """
    daily_duration_stats = df.groupby('pickup_date')['trip_duration'].agg(
        daily_median_duration='median',
        daily_p95_duration=lambda x: x.quantile(0.95)
    ).reset_index()
    dow_duration_stats = df.groupby('dayofweek')['trip_duration'].agg(
        dow_median_duration='median',
        dow_p95_duration=lambda x: x.quantile(0.95)
    ).reset_index()

    if is_training:
        train_stats['daily_duration_stats'] = daily_duration_stats
        train_stats['dow_duration_stats']   = dow_duration_stats
    else:
        daily_duration_stats = train_stats.get('daily_duration_stats', daily_duration_stats)
        dow_duration_stats   = train_stats.get('dow_duration_stats',   dow_duration_stats)

    df = df.merge(daily_duration_stats, on='pickup_date', how='left')
    df = df.merge(dow_duration_stats,   on='dayofweek',   how='left')
    df['daily_median_duration'].fillna(df['dow_median_duration'], inplace=True)
    df['daily_p95_duration'].fillna(df['dow_p95_duration'],       inplace=True)
    df.drop(columns=['dow_median_duration', 'dow_p95_duration'], inplace=True)
    return df


def _step_week_hour_stats(df, train_stats, is_training):
    """
    Merge normalised trip-count and median duration for each (dayofweek, hour) slot.
    Training  : fits and stores the aggregation table.
    Inference : loads stored table; fills missing slots with the training median.
    """
    week_hour_stats = df.groupby('week_hour')['trip_duration'].agg(
        week_hour_trip_ct='size',
        week_hour_median_duration='median'
    ).reset_index()
    week_hour_stats['week_hour_trip_ct_norm'] = (
        week_hour_stats['week_hour_trip_ct'] / week_hour_stats['week_hour_trip_ct'].max()
    )

    if is_training:
        train_stats['week_hour_stats']           = week_hour_stats
        train_stats['week_hour_median_fallback'] = week_hour_stats['week_hour_median_duration'].median()
    else:
        week_hour_stats = train_stats.get('week_hour_stats', week_hour_stats)

    df = df.merge(
        week_hour_stats[['week_hour', 'week_hour_trip_ct_norm', 'week_hour_median_duration']],
        on='week_hour', how='left'
    )
    df['week_hour_trip_ct_norm'].fillna(0, inplace=True)
    df['week_hour_median_duration'].fillna(
        train_stats.get('week_hour_median_fallback', df['week_hour_median_duration'].median()),
        inplace=True
    )
    return df


def _step_traffic_congestion(df, train_stats, is_training):
    """
    Compute traffic congestion features:
      - congestion_rate            : fraction of slow trips in each (dayofweek, hour) slot
      - pickup_congestion_rate     : same, split further by 1 km pickup grid zone
      - dropoff_congestion_rate    : same, split further by 1 km dropoff grid zone

    A trip is 'slow' when its speed is > 1 std below the slot mean (z < -1).
    Training  : fits and stores speed baselines and zone-level congestion tables.
    Inference : loads stored tables.
    Intermediate columns are dropped in _step_cleanup.
    """
    # Slot-level speed baseline
    if is_training:
        speed_stats = (
            df.groupby(['dayofweek', 'hour'])['speed_kmph']
              .agg(mean_speed='mean', std_speed='std')
              .reset_index()
        )
        train_stats['speed_stats'] = speed_stats.copy()
    else:
        speed_stats = train_stats.get('speed_stats')
        if speed_stats is None:
            raise KeyError("Missing speed_stats in train_stats. Rerun training feature engineering first.")

    df = df.merge(speed_stats, on=['dayofweek', 'hour'], how='left')
    df['std_speed']          = df['std_speed'].replace(0, np.nan)
    df['speed_z']            = ((df['speed_kmph'] - df['mean_speed']) / df['std_speed']).fillna(0)
    df['traffic_congestion'] = (df['speed_z'] < -1).astype(int)
    df['congestion_rate']    = df.groupby(['dayofweek', 'hour'])['traffic_congestion'].transform('mean')

    # Zone-level congestion
    df['pickup_zone']  = _encode_zone(df['pickup_latitude'],  df['pickup_longitude'])
    df['dropoff_zone'] = _encode_zone(df['dropoff_latitude'], df['dropoff_longitude'])

    if is_training:
        pickup_congestion = (
            df.groupby(['pickup_zone',  'dayofweek', 'hour'])['traffic_congestion']
              .mean().reset_index(name='pickup_congestion_rate')
        )
        dropoff_congestion = (
            df.groupby(['dropoff_zone', 'dayofweek', 'hour'])['traffic_congestion']
              .mean().reset_index(name='dropoff_congestion_rate')
        )
        train_stats['pickup_congestion_stats']  = pickup_congestion
        train_stats['dropoff_congestion_stats'] = dropoff_congestion
    else:
        pickup_congestion  = train_stats.get('pickup_congestion_stats')
        dropoff_congestion = train_stats.get('dropoff_congestion_stats')
        if pickup_congestion is None or dropoff_congestion is None:
            raise KeyError("Missing congestion statistics in train_stats.")

    df = (
        df.merge(pickup_congestion,  on=['pickup_zone',  'dayofweek', 'hour'], how='left')
          .merge(dropoff_congestion, on=['dropoff_zone', 'dayofweek', 'hour'], how='left')
    )
    df['pickup_congestion_rate'].fillna(0,  inplace=True)
    df['dropoff_congestion_rate'].fillna(0, inplace=True)
    df['congestion_rate'].fillna(df['congestion_rate'].median(), inplace=True)
    return df


def _step_airport_flags(df, train_stats, is_training):
    """Binary flags: is the pickup or dropoff within 1 km of JFK, LGA, or EWR."""
    df['is_pickup_at_airport'] = df.apply(
        lambda row: is_near_airport(row['pickup_longitude'], row['pickup_latitude']), axis=1
    ).astype(int)
    df['is_dropoff_at_airport'] = df.apply(
        lambda row: is_near_airport(row['dropoff_longitude'], row['dropoff_latitude']), axis=1
    ).astype(int)
    return df


def _step_iqr_filter(df, train_stats, is_training, iqr_factor):
    """
    Statistical outlier removal via IQR bounds on four key columns.
    Training  : computes bounds from training distribution and stores them.
    Inference : applies stored bounds (rows outside bounds are dropped).
    """
    cols   = ['trip_duration', 'haversine_distance', 'manhattan_distance', 'speed_kmph']
    bounds = train_stats.setdefault('outlier_bounds', {})
    df_out = df.copy()

    if is_training:
        for col in cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR    = Q3 - Q1
            lo, hi = Q1 - iqr_factor * IQR, Q3 + iqr_factor * IQR
            bounds[col] = (lo, hi)
            df_out = df_out[(df_out[col] >= lo) & (df_out[col] <= hi)]
    else:
        missing = [c for c in cols if c not in bounds]
        if missing:
            raise KeyError(f"Missing outlier bounds for columns: {missing}")
        for col in cols:
            lo, hi = bounds[col]
            df_out = df_out[(df_out[col] >= lo) & (df_out[col] <= hi)]

    return df_out


def _step_cleanup(df, train_stats, is_training):
    """
    Drop all intermediate / raw columns that must not reach the model,
    cast remaining bool columns to int, and add the log-transformed target.

    Dropped:
      - congestion intermediates: traffic_congestion, speed_z, mean_speed, std_speed
      - zone helpers: pickup_zone, dropoff_zone
      - raw cluster IDs (replaced by derived stats): pickup_cluster, dropoff_cluster
      - raw cluster trip counts (replaced by derived stats):
          pickup_cluster_trip_ct, dropoff_cluster_trip_ct
      - raw coordinates and datetime fields
      - 'id' column if present
    """
    helper_cols = [
        'traffic_congestion', 'speed_z', 'mean_speed', 'std_speed',
        'pickup_zone', 'dropoff_zone',
        'pickup_cluster', 'dropoff_cluster',
        'pickup_cluster_trip_ct', 'dropoff_cluster_trip_ct',
    ]
    df.drop(columns=[c for c in helper_cols if c in df.columns], inplace=True)
    df.drop(columns=['id'], inplace=True, errors='ignore')
    df.drop(columns=[
        'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'pickup_date',
    ], inplace=True)

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    df['log_trip_duration'] = np.log1p(df['trip_duration'])
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def apply_feature_engineering(data, train_stats=None, iqr_factor=2.5):
    """
    Full feature-engineering pipeline for both training and inference.

    Args:
        data:        Raw input DataFrame.
        train_stats: None on the first (training) call; the dict returned by
                     that call on all subsequent (inference) calls.
        iqr_factor:  IQR multiplier for statistical outlier removal.

    Returns:
        (df, train_stats)
    """
    df          = data.copy()
    is_training = train_stats is None
    train_stats = train_stats if train_stats is not None else {}

    df = _step_filter_nyc_bounds(df,         train_stats, is_training)
    df = _step_haversine_distance(df,         train_stats, is_training)
    df = _step_filter_hard_bounds(df,         train_stats, is_training)
    df = _step_encode_flag(df,                train_stats, is_training)
    df = _step_temporal_features(df,          train_stats, is_training)
    df = _step_spatial_distance_features(df,  train_stats, is_training)
    df = _step_speed(df,                      train_stats, is_training)
    df = _step_spatial_clusters(df,           train_stats, is_training)
    df = _step_anomaly_detection(df,          train_stats, is_training)
    df = _step_duration_stats(df,             train_stats, is_training)
    df = _step_week_hour_stats(df,            train_stats, is_training)
    df = _step_traffic_congestion(df,         train_stats, is_training)
    df = _step_airport_flags(df,              train_stats, is_training)
    df = _step_iqr_filter(df,                 train_stats, is_training, iqr_factor)
    df = _step_cleanup(df,                    train_stats, is_training)

    return df, train_stats


# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------

def get_feature_lists():
    """Return categorised feature lists consumed by the sklearn pipeline."""
    categorical_features = [
        'passenger_count',
        'vendor_id',
        'hour_weekday_interaction',        # hour × weekday interaction (EDA)
    ]
    cyclic_features = ['hour', 'dayofweek', 'month', 'day']
    numeric_features = [
        'haversine_distance', 'manhattan_distance', 'bearing',
        'distance_per_passenger', 'log_distance', 'week_hour',
        'week_hour_trip_ct_norm', 'week_hour_median_duration',
        'daily_median_duration', 'daily_p95_duration',
        'pickup_congestion_rate', 'dropoff_congestion_rate',
        'congestion_rate', 'speed_kmph',
        # spatial cluster derived features (notebook)
        'pickup_cluster_median_trip', 'pickup_cluster_median_speed',
        'dropoff_cluster_median_trip', 'dropoff_cluster_median_speed',
        'corridor_median_duration',
        'vendor_hour_speed_median',
        'speed_minus_cluster_median',
    ]
    binary_features = [
        'store_and_fwd_flag', 'is_holiday', 'is_anomaly',
        'is_pickup_at_airport', 'is_dropoff_at_airport',
        'night_flag', 'is_weekend', 'rush_hours', 'weekend_rush',
        'night_weekday_flag',               # night × weekday (notebook)
    ]

    all_features = categorical_features + numeric_features + cyclic_features + binary_features

    return {
        'categorical': categorical_features,
        'cyclic':      cyclic_features,
        'numeric':     numeric_features,
        'binary':      binary_features,
        'all':         all_features,
    }