# src/utils_data.py
import pandas as pd
import numpy as np
import holidays


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


def haversine(lon1, lat1, lon2, lat2):
    """Calculate haversine distance between two points in kilometers."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # earth radius in km
    return c * r


def bearing(lon1, lat1, lon2, lat2):
    """Calculate bearing angle between two points."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def is_near_airport(lon, lat, airports=AIRPORTS, threshold=1):
    """Check if coordinates are within threshold km of any airport."""
    for airport_coords in airports.values():
        if haversine(lon, lat, *airport_coords) <= threshold:
            return True
    return False


def _filter_nyc_bounds(df):
    mask = (
        df['pickup_latitude'].between(NYC_BOUNDS['lat_min'], NYC_BOUNDS['lat_max'])
        & df['pickup_longitude'].between(NYC_BOUNDS['lon_min'], NYC_BOUNDS['lon_max'])
        & df['dropoff_latitude'].between(NYC_BOUNDS['lat_min'], NYC_BOUNDS['lat_max'])
        & df['dropoff_longitude'].between(NYC_BOUNDS['lon_min'], NYC_BOUNDS['lon_max'])
    )
    return df.loc[mask].copy()


def _encode_zone(lat_col, lon_col):
    return (lat_col // 0.01).astype(int).astype(str) + "_" + (lon_col // 0.01).astype(int).astype(str)


def apply_feature_engineering(data, train_stats=None, iqr_factor=2.5):
    """
    Apply deterministic feature engineering used by both training and inference.

    Args:
        data: Input dataframe.
        train_stats: Optional dictionary populated during training with
            statistics required for inference.
        iqr_factor: Outlier clipping multiplier applied around the IQR.
    """
    df = data.copy()
    df = _filter_nyc_bounds(df)

    is_training = train_stats is None
    if train_stats is None:
        train_stats = {}

    df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['pickup_date'] = df['pickup_datetime'].dt.floor('D')
    df['is_holiday'] = df['pickup_date'].isin(US_HOLIDAYS).astype(int)
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['rush_hours'] = (
        df['hour'].between(8, 10) | df['hour'].between(17, 21)
    ).astype(int)
    df['night_flag'] = ((df['hour'] >= 21) | (df['hour'] < 5)).astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['weekend_rush'] = df['is_weekend'] * df['rush_hours']
    df['week_hour'] = df['dayofweek'] * 24 + df['hour']

    df['haversine_distance'] = haversine(
        df['pickup_longitude'], df['pickup_latitude'],
        df['dropoff_longitude'], df['dropoff_latitude']
    )
    df['bearing'] = bearing(
        df['pickup_longitude'], df['pickup_latitude'],
        df['dropoff_longitude'], df['dropoff_latitude']
    )
    df['manhattan_distance'] = (
        np.abs(df['dropoff_longitude'] - df['pickup_longitude']) +
        np.abs(df['dropoff_latitude'] - df['pickup_latitude'])
    )

    df['distance_per_passenger'] = df['haversine_distance'] / df['passenger_count'].clip(lower=1)
    df['log_distance'] = np.log1p(df['haversine_distance'])

    duration_hours = df['trip_duration'].clip(lower=1) / 3600
    df['speed_kmph'] = df['haversine_distance'] / duration_hours
    df['speed_kmph'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['speed_kmph'].fillna(df['speed_kmph'].median(), inplace=True)

    # Day-level stats (stored for fallbacks)
    daily_trip_count = df.groupby('pickup_date').size()
    if is_training:
        train_stats['daily_trip_mean'] = daily_trip_count.mean()
        train_stats['daily_trip_std'] = daily_trip_count.std(ddof=0) or 1.0
    else:
        if 'daily_trip_mean' not in train_stats or 'daily_trip_std' not in train_stats:
            raise KeyError("Missing daily trip stats. Refit feature engineering on training data first.")
    mean_daily = train_stats['daily_trip_mean']
    std_daily = train_stats['daily_trip_std'] or 1.0
    z_scores = (daily_trip_count - mean_daily) / std_daily
    anomalous_days = z_scores[abs(z_scores) > 2].index
    df['is_anomaly'] = df['pickup_date'].isin(anomalous_days).astype(int)

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
        train_stats['dow_duration_stats'] = dow_duration_stats
    else:
        daily_duration_stats = train_stats.get('daily_duration_stats', daily_duration_stats)
        dow_duration_stats = train_stats.get('dow_duration_stats', dow_duration_stats)

    df = df.merge(daily_duration_stats, on='pickup_date', how='left')
    df = df.merge(dow_duration_stats, on='dayofweek', how='left')
    df['daily_median_duration'].fillna(df['dow_median_duration'], inplace=True)
    df['daily_p95_duration'].fillna(df['dow_p95_duration'], inplace=True)
    df.drop(columns=['dow_median_duration', 'dow_p95_duration'], inplace=True)

    week_hour_stats = df.groupby('week_hour')['trip_duration'].agg(
        week_hour_trip_ct='size',
        week_hour_median_duration='median'
    ).reset_index()
    week_hour_stats['week_hour_trip_ct_norm'] = (
        week_hour_stats['week_hour_trip_ct'] / week_hour_stats['week_hour_trip_ct'].max()
    )
    if is_training:
        train_stats['week_hour_stats'] = week_hour_stats
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

    # Traffic congestion proxy
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
    df['std_speed'] = df['std_speed'].replace(0, np.nan)
    df['speed_z'] = ((df['speed_kmph'] - df['mean_speed']) / df['std_speed']).fillna(0)
    df['traffic_congestion'] = (df['speed_z'] < -1).astype(int)
    df['congestion_rate'] = df.groupby(['dayofweek', 'hour'])['traffic_congestion'].transform('mean')

    df['pickup_zone'] = _encode_zone(df['pickup_latitude'], df['pickup_longitude'])
    df['dropoff_zone'] = _encode_zone(df['dropoff_latitude'], df['dropoff_longitude'])

    if is_training:
        pickup_congestion = df.groupby(['pickup_zone', 'dayofweek', 'hour'])['traffic_congestion'] \
                              .mean().reset_index(name='pickup_congestion_rate')
        dropoff_congestion = df.groupby(['dropoff_zone', 'dayofweek', 'hour'])['traffic_congestion'] \
                               .mean().reset_index(name='dropoff_congestion_rate')
        train_stats['pickup_congestion_stats'] = pickup_congestion
        train_stats['dropoff_congestion_stats'] = dropoff_congestion
    else:
        pickup_congestion = train_stats.get('pickup_congestion_stats')
        dropoff_congestion = train_stats.get('dropoff_congestion_stats')
        if pickup_congestion is None or dropoff_congestion is None:
            raise KeyError("Missing congestion statistics in train_stats.")

    df = df.merge(
        pickup_congestion,
        on=['pickup_zone', 'dayofweek', 'hour'],
        how='left'
    ).merge(
        dropoff_congestion,
        on=['dropoff_zone', 'dayofweek', 'hour'],
        how='left'
    )
    df['pickup_congestion_rate'].fillna(0, inplace=True)
    df['dropoff_congestion_rate'].fillna(0, inplace=True)

    df['congestion_rate'].fillna(df['congestion_rate'].median(), inplace=True)

    df['is_pickup_at_airport'] = df.apply(
        lambda row: is_near_airport(row['pickup_longitude'], row['pickup_latitude']),
        axis=1
    ).astype(int)
    df['is_dropoff_at_airport'] = df.apply(
        lambda row: is_near_airport(row['dropoff_longitude'], row['dropoff_latitude']),
        axis=1
    ).astype(int)

    cols_for_clipping = ['trip_duration', 'haversine_distance', 'manhattan_distance', 'speed_kmph']
    bounds = train_stats.setdefault('outlier_bounds', {})

    df_clean = df.copy()
    if is_training:
        for col in cols_for_clipping:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - iqr_factor * IQR, Q3 + iqr_factor * IQR
            bounds[col] = (lower, upper)
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    else:
        missing = [col for col in cols_for_clipping if col not in bounds]
        if missing:
            raise KeyError(f"Missing outlier bounds for columns: {missing}")
        for col in cols_for_clipping:
            lower, upper = bounds[col]
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    df = df_clean

    helper_cols = [
        'traffic_congestion', 'speed_z', 'mean_speed', 'std_speed',
        'pickup_zone', 'dropoff_zone'
    ]
    df.drop(columns=[c for c in helper_cols if c in df.columns], inplace=True)

    df.drop(columns=['id'], inplace=True, errors='ignore')
    df.drop(columns=[
        'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'pickup_date'
    ], inplace=True)

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    df['log_trip_duration'] = np.log1p(df['trip_duration'])

    return df, train_stats


def get_feature_lists():
    """Return categorized feature lists for modeling."""
    categorical_features = ['passenger_count', 'vendor_id']
    cyclic_features = ['hour', 'dayofweek', 'month', 'day']
    numeric_features = [
        'haversine_distance', 'manhattan_distance', 'bearing',
        'distance_per_passenger', 'log_distance', 'week_hour',
        'week_hour_trip_ct_norm', 'week_hour_median_duration',
        'daily_median_duration', 'daily_p95_duration',
        'pickup_congestion_rate', 'dropoff_congestion_rate',
        'congestion_rate', 'speed_kmph'
    ]
    binary_features = [
        'store_and_fwd_flag', 'is_holiday', 'is_anomaly',
        'is_pickup_at_airport', 'is_dropoff_at_airport',
        'night_flag', 'is_weekend', 'rush_hours', 'weekend_rush'
    ]

    all_features = categorical_features + numeric_features + cyclic_features + binary_features

    return {
        'categorical': categorical_features,
        'cyclic': cyclic_features,
        'numeric': numeric_features,
        'binary': binary_features,
        'all': all_features
    }