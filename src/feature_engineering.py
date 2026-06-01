import pandas as pd
import numpy as np
import pygeohash as pgh


def create_features(df):

    df['timestamp'] = pd.to_datetime(
        df['timestamp'],
        errors='coerce',
        format='mixed'
    )

    # =========================
    # BASIC FEATURES
    # =========================

    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['weekofyear'] = (
        df['timestamp'].dt.isocalendar().week.astype(int)
    )

    # =========================
    # WEEKEND
    # =========================

    df['is_weekend'] = (
        df['dayofweek'].isin([5, 6]).astype(int)
    )

    # =========================
    # PEAK HOURS
    # =========================

    df['peak_hour'] = (
        df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    )

    # =========================
    # NIGHT
    # =========================

    df['is_night'] = (
        df['hour'].isin([0, 1, 2, 3, 4, 23]).astype(int)
    )

    # =========================
    # GEOHASH
    # =========================

    df['lat'] = df['geohash'].apply(
        lambda x: pgh.decode(x)[0]
    )

    df['lon'] = df['geohash'].apply(
        lambda x: pgh.decode(x)[1]
    )

    # =========================
    # INTERACTIONS
    # =========================

    df['temp_lane'] = (
        df['Temperature'] *
        df['NumberofLanes']
    )

    df['temp_squared'] = (
        df['Temperature'] ** 2
    )

    df['lane_squared'] = (
        df['NumberofLanes'] ** 2
    )

    # =========================
    # CYCLICAL ENCODING
    # =========================

    df['hour_sin'] = np.sin(
        2 * np.pi * df['hour'] / 24
    )

    df['hour_cos'] = np.cos(
        2 * np.pi * df['hour'] / 24
    )

    df.fillna(0, inplace=True)

    return df


def add_aggregation_features(train, test):

    geo_mean = train.groupby(
        'geohash'
    )['demand'].mean()

    geo_std = train.groupby(
        'geohash'
    )['demand'].std()

    geo_max = train.groupby(
        'geohash'
    )['demand'].max()

    geo_min = train.groupby(
        'geohash'
    )['demand'].min()

    geo_median = train.groupby(
        'geohash'
    )['demand'].median()

    train['geo_mean_demand'] = (
        train['geohash'].map(geo_mean)
    )

    test['geo_mean_demand'] = (
        test['geohash'].map(geo_mean)
    )

    train['geo_std_demand'] = (
        train['geohash'].map(geo_std)
    )

    test['geo_std_demand'] = (
        test['geohash'].map(geo_std)
    )

    train['geo_max_demand'] = (
        train['geohash'].map(geo_max)
    )

    test['geo_max_demand'] = (
        test['geohash'].map(geo_max)
    )

    train['geo_min_demand'] = (
        train['geohash'].map(geo_min)
    )

    test['geo_min_demand'] = (
        test['geohash'].map(geo_min)
    )

    train['geo_median_demand'] = (
        train['geohash'].map(geo_median)
    )

    test['geo_median_demand'] = (
        test['geohash'].map(geo_median)
    )

    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    return train, test