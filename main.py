import pandas as pd
import numpy as np

from src.data_loader import load_data

from src.preprocessing import (
    handle_missing,
    target_encoding
)

from src.feature_engineering import (
    create_features,
    add_aggregation_features
)

from src.train_model import train_model
from src.predict import create_submission

from src.config import TARGET, DROP_COLS


# =========================
# LOAD DATA
# =========================
train, test, sample = load_data()


# =========================
# TIMESTAMP PROCESSING
# =========================
train['timestamp'] = pd.to_datetime(
    train['timestamp'],
    errors='coerce',
    format='mixed'
)

test['timestamp'] = pd.to_datetime(
    test['timestamp'],
    errors='coerce',
    format='mixed'
)

train = train.sort_values('timestamp')


# =========================
# HANDLE MISSING VALUES
# =========================
train, test = handle_missing(train, test)


# =========================
# FEATURE ENGINEERING
# =========================
train = create_features(train)
test = create_features(test)


# =========================
# AGGREGATION FEATURES
# =========================
train, test = add_aggregation_features(
    train,
    test
)


# =========================
# TARGET ENCODING
# =========================
train, test = target_encoding(
    train,
    test
)


# =========================
# KEEP GEOHASH AS STRING
# =========================
train['geohash'] = train['geohash'].astype(str)
test['geohash'] = test['geohash'].astype(str)

categorical_cols = [
    'geohash',
    'RoadType',
    'Weather',
    'Landmarks',
    'LargeVehicles',
    'day'
]

for col in categorical_cols:

    if col in train.columns:
        train[col] = train[col].astype(str)

    if col in test.columns:
        test[col] = test[col].astype(str)


# =========================
# PREPARE TRAIN DATA
# =========================
X = train.drop(
    columns=[
        col for col in DROP_COLS
        if col in train.columns
    ]
)

y = np.log1p(train[TARGET])


# =========================
# PREPARE TEST DATA
# =========================
X_test = test.drop(
    columns=[
        col for col in DROP_COLS
        if col in test.columns
    ]
)

X_test = X_test[X.columns]


# =========================
# TRAIN MODELS
# =========================
cat_models, lgb_model, xgb_model = train_model(
    X,
    y
)


# =========================
# CREATE SUBMISSION
# =========================
create_submission(
    cat_models,
    lgb_model,
    xgb_model,
    X_test,
    test
)