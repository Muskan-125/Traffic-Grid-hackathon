import pandas as pd
import numpy as np

from sklearn.model_selection import KFold


def handle_missing(train, test):

    num_cols = [
        'Temperature',
        'NumberofLanes'
    ]

    for col in num_cols:

        train[col] = train[col].fillna(
            train[col].median()
        )

        test[col] = test[col].fillna(
            test[col].median()
        )

    cat_cols = [
        'RoadType',
        'Weather',
        'Landmarks',
        'LargeVehicles',
        'day'
    ]

    for col in cat_cols:

        train[col] = train[col].fillna(
            train[col].mode()[0]
        )

        test[col] = test[col].fillna(
            test[col].mode()[0]
        )

    return train, test


def target_encoding(train, test):

    cols = [
        'Weather',
        'RoadType'
    ]

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for col in cols:

        train[f'{col}_target_enc'] = np.zeros(
            len(train),
            dtype=float
        )

        for train_idx, val_idx in kf.split(train):

            X_train = train.iloc[train_idx]
            X_val = train.iloc[val_idx]

            means = X_train.groupby(col)[
                'demand'
            ].mean()

            train.loc[
                train.index[val_idx],
                f'{col}_target_enc'
            ] = X_val[col].map(means)

        overall_mean = train['demand'].mean()

        train[f'{col}_target_enc'] = train[
            f'{col}_target_enc'
        ].fillna(overall_mean)

        means = train.groupby(col)[
            'demand'
        ].mean()

        test[f'{col}_target_enc'] = (
            test[col].map(means)
        )

        test[f'{col}_target_enc'] = test[
            f'{col}_target_enc'
        ].fillna(overall_mean)

    return train, test