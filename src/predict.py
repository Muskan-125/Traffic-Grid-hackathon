import pandas as pd
import numpy as np


def create_submission(
    cat_models,
    lgb_model,
    xgb_model,
    X_test,
    test
):

    # =========================
    # CATEGORICAL COLUMNS
    # =========================

    cat_cols = [
        'geohash',
        'RoadType',
        'Weather',
        'Landmarks',
        'LargeVehicles',
        'day'
    ]

    # =========================
    # CATBOOST TEST DATA
    # =========================

    X_test_cat = X_test.copy()

    for col in cat_cols:

        if col in X_test_cat.columns:

            X_test_cat[col] = (
                X_test_cat[col]
                .astype(str)
            )

    # =========================
    # LIGHTGBM/XGB TEST DATA
    # =========================

    X_test_lgb = X_test.copy()

    for col in cat_cols:

        if col in X_test_lgb.columns:

            X_test_lgb[col] = (
                X_test_lgb[col]
                .astype('category')
                .cat.codes
            )

    # =========================
    # CATBOOST PREDICTIONS
    # =========================

    cat_preds = np.mean([

        model.predict(X_test_cat)

        for model in cat_models

    ], axis=0)

    # =========================
    # LIGHTGBM PREDICTIONS
    # =========================

    lgb_preds = lgb_model.predict(
        X_test_lgb
    )

    # =========================
    # XGBOOST PREDICTIONS
    # =========================

    xgb_preds = xgb_model.predict(
        X_test_lgb
    )

    # =========================
    # FINAL ENSEMBLE
    # =========================

    predictions = (
        0.60 * cat_preds +
        0.25 * lgb_preds +
        0.15 * xgb_preds
    )

    # =========================
    # REVERSE LOG TRANSFORM
    # =========================

    predictions = np.expm1(
        predictions
    )

    # =========================
    # CREATE SUBMISSION
    # =========================

    submission = pd.DataFrame({

        'Index': test['Index'],
        'demand': predictions

    })

    submission.to_csv(
        "submissions/final_submission.csv",
        index=False
    )

    print(
        "Submission created successfully."
    )