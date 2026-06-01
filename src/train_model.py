import pandas as pd
import numpy as np
import joblib

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score


def train_model(X, y):

    tscv = TimeSeriesSplit(n_splits=5)

    scores = []

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

    for train_idx, val_idx in tscv.split(X):

        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # =========================
        # CATBOOST DATA
        # =========================

        X_train_cat = X_train.copy()
        X_val_cat = X_val.copy()

        for col in cat_cols:

            if col in X_train_cat.columns:

                X_train_cat[col] = (
                    X_train_cat[col]
                    .astype(str)
                )

                X_val_cat[col] = (
                    X_val_cat[col]
                    .astype(str)
                )

        # =========================
        # LIGHTGBM/XGB DATA
        # =========================

        X_train_lgb = X_train.copy()
        X_val_lgb = X_val.copy()

        for col in cat_cols:

            if col in X_train_lgb.columns:

                X_train_lgb[col] = (
                    X_train_lgb[col]
                    .astype('category')
                    .cat.codes
                )

                X_val_lgb[col] = (
                    X_val_lgb[col]
                    .astype('category')
                    .cat.codes
                )

        # =========================
        # MULTI-SEED CATBOOST
        # =========================

        cat_models = []

        seeds = [42, 100, 2025]

        for seed in seeds:

            model = CatBoostRegressor(
                iterations=2500,
                learning_rate=0.02,
                depth=7,
                loss_function='RMSE',
                verbose=False,
                random_seed=seed
            )

            model.fit(
                X_train_cat,
                y_train,
                cat_features=cat_cols,
                eval_set=(X_val_cat, y_val),
                early_stopping_rounds=100
            )

            cat_models.append(model)

        # =========================
        # LIGHTGBM
        # =========================

        lgb_model = LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.02,
            max_depth=10,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        lgb_model.fit(
            X_train_lgb,
            y_train
        )

        # =========================
        # XGBOOST
        # =========================

        xgb_model = XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        )

        xgb_model.fit(
            X_train_lgb,
            y_train,
            verbose=False
        )

        # =========================
        # PREDICTIONS
        # =========================

        cat_preds = np.mean([

            model.predict(X_val_cat)

            for model in cat_models

        ], axis=0)

        lgb_preds = lgb_model.predict(X_val_lgb)

        xgb_preds = xgb_model.predict(X_val_lgb)

        # =========================
        # ENSEMBLE
        # =========================

        preds = (
            0.60 * cat_preds +
            0.25 * lgb_preds +
            0.15 * xgb_preds
        )

        score = r2_score(y_val, preds)

        scores.append(score)

        print("Fold Score:", score)

    print("\nMean Score:", np.mean(scores))

    # =========================
    # FINAL FULL DATASET TRAINING
    # =========================

    X_cat = X.copy()
    X_lgb = X.copy()

    for col in cat_cols:

        if col in X_cat.columns:

            X_cat[col] = (
                X_cat[col]
                .astype(str)
            )

        if col in X_lgb.columns:

            X_lgb[col] = (
                X_lgb[col]
                .astype('category')
                .cat.codes
            )

    # =========================
    # FINAL CATBOOST
    # =========================

    final_cat_models = []

    seeds = [42, 100, 2025]

    for seed in seeds:

        model = CatBoostRegressor(
            iterations=2500,
            learning_rate=0.02,
            depth=7,
            loss_function='RMSE',
            verbose=False,
            random_seed=seed
        )

        model.fit(
            X_cat,
            y,
            cat_features=cat_cols
        )

        final_cat_models.append(model)

    # =========================
    # FINAL LIGHTGBM
    # =========================

    lgb_model.fit(X_lgb, y)

    # =========================
    # FINAL XGBOOST
    # =========================

    xgb_model.fit(X_lgb, y)

    # =========================
    # SAVE MODELS
    # =========================

    joblib.dump(
        final_cat_models,
        "models/cat_model.pkl"
    )

    joblib.dump(
        lgb_model,
        "models/lgb_model.pkl"
    )

    joblib.dump(
        xgb_model,
        "models/xgb_model.pkl"
    )

    # =========================
    # FEATURE IMPORTANCE
    # =========================

    importance = pd.DataFrame({

        'feature': X_lgb.columns,

        'importance': final_cat_models[0]
        .feature_importances_

    })

    importance = importance.sort_values(
        by='importance',
        ascending=False
    )

    print("\nFeature Importance:\n")

    print(importance)

    return final_cat_models, lgb_model, xgb_model