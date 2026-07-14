"""
Train a LightGBM model for shipping cost prediction.
Uses the same data pipeline as train.py for a direct apples-to-apples comparison.

Usage:
    python train_lgbm.py

Saves model to artifacts/lgbm_model.txt
Also saves lookups/encoders so predict.py can use either model.
"""

import os
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from data_prep import (
    load_and_clean, build_lookup_tables, enrich_features,
    build_encoders, apply_encoders, save_artifacts,
    EXCEL_PATH, ARTIFACTS_DIR, CAT_COLS, NUM_COLS,
)

RANDOM_STATE = 42
VAL_SPLIT    = 0.1

PARAMS = {
    'objective':        'huber',
    'metric':           'mae',
    'learning_rate':    0.05,
    'num_leaves':       127,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     1,
    'min_child_samples': 20,
    'verbose':          -1,
    'n_jobs':           -1,
    'seed':             RANDOM_STATE,
}


def main():
    # ── Data (identical pipeline to train.py) ────────────────────────────────
    df = load_and_clean(EXCEL_PATH)
    print(f'Rows after cleaning: {len(df):,}')

    print('Building lookup tables ...')
    lookups, df = build_lookup_tables(df)

    print('Enriching features ...')
    df = enrich_features(df, lookups)
    df = df.dropna(subset=['log_cost'])
    print(f'Rows for training: {len(df):,}')

    print('Fitting encoders ...')
    encoders, scaler = build_encoders(df)
    X_cat, _, y = apply_encoders(df, encoders, scaler)

    # LightGBM is tree-based — no scaling needed for numerics
    X_num = df[NUM_COLS].values.astype(np.float32)
    X = np.hstack([X_cat, X_num])

    feature_names = CAT_COLS + NUM_COLS
    cat_indices   = list(range(len(CAT_COLS)))

    idx_train, idx_val = train_test_split(
        np.arange(len(y)), test_size=VAL_SPLIT, random_state=RANDOM_STATE
    )

    train_set = lgb.Dataset(
        X[idx_train], label=y[idx_train],
        feature_name=feature_names,
        categorical_feature=cat_indices,
    )
    val_set = lgb.Dataset(
        X[idx_val], label=y[idx_val],
        feature_name=feature_names,
        categorical_feature=cat_indices,
        reference=train_set,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print('\nTraining LightGBM (up to 2000 rounds, early stop at 50) ...')
    model = lgb.train(
        PARAMS,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    preds_log  = model.predict(X[idx_val])
    preds_cost = np.expm1(preds_log)
    true_cost  = np.expm1(y[idx_val])

    mae = mean_absolute_error(true_cost, preds_cost)
    r2  = r2_score(true_cost, preds_cost)

    print(f'\n{"="*40}')
    print(f'  LightGBM — best round: {model.best_iteration}')
    print(f'  Val MAE : ${mae:.2f}')
    print(f'  Val R²  : {r2:.4f}')
    print(f'{"="*40}')
    print('  (Neural net was MAE ~$14.71, R² 0.8127)')

    # ── Feature importance ───────────────────────────────────────────────────
    importances = model.feature_importance(importance_type='gain')
    sorted_idx  = np.argsort(importances)[::-1]
    print('\nTop feature importances (by gain):')
    for rank, i in enumerate(sorted_idx[:10], 1):
        bar = '█' * int(importances[i] / importances[sorted_idx[0]] * 30)
        print(f'  {rank:>2}. {feature_names[i]:<20} {bar}')

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACTS_DIR, 'lgbm_model.txt')
    model.save_model(model_path)

    # Save shared artifacts so predict.py can use this model too
    model_config = {'model_type': 'lgbm'}
    save_artifacts(lookups, encoders, scaler, model_config)

    print(f'\nModel saved to: {model_path}')
    print('Run predict.py with model_type="lgbm" to use this model.')


if __name__ == '__main__':
    main()
