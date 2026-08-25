"""
Segment-level error analysis for the trained PARCEL and Freight models.

Reproduces the exact validation split each model was trained on (same
random_state as _train_core.py), scores it with the already-saved artifacts,
and reports MAE/MAPE broken out by speed tier, weight, distance, and origin
so we know where to focus improvement effort.

Usage:
    python error_analysis.py
"""

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from data_prep import (
    load_and_clean, build_lookup_tables, aggregate_to_tickets, enrich_features, apply_encoders,
    load_artifacts, EXCEL_PATH, PARCEL_ARTIFACTS_DIR, FREIGHT_ARTIFACTS_DIR,
    NUM_COLS,
)
from model import ShippingCostNN

VAL_SPLIT    = 0.1
RANDOM_STATE = 42
DEVICE       = 'cpu'

MODE_GROUPS = [
    ('PARCEL',  ['PARCEL'],       PARCEL_ARTIFACTS_DIR),
    ('Freight', ['LTL', 'FTL'],   FREIGHT_ARTIFACTS_DIR),
]


def _load_models(artifacts_dir, model_config):
    nn = ShippingCostNN(
        embedding_sizes=model_config['embedding_sizes'],
        n_numeric=model_config['n_numeric'],
        hidden_sizes=model_config.get('hidden_sizes', (256, 128, 64)),
        dropout=model_config.get('dropout', 0.3),
    )
    nn.load_state_dict(torch.load(f'{artifacts_dir}/best_model.pt', map_location=DEVICE))
    nn.eval()
    gbm = lgb.Booster(model_file=f'{artifacts_dir}/lgbm_model.txt')
    return nn, gbm


def _report(val_df, col, title):
    g = val_df.groupby(col, observed=True).agg(
        n=('abs_err', 'size'),
        mae=('abs_err', 'mean'),
        bias=('signed_err', 'mean'),
        mape=('pct_err', 'mean'),
        avg_cost=('true_cost', 'mean'),
    ).sort_values('n', ascending=False)
    print(f'\n-- {title} --')
    print(g.round(2).to_string())


def analyze(label, carrier_modes, artifacts_dir):
    print(f'\n{"="*70}\n  {label}\n{"="*70}')

    df = load_and_clean(EXCEL_PATH, carrier_modes=carrier_modes)
    lookups, df_lines = build_lookup_tables(df)
    df = aggregate_to_tickets(df_lines)
    df = enrich_features(df, lookups)
    df = df.dropna(subset=['log_cost']).reset_index(drop=True)

    _, encoders, scaler, model_config = load_artifacts(artifacts_dir)
    X_cat, X_num, y = apply_encoders(df, encoders, scaler)

    idx_train, idx_val = train_test_split(
        np.arange(len(y)), test_size=VAL_SPLIT, random_state=RANDOM_STATE
    )

    nn, gbm = _load_models(artifacts_dir, model_config)

    with torch.no_grad():
        nn_log = nn(
            torch.tensor(X_cat[idx_val], dtype=torch.long),
            torch.tensor(X_num[idx_val], dtype=torch.float32),
        ).numpy()

    X_num_raw = df[NUM_COLS].values.astype(np.float32)
    X_lgbm    = np.hstack([X_cat, X_num_raw])
    lgbm_log  = gbm.predict(X_lgbm[idx_val])

    ens_log   = (nn_log + lgbm_log) / 2
    pred_cost = np.expm1(ens_log)
    true_cost = np.expm1(y[idx_val])
    signed_err = pred_cost - true_cost  # negative = underprediction
    abs_err    = np.abs(signed_err)
    pct_err    = abs_err / true_cost * 100

    val_df = df.iloc[idx_val].copy()
    val_df['true_cost']  = true_cost
    val_df['pred_cost']  = pred_cost
    val_df['signed_err'] = signed_err
    val_df['abs_err']    = abs_err
    val_df['pct_err']    = pct_err

    print(f'Overall: n={len(val_df):,}  MAE=${abs_err.mean():.2f}  bias=${signed_err.mean():+.2f}  '
          f'MAPE={pct_err.mean():.1f}%  R²={r2_score(true_cost, pred_cost):.4f}')

    if len(carrier_modes) > 1:
        _report(val_df, 'Carrier Mode', 'By carrier mode')

    if 'speed_tier' in val_df.columns:
        _report(val_df, 'speed_tier', 'By speed tier')

    val_df['weight_bucket'] = pd.cut(
        val_df['billable_weight'],
        bins=[0, 10, 50, 150, 500, 2000, np.inf],
        labels=['0-10', '10-50', '50-150', '150-500', '500-2000', '2000+'],
        include_lowest=True,
    )
    _report(val_df, 'weight_bucket', 'By billable weight (lb)')

    val_df['distance_bucket'] = pd.cut(
        val_df['estimated_miles'],
        bins=[0, 100, 300, 750, 1500, np.inf],
        labels=['0-100', '100-300', '300-750', '750-1500', '1500+'],
        include_lowest=True,
    )
    _report(val_df, 'distance_bucket', 'By distance (mi)')

    _report(val_df, 'ship_from_location_name', 'By origin location')

    top_decile_cutoff = val_df['true_cost'].quantile(0.9)
    tail = val_df[val_df['true_cost'] >= top_decile_cutoff]
    print(f'\n-- Top 10% most expensive shipments (true_cost >= ${top_decile_cutoff:.2f}) --')
    print(f'n={len(tail):,}  MAE=${tail["abs_err"].mean():.2f}  bias=${tail["signed_err"].mean():+.2f}  '
          f'MAPE={tail["pct_err"].mean():.1f}%')

    print('\n-- 10 worst absolute errors --')
    worst = val_df.nlargest(10, 'abs_err')[
        ['ship_from_location_name', 'to_zip3', 'item_id', 'true_cost', 'pred_cost', 'signed_err']
    ]
    print(worst.round(2).to_string(index=False))


if __name__ == '__main__':
    for label, carrier_modes, artifacts_dir in MODE_GROUPS:
        analyze(label, carrier_modes, artifacts_dir)
