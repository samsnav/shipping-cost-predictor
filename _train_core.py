"""
Shared training logic for mode-specific models (PARCEL, Freight).
Called by train_parcel.py and train_freight.py — not run directly.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from data_prep import (
    load_and_clean, build_lookup_tables, enrich_features,
    build_encoders, apply_encoders, save_artifacts,
    EXCEL_PATH, CAT_COLS, NUM_COLS,
)
from model import ShippingCostNN, get_embedding_sizes

EPOCHS       = 40
BATCH_SIZE   = 2048
LR           = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_SIZES = (256, 128, 64)
DROPOUT      = 0.3
VAL_SPLIT    = 0.1
RANDOM_STATE = 42
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

LGBM_PARAMS = {
    'objective':         'huber',
    'metric':            'mae',
    'learning_rate':     0.05,
    'num_leaves':        127,
    'feature_fraction':  0.8,
    'bagging_fraction':  0.8,
    'bagging_freq':      1,
    'min_child_samples': 20,
    'verbose':           -1,
    'n_jobs':            -1,
    'seed':              RANDOM_STATE,
}


def run_training(carrier_modes: list, artifacts_dir: str, label: str):
    """Train NN + LightGBM for the given carrier modes and save artifacts."""

    print(f'\n{"="*55}')
    print(f'  Training {label} model  ({", ".join(carrier_modes)})')
    print(f'  Artifacts → {artifacts_dir}')
    print(f'{"="*55}')

    # ── Data ─────────────────────────────────────────────────────────────────
    df = load_and_clean(EXCEL_PATH, carrier_modes=carrier_modes)
    print(f'Rows after cleaning: {len(df):,}')

    lookups, df = build_lookup_tables(df)
    df = enrich_features(df, lookups)
    df = df.dropna(subset=['log_cost'])
    print(f'Rows for training:   {len(df):,}')

    encoders, scaler = build_encoders(df)
    X_cat, _, y = apply_encoders(df, encoders, scaler)
    X_num_raw    = df[NUM_COLS].values.astype(np.float32)
    X_num_scaled = scaler.transform(X_num_raw).astype(np.float32)

    idx_train, idx_val = train_test_split(
        np.arange(len(y)), test_size=VAL_SPLIT, random_state=RANDOM_STATE
    )

    # ── Neural Network ────────────────────────────────────────────────────────
    print(f'\n--- Neural Network ---')
    emb_sizes = get_embedding_sizes(encoders, CAT_COLS)
    model = ShippingCostNN(emb_sizes, n_numeric=X_num_scaled.shape[1],
                           hidden_sizes=HIDDEN_SIZES, dropout=DROPOUT).to(DEVICE)

    def make_loader(idx, shuffle):
        ds = TensorDataset(
            torch.tensor(X_cat[idx], dtype=torch.long),
            torch.tensor(X_num_scaled[idx], dtype=torch.float32),
            torch.tensor(y[idx], dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(idx_train, shuffle=True)
    val_loader   = make_loader(idx_val,   shuffle=False)

    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)
    criterion  = nn.HuberLoss()
    best_loss  = float('inf')

    print(f'{"Epoch":>5}  {"Train":>8}  {"Val":>8}  {"MAE":>9}  {"R²":>7}')
    print('-' * 45)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for xc, xn, yt in train_loader:
            xc, xn, yt = xc.to(DEVICE), xn.to(DEVICE), yt.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xc, xn), yt)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(yt)
        t_loss /= len(idx_train)

        model.eval()
        v_loss, preds, trues = 0.0, [], []
        with torch.no_grad():
            for xc, xn, yt in val_loader:
                xc, xn, yt = xc.to(DEVICE), xn.to(DEVICE), yt.to(DEVICE)
                p = model(xc, xn)
                v_loss += criterion(p, yt).item() * len(yt)
                preds.append(p.cpu().numpy())
                trues.append(yt.cpu().numpy())
        v_loss /= len(idx_val)
        mae = mean_absolute_error(np.expm1(np.concatenate(trues)),
                                  np.expm1(np.concatenate(preds)))
        r2  = r2_score(np.expm1(np.concatenate(trues)),
                       np.expm1(np.concatenate(preds)))
        scheduler.step(v_loss)

        if v_loss < best_loss:
            best_loss = v_loss
            os.makedirs(artifacts_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(artifacts_dir, 'best_model.pt'))

        print(f'{epoch:>5}  {t_loss:>8.4f}  {v_loss:>8.4f}  ${mae:>8.2f}  {r2:>7.4f}')

    # Reload best weights for evaluation
    model.load_state_dict(torch.load(os.path.join(artifacts_dir, 'best_model.pt'),
                                     map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        nn_log = model(
            torch.tensor(X_cat[idx_val], dtype=torch.long).to(DEVICE),
            torch.tensor(X_num_scaled[idx_val], dtype=torch.float32).to(DEVICE),
        ).cpu().numpy()

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print(f'\n--- LightGBM ---')
    X_lgbm  = np.hstack([X_cat, X_num_raw])
    train_d = lgb.Dataset(X_lgbm[idx_train], label=y[idx_train],
                          feature_name=CAT_COLS + NUM_COLS,
                          categorical_feature=list(range(len(CAT_COLS))))
    val_d   = lgb.Dataset(X_lgbm[idx_val], label=y[idx_val],
                          feature_name=CAT_COLS + NUM_COLS,
                          categorical_feature=list(range(len(CAT_COLS))),
                          reference=train_d)

    gbm = lgb.train(
        LGBM_PARAMS, train_d, num_boost_round=2000, valid_sets=[val_d],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    gbm.save_model(os.path.join(artifacts_dir, 'lgbm_model.txt'))
    lgbm_log = gbm.predict(X_lgbm[idx_val])

    # ── Final comparison ──────────────────────────────────────────────────────
    true_cost = np.expm1(y[idx_val])
    ens_log   = (nn_log + lgbm_log) / 2

    print(f'\n{"Model":<12} {"MAE":>10}  {"R²":>8}')
    print('-' * 34)
    for name, preds in [('Neural Net', np.expm1(nn_log)),
                         ('LightGBM',  np.expm1(lgbm_log)),
                         ('Ensemble',  np.expm1(ens_log))]:
        print(f'{name:<12} ${mean_absolute_error(true_cost, preds):>9.2f}  '
              f'{r2_score(true_cost, preds):>8.4f}')

    # ── Save shared artifacts ─────────────────────────────────────────────────
    model_config = {
        'embedding_sizes': emb_sizes,
        'n_numeric':       X_num_scaled.shape[1],
        'hidden_sizes':    HIDDEN_SIZES,
        'dropout':         DROPOUT,
        'carrier_modes':   carrier_modes,
    }
    save_artifacts(lookups, encoders, scaler, model_config, artifacts_dir)
    print(f'\nAll artifacts saved to: {artifacts_dir}')
