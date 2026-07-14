"""
Train the shipping cost neural network.

Usage:
    python train.py

Saves trained model and all inference artifacts to ./artifacts/
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from data_prep import (
    load_and_clean, build_lookup_tables, enrich_features,
    build_encoders, apply_encoders, save_artifacts,
    EXCEL_PATH, ARTIFACTS_DIR, CAT_COLS,
)
from model import ShippingCostNN, get_embedding_sizes

# ── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS = 40
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_SIZES = (256, 128, 64)
DROPOUT = 0.3
VAL_SPLIT = 0.1
RANDOM_STATE = 42
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_loader(X_cat, X_num, y, idx, shuffle):
    ds = TensorDataset(
        torch.tensor(X_cat[idx], dtype=torch.long),
        torch.tensor(X_num[idx], dtype=torch.float32),
        torch.tensor(y[idx], dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def evaluate(model, loader):
    model.eval()
    total_loss, preds_list, true_list = 0.0, [], []
    criterion = nn.HuberLoss()
    with torch.no_grad():
        for xc, xn, yt in loader:
            xc, xn, yt = xc.to(DEVICE), xn.to(DEVICE), yt.to(DEVICE)
            pred = model(xc, xn)
            total_loss += criterion(pred, yt).item() * len(yt)
            preds_list.append(pred.cpu().numpy())
            true_list.append(yt.cpu().numpy())
    n = sum(len(p) for p in preds_list)
    preds_cost = np.expm1(np.concatenate(preds_list))
    true_cost = np.expm1(np.concatenate(true_list))
    return total_loss / n, mean_absolute_error(true_cost, preds_cost), r2_score(true_cost, preds_cost)


def main():
    print(f'Device: {DEVICE}')

    # ── Data ──────────────────────────────────────────────────────────────────
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
    X_cat, X_num, y = apply_encoders(df, encoders, scaler)

    idx_train, idx_val = train_test_split(
        np.arange(len(y)), test_size=VAL_SPLIT, random_state=RANDOM_STATE
    )
    train_loader = make_loader(X_cat, X_num, y, idx_train, shuffle=True)
    val_loader   = make_loader(X_cat, X_num, y, idx_val,   shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    emb_sizes = get_embedding_sizes(encoders, CAT_COLS)
    print('Embedding sizes:', list(zip(CAT_COLS, emb_sizes)))

    model = ShippingCostNN(
        embedding_sizes=emb_sizes,
        n_numeric=X_num.shape[1],
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {total_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=4, factor=0.5
    )
    criterion = nn.HuberLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    best_epoch = 0

    print(f'\n{"Epoch":>5}  {"Train Loss":>10}  {"Val Loss":>10}  {"Val MAE":>10}  {"Val R²":>8}')
    print('-' * 55)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xc, xn, yt in train_loader:
            xc, xn, yt = xc.to(DEVICE), xn.to(DEVICE), yt.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xc, xn), yt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yt)
        train_loss /= len(idx_train)

        val_loss, val_mae, val_r2 = evaluate(model, val_loader)
        scheduler.step(val_loss)

        print(f'{epoch:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}  ${val_mae:>9.2f}  {val_r2:>8.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ARTIFACTS_DIR, 'best_model.pt'))

    print(f'\nBest model at epoch {best_epoch} (val loss {best_val_loss:.4f})')

    # ── Save artifacts ────────────────────────────────────────────────────────
    model_config = {
        'embedding_sizes': emb_sizes,
        'n_numeric': X_num.shape[1],
        'hidden_sizes': HIDDEN_SIZES,
        'dropout': DROPOUT,
    }
    save_artifacts(lookups, encoders, scaler, model_config)
    print(f'Artifacts saved to: {ARTIFACTS_DIR}')


if __name__ == '__main__':
    main()
