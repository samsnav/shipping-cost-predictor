"""
Predict shipping cost from the four user-supplied inputs.

Usage (script):
    python predict.py

Usage (import):
    from predict import predict_cost
    cost = predict_cost('Keystone Technologies PA', '10001', 'KT-LED17PLL-22GC-840-D /G2', 100)
    print(f'${cost:.2f}')
"""

import os
import numpy as np
import torch
import lightgbm as lgb

from data_prep import (
    load_artifacts, classify_residential, ARTIFACTS_DIR, CAT_COLS, NUM_COLS
)
from model import ShippingCostNN

DEVICE = 'cpu'
_cache: dict = {}


def _load(model_type: str = 'nn'):
    cache_key = f'loaded_{model_type}'
    if cache_key in _cache:
        return _cache

    lookups, encoders, scaler, model_config = load_artifacts()

    def _load_nn():
        m = ShippingCostNN(
            embedding_sizes=model_config['embedding_sizes'],
            n_numeric=model_config['n_numeric'],
            hidden_sizes=model_config.get('hidden_sizes', (256, 128, 64)),
            dropout=model_config.get('dropout', 0.3),
        )
        m.load_state_dict(
            torch.load(os.path.join(ARTIFACTS_DIR, 'best_model.pt'), map_location=DEVICE)
        )
        m.eval()
        return m

    def _load_lgbm():
        return lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, 'lgbm_model.txt'))

    if model_type == 'lgbm':
        model = _load_lgbm()
    elif model_type == 'ensemble':
        model = {'nn': _load_nn(), 'lgbm': _load_lgbm()}
    else:
        model = _load_nn()

    _cache.update({
        'lookups': lookups,
        'encoders': encoders,
        'scaler': scaler,
        'model': model,
        'model_type': model_type,
        cache_key: True,
    })
    return _cache


def _encode_val(encoders, col, val):
    val = str(val)
    classes = encoders[col].classes_
    if val in classes:
        return int(encoders[col].transform([val])[0])
    return int(encoders[col].transform(['__unknown__'])[0])


def predict_cost(
    ship_from_location_name: str,
    ship_to_zip: str,
    item_id: str,
    qty: int | float,
    carrier_mode: str = 'PARCEL',
    carrier_name: str = 'UPS',
    model_type: str = 'nn',
) -> float:
    """
    Predict the shipping cost for a single shipment.

    Parameters
    ----------
    ship_from_location_name : e.g. 'Keystone Technologies PA'
    ship_to_zip             : destination zip code, e.g. '10001'
    item_id                 : SKU / item identifier
    qty                     : number of units
    carrier_mode            : e.g. 'PARCEL', 'LTL', 'VOLUME'  (default: 'PARCEL')
    carrier_name            : e.g. 'UPS', 'FEDEX', 'ESTES'    (default: 'UPS')
    model_type              : 'nn', 'lgbm', or 'ensemble'      (default: 'nn')

    Returns
    -------
    float : predicted cost in dollars
    """
    arts = _load(model_type)
    lookups  = arts['lookups']
    encoders = arts['encoders']
    scaler   = arts['scaler']
    model    = arts['model']

    item_lookup   = lookups['item_lookup']
    location_zip3 = lookups['location_zip3']
    zip_miles     = lookups['zip_miles']

    # ── Enrich item-level features ────────────────────────────────────────────
    item_id_str = str(item_id)
    if item_id_str in item_lookup.index:
        row = item_lookup.loc[item_id_str]
        cbft_per_unit   = float(row['avg_cbft_per_unit'])
        weight_per_unit = float(row['avg_weight_per_unit'])
        item_class1     = str(row['Item_Class1'])
        item_class2     = str(row['Item_Class2'])
        nfmc_code       = str(row['NFMC_code'])
    else:
        cbft_per_unit   = lookups['global_cbft_median']
        weight_per_unit = lookups['global_weight_median']
        item_class1     = '__unknown__'
        item_class2     = '__unknown__'
        nfmc_code       = '__unknown__'

    estimated_cbft   = max(qty * cbft_per_unit, 0.0)
    estimated_weight = max(qty * weight_per_unit, 0.0)

    # ── Estimate distance from zip-prefix pair ────────────────────────────────
    loc_str  = str(ship_from_location_name)
    from_zip3 = str(location_zip3.get(loc_str, '194'))[:3]
    to_zip3   = str(ship_to_zip).strip()[:3]

    try:
        estimated_miles = float(zip_miles.loc[(from_zip3, to_zip3)])
    except KeyError:
        estimated_miles = lookups['default_miles']

    # ── Build feature row ─────────────────────────────────────────────────────
    density = estimated_weight / estimated_cbft if estimated_cbft > 0 else 0.0

    # Residential flag — automatic from ship_to_zip, not a user input
    res_lookup = lookups.get('residential_lookup', {})
    is_residential = res_lookup.get(str(ship_to_zip).strip(), classify_residential(ship_to_zip))

    cat_vals = {
        'ship_from_location_name': loc_str,
        'Carrier Mode':            str(carrier_mode).strip(),
        'Carrier Name':            str(carrier_name).strip(),
        'to_zip3':                 to_zip3,
        'item_id':                 item_id_str,
        'Item_Class1':             item_class1,
        'Item_Class2':             item_class2,
        'NFMC_code':               nfmc_code,
    }
    num_vals = {
        'log_qty':        np.log1p(qty),
        'log_cbft':       np.log1p(estimated_cbft),
        'log_weight':     np.log1p(estimated_weight),
        'log_density':    np.log1p(density),
        'log_miles':      np.log1p(estimated_miles),
        'is_residential': float(is_residential),
    }

    x_cat = np.array(
        [[_encode_val(encoders, col, cat_vals[col]) for col in CAT_COLS]],
        dtype=np.int64,
    )

    # ── Forward pass ─────────────────────────────────────────────────────────
    x_num_raw    = np.array([[num_vals[col] for col in NUM_COLS]], dtype=np.float32)
    x_num_scaled = scaler.transform(x_num_raw).astype(np.float32)
    X_lgbm       = np.hstack([x_cat, x_num_raw])

    def _nn_log_pred(m):
        with torch.no_grad():
            return m(
                torch.tensor(x_cat, dtype=torch.long),
                torch.tensor(x_num_scaled, dtype=torch.float32),
            ).item()

    def _lgbm_log_pred(m):
        return float(m.predict(X_lgbm)[0])

    if model_type == 'ensemble':
        log_pred = (_nn_log_pred(model['nn']) + _lgbm_log_pred(model['lgbm'])) / 2
    elif model_type == 'lgbm':
        log_pred = _lgbm_log_pred(model)
    else:
        log_pred = _nn_log_pred(model)

    return float(np.expm1(log_pred))


def predict_batch(records: list) -> list:
    """
    Predict costs for multiple shipments.

    Each record is a dict with keys:
        ship_from_location_name, ship_to_zip, item_id, qty, carrier_mode (optional)

    Returns list of predicted costs (dollars), in the same order.
    """
    return [predict_cost(**r) for r in records]


if __name__ == '__main__':
    examples = [
        {
            'ship_from_location_name': 'Keystone Technologies PA',
            'ship_to_zip': '10001',
            'item_id': 'KT-LED17PLL-22GC-840-D /G2',
            'qty': 100,
            'carrier_mode': 'PARCEL',
        },
        {
            'ship_from_location_name': 'Keystone Technologies New KC',
            'ship_to_zip': '90210',
            'item_id': 'KT-HBLED90-1.5F-850-VDIM-P /G2',
            'qty': 50,
            'carrier_mode': 'LTL',
        },
        {
            'ship_from_location_name': 'Keystone Technologies PHX',
            'ship_to_zip': '30301',
            'item_id': 'KT-SOCKET-T8-U-S-2-W',
            'qty': 500,
            'carrier_mode': 'PARCEL',
        },
    ]

    print(f'{"Ship From":<35} {"Mode":>8}  {"Ship To":>8}  {"Item":<35} {"Qty":>5}  {"NN":>10}  {"LightGBM":>10}  {"Ensemble":>10}')
    print('-' * 130)
    for r in examples:
        cost_nn  = predict_cost(**r, model_type='nn')
        cost_lgb = predict_cost(**r, model_type='lgbm')
        cost_ens = predict_cost(**r, model_type='ensemble')
        print(
            f'{r["ship_from_location_name"]:<35} {r["carrier_mode"]:>8}  {r["ship_to_zip"]:>8}  '
            f'{r["item_id"]:<35} {r["qty"]:>5}  ${cost_nn:>9.2f}  ${cost_lgb:>9.2f}  ${cost_ens:>9.2f}'
        )
