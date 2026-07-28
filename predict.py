"""
Predict shipping cost from the four user-supplied inputs.

Estimates cost across every realistic Mode x Speed Tier combination instead of
requiring the caller to pick a carrier mode up front.

Usage (script):
    python predict.py

Usage (import):
    from predict import predict_options
    options = predict_options('Keystone Technologies PA', '10001', 'KT-LED17PLL-22GC-840-D /G2', 100)
    # {'PARCEL': {'Ground': 12.34, '2 Day': 18.50, 'Next Day': 32.10, '3 Day': 15.75},
    #  'LTL':    {'Ground': 210.40}}
"""

import os
import datetime
import numpy as np
import torch
import lightgbm as lgb

from data_prep import (
    load_artifacts, classify_residential,
    PARCEL_ARTIFACTS_DIR, FREIGHT_ARTIFACTS_DIR,
    CAT_COLS, NUM_COLS, DIM_DIVISOR,
)
from model import ShippingCostNN

DEVICE = 'cpu'
FREIGHT_MODES = {'LTL', 'FTL'}

# Speed tiers to show per mode, restricted to tiers with meaningful training volume
# (see data_prep.classify_speed_tier — LTL Economy/Expedited have <100 rows each, too
# thin to trust, so LTL is Ground-only here).
MODE_SPEED_TIERS = {
    'PARCEL': ['Ground', '2 Day', 'Next Day', '3 Day'],
    'LTL':    ['Ground'],
}

# Cache keyed by artifacts_dir to support both models in the same process
_cache: dict = {}


def _artifacts_dir_for_mode(carrier_mode: str) -> str:
    if str(carrier_mode).upper().strip() in FREIGHT_MODES:
        return FREIGHT_ARTIFACTS_DIR
    return PARCEL_ARTIFACTS_DIR


def _load(artifacts_dir: str, model_type: str = 'nn') -> dict:
    slot = _cache.setdefault(artifacts_dir, {})

    if 'lookups' not in slot:
        lookups, encoders, scaler, model_config = load_artifacts(artifacts_dir)
        slot.update({'lookups': lookups, 'encoders': encoders,
                     'scaler': scaler, 'model_config': model_config})

    need_nn   = model_type in ('nn', 'ensemble')
    need_lgbm = model_type in ('lgbm', 'ensemble')

    if need_nn and 'model_nn' not in slot:
        cfg = slot['model_config']
        m = ShippingCostNN(
            embedding_sizes=cfg['embedding_sizes'],
            n_numeric=cfg['n_numeric'],
            hidden_sizes=cfg.get('hidden_sizes', (256, 128, 64)),
            dropout=cfg.get('dropout', 0.3),
        )
        m.load_state_dict(
            torch.load(os.path.join(artifacts_dir, 'best_model.pt'), map_location=DEVICE)
        )
        m.eval()
        slot['model_nn'] = m

    if need_lgbm and 'model_lgbm' not in slot:
        slot['model_lgbm'] = lgb.Booster(
            model_file=os.path.join(artifacts_dir, 'lgbm_model.txt')
        )

    return slot


def _encode_val(encoders, col, val):
    val = str(val)
    classes = encoders[col].classes_
    if val in classes:
        return int(encoders[col].transform([val])[0])
    return int(encoders[col].transform(['__unknown__'])[0])


def _enrich(artifacts_dir, ship_from_location_name, ship_to_zip, item_id, qty, ship_date, model_type):
    """Build the mode-specific (cat_vals, num_vals) feature set, minus Carrier Mode/speed_tier."""
    if ship_date is None:
        _date = datetime.date.today()
    elif isinstance(ship_date, str):
        _date = datetime.date.fromisoformat(ship_date)
    else:
        _date = ship_date

    arts     = _load(artifacts_dir, model_type)
    lookups  = arts['lookups']

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

    # Billable weight — carriers charge on whichever is greater, actual or dimensional
    dim_weight       = estimated_cbft / DIM_DIVISOR
    billable_weight  = max(dim_weight, estimated_weight)

    # ── Estimate distance from zip-prefix pair ────────────────────────────────
    loc_str   = str(ship_from_location_name)
    from_zip3 = str(location_zip3.get(loc_str, '194'))[:3]
    to_zip3   = str(ship_to_zip).strip()[:3]

    try:
        estimated_miles = float(zip_miles.loc[(from_zip3, to_zip3)])
    except KeyError:
        estimated_miles = lookups['default_miles']

    # ── Build feature row ─────────────────────────────────────────────────────
    density = billable_weight / estimated_cbft if estimated_cbft > 0 else 0.0

    res_lookup = lookups.get('residential_lookup', {})
    is_residential = res_lookup.get(str(ship_to_zip).strip(), classify_residential(ship_to_zip))

    cat_vals = {
        'ship_from_location_name': loc_str,
        'to_zip3':                 to_zip3,
        'item_id':                 item_id_str,
        'Item_Class1':             item_class1,
        'Item_Class2':             item_class2,
        'NFMC_code':               nfmc_code,
        'ship_month':              str(_date.month),
    }
    num_vals = {
        'log_qty':             np.log1p(qty),
        'log_cbft':            np.log1p(estimated_cbft),
        'log_billable_weight': np.log1p(billable_weight),
        'log_density':         np.log1p(density),
        'log_miles':           np.log1p(estimated_miles),
        'is_residential':      float(is_residential),
        'ship_year':           float(_date.year),
    }
    return arts, cat_vals, num_vals


def _forward(arts, cat_vals, num_vals, model_type):
    """Encode a complete feature row and return the predicted cost in dollars."""
    encoders = arts['encoders']
    scaler   = arts['scaler']

    x_cat = np.array(
        [[_encode_val(encoders, col, cat_vals[col]) for col in CAT_COLS]],
        dtype=np.int64,
    )
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
        log_pred = (_nn_log_pred(arts['model_nn']) + _lgbm_log_pred(arts['model_lgbm'])) / 2
    elif model_type == 'lgbm':
        log_pred = _lgbm_log_pred(arts['model_lgbm'])
    else:
        log_pred = _nn_log_pred(arts['model_nn'])

    return float(np.expm1(log_pred))


def predict_options(
    ship_from_location_name: str,
    ship_to_zip: str,
    item_id: str,
    qty: int | float,
    ship_date=None,
    model_type: str = 'ensemble',
) -> dict:
    """
    Estimate shipping cost across every realistic Mode x Speed Tier combination —
    no carrier mode required as input.

    Parameters
    ----------
    ship_from_location_name : e.g. 'Keystone Technologies PA'
    ship_to_zip             : destination zip code, e.g. '10001'
    item_id                 : SKU / item identifier
    qty                     : number of units
    model_type              : 'nn', 'lgbm', or 'ensemble'  (default: 'ensemble')

    Returns
    -------
    dict : {mode: {speed_tier: predicted_cost_dollars, ...}, ...}
           e.g. {'PARCEL': {'Ground': 12.34, '2 Day': 18.50, 'Next Day': 32.10, '3 Day': 15.75},
                 'LTL':    {'Ground': 210.40}}
    """
    results = {}
    for mode, tiers in MODE_SPEED_TIERS.items():
        artifacts_dir = _artifacts_dir_for_mode(mode)
        arts, cat_vals, num_vals = _enrich(
            artifacts_dir, ship_from_location_name, ship_to_zip, item_id, qty, ship_date, model_type
        )
        results[mode] = {}
        for tier in tiers:
            row = dict(cat_vals, **{'Carrier Mode': mode, 'speed_tier': tier})
            results[mode][tier] = _forward(arts, row, num_vals, model_type)
    return results


if __name__ == '__main__':
    examples = [
        {
            'ship_from_location_name': 'Keystone Technologies PA',
            'ship_to_zip': '10001',
            'item_id': 'KT-LED17PLL-22GC-840-D /G2',
            'qty': 100,
        },
        {
            'ship_from_location_name': 'Keystone Technologies New KC',
            'ship_to_zip': '11788',
            'item_id': 'KT-HBLED90-1.5F-850-VDIM-P /G2',
            'qty': 8,
        },
        {
            'ship_from_location_name': 'Keystone Technologies PHX',
            'ship_to_zip': '30301',
            'item_id': 'KT-SOCKET-T8-U-S-2-W',
            'qty': 500,
        },
    ]

    for r in examples:
        options = predict_options(**r)
        print(f'\n{r["ship_from_location_name"]} -> {r["ship_to_zip"]}  '
              f'{r["item_id"]} x{r["qty"]}')
        for mode, tiers in options.items():
            tier_str = '  '.join(f'{tier}: ${cost:.2f}' for tier, cost in tiers.items())
            print(f'  {mode:<8} {tier_str}')
