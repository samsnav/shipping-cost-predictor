"""
Predict shipping cost from the four user-supplied inputs.

Estimates cost across every realistic Mode x Speed Tier combination instead of
requiring the caller to pick a carrier mode up front.

Usage (script):
    python predict.py

Usage (import):
    from predict import predict_options
    options = predict_options('KT PA', '10001', 'KT-LED17PLL-22GC-840-D /G2', 100)
    # {'PARCEL': {'Ground': 12.34, '2 Day': 18.50, 'Next Day': 32.10, '3 Day': 15.75},
    #  'LTL':    {'Ground': 210.40},
    #  'recommendation': {'flag': None, 'message': None}}
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


def _lookup_item(item_lookup, lookups, item_id):
    """Per-unit cbft/weight and class info for one item_id, falling back to global medians."""
    item_id_str = str(item_id)
    if item_id_str in item_lookup.index:
        row = item_lookup.loc[item_id_str]
        return {
            'cbft_per_unit':   float(row['avg_cbft_per_unit']),
            'weight_per_unit': float(row['avg_weight_per_unit']),
            'item_class1':     str(row['Item_Class1']),
            'item_class2':     str(row['Item_Class2']),
            'nfmc_code':       str(row['NFMC_code']),
        }
    return {
        'cbft_per_unit':   lookups['global_cbft_median'],
        'weight_per_unit': lookups['global_weight_median'],
        'item_class1':     '__unknown__',
        'item_class2':     '__unknown__',
        'nfmc_code':       '__unknown__',
    }


def _enrich(artifacts_dir, ship_from_location_name, ship_to_zip, items, ship_date, model_type):
    """
    Build the mode-specific (cat_vals, num_vals) feature set, minus Carrier Mode/speed_tier.

    items : list of (item_id, qty) pairs — a shipment can be multiple line items. Their
    weight/cubic footage are combined into one shipment; the categorical item/class/NMFC
    fields (which the model was trained on as one-per-row) come from whichever item
    contributes the most weight, since that item's freight characteristics dominate how
    a mixed shipment is typically classified and priced.
    """
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

    # ── Combine line items into one shipment ──────────────────────────────────
    total_qty = 0.0
    total_cbft = 0.0
    total_weight = 0.0
    dominant = None  # (weight_contribution, item_id, class1, class2, nfmc_code)
    for item_id, qty in items:
        info = _lookup_item(item_lookup, lookups, item_id)
        item_cbft   = max(qty * info['cbft_per_unit'], 0.0)
        item_weight = max(qty * info['weight_per_unit'], 0.0)
        total_qty    += qty
        total_cbft   += item_cbft
        total_weight += item_weight
        if dominant is None or item_weight > dominant[0]:
            dominant = (item_weight, str(item_id), info['item_class1'], info['item_class2'], info['nfmc_code'])

    _, item_id_str, item_class1, item_class2, nfmc_code = dominant
    estimated_cbft   = total_cbft
    estimated_weight = total_weight

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
        'log_qty':             np.log1p(total_qty),
        'log_cbft':            np.log1p(estimated_cbft),
        'log_billable_weight': np.log1p(billable_weight),
        'log_density':         np.log1p(density),
        'log_miles':           np.log1p(estimated_miles),
        'is_residential':      float(is_residential),
        'ship_year':           float(_date.year),
        'log_n_line_items':    np.log1p(len(items)),
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


def _mode_recommendation(billable_weight_by_mode: dict, lookups_by_mode: dict, n_items: int) -> dict:
    """
    Flag shipments that fall outside the typical weight range for their mode, using
    each mode's own historical billable-weight distribution (computed in
    data_prep.enrich_features during training). A PARCEL shipment heavier than 95%
    of historical PARCEL shipments is usually cheaper via LTL; an LTL shipment
    lighter than 95% of historical LTL shipments (i.e. below the 5th percentile)
    is usually cheaper via PARCEL.

    Thresholds are chosen by line-item count (n_items): a heavy multi-item shipment is
    usually many ordinary packages summed (cheap, legitimate), while a heavy SINGLE-item
    shipment is one genuinely oversized package — blending both populations into one
    threshold badly miscalibrates it for both, so single- and multi-item tickets get
    their own percentile computed from the matching historical population. Falls back to
    the blended threshold if the mode has no historical tickets in that segment.
    """
    seg = 'single' if n_items == 1 else 'multi'
    parcel_weight = billable_weight_by_mode.get('PARCEL')
    ltl_weight    = billable_weight_by_mode.get('LTL')
    parcel_lookups = lookups_by_mode.get('PARCEL', {})
    ltl_lookups    = lookups_by_mode.get('LTL', {})
    parcel_p95     = parcel_lookups.get(f'billable_weight_p95_{seg}', parcel_lookups.get('billable_weight_p95'))
    ltl_p05        = ltl_lookups.get(f'billable_weight_p05_{seg}', ltl_lookups.get('billable_weight_p05'))

    if parcel_weight is not None and parcel_p95 is not None and parcel_weight > parcel_p95:
        return {
            'flag': 'consider_ltl',
            'message': (f'Estimated billable weight ({parcel_weight:.0f} lb) exceeds the 95th '
                        f'percentile of historical {seg}-item PARCEL shipments ({parcel_p95:.0f} lb) — '
                        f'LTL is likely more cost-effective.'),
        }
    if ltl_weight is not None and ltl_p05 is not None and ltl_weight < ltl_p05:
        return {
            'flag': 'consider_parcel',
            'message': (f'Estimated billable weight ({ltl_weight:.0f} lb) is below the 5th '
                        f'percentile of historical {seg}-item LTL shipments ({ltl_p05:.0f} lb) — '
                        f'PARCEL is likely more cost-effective.'),
        }
    return {'flag': None, 'message': None}


def predict_options(
    ship_from_location_name: str,
    ship_to_zip: str,
    item_id: str,
    qty: int | float,
    ship_date=None,
    model_type: str = 'ensemble',
) -> dict:
    """
    Estimate shipping cost across every realistic Mode x Speed Tier combination for a
    single item/qty — no carrier mode required as input. See predict_options_multi for
    shipments with more than one line item.

    Parameters
    ----------
    ship_from_location_name : e.g. 'KT PA'
    ship_to_zip             : destination zip code, e.g. '10001'
    item_id                 : SKU / item identifier
    qty                     : number of units
    model_type              : 'nn', 'lgbm', or 'ensemble'  (default: 'ensemble')

    Returns
    -------
    See predict_options_multi.
    """
    return predict_options_multi(
        ship_from_location_name, ship_to_zip, [(item_id, qty)], ship_date, model_type
    )


def predict_options_multi(
    ship_from_location_name: str,
    ship_to_zip: str,
    items,
    ship_date=None,
    model_type: str = 'ensemble',
) -> dict:
    """
    Estimate shipping cost for a multi-line shipment (several items/qtys shipped together)
    across every realistic Mode x Speed Tier combination — no carrier mode required as input.

    Line items are combined into one shipment (weight and cubic footage summed) rather than
    predicted separately and added up — carriers price a shared shipment more efficiently
    than several separate ones, so summing single-item predictions would overestimate cost.
    The categorical item/class/NMFC signal comes from whichever line item contributes the
    most weight, since that's what dominates how a mixed shipment gets classified and priced.

    Parameters
    ----------
    ship_from_location_name : e.g. 'KT PA'
    ship_to_zip             : destination zip code, e.g. '10001'
    items                    : list of (item_id, qty) pairs, or [{'item_id': ..., 'qty': ...}, ...]
    model_type               : 'nn', 'lgbm', or 'ensemble'  (default: 'ensemble')

    Returns
    -------
    dict : {mode: {speed_tier: predicted_cost_dollars, ...} | str, ...,
            'recommendation': {'flag': 'consider_ltl' | 'consider_parcel' | None, 'message': str | None}}
           e.g. {'PARCEL': {'Ground': 12.34, '2 Day': 18.50, 'Next Day': 32.10, '3 Day': 15.75},
                 'LTL':    {'Ground': 210.40},
                 'recommendation': {'flag': None, 'message': None}}

           When a shipment falls outside a mode's typical weight range, that mode's prices
           are withheld and replaced with a plain-string steer ('Use LTL' / 'Use PARCEL')
           instead — e.g. {'PARCEL': 'Use LTL', 'LTL': {'Ground': 563.69}, 'recommendation': {...}}.
    """
    items = [(it['item_id'], it['qty']) if isinstance(it, dict) else (it[0], it[1]) for it in items]
    if not items:
        raise ValueError('items must contain at least one (item_id, qty) line')

    results = {}
    billable_weight_by_mode = {}
    lookups_by_mode = {}
    for mode, tiers in MODE_SPEED_TIERS.items():
        artifacts_dir = _artifacts_dir_for_mode(mode)
        arts, cat_vals, num_vals = _enrich(
            artifacts_dir, ship_from_location_name, ship_to_zip, items, ship_date, model_type
        )
        billable_weight_by_mode[mode] = float(np.expm1(num_vals['log_billable_weight']))
        lookups_by_mode[mode] = arts['lookups']
        results[mode] = {}
        for tier in tiers:
            row = dict(cat_vals, **{'Carrier Mode': mode, 'speed_tier': tier})
            results[mode][tier] = _forward(arts, row, num_vals, model_type)

    recommendation = _mode_recommendation(billable_weight_by_mode, lookups_by_mode, len(items))
    if recommendation['flag'] == 'consider_ltl':
        results['PARCEL'] = 'Use LTL'
    elif recommendation['flag'] == 'consider_parcel':
        results['LTL'] = 'Use PARCEL'

    results['recommendation'] = recommendation
    return results


if __name__ == '__main__':
    examples = [
        {
            'ship_from_location_name': 'KT PA',
            'ship_to_zip': '10001',
            'items': [('KT-LED17PLL-22GC-840-D /G2', 100)],
        },
        {
            'ship_from_location_name': 'KT New KC',
            'ship_to_zip': '11788',
            'items': [('KT-HBLED90-1.5F-850-VDIM-P /G2', 200)],
        },
        {
            'ship_from_location_name': 'KT PHX',
            'ship_to_zip': '30301',
            'items': [('KT-SOCKET-T8-U-S-2-W', 500)],
        },
        {
            'ship_from_location_name': 'KT PA',
            'ship_to_zip': '10001',
            'items': [
                ('KT-LED17PLL-22GC-840-D /G2', 100),
                ('KT-HBLED90-1.5F-850-VDIM-P /G2', 20),
            ],
        },
    ]

    for r in examples:
        options = predict_options_multi(**r)
        options.pop('recommendation')
        items_str = ', '.join(f'{item_id} x{qty}' for item_id, qty in r['items'])
        print(f'\n{r["ship_from_location_name"]} -> {r["ship_to_zip"]}  [{items_str}]')
        for mode, tiers in options.items():
            if isinstance(tiers, str):
                print(f'  {mode:<8} {tiers}')
            else:
                tier_str = '  '.join(f'{tier}: ${cost:.2f}' for tier, cost in tiers.items())
                print(f'  {mode:<8} {tier_str}')
