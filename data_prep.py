"""
Data loading, feature engineering, and encoding for the shipping cost predictor.

Inputs at inference time: ship_from_location_name, ship_to_zip, item_id, qty
Target: Fixed Total Cost (dollars)
"""

import os
import pickle
import shutil
import tempfile
import numpy as np
import zipcodes as zc
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

EXCEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_shipping_data.xlsx')
_BASE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR         = os.path.join(_BASE, 'artifacts')
PARCEL_ARTIFACTS_DIR  = os.path.join(_BASE, 'artifacts', 'parcel')
FREIGHT_ARTIFACTS_DIR = os.path.join(_BASE, 'artifacts', 'freight')

# Categorical features fed to embedding layers
CAT_COLS = ['ship_from_location_name', 'Carrier Mode', 'Carrier Name', 'to_zip3', 'item_id', 'Item_Class1', 'Item_Class2', 'NFMC_code']
# Log-transformed numeric features
NUM_COLS = ['log_qty', 'log_cbft', 'log_weight', 'log_density', 'log_miles', 'is_residential']
TARGET_COL = 'log_cost'


def load_and_clean(excel_path=EXCEL_PATH, carrier_modes=None):
    print(f'Reading {excel_path} ...')
    # Copy to a temp file first so this works even when Excel has the file open
    tmp = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    tmp.close()
    shutil.copy2(excel_path, tmp.name)
    df = pd.read_excel(tmp.name)
    os.unlink(tmp.name)
    df.columns = [c.strip('[]') for c in df.columns]

    required = ['Fixed Total Cost', 'ship_from_location_name', 'ship_to_zip',
                'item_id', 'PT Total Quantity']
    df = df.dropna(subset=required)
    df = df[df['Fixed Total Cost'] > 0]
    df = df[df['Fixed Total Cost'] <= 1000]
    df = df[df['PT Total Quantity'] > 0]

    allowed_locations = {
        'Keystone Technologies PA',
        'Keystone Technologies PHX',
        'Keystone Technologies New KC',
    }
    df = df[df['ship_from_location_name'].isin(allowed_locations)]
    df = df[df['pt_type'] == 'PT']

    df['ship_to_zip'] = df['ship_to_zip'].astype(str).str.strip()
    df['ship_from_zip'] = df['ship_from_zip'].astype(str).str.strip()
    df['to_zip3'] = df['ship_to_zip'].str[:3]
    df['from_zip3'] = df['ship_from_zip'].str[:3]
    df['NFMC_code'] = df['NFMC code'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    df['ship_from_location_name'] = df['ship_from_location_name'].astype(str)
    df['Carrier Mode'] = df['Carrier Mode'].fillna('Unknown').astype(str).str.strip()
    df['Carrier Name'] = df['Carrier Name'].fillna('Unknown').astype(str).str.strip()

    if carrier_modes is not None:
        df = df[df['Carrier Mode'].isin(carrier_modes)]
        print(f'  Filtered to {carrier_modes}: {len(df):,} rows')

    return df


def classify_residential(zip_code: str) -> float:
    """Return 1.0 if zip is residential, 0.0 if commercial/military/PO Box."""
    try:
        matches = zc.matching(str(zip_code).strip().zfill(5))
        if not matches:
            return 1.0  # unknown → default residential (safer for cost estimation)
        zip_type = matches[0]['zip_code_type']
        return 0.0 if zip_type in ('UNIQUE', 'PO BOX', 'MILITARY') else 1.0
    except Exception:
        return 1.0


def build_residential_lookup(zip_codes):
    """Build a zip → residential flag dict for all unique zip codes in the dataset."""
    print(f'  Building residential lookup for {len(zip_codes):,} unique zip codes ...')
    return {z: classify_residential(z) for z in zip_codes}


def build_lookup_tables(df):
    """
    Build lookup tables from training data. These are saved to disk and
    used at inference time to enrich the 4 user-provided inputs.
    """
    df = df.copy()

    # Per-unit cubic footage and weight (used to estimate totals from qty at inference)
    df['cbft_per_unit'] = df['PT Total Cb Ft'] / df['PT Total Quantity']
    df['cbft_per_unit'] = df['cbft_per_unit'].replace([np.inf, -np.inf], np.nan)
    global_cbft_median = df['cbft_per_unit'].median()

    df['weight_per_unit'] = df['PT Total Weight'] / df['PT Total Quantity']
    df['weight_per_unit'] = df['weight_per_unit'].replace([np.inf, -np.inf], np.nan)
    global_weight_median = df['weight_per_unit'].median()

    item_lookup = df.groupby('item_id').agg(
        avg_cbft_per_unit=('cbft_per_unit', 'median'),
        avg_weight_per_unit=('weight_per_unit', 'median'),
        Item_Class1=('Item_Class1', lambda x: x.mode().iloc[0]),
        Item_Class2=('Item_Class2', lambda x: x.mode().iloc[0]),
        NFMC_code=('NFMC_code', lambda x: x.mode().iloc[0]),
    )
    item_lookup['avg_cbft_per_unit'] = item_lookup['avg_cbft_per_unit'].fillna(global_cbft_median)
    item_lookup['avg_weight_per_unit'] = item_lookup['avg_weight_per_unit'].fillna(global_weight_median)

    # Ship-from location name → 3-digit zip prefix
    location_zip3 = df.groupby('ship_from_location_name')['from_zip3'].agg(
        lambda x: x.mode().iloc[0]
    )

    # (from_zip3, to_zip3) → median miles; used to estimate distance at inference
    zip_miles = df.dropna(subset=['PT Actual Miles']).groupby(
        ['from_zip3', 'to_zip3']
    )['PT Actual Miles'].median()
    default_miles = float(df['PT Actual Miles'].median())

    residential_lookup = build_residential_lookup(df['ship_to_zip'].unique())

    lookups = {
        'item_lookup': item_lookup,
        'location_zip3': location_zip3,
        'zip_miles': zip_miles,
        'default_miles': default_miles,
        'global_cbft_median': float(global_cbft_median),
        'global_weight_median': float(global_weight_median),
        'residential_lookup': residential_lookup,
    }
    return lookups, df


def enrich_features(df, lookups):
    """
    Add estimated_cbft, estimated_weight, estimated_miles, density, and log features.
    Uses actual column values when present (training), lookup estimates otherwise (inference).
    """
    df = df.copy()
    item_lookup = lookups['item_lookup']
    zip_miles = lookups['zip_miles']

    # Use actual values when available (training); fall back to lookup estimates at inference

    # Cubic footage — actual value preferred, lookup estimate fills any nulls
    cbft_map = item_lookup['avg_cbft_per_unit'].to_dict()
    cbft_estimate = (df['PT Total Quantity'] * df['item_id'].map(cbft_map).fillna(lookups['global_cbft_median']))
    if 'PT Total Cb Ft' in df.columns:
        df['estimated_cbft'] = df['PT Total Cb Ft'].fillna(cbft_estimate).clip(lower=0)
    else:
        df['estimated_cbft'] = cbft_estimate.clip(lower=0)

    # Weight — actual value preferred, lookup estimate fills any nulls
    weight_map = item_lookup['avg_weight_per_unit'].to_dict()
    weight_estimate = (df['PT Total Quantity'] * df['item_id'].map(weight_map).fillna(lookups['global_weight_median']))
    if 'PT Total Weight' in df.columns:
        df['estimated_weight'] = df['PT Total Weight'].fillna(weight_estimate).clip(lower=0)
    else:
        df['estimated_weight'] = weight_estimate.clip(lower=0)

    # Miles
    if 'PT Actual Miles' in df.columns:
        zip_miles_df = zip_miles.reset_index()
        zip_miles_df.columns = ['from_zip3', 'to_zip3', '_miles']
        df = df.merge(zip_miles_df, on=['from_zip3', 'to_zip3'], how='left')
        df['estimated_miles'] = df['PT Actual Miles'].fillna(df['_miles']).fillna(lookups['default_miles']).clip(lower=0)
        df = df.drop(columns=['_miles'])
    else:
        zip_miles_df = zip_miles.reset_index()
        zip_miles_df.columns = ['from_zip3', 'to_zip3', '_miles']
        df = df.merge(zip_miles_df, on=['from_zip3', 'to_zip3'], how='left')
        df['estimated_miles'] = df['_miles'].fillna(lookups['default_miles']).clip(lower=0)
        df = df.drop(columns=['_miles'])

    # Fill item class columns from lookup for any missing rows
    for col in ['Item_Class1', 'Item_Class2', 'NFMC_code']:
        if df[col].isnull().any():
            fallback = item_lookup[col].to_dict()
            df[col] = df[col].fillna(df['item_id'].map(fallback)).fillna('Unknown')

    # Density = lbs per cubic foot — signals whether carrier bills on weight or DIM
    df['density'] = (df['estimated_weight'] / df['estimated_cbft'].replace(0, np.nan)).fillna(0).clip(lower=0)

    # Residential delivery flag — looked up from ship_to_zip, no user input needed
    res_lookup = lookups['residential_lookup']
    df['is_residential'] = df['ship_to_zip'].map(res_lookup).fillna(1.0)

    # Log-transform inputs and target
    df['log_qty'] = np.log1p(df['PT Total Quantity'])
    df['log_cbft'] = np.log1p(df['estimated_cbft'])
    df['log_weight'] = np.log1p(df['estimated_weight'])
    df['log_density'] = np.log1p(df['density'])
    df['log_miles'] = np.log1p(df['estimated_miles'])
    if 'Fixed Total Cost' in df.columns:
        df['log_cost'] = np.log1p(df['Fixed Total Cost'])

    return df


def build_encoders(df):
    """Fit LabelEncoders (index 0 = unknown) and StandardScaler on training data."""
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        vals = df[col].astype(str).unique().tolist()
        le.fit(['__unknown__'] + vals)
        encoders[col] = le

    scaler = StandardScaler()
    scaler.fit(df[NUM_COLS].values.astype(np.float32))

    return encoders, scaler


def apply_encoders(df, encoders, scaler):
    """Return (X_cat, X_num, y) numpy arrays ready for the model."""
    n = len(df)
    X_cat = np.zeros((n, len(CAT_COLS)), dtype=np.int64)
    for i, col in enumerate(CAT_COLS):
        vals = df[col].astype(str).values.copy()
        unknown_mask = ~np.isin(vals, encoders[col].classes_)
        vals[unknown_mask] = '__unknown__'
        X_cat[:, i] = encoders[col].transform(vals)

    X_num = scaler.transform(df[NUM_COLS].values.astype(np.float32)).astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32) if TARGET_COL in df.columns else None

    return X_cat, X_num, y


def save_artifacts(lookups, encoders, scaler, model_config, artifacts_dir=ARTIFACTS_DIR):
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, 'lookups.pkl'), 'wb') as f:
        pickle.dump(lookups, f)
    with open(os.path.join(artifacts_dir, 'encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
    with open(os.path.join(artifacts_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(artifacts_dir, 'model_config.pkl'), 'wb') as f:
        pickle.dump(model_config, f)


def load_artifacts(artifacts_dir=ARTIFACTS_DIR):
    def _load(name):
        with open(os.path.join(artifacts_dir, name), 'rb') as f:
            return pickle.load(f)
    return _load('lookups.pkl'), _load('encoders.pkl'), _load('scaler.pkl'), _load('model_config.pkl')
