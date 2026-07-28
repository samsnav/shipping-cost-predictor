# Shipping Cost Predictor

Predicts shipping cost from four inputs — origin location, destination zip, item ID, and
quantity — without requiring the caller to know the carrier mode or speed tier up front.
Returns an estimate for every realistic Mode x Speed Tier combination (PARCEL Ground/2 Day/
Next Day/3 Day, LTL Ground) so the caller can compare options.

Each estimate is an ensemble of a neural network (entity embeddings + MLP) and a LightGBM
model, trained separately for PARCEL vs. Freight (LTL/FTL) since the two modes have very
different cost drivers.

## Setup

```
pip install -r requirements.txt
```

Place the raw data export at `raw_shipping_data.xlsx` in the project root (expected columns
include `Fixed Total Cost`, `ship_from_location_name`, `ship_to_zip`, `item_id`,
`PT Total Quantity`, `PT Total Cb Ft`, `PT Total Weight`, `PT Actual Miles`, `Carrier Mode`,
`Carrier Name`, `NFMC code`, `Item_Class1`, `Item_Class2`).

## Usage

Train both mode-specific models (each saves NN + LightGBM + encoders/lookups to
`artifacts/parcel/` or `artifacts/freight/`):

```
python train_parcel.py
python train_freight.py
```

Get predictions:

```python
from predict import predict_options

options = predict_options('Keystone Technologies PA', '10001', 'KT-LED17PLL-22GC-840-D /G2', 100)
# {'PARCEL': {'Ground': 12.34, '2 Day': 18.50, 'Next Day': 32.10, '3 Day': 15.75},
#  'LTL':    {'Ground': 210.40}}
```

Or run `python predict.py` directly for a few worked examples.

## Project layout

| File | Purpose |
|---|---|
| `data_prep.py` | Load/clean the raw Excel export, engineer features (billable weight, distance, residential flag, speed tier), fit encoders, save/load artifacts |
| `model.py` | `ShippingCostNN` — entity-embedding + MLP architecture shared by both mode-specific models |
| `_train_core.py` | Shared training loop (NN + LightGBM) — not run directly |
| `train_parcel.py` / `train_freight.py` | Entry points that call `_train_core.run_training` for PARCEL vs. LTL/FTL, writing to separate `artifacts/` subdirectories |
| `predict.py` | Loads trained artifacts and serves predictions across all mode/speed-tier combinations |

## How it works

- **Feature engineering** (`data_prep.py`): cubic footage and weight are estimated per-unit
  from historical item data, since inference time only gets an item ID and quantity. Billable
  weight takes the greater of actual weight and dimensional weight (cubic feet / 225). Distance
  is estimated from a (origin zip3, destination zip3) → median-miles lookup. Speed tier is
  derived from keyword matching on the raw carrier name (Next Day, 2 Day, 3 Day, Economy,
  Expedited, International, else Ground).
- **Training** (`_train_core.py`): each mode gets its own NN (via PyTorch, entity embeddings
  for categoricals + log-scaled numerics) and LightGBM model, trained on an 90/10 split with
  early stopping. MAE/R² are reported for NN, LightGBM, and their ensemble average.
- **Inference** (`predict.py`): re-derives the same features from the four user inputs using
  saved lookup tables, then evaluates both models per mode/speed-tier and averages their
  log-cost predictions before converting back to dollars.
