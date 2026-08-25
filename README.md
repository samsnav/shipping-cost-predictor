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

Place the raw data export at `raw_shipping_info.xlsx` in the project root, with two tabs:

- `OutboundFreightSpend` — one row per shipment. Expected columns include `Fixed Total Cost`,
  `ship_from_location_name`, `ship_to_zip`, `item_id`, `PT Total Quantity`, `PT Total Cb Ft`,
  `PT Total Weight`, `unit_weight`, `PT Actual Miles`, `Carrier Mode`, `Carrier Name`, `NMFC`,
  `Item_Class1`, `Item_Class2`.
- `Item Unit Dims and Cartons` — one row per item (`item_id`, `unit_length`, `unit_width`,
  `unit_height`, `nmfc_code`, `carton_qty`). Used as a fallback when an item's shipment history
  doesn't yield a usable cubic-footage or NMFC estimate.

Training is restricted to Keystone's own ship-from locations (`KT PA`, `KT PHX`, `KT New KC`) —
drop-ship vendor locations in the export are excluded.

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

options = predict_options('KT PA', '10001', 'KT-LED17PLL-22GC-840-D /G2', 100)
# {'PARCEL': {'Ground': 12.34, '2 Day': 18.50, 'Next Day': 32.10, '3 Day': 15.75},
#  'LTL':    {'Ground': 210.40},
#  'recommendation': {'flag': None, 'message': None}}
```

`recommendation` flags shipments that fall outside the typical weight range for the mode
they'd naturally ship in — a PARCEL-sized shipment heavier than 95% of historical PARCEL
shipments gets `flag: 'consider_ltl'`; an LTL shipment lighter than 95% of historical LTL
shipments (below the 5th percentile) gets `flag: 'consider_parcel'`. `flag` is `None` when
the shipment is typical for both modes. When a mode's prices are withheld this way, that
mode's key holds a plain string steer instead of a tier dict, e.g. `{'PARCEL': 'Use LTL',
'LTL': {'Ground': 563.69}, 'recommendation': {'flag': 'consider_ltl', 'message': '...'}}`.

The 95th/5th-percentile thresholds are computed separately for single- vs. multi-item
shipments, not from one blended population — a heavy multi-item shipment is usually many
ordinary packages summed (cheap, legitimate), while a heavy single-item shipment is one
genuinely oversized package with far less historical precedent. Blending them badly
miscalibrates the threshold for both (see `predict._mode_recommendation`).

For shipments with more than one line item, use `predict_options_multi` instead:

```python
from predict import predict_options_multi

options = predict_options_multi('KT PA', '10001', [
    ('KT-LED17PLL-22GC-840-D /G2', 100),
    ('KT-HBLED90-1.5F-850-VDIM-P /G2', 20),
])
```

Line items are combined into one shipment (weight/cubic footage summed) rather than priced
separately and added up, since carriers price one shared shipment more efficiently than
several individual ones. The categorical item/class/NMFC signal comes from whichever line
item contributes the most weight.

Or run `python predict.py` directly for a few worked examples.

## Project layout

| File | Purpose |
|---|---|
| `data_prep.py` | Load/clean the raw Excel export, collapse line items into physical shipments, engineer features (billable weight, distance, residential flag, speed tier), fit encoders, save/load artifacts |
| `model.py` | `ShippingCostNN` — entity-embedding + MLP architecture shared by both mode-specific models |
| `_train_core.py` | Shared training loop (NN + LightGBM) — not run directly |
| `train_parcel.py` / `train_freight.py` | Entry points that call `_train_core.run_training` for PARCEL vs. LTL/FTL, writing to separate `artifacts/` subdirectories |
| `predict.py` | Loads trained artifacts and serves predictions across all mode/speed-tier combinations |

## How it works

- **Line items → tickets** (`data_prep.aggregate_to_tickets`): the raw export has one row per
  line item, and a line's `Fixed Total Cost` is a weight-proportional allocation of its pick
  ticket's real total cost (verified against the raw data — see git history), not a standalone
  price. Training rows are therefore built by grouping lines on `PT or TR no` and summing
  quantity/weight/cubic-footage/cost into one row per physical shipment, so the model learns
  from real shipment totals instead of fragmented allocations. The categorical item/class/NMFC
  fields come from whichever line contributes the most weight — the same heuristic
  `predict._enrich` uses to combine a caller-supplied multi-item list at inference time, so
  training and inference build features the same way. `n_line_items` (how many distinct lines
  made up the ticket) is fed to the model as a feature.
- **Feature engineering** (`data_prep.py`): cubic footage is estimated per-unit from historical
  `PT Total Cb Ft` / quantity (falling back to the item master's unit dimensions, then a global
  median, when a given item has no usable shipment history). Weight uses the item master's fixed
  `unit_weight` directly. Billable weight takes the greater of actual weight and dimensional
  weight (cubic feet / 225). Distance is estimated from a (origin zip3, destination zip3) →
  median-miles lookup. Speed tier is derived from keyword matching on the raw carrier name (Next
  Day, 2 Day, 3 Day, Economy, Expedited, International, else Ground).
- **Training** (`_train_core.py`): each mode gets its own NN (via PyTorch, entity embeddings
  for categoricals + log-scaled numerics) and LightGBM model, trained on an 90/10 split with
  early stopping. MAE/R² are reported for NN, LightGBM, and their ensemble average. Each mode's
  5th/95th-percentile billable weight is saved alongside its lookups for the mode recommendation
  — once from all tickets (`billable_weight_p05`/`p95`, used as a fallback) and once each from
  single-line-item tickets only and multi-line-item tickets only (`..._single`/`..._multi`).
- **Inference** (`predict.py`): re-derives the same features from the user's inputs using saved
  lookup tables, then evaluates both models per mode/speed-tier and averages their log-cost
  predictions before converting back to dollars. Also compares the estimated billable weight
  against each mode's saved percentile thresholds to flag when the other mode is likely cheaper
  (see `_mode_recommendation`).
