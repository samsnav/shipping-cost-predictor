"""
Train PARCEL-specific NN + LightGBM models.

Usage:
    python train_parcel.py

Saves artifacts to ./artifacts/parcel/
"""

from _train_core import run_training
from data_prep import PARCEL_ARTIFACTS_DIR

if __name__ == '__main__':
    run_training(carrier_modes=['PARCEL'], artifacts_dir=PARCEL_ARTIFACTS_DIR, label='PARCEL')
