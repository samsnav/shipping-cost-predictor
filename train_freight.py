"""
Train Freight-specific (LTL + FTL) NN + LightGBM models.

Usage:
    python train_freight.py

Saves artifacts to ./artifacts/freight/
"""

from _train_core import run_training
from data_prep import FREIGHT_ARTIFACTS_DIR

if __name__ == '__main__':
    run_training(carrier_modes=['LTL', 'FTL'], artifacts_dir=FREIGHT_ARTIFACTS_DIR, label='Freight')
