"""
Neural network for shipping cost regression.

Uses entity embeddings for categorical features (ship-from location, destination
zip prefix, item ID, item classes, NFMC freight code) combined with log-scaled
numeric features (qty, cubic footage, distance).
"""

import torch
import torch.nn as nn


class ShippingCostNN(nn.Module):
    """
    embedding_sizes: list of (n_categories, embed_dim) per categorical column,
                     in the same order as CAT_COLS in data_prep.py
    n_numeric:       number of numeric input features
    hidden_sizes:    sizes of fully-connected hidden layers
    dropout:         dropout rate applied after the first two hidden layers
    """

    def __init__(
        self,
        embedding_sizes: list,
        n_numeric: int,
        hidden_sizes: tuple = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cats, emb_dim, padding_idx=0)
            for n_cats, emb_dim in embedding_sizes
        ])

        emb_total = sum(emb_dim for _, emb_dim in embedding_sizes)
        in_features = emb_total + n_numeric

        layers = []
        drop_rates = [dropout, dropout * 0.67, 0.0]
        for hidden, drop in zip(hidden_sizes, drop_rates):
            layers += [
                nn.Linear(in_features, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
            ]
            if drop > 0:
                layers.append(nn.Dropout(drop))
            in_features = hidden

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        """
        x_cat: (batch, n_cat_cols)  int64 encoded indices
        x_num: (batch, n_numeric)   float32 scaled values
        returns: (batch,) predicted log1p(cost)
        """
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.network(x).squeeze(1)


def get_embedding_sizes(encoders, cat_cols):
    """
    Compute (n_categories, embed_dim) for each categorical column.
    Embedding dimension rule: min(50, max(4, (n + 1) // 2))
    """
    sizes = []
    for col in cat_cols:
        n = len(encoders[col].classes_)
        dim = min(50, max(4, (n + 1) // 2))
        sizes.append((n, dim))
    return sizes
