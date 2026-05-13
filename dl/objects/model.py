import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AmesDNN(nn.Module):
    def __init__(self, num_features: int, cat_sizes: list, cfg):
        super().__init__()
        D       = cfg.model.hidden_dim
        emb_dim = cfg.model.emb_dim

        self.cat_embeds = nn.ModuleList([nn.Embedding(sz + 1, emb_dim) for sz in cat_sizes])

        input_dim = num_features + len(cat_sizes) * emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, D),
            nn.BatchNorm1d(D),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[
            ResBlock(D, cfg.model.dropout) for _ in range(cfg.model.num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(D // 2, 1),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        x_num, x_cat = batch['num'], batch['cat']
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeds)]
        x = torch.cat([x_num] + cat_embs, dim=1)
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x)
