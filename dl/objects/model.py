import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed      = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class AmesTabTransformer(nn.Module):
    def __init__(self, num_features: int, cat_sizes: list, cfg):
        super().__init__()
        D = cfg.model.hidden_dim

        self.cat_embeds  = nn.ModuleList([nn.Embedding(sz, D) for sz in cat_sizes])
        self.num_projs   = nn.ModuleList([nn.Linear(1, D) for _ in range(num_features)])
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, D))
        self.transformer = nn.Sequential(*[
            TransformerBlock(D, cfg.model.num_heads, cfg.model.mlp_ratio, cfg.model.dropout)
            for _ in range(cfg.model.num_layers)
        ])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Sequential(
            nn.Linear(D, D // 2), nn.GELU(), nn.Dropout(cfg.model.dropout),
            nn.Linear(D // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, batch: dict) -> torch.Tensor:
        x_num, x_cat = batch['num'], batch['cat']
        B = x_num.size(0)
        num_tokens = [proj(x_num[:, i:i+1]).unsqueeze(1) for i, proj in enumerate(self.num_projs)]
        cat_tokens = [emb(x_cat[:, i]).unsqueeze(1)      for i, emb  in enumerate(self.cat_embeds)]
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls] + num_tokens + cat_tokens, dim=1)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens[:, 0])
