# Ames Housing Predictions

Tab-Transformer deep learning pipeline for predicting house prices on the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).


## Quickstart

```bash
# install dependencies
uv sync

# run training (dataset downloads automatically on first run)
uv run python -m dl.main
```

Training auto-downloads the Ames dataset from OpenML if `train.csv` is not present.  
Checkpoints and the loss curve plot are saved to `checkpoints/tab_transformer_v1/`.

## Training config

| Parameter | Default |
|---|---|
| `hidden_dim` | 128 |
| `num_heads` | 4 |
| `num_layers` | 3 |
| `mlp_ratio` | 2.0 |
| `dropout` | 0.10 |
| `epochs` | 300 |
| `lr` | 1e-3 |
| `weight_decay` | 1e-4 |
| `warmup_epochs` | 10 |
| `patience` | 40 |
| `batch_size` | 64 |

Edit [dl/config.py](dl/config.py) to change any of these.


