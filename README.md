# Ames Housing Price Prediction

Regression pipeline for the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).  
Target: `SalePrice` (MAPE on held-out test set).

## Results

| Model | MAPE (%) | RMSE (log) |
|---|---|---|
| **Stacking** | **7.75** | **0.1140** |
| SVR | 7.92 | 0.1171 |
| Simple Average | 7.94 | 0.1172 |
| Voting Ensemble | 7.94 | 0.1172 |
| CatBoost | 8.20 | 0.1190 |
| XGBoost | 8.57 | 0.1227 |
| Ridge | 8.69 | 0.1221 |
| ElasticNet | 8.70 | 0.1230 |
| LightGBM | 9.00 | 0.1308 |
| Linear Regression | 9.13 | 0.1303 |
| Random Forest | 9.44 | 0.1368 |
| DNN (AmesDNN)¹ | 10.11 | 0.1344 |
| KNN | 11.35 | 0.1640 |
| Lasso | 12.67 | 0.1712 |
| Decision Tree | 14.39 | 0.1990 |

¹ OOF MAPE across 5-fold CV, measured by `dl/main.py` without feature interaction terms.

## Quickstart

```bash
# install dependencies
uv sync

# run DNN training (dataset downloads automatically on first run)
uv run python -m dl.main
```

Checkpoints and the training curve are saved to `checkpoints/tab_transformer_v1/`.

## Project structure

```
dl/                  # DNN pipeline (runnable standalone)
├── config.py        # all hyperparameters
├── main.py          # entry point — trains and plots
├── data.py          # data loading, feature engineering, fold loaders
├── train_functions.py
└── objects/
    ├── model.py     # AmesDNN (ResBlock + Embeddings)
    └── scheduler.py # warmup + cosine annealing
Ames.ipynb           # EDA, all models, experiments
pyproject.toml       # dependencies (uv / pip)
```

## DNN config

Edit [dl/config.py](dl/config.py) to change any of these.

| Parameter | Default | Description |
|---|---|---|
| `hidden_dim` | 384 | Width of each residual block |
| `emb_dim` | 32 | Categorical embedding size |
| `num_layers` | 4 | Number of ResBlocks |
| `dropout` | 0.05 | Dropout rate |
| `epochs` | 500 | Max training epochs |
| `patience` | 60 | Early stopping patience |
| `lr` | 5e-4 | AdamW learning rate |
| `weight_decay` | 5e-5 | AdamW weight decay |
| `warmup_epochs` | 20 | Linear warmup before cosine annealing |
| `n_folds` | 5 | Stratified K-Fold splits |
| `batch_size` | 128 | Dataloader batch size |
