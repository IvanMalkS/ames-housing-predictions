import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from dl.config import config
from dl.train_functions import run


def fit(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.training.device = device

    print(f'Experiment : {cfg.general.experiment_name}')
    print(f'Device     : {device}')
    print(f'Config     : hidden_dim={cfg.model.hidden_dim}, heads={cfg.model.num_heads}, layers={cfg.model.num_layers}')
    print(f'Training   : epochs={cfg.training.epochs}, lr={cfg.training.lr}, warmup={cfg.training.warmup_epochs}')
    print()

    model, meta, train_losses, val_losses = run(cfg)

    best_ep = int(np.argmin(val_losses))
    print(f'\nStopped at epoch {len(train_losses)}, best epoch {best_ep + 1}')

    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Train MSE')
    plt.plot(val_losses,   label='Val MSE')
    plt.axvline(best_ep, color='red',    linestyle='--', label=f'Best epoch {best_ep + 1}')
    plt.axvline(cfg.training.warmup_epochs, color='orange', linestyle=':', label=f'Warmup end ({cfg.training.warmup_epochs})')
    plt.title(f'Tab-Transformer — {cfg.general.experiment_name}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (log-space)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{cfg.paths.checkpoints}/training_curve.png', dpi=120)
    plt.show()

    return model, meta


if __name__ == '__main__':
    fit(config)
