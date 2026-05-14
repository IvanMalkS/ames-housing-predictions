import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

from dl.config import config
from dl.train_functions import run


def fit(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.training.device = device

    print(f'Experiment : {cfg.general.experiment_name}')
    print(f'Device     : {device}')
    print(f'Config     : hidden_dim={cfg.model.hidden_dim}, emb_dim={cfg.model.emb_dim}, layers={cfg.model.num_layers}')
    print(f'Training   : epochs={cfg.training.epochs}, lr={cfg.training.lr}, folds={cfg.training.n_folds}')
    print()

    model, meta, all_train_losses, all_val_losses, oof_preds, oof_labels, fold_mapes = run(cfg)

    oof_mape = mean_absolute_percentage_error(np.expm1(oof_labels), np.expm1(oof_preds))
    print(f'\nFinal OOF MAPE : {oof_mape * 100:.2f}%')

    fig, axes = plt.subplots(1, cfg.training.n_folds, figsize=(4 * cfg.training.n_folds, 4), sharey=True)
    if cfg.training.n_folds == 1:
        axes = [axes]

    for i, (train_losses, val_losses) in enumerate(zip(all_train_losses, all_val_losses)):
        best_ep = int(np.argmin(val_losses))
        ax = axes[i]
        ax.plot(train_losses, label='Train MSE', alpha=0.7)
        ax.plot(val_losses,   label='Val MSE',   alpha=0.9)
        ax.axvline(best_ep, color='red', linestyle='--', linewidth=1, label=f'Best ep {best_ep + 1}')
        ax.set_title(f'Fold {i + 1}  MAPE={fold_mapes[i]*100:.1f}%')
        ax.set_xlabel('Epoch')
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax.set_ylabel('MSE (log-space)')
            ax.legend(fontsize=7)

    fig.suptitle(f'AmesDNN — {cfg.general.experiment_name} | OOF MAPE={oof_mape*100:.2f}%')
    plt.tight_layout()
    plt.savefig(f'{cfg.paths.checkpoints}/training_curve.png', dpi=120)
    plt.show()

    return model, meta


if __name__ == '__main__':
    fit(config)
