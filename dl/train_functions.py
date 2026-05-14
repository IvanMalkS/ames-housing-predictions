import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_percentage_error

from dl.data import prepare_data, get_fold_loaders
from dl.objects.model import AmesDNN
from dl.objects.scheduler import get_scheduler


def train(config, model, loader, optimizer, scheduler, criterion, scaler):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    use_amp = config.training.mixed_precision and config.training.device == 'cuda'

    for step, batch in enumerate(loader):
        batch = {k: v.to(config.training.device) for k, v in batch.items()}

        if use_amp:
            with torch.amp.autocast('cuda'):
                loss = criterion(model(batch), batch['label'])
                if config.training.gradient_accumulation:
                    loss = loss / config.training.gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            loss = criterion(model(batch), batch['label'])
            if config.training.gradient_accumulation:
                loss = loss / config.training.gradient_accumulation_steps
            loss.backward()

        total_loss += loss.item() * batch['num'].size(0)

        last_step  = (step + 1) == len(loader)
        accum_ready = (step + 1) % config.training.gradient_accumulation_steps == 0

        if not config.training.gradient_accumulation or accum_ready or last_step:
            if config.training.gradient_clipping:
                if use_amp:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_value)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

    scheduler.step()
    accum = config.training.gradient_accumulation_steps if config.training.gradient_accumulation else 1
    return total_loss * accum / len(loader.dataset)


def validation(config, model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(config.training.device) for k, v in batch.items()}
            out = model(batch)
            total_loss += criterion(out, batch['label']).item() * batch['num'].size(0)
            preds.append(out.cpu())
            targets.append(batch['label'].cpu())

    return (
        total_loss / len(loader.dataset),
        torch.cat(preds).numpy(),
        torch.cat(targets).numpy(),
    )


def _train_fold(config, fold_idx, train_loader, val_loader, meta):
    torch.manual_seed(config.general.seed + fold_idx)
    torch.cuda.manual_seed_all(config.general.seed + fold_idx)

    model     = AmesDNN(meta['num_features'], meta['cat_sizes'], config).to(config.training.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scheduler = get_scheduler(config, optimizer)
    use_amp   = config.training.mixed_precision and config.training.device == 'cuda'
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_loss, no_improve, best_weights = float('inf'), 0, None
    best_preds, best_labels = None, None
    train_losses, val_losses = [], []

    for epoch in range(config.training.epochs):
        train_loss              = train(config, model, train_loader, optimizer, scheduler, criterion, scaler)
        val_loss, preds, labels = validation(config, model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds    = preds
            best_labels   = labels
            no_improve    = 0
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if config.training.verbose and (epoch + 1) % 20 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f'  Epoch {epoch+1:3d}/{config.training.epochs} | train={train_loss:.5f} | val={val_loss:.5f} | lr={lr:.2e} | patience={no_improve}/{config.training.patience}')

        if no_improve >= config.training.patience:
            print(f'  Early stopping at epoch {epoch + 1}')
            break

        gc.collect()

    model.load_state_dict(best_weights)
    return model, train_losses, val_losses, best_preds, best_labels, best_val_loss


def run(config):
    seed = config.general.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    os.makedirs(config.paths.checkpoints, exist_ok=True)

    X, y, num_cols, cat_cols = prepare_data(config, config.paths.train_csv)

    n_folds = config.training.n_folds
    y_bins = pd.cut(y, bins=10, labels=False)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_preds  = np.zeros(len(y))
    oof_labels = y.copy()

    fold_mapes      = []
    fold_val_losses = []
    all_train_losses = []
    all_val_losses   = []

    best_model     = None
    best_meta      = None
    best_fold_loss = float('inf')

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        print(f'\n=== Fold {fold_idx + 1}/{n_folds} ===')

        train_loader, val_loader, meta = get_fold_loaders(
            config, X, y, train_idx, val_idx, num_cols, cat_cols
        )

        model, train_losses, val_losses, preds, labels, best_val_loss = _train_fold(
            config, fold_idx, train_loader, val_loader, meta
        )

        oof_preds[val_idx] = preds.flatten()
        fold_mape = mean_absolute_percentage_error(np.expm1(labels.flatten()), np.expm1(preds.flatten()))
        fold_mapes.append(fold_mape)
        fold_val_losses.append(best_val_loss)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        print(f'  Best val MSE : {best_val_loss:.5f} | MAPE : {fold_mape * 100:.2f}%')

        if best_val_loss < best_fold_loss:
            best_fold_loss = best_val_loss
            best_model     = model
            best_meta      = meta
            if config.training.save_best:
                torch.save(
                    {'fold': fold_idx + 1, 'model_state': model.state_dict(), 'val_loss': best_val_loss},
                    os.path.join(config.paths.checkpoints, 'best.pt'),
                )

    oof_mape = mean_absolute_percentage_error(np.expm1(oof_labels), np.expm1(oof_preds))
    oof_rmse = np.sqrt(np.mean((oof_preds - oof_labels) ** 2))

    print(f'\n{"="*45}')
    print(f'Per-fold MAPE : {[f"{m*100:.2f}%" for m in fold_mapes]}')
    print(f'Mean MAPE     : {np.mean(fold_mapes)*100:.2f}% ± {np.std(fold_mapes)*100:.2f}%')
    print(f'OOF  MAPE     : {oof_mape*100:.2f}%')
    print(f'OOF  RMSE     : {oof_rmse:.5f} (log-space)')
    print(f'{"="*45}')

    best_fold_idx = int(np.argmin(fold_val_losses))
    return (
        best_model,
        best_meta,
        all_train_losses,
        all_val_losses,
        oof_preds,
        oof_labels,
        fold_mapes,
    )
