import os
import gc
import torch
import torch.nn as nn
from torch.cuda import amp

from dl.data import get_loaders
from dl.objects.model import AmesTabTransformer
from dl.objects.scheduler import get_scheduler


def train(config, model, loader, optimizer, scheduler, criterion, scaler):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        batch = {k: v.to(config.training.device) for k, v in batch.items()}

        if config.training.mixed_precision:
            with amp.autocast():
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

        last_step = (step + 1) == len(loader)
        accum_ready = (step + 1) % config.training.gradient_accumulation_steps == 0

        if not config.training.gradient_accumulation or accum_ready or last_step:
            if config.training.gradient_clipping:
                if config.training.mixed_precision:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_value)

            if config.training.mixed_precision:
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


def run(config):
    os.makedirs(config.paths.checkpoints, exist_ok=True)

    train_loader, val_loader, meta = get_loaders(config, config.paths.train_csv)

    model     = AmesTabTransformer(meta['num_features'], meta['cat_sizes'], config).to(config.training.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scheduler = get_scheduler(config, optimizer)
    scaler    = amp.GradScaler(enabled=(config.training.mixed_precision and config.training.device == 'cuda'))

    best_val_loss, no_improve, best_weights = float('inf'), 0, None
    train_losses, val_losses = [], []

    for epoch in range(config.training.epochs):
        train_loss              = train(config, model, train_loader, optimizer, scheduler, criterion, scaler)
        val_loss, preds, labels = validation(config, model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            if config.training.save_best:
                torch.save(
                    {'epoch': epoch + 1, 'model_state': best_weights, 'val_loss': best_val_loss},
                    os.path.join(config.paths.checkpoints, 'best.pt'),
                )
        else:
            no_improve += 1

        if config.training.verbose and (epoch + 1) % 20 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1:3d}/{config.training.epochs} | train={train_loss:.5f} | val={val_loss:.5f} | lr={lr:.2e} | patience={no_improve}/{config.training.patience}')

        if no_improve >= config.training.patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        gc.collect()

    model.load_state_dict(best_weights)
    print(f'Best val MSE : {best_val_loss:.5f}')
    return model, meta, train_losses, val_losses
