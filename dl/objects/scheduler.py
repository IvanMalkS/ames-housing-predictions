from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def get_scheduler(config, optimizer: Optimizer):
    if config.training.warmup_scheduler:
        warmup = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=config.training.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs - config.training.warmup_epochs,
            eta_min=1e-5,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[config.training.warmup_epochs],
        )
    return CosineAnnealingLR(optimizer, T_max=config.training.epochs, eta_min=1e-5)
