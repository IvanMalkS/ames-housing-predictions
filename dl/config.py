from omegaconf import OmegaConf

config = {
    'general': {
        'experiment_name': 'tab_transformer_v1',
        'seed': 42,
    },
    'paths': {
        'train_csv': './train.csv',
        'checkpoints': './checkpoints/${general.experiment_name}',
    },
    'training': {
        'epochs': 300,
        'patience': 40,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'mixed_precision': True,
        'gradient_accumulation': True,
        'gradient_accumulation_steps': 2,
        'gradient_clipping': True,
        'clip_value': 1.0,
        'warmup_scheduler': True,
        'warmup_epochs': 10,
        'device': 'cpu',
        'save_best': True,
        'verbose': True,
    },
    'dataloader': {
        'batch_size': 64,
        'num_workers': 0,
        'shuffle': True,
        'drop_last': True,
    },
    'model': {
        'hidden_dim': 256,
        'emb_dim': 16,
        'num_layers': 4,
        'dropout': 0.15,
    },
}

config = OmegaConf.create(config)
