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
        'epochs': 500,
        'patience': 60,
        'n_folds': 5,
        'lr': 5e-4,
        'weight_decay': 5e-5,
        'mixed_precision': True,
        'gradient_accumulation': True,
        'gradient_accumulation_steps': 2,
        'gradient_clipping': True,
        'clip_value': 1.0,
        'warmup_scheduler': True,
        'warmup_epochs': 20,
        'device': 'cpu',
        'save_best': True,
        'verbose': True,
    },
    'dataloader': {
        'batch_size': 128,
        'num_workers': 0,
        'shuffle': True,
        'drop_last': True,
    },
    'model': {
        'hidden_dim': 384,
        'emb_dim': 32,
        'num_layers': 4,
        'dropout': 0.05,
    },
}

config = OmegaConf.create(config)
