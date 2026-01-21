"""
Training entry point for Attn-JGNN.

Usage:
    python train.py
    python train.py model.num_iterations=10
    python train.py training.lr=0.0001
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import logging
import os

from torch_geometric.loader import DataLoader

from data.datasets.satlib_dataset import SATLIBDataset
from models.attn_jgnn_model import AttnJGNN
from models.wrapper import AttnJGNNWrapper

logger = logging.getLogger(__name__)


def create_dataloaders(cfg: DictConfig):
    """Create train/val/test dataloaders."""
    data_cfg = cfg.data
    training_cfg = cfg.training

    common_args = dict(
        root=data_cfg.root,
        categories=list(data_cfg.get('categories', ['rnd3sat'])),
        use_join_graph=True,  # Required for Attn-JGNN
        feature_dim=cfg.model.feature_dim,
        variable_range=dict(data_cfg.get('variable_range', {'min': 100, 'max': 600})),
        min_instances_per_category=data_cfg.get('min_instances_per_category', 100),
        seed=cfg.get('seed', 42),
    )

    train_dataset = SATLIBDataset(split='train', **common_args)
    val_dataset = SATLIBDataset(split='val', **common_args)
    test_dataset = SATLIBDataset(split='test', **common_args)

    loader_args = dict(
        batch_size=training_cfg.get('batch_size', 1),
        num_workers=training_cfg.get('num_workers', 0),
        pin_memory=True,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def create_model(cfg: DictConfig) -> AttnJGNNWrapper:
    """Create Attn-JGNN model with Lightning wrapper."""
    model_cfg = cfg.model

    model = AttnJGNN(
        feature_dim=model_cfg.feature_dim,
        num_iterations=model_cfg.num_iterations,
        initial_heads=model_cfg.initial_heads,
        max_heads=model_cfg.max_heads,
        head_increase_interval=model_cfg.head_increase_interval,
        constraint_gamma=model_cfg.constraint_gamma,
        mlp_hidden_dim=model_cfg.mlp_hidden_dim,
        use_constraint_aware=model_cfg.get('use_constraint_aware', True),
        use_attention=model_cfg.get('use_attention', True),
    )

    # Merge model and training configs for wrapper
    # Enable struct flag to False to allow adding new keys during merge
    OmegaConf.set_struct(model_cfg, False)
    wrapper_config = OmegaConf.merge(model_cfg, cfg.training)
    wrapper = AttnJGNNWrapper(model=model, config=wrapper_config)

    return wrapper


def create_callbacks(cfg: DictConfig):
    """Create training callbacks."""
    training_cfg = cfg.training

    callbacks = [
        ModelCheckpoint(
            monitor='val/rmse',
            mode='min',
            save_top_k=1,
            filename='best-{epoch:02d}-{val/rmse:.4f}',
            save_last=True,
        ),
        EarlyStopping(
            monitor='val/rmse',
            mode='min',
            patience=training_cfg.get('early_stopping_patience', 20),
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    return callbacks


@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig):
    """Main training function."""
    # Print config
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    seed = cfg.get('seed', 42)
    pl.seed_everything(seed, workers=True)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(cfg)

    # Check if we have data
    if len(train_loader.dataset) == 0:
        logger.warning("No training data found. Please ensure SATLIB data is available.")
        logger.info(f"Expected data location: {cfg.data.root}")
        logger.info("Run label generation first if needed.")
        return

    # Create model
    wrapper = create_model(cfg)
    logger.info(f"Model created with {sum(p.numel() for p in wrapper.parameters())} parameters")

    # Create callbacks
    callbacks = create_callbacks(cfg)

    # Create logger
    tb_logger = TensorBoardLogger(
        save_dir='.',
        name='logs',
        default_hp_metric=False,
    )

    # Create trainer
    training_cfg = cfg.training
    
    # Check if GPU is actually available
    import torch
    use_gpu = training_cfg.get('gpus', 0) > 0 and torch.cuda.is_available()
    
    trainer = pl.Trainer(
        max_epochs=training_cfg.get('max_epochs', 200),
        accelerator='gpu' if use_gpu else 'cpu',
        devices=training_cfg.get('gpus', 1) if use_gpu else 'auto',
        precision=training_cfg.get('precision', 32),
        gradient_clip_val=training_cfg.get('gradient_clip_val', None),
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=10,
        deterministic=True,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(wrapper, train_loader, val_loader)

    # Test
    logger.info("Running test evaluation...")
    test_results = trainer.test(wrapper, test_loader, ckpt_path='best')

    logger.info(f"Test results: {test_results}")

    return test_results


if __name__ == '__main__':
    train()
