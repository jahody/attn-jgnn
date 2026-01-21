"""
PyTorch Lightning wrapper for Attn-JGNN model.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from omegaconf import DictConfig

from .attn_jgnn_model import AttnJGNN


class AttnJGNNWrapper(pl.LightningModule):
    """
    PyTorch Lightning wrapper for Attn-JGNN.

    Handles training, validation, testing loops and optimization configuration.
    """

    def __init__(
        self,
        model: Optional[AttnJGNN] = None,
        config: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__()

        # Handle config
        if config is not None:
            self.config = config
            self.save_hyperparameters(dict(config) if hasattr(config, 'items') else config)
        else:
            self.config = DictConfig(kwargs)
            self.save_hyperparameters(kwargs)

        # Create model if not provided
        if model is None:
            model = AttnJGNN(
                feature_dim=self.config.get('feature_dim', 64),
                num_iterations=self.config.get('num_iterations', 5),
                initial_heads=self.config.get('initial_heads', 4),
                max_heads=self.config.get('max_heads', 8),
                head_increase_interval=self.config.get('head_increase_interval', 1000),
                constraint_gamma=self.config.get('constraint_gamma', 1.0),
                mlp_hidden_dim=self.config.get('mlp_hidden_dim', 64),
                use_constraint_aware=self.config.get('use_constraint_aware', True),
                use_attention=self.config.get('use_attention', True),
            )
        self.model = model

        # Loss configuration
        self.constraint_delta = self.config.get('constraint_delta', 0.1)

    def forward(self, batch):
        return self.model(batch)

    def _compute_loss(self, batch, logZ_pred):
        """Compute total loss (RMSE + constraint-aware regularization)."""
        logZ_true = batch.y

        # RMSE loss
        mse = F.mse_loss(logZ_pred, logZ_true)
        rmse_loss = torch.sqrt(mse + 1e-8)

        # Constraint-aware loss
        cons_loss = torch.tensor(0.0, device=logZ_pred.device)
        if self.model.satisfaction_scores is not None and self.constraint_delta > 0:
            s = self.model.satisfaction_scores
            # L_cons = -δ * sum ln(s_i)
            cons_loss = -self.constraint_delta * torch.log(s + 1e-8).mean()

        # Total loss
        total_loss = rmse_loss + cons_loss

        return total_loss, rmse_loss, cons_loss

    def training_step(self, batch, batch_idx):
        # Update dynamic attention heads
        self.model.update_num_heads(self.global_step)

        # Forward pass
        logZ_pred = self(batch)

        # Compute loss
        loss, rmse_loss, cons_loss = self._compute_loss(batch, logZ_pred)

        # Logging
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/rmse', rmse_loss)
        self.log('train/cons_loss', cons_loss)
        self.log('train/active_heads', float(self.model.current_heads.item()))

        return loss

    def validation_step(self, batch, batch_idx):
        logZ_pred = self(batch)
        logZ_true = batch.y

        # Compute RMSE
        mse = F.mse_loss(logZ_pred, logZ_true)
        rmse = torch.sqrt(mse + 1e-8)

        self.log('val/rmse', rmse, prog_bar=True)

        return rmse

    def test_step(self, batch, batch_idx):
        logZ_pred = self(batch)
        logZ_true = batch.y

        # Compute RMSE
        mse = F.mse_loss(logZ_pred, logZ_true)
        rmse = torch.sqrt(mse + 1e-8)

        self.log('test/rmse', rmse)

        return rmse

    def configure_optimizers(self):
        # Optimizer
        lr = self.config.get('lr', 1e-3)
        weight_decay = self.config.get('weight_decay', 0.0)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # LR Scheduler (optional)
        use_scheduler = self.config.get('use_scheduler', True)
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config.get('scheduler_factor', 0.5),
                patience=self.config.get('scheduler_patience', 10),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/rmse',
                    'interval': 'epoch',
                },
            }

        return optimizer
