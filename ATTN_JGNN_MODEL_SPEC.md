# Attn-JGNN Model & Training Specification

---
AGENT SCOPE: Implementing ONLY model, training, loss components. Dataset exists in data/.

Repository structure:
- conf/ - Hydra yaml configs
- data/ - dataset implementation (EXISTS - read first, modify only if incompatible)
- models/ - model architecture + Lightning wrapper (CREATE)
- losses/ - custom loss functions (CREATE if needed)
- train.py - training entry point (CREATE)

Tech stack: PyTorch Lightning, Hydra, PyTorch Geometric (if applicable)

BEFORE IMPLEMENTING: Read data/ files to understand batch structure, shapes, types.
---

## SECTION 1 - EXPERIMENT OVERVIEW

**Attn-JGNN (Attention Enhanced Join-Graph Neural Networks)** is a specialized neural network model for solving #SAT (Model Counting) problems. It is a Graph Neural Network that:

1. Encodes CNF formulas as factor graphs
2. Uses tree decomposition to create join-graphs with cluster structure
3. Applies hierarchical attention mechanisms (intra-cluster and inter-cluster)
4. Performs iterative message passing inspired by IJGP (Iterative Join-Graph Propagation)
5. Estimates the partition function (log model count) via an MLP readout

**Learning objective**: Regression - predict log(Z) where Z is the model count (number of satisfying assignments)

**Data**: RND3SAT from SATLIB benchmark (uniform random 3-SAT on phase transition region). Variables range from 100-600. Dataset split: 60%/20%/20% (train/val/test).

**Metric**: RMSE between estimated log countings and ground truth

---

## SECTION 2 - MODEL ARCHITECTURE

### Model Type
Hierarchical Graph Attention Network with two-level message passing:
- **GAT1 (Intra-cluster)**: Message passing between variables and clauses within clusters
- **GAT2 (Inter-cluster)**: Message passing between clusters via shared variables
- **MLP**: Final readout layer to predict log(Z)

### Architecture Overview

```
Input: Join-graph (from tree decomposition of CNF factor graph)
       ├── Variable features: x_var [num_vars, d=64]
       └── Clause features: x_clause [num_clauses, d=64]

For t = 1 to T (message-passing iterations):
    ├── GAT1: Intra-cluster attention (variable-clause within clusters)
    │         LeakyReLU activation
    └── GAT2: Inter-cluster attention (cross-cluster via shared variables)
              LeakyReLU activation

Pooling: Mean pooling over cluster features to get global representation h_G
Output: MLP(h_G) → logZ prediction (scalar)
```

### Layer Specifications

#### Feature Dimension
- **d = 64** (feature dimension for all node embeddings)

#### Message-Passing Iterations
- **T = 5** iterations of GAT1 + GAT2

#### GAT1 (Intra-Cluster Attention)
Computes attention between variables and clauses within the same cluster.

Attention weight formula:
```
α_intra = LeakyReLU((W_Q * h_i)^T * (W_K * h_j) / sqrt(d))
```

For constraint-aware modification (adds clause satisfaction score):
```
α_intra = LeakyReLU((W_Q * h_i)^T * (W_K * h_j) + γ * s_i) / sqrt(d))
```

Where:
- W_Q, W_K, W_V are learnable weight matrices
- h_i, h_j are node features
- s_i is clause satisfaction score (see constraint-aware mechanism)
- γ is a hyperparameter for constraint awareness

Feature update:
```
h_j = sum_{x_i in C_k} α_intra * W_V * h_i
```

#### GAT2 (Inter-Cluster Attention)
Computes attention between clusters through shared variables.

Attention weight formula:
```
α_inter = LeakyReLU((W_Q * h_C1)^T * (W_K * h_C2) / sqrt(d))
```

Shared variable feature update:
```
h_x = h_x^(C1) + α_inter * W_V * h_x^(C2)
```

#### Attention Heads (Dynamic)
- **H_init = 4** (initial number of attention heads)
- **H_max = 8** (maximum number of attention heads)
- Attention heads increase by 1 every 1000 training steps
- Dynamic formula: `H(t) = min(H_max, H_init + floor(t/T))`
- Each head has learnable weight λ_h for its contribution
- Multi-head aggregation: `α_dy = (1/H(t)) * sum_{h=1}^{H(t)} λ_h * Attention(Q, K, V)`

#### MLP Readout Layer
- **Input**: h_G (pooled cluster representation, dim=2 based on Bethe free energy components)
- **Hidden**: d=64 neurons with ReLU activation
- **Output**: 1 scalar (logZ prediction)

Architecture:
```
h_C_α = [H(b_C_α), sum_{v in C_α} (d_v^α - 1) * H(b_v)]  # dim=2 per cluster
h_G = (1/|C_α|) * sum_α h_C_α  # Global representation (dim=2)
F_Bethe_Join = W2 * ReLU(W1 * h_G + b1) + b2
logZ = -F_Bethe_Join
```

Where:
- W1 ∈ R^{d×2}, b1 ∈ R^d
- W2 ∈ R^{1×d}, b2 ∈ R
- H(b_C_α) = GAT1 output (cluster entropy estimate)
- H(b_v) = GAT2 output (variable entropy estimate)

### Activations
- **GAT layers**: LeakyReLU for attention computation
- **Feature update aggregation**: Implicit (weighted sum)
- **MLP hidden layer**: ReLU
- **Pooling output**: Tanh (as shown in Figure 1)
- **Final output**: Softmax for normalization, then linear

### Normalization
Not explicitly mentioned in paper. Suggest: None or LayerNorm after attention aggregation.

### Dropout
Not explicitly specified in paper.

### Residual Connections
Not explicitly specified in paper.

### Pooling
- **Mean pooling**: Over cluster features to obtain global graph representation
- Pooling is applied after GAT message passing to compress variable and clause node features

### Input/Output Format
- **Input**:
  - `x_var`: [num_variables, 64] - Variable node features (initialized to zeros/learnable)
  - `x_clause`: [num_clauses, 64] - Clause node features (self-identifying initialization)
  - `var_clause_edge_index`: [2, num_edges] - Bipartite edges between variables and clauses
  - `edge_polarity`: [num_edges, 1] - Polarity (+1/-1) of each literal in clause
  - `cluster_edge_index`: [2, num_cluster_edges] - Edges between clusters
  - `cluster_var_ids`: List[List[int]] - Variable indices per cluster
  - `cluster_clause_ids`: List[List[int]] - Clause indices per cluster
  - `shared_vars`: List[List[int]] - Shared variables for each cluster edge

- **Output**:
  - `logZ`: [batch_size, 1] - Predicted log model count (scalar per graph)

### Parameter Count
Not explicitly mentioned in paper.

---

## SECTION 3 - LOSS FUNCTIONS

### Primary Loss: RMSE
Root Mean Square Error between predicted and ground truth log model counts:
```python
L_RMSE = sqrt(mean((logZ_pred - logZ_true)^2))
```

### Auxiliary Loss: Constraint-Aware Regularization
Encourages the model to satisfy clause constraints. For each clause φ_i, compute satisfaction score:

```python
s_i = sigmoid(sum_{x_j in φ_i} (2*b_j(x_j) - 1) * polarity(x_j, φ_i))
```

Where:
- `b_j(x_j)` is the current assignment probability of variable x_j
- `polarity(x_j, φ_i)` is +1 if x_j appears positive in φ_i, -1 if negated
- `s_i ∈ (0, 1)` - closer to 1 means clause is more likely satisfied

Regularization term:
```python
L_cons = -δ * sum_{i=1}^{m} ln(s_i)
```

Where:
- δ is the constraint-aware coefficient (hyperparameter)
- m is the number of clauses

### Combined Loss Formula
```python
L_total = L_RMSE + L_cons
```

### Special Considerations
- The constraint-aware mechanism also modifies attention weights (see Section 2)
- Satisfaction scores s_i are used to weight messages during propagation

---

## SECTION 4 - TRAINING PROCEDURE

### Optimizer
- **Type**: Not explicitly specified
- **Suggested**: Adam optimizer (standard for GNN training)

### Learning Rate
- Not explicitly specified in paper

### LR Schedule
- Not explicitly specified

### Duration
- Not explicitly specified (paper mentions "training time/convergence" in ablation table)
- Ablation shows convergence times around 113-185 time units

### Early Stopping
- Not explicitly specified

### Batch Size
- Not explicitly specified

### Gradient Clipping
- Not explicitly specified

### Checkpointing
- **Metric**: RMSE (lower is better)
- **Mode**: min

### Hardware
- **GPU**: NVIDIA A100 (single)
- **CPU**: 8 cores

### Precision
- Not explicitly specified (suggest: float32)

### Random Seed
- Not explicitly specified

---

## SECTION 5 - IMPLEMENTATION INSTRUCTIONS

### models/attn_jgnn_model.py

```python
class IntraClusterGAT(nn.Module):
    """
    GAT1: Intra-cluster attention for variable-clause message passing.

    __init__ params:
        in_dim: int = 64 (input feature dimension)
        out_dim: int = 64 (output feature dimension)
        num_heads: int = 4 (number of attention heads, will be dynamic)
        negative_slope: float = 0.2 (LeakyReLU negative slope)
        use_constraint_aware: bool = True
        constraint_gamma: float = 1.0 (constraint awareness coefficient)

    Layer definitions:
        - W_Q: Linear(in_dim, out_dim * num_heads)
        - W_K: Linear(in_dim, out_dim * num_heads)
        - W_V: Linear(in_dim, out_dim * num_heads)
        - head_weights: Parameter(num_heads) - learnable λ_h weights

    forward(x_var, x_clause, cluster_var_ids, cluster_clause_ids,
            edge_polarity, satisfaction_scores=None):
        For each cluster:
            1. Extract variable and clause features in cluster
            2. Compute Q, K, V projections
            3. Compute attention: α = LeakyReLU(Q @ K.T / sqrt(d))
            4. If constraint-aware: α += γ * s_i
            5. Apply softmax normalization
            6. Aggregate: h_new = α @ V
            7. Multi-head: weighted average with λ_h
        Return updated features
```

```python
class InterClusterGAT(nn.Module):
    """
    GAT2: Inter-cluster attention for cross-cluster message passing.

    __init__ params:
        in_dim: int = 64
        out_dim: int = 64
        num_heads: int = 4
        negative_slope: float = 0.2

    Layer definitions:
        - W_Q: Linear(in_dim, out_dim * num_heads)
        - W_K: Linear(in_dim, out_dim * num_heads)
        - W_V: Linear(in_dim, out_dim * num_heads)
        - head_weights: Parameter(num_heads)

    forward(cluster_features, cluster_edge_index, shared_vars, x_var):
        1. For each cluster edge (C1, C2):
           - Get cluster representations (mean of node features)
           - Compute attention α_inter between clusters
        2. For shared variables between clusters:
           - Update: h_x = h_x^(C1) + α_inter * W_V * h_x^(C2)
        Return updated variable features
```

```python
class AttnJGNN(nn.Module):
    """
    Main Attn-JGNN model.

    __init__ params:
        feature_dim: int = 64
        num_iterations: int = 5 (T - message passing iterations)
        initial_heads: int = 4 (H_init)
        max_heads: int = 8 (H_max)
        head_increase_interval: int = 1000 (steps between head increases)
        constraint_gamma: float = 1.0 (δ for constraint awareness)
        mlp_hidden_dim: int = 64

    Layer definitions:
        - gat1: IntraClusterGAT(feature_dim, feature_dim, initial_heads)
        - gat2: InterClusterGAT(feature_dim, feature_dim, initial_heads)
        - mlp: Sequential(
            Linear(2, mlp_hidden_dim),
            ReLU(),
            Linear(mlp_hidden_dim, 1)
          )

    forward(batch):
        # batch contains: x_var, x_clause, var_clause_edge_index,
        #                 edge_polarity, cluster_edge_index,
        #                 cluster_var_ids, cluster_clause_ids, shared_vars

        1. Initialize features:
           h_var = batch.x_var  # [num_vars, 64]
           h_clause = batch.x_clause  # [num_clauses, 64]

        2. For t in range(num_iterations):
           # Compute satisfaction scores for constraint awareness
           s = compute_satisfaction_scores(h_var, batch)

           # GAT1: Intra-cluster message passing
           h_var, h_clause = gat1(h_var, h_clause, cluster_var_ids,
                                   cluster_clause_ids, edge_polarity, s)

           # GAT2: Inter-cluster message passing
           h_var = gat2(cluster_features, cluster_edge_index,
                        shared_vars, h_var)

        3. Compute Bethe free energy approximation:
           # For each cluster, compute entropy terms
           H_cluster = gat1_output_pooled  # Cluster joint entropy
           H_var = gat2_output  # Variable entropy

           # Pool to graph-level
           h_C = [H_cluster, sum (d_v - 1) * H_var]  # Per cluster
           h_G = mean(h_C)  # Global representation [2]

        4. MLP prediction:
           logZ = -mlp(h_G)

        Return logZ

    update_num_heads(global_step):
        # Dynamic attention head adjustment
        new_heads = min(max_heads, initial_heads + global_step // head_increase_interval)
        # Update GAT layers if needed
```

```python
def compute_satisfaction_scores(h_var, batch):
    """
    Compute clause satisfaction scores for constraint-aware attention.

    Args:
        h_var: Variable features [num_vars, d]
        batch: Data batch with edge information

    Returns:
        s: Satisfaction scores [num_clauses]
    """
    # Get assignment probabilities from variable features
    b = torch.sigmoid(h_var.mean(dim=-1))  # [num_vars]

    # For each clause, compute satisfaction score
    # s_i = sigmoid(sum_{x_j in φ_i} (2*b_j - 1) * polarity(x_j, φ_i))

    s = []
    for clause_idx in range(batch.num_clauses):
        # Get variables and polarities for this clause
        mask = batch.var_clause_edge_index[1] == clause_idx
        var_ids = batch.var_clause_edge_index[0, mask]
        polarities = batch.edge_polarity[mask].squeeze()

        # Compute satisfaction score
        score = torch.sigmoid(((2 * b[var_ids] - 1) * polarities).sum())
        s.append(score)

    return torch.stack(s)
```

### models/wrapper.py

```python
class AttnJGNNWrapper(pl.LightningModule):
    """
    PyTorch Lightning wrapper for Attn-JGNN.

    __init__(model, config):
        self.model = model
        self.config = config
        self.save_hyperparameters(config)
        self.constraint_delta = config.get('constraint_delta', 0.1)

    forward(batch):
        return self.model(batch)

    training_step(batch, batch_idx):
        # Forward pass
        logZ_pred = self(batch)
        logZ_true = batch.y

        # RMSE loss
        rmse_loss = torch.sqrt(F.mse_loss(logZ_pred, logZ_true))

        # Constraint-aware loss
        s = compute_satisfaction_scores(self.model.h_var, batch)
        cons_loss = -self.constraint_delta * torch.log(s + 1e-8).sum()

        # Total loss
        loss = rmse_loss + cons_loss

        # Update dynamic attention heads
        self.model.update_num_heads(self.global_step)

        # Logging
        self.log('train/loss', loss)
        self.log('train/rmse', rmse_loss)
        self.log('train/cons_loss', cons_loss)

        return loss

    validation_step(batch, batch_idx):
        logZ_pred = self(batch)
        logZ_true = batch.y

        rmse = torch.sqrt(F.mse_loss(logZ_pred, logZ_true))

        self.log('val/rmse', rmse)
        return rmse

    test_step(batch, batch_idx):
        # Same as validation
        logZ_pred = self(batch)
        logZ_true = batch.y

        rmse = torch.sqrt(F.mse_loss(logZ_pred, logZ_true))

        self.log('test/rmse', rmse)
        return rmse

    configure_optimizers():
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.get('lr', 1e-3),
            weight_decay=self.config.get('weight_decay', 0)
        )

        # Optional: LR scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/rmse'
            }
        }
```

### losses/constraint_loss.py (if custom needed)

```python
def constraint_aware_loss(satisfaction_scores, delta=0.1):
    """
    Constraint-aware regularization loss.

    Args:
        satisfaction_scores: Tensor [num_clauses] of s_i values in (0, 1)
        delta: Coefficient for constraint awareness

    Returns:
        L_cons: Scalar loss value
    """
    # L_cons = -δ * sum_{i=1}^{m} ln(s_i)
    return -delta * torch.log(satisfaction_scores + 1e-8).sum()


def compute_satisfaction_scores(var_probs, var_clause_edges, polarities, num_clauses):
    """
    Compute clause satisfaction scores.

    Args:
        var_probs: Variable assignment probabilities [num_vars]
        var_clause_edges: Edge indices [2, num_edges]
        polarities: Polarity for each edge [num_edges]
        num_clauses: Number of clauses

    Returns:
        s: Satisfaction scores [num_clauses]
    """
    s = torch.zeros(num_clauses, device=var_probs.device)

    for clause_idx in range(num_clauses):
        mask = var_clause_edges[1] == clause_idx
        var_ids = var_clause_edges[0, mask]
        pols = polarities[mask]

        # s_i = sigmoid(sum (2*b_j - 1) * polarity_j)
        s[clause_idx] = torch.sigmoid(
            ((2 * var_probs[var_ids] - 1) * pols).sum()
        )

    return s
```

### train.py

```python
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data.datasets.satlib_dataset import SATLIBDataset
from torch_geometric.loader import DataLoader
from models.attn_jgnn_model import AttnJGNN
from models.wrapper import AttnJGNNWrapper


@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig):
    # Set seed for reproducibility
    pl.seed_everything(cfg.get('seed', 42))

    # Create datasets
    train_dataset = SATLIBDataset(
        root=cfg.data.root,
        categories=cfg.data.get('categories', ['rnd3sat']),
        split='train',
        use_join_graph=True,  # IMPORTANT: Use join-graph for Attn-JGNN
        feature_dim=cfg.model.feature_dim,
    )

    val_dataset = SATLIBDataset(
        root=cfg.data.root,
        categories=cfg.data.get('categories', ['rnd3sat']),
        split='val',
        use_join_graph=True,
        feature_dim=cfg.model.feature_dim,
    )

    test_dataset = SATLIBDataset(
        root=cfg.data.root,
        categories=cfg.data.get('categories', ['rnd3sat']),
        split='test',
        use_join_graph=True,
        feature_dim=cfg.model.feature_dim,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.get('num_workers', 4),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.get('num_workers', 4),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.get('num_workers', 4),
    )

    # Create model
    model = AttnJGNN(
        feature_dim=cfg.model.feature_dim,
        num_iterations=cfg.model.num_iterations,
        initial_heads=cfg.model.initial_heads,
        max_heads=cfg.model.max_heads,
        head_increase_interval=cfg.model.head_increase_interval,
        constraint_gamma=cfg.model.constraint_gamma,
        mlp_hidden_dim=cfg.model.mlp_hidden_dim,
    )

    # Create Lightning wrapper
    wrapper = AttnJGNNWrapper(model, cfg)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val/rmse',
            mode='min',
            save_top_k=1,
            filename='best-{epoch}-{val/rmse:.4f}',
        ),
        EarlyStopping(
            monitor='val/rmse',
            mode='min',
            patience=cfg.training.get('early_stopping_patience', 20),
        ),
    ]

    # Logger
    logger = TensorBoardLogger(save_dir='logs', name='attn_jgnn')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='gpu' if cfg.training.get('gpus', 1) > 0 else 'cpu',
        devices=cfg.training.get('gpus', 1),
        precision=cfg.training.get('precision', 32),
        gradient_clip_val=cfg.training.get('gradient_clip_val', None),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(wrapper, train_loader, val_loader)

    # Test
    trainer.test(wrapper, test_loader, ckpt_path='best')


if __name__ == '__main__':
    train()
```

### Config Files

#### conf/model/attn_jgnn.yaml

```yaml
# Attn-JGNN Model Configuration
_target_: models.attn_jgnn_model.AttnJGNN

feature_dim: 64
num_iterations: 5  # T - message passing iterations
initial_heads: 4   # H_init
max_heads: 8       # H_max
head_increase_interval: 1000  # Steps between head increases
constraint_gamma: 1.0  # γ for constraint-aware attention
mlp_hidden_dim: 64
```

#### conf/training/attn_jgnn.yaml

```yaml
# Training Configuration
batch_size: 32  # Not specified in paper - suggest starting value
max_epochs: 200  # Not specified - suggest starting value
lr: 0.001  # Not specified - suggest Adam default
weight_decay: 0.0  # Not specified
constraint_delta: 0.1  # δ for constraint-aware loss

# Hardware
gpus: 1
precision: 32
num_workers: 4

# Optimization
gradient_clip_val: null  # Not specified
early_stopping_patience: 20

# Scheduler (optional)
use_scheduler: true
scheduler_factor: 0.5
scheduler_patience: 10
```

#### conf/data/satlib.yaml

```yaml
# SATLIB Dataset Configuration
root: data/satlib
categories:
  - rnd3sat  # Target benchmark for this implementation

split_ratio:
  train: 0.6
  val: 0.2
  test: 0.2

variable_range:
  min: 100
  max: 600

min_instances_per_category: 100
seed: 42
```

#### conf/config.yaml

```yaml
defaults:
  - model: attn_jgnn
  - training: attn_jgnn
  - data: satlib

seed: 42

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

---

## SECTION 6 - DATA INTERFACE REQUIREMENTS

Based on reading the existing data/ files, the dataset provides:

### Required Batch Fields (for join-graph mode)

| Field | Shape | Dtype | Purpose |
|-------|-------|-------|---------|
| `x_var` | [num_vars, 64] | float32 | Variable node features |
| `x_clause` | [num_clauses, 64] | float32 | Clause node features |
| `var_clause_edge_index` | [2, num_edges] | long | Bipartite edges (var_idx, clause_idx) |
| `edge_polarity` | [num_edges, 1] | float32 | Literal polarity (+1/-1) |
| `cluster_edge_index` | [2, num_cluster_edges] | long | Inter-cluster edges |
| `cluster_var_ids` | List[List[int]] | Python list | Variable indices per cluster |
| `cluster_clause_ids` | List[List[int]] | Python list | Clause indices per cluster |
| `shared_vars` | List[List[int]] | Python list | Shared variables per cluster edge |
| `num_variables` | int | - | Number of variables |
| `num_clauses` | int | - | Number of clauses |
| `num_clusters` | int | - | Number of clusters |
| `y` | [1] | float32 | Ground truth log(model_count) |

### Compatibility Notes

The existing dataset implementation (`data/datasets/satlib_dataset.py`) already supports:
- Join-graph mode via `use_join_graph=True` parameter
- Feature dimension configuration via `feature_dim` parameter
- All required fields are generated by `join_graph_to_pyg()` function

**No dataset modifications required** - the existing implementation is compatible with Attn-JGNN requirements.

### Collation
PyTorch Geometric's default `DataLoader` handles batching of graph data automatically. The model should handle variable-sized graphs in a batch.

---

## SECTION 7 - VERIFICATION CHECKLIST

| Parameter | Value from Paper | Notes |
|-----------|------------------|-------|
| Feature dimension d | 64 | Explicit |
| Message-passing iterations T | 5 | Explicit |
| Initial attention heads H_init | 4 | Explicit |
| Maximum attention heads H_max | 8 | Explicit |
| Head increase interval | 1000 steps | Explicit |
| GAT layers | 2 (GAT1 + GAT2) | Explicit |
| MLP layers | 1 hidden layer | Explicit (from equations) |
| Activation (attention) | LeakyReLU | Explicit |
| Activation (MLP) | ReLU | Explicit |
| Loss function | RMSE + Constraint-aware | Explicit |
| Hardware | NVIDIA A100, 8 CPU cores | Explicit |
| Target benchmark | RND3SAT from SATLIB | Explicit |
| Dataset split | 60/20/20 | Explicit |
| Metric | RMSE | Explicit |
| Expected RND3SAT RMSE | 1.15 | From Table 2 |

---

## SECTION 8 - MISSING INFORMATION

The following hyperparameters/details are **not explicitly specified** in the paper:

| Missing Parameter | Suggested Default | Rationale |
|-------------------|-------------------|-----------|
| Optimizer type | Adam | Standard for GNN training |
| Learning rate | 1e-3 | Common Adam default |
| Weight decay | 0 | Start without regularization |
| Batch size | 32 | Common starting point |
| Number of epochs | 200 | Sufficient for convergence |
| Dropout rate | 0 | Not mentioned |
| Gradient clipping | None | Not mentioned |
| Early stopping patience | 20 | Reasonable default |
| Constraint delta δ | 0.1 | Needs tuning |
| Constraint gamma γ | 1.0 | Needs tuning |
| MLP hidden dimension | 64 | Match feature_dim |
| LeakyReLU negative slope | 0.2 | PyTorch default |
| Random seed | 42 | Common default |

### Implementation Decisions Required

1. **Variable probability extraction**: The paper mentions `b_j(x_j)` (assignment probability) but doesn't specify how to extract this from features. Suggest: sigmoid of mean-pooled feature or learned projection.

2. **Cluster feature aggregation**: The paper mentions cluster representations but doesn't specify exactly how to aggregate node features to cluster level. Suggest: mean pooling over nodes in cluster.

3. **Bethe free energy approximation**: The MLP input dimension is derived from the Bethe formula but exact computation needs implementation. The formula suggests a 2-dimensional input (cluster entropy + weighted variable entropy sum).

4. **Dynamic head update mechanism**: Need to implement mechanism to increase attention heads during training and handle weight matrix resizing or head pruning.

5. **Softmax temperature**: Not specified for attention normalization.

---

## Additional Implementation Notes

### Tree Decomposition
The existing `tree_decomposition.py` provides FlowCutter integration with fallback heuristic. The paper mentions using FlowCutter for tree decomposition - this is already implemented.

### Constraint-Aware Integration
The constraint-aware mechanism affects:
1. Loss function (regularization term)
2. Attention weights (additive term γ*s_i)

Both need to be implemented together for full effect.

### Dynamic Attention Heads
The dynamic mechanism requires:
1. Tracking global training step
2. Computing current number of heads: `H(t) = min(8, 4 + t//1000)`
3. Learnable weights λ_h for each head
4. Proper handling when adding new heads during training

### Expected Performance
On RND3SAT benchmark:
- NSNet baseline: RMSE = 1.57
- Attn-JGNN: RMSE = 1.15 (target)
- Improvement: ~27% reduction in RMSE
