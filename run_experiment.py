"""
Run Attn-JGNN experiment on synthetic RND3SAT data.

This script:
1. Loads synthetic CNF data
2. Processes into join-graph format
3. Trains Attn-JGNN (with or without attention)
4. Reports RMSE on test set
"""

import os
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import logging

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessing.cnf_parser import parse_dimacs
from data.preprocessing.graph_builder import build_factor_graph, build_join_graph
from data.preprocessing.tree_decomposition import decompose_cnf
from models.attn_jgnn_model import AttnJGNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CNFDataset(Dataset):
    """Simple dataset for CNF files with labels."""

    def __init__(
        self,
        cnf_dir: str,
        labels_file: str,
        feature_dim: int = 64,
        use_join_graph: bool = True,
        max_instances: Optional[int] = None,
    ):
        self.feature_dim = feature_dim
        self.use_join_graph = use_join_graph

        # Load labels
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)

        # Find CNF files with labels
        cnf_path = Path(cnf_dir)
        self.files = []
        for cnf_file in cnf_path.rglob("*.cnf"):
            if cnf_file.name in self.labels:
                self.files.append(cnf_file)

        if max_instances:
            self.files = self.files[:max_instances]

        logger.info(f"Loaded {len(self.files)} CNF files with labels")

        # Pre-process all files
        self.data_list = []
        for cnf_file in tqdm(self.files, desc="Processing CNF files"):
            try:
                data = self._process_file(cnf_file)
                if data is not None:
                    self.data_list.append(data)
            except Exception as e:
                logger.warning(f"Failed to process {cnf_file}: {e}")

    def _process_file(self, cnf_file: Path) -> Optional[Data]:
        """Process a single CNF file into PyG Data."""
        # Parse CNF
        cnf = parse_dimacs(cnf_file)

        # Build factor graph
        factor_graph = build_factor_graph(cnf)

        # Get label
        label = self.labels[cnf_file.name]
        if math.isinf(label):
            return None  # Skip UNSAT instances

        num_vars = factor_graph.num_variables
        num_clauses = factor_graph.num_clauses

        # Initialize features
        x_var = torch.zeros(num_vars, self.feature_dim)
        x_clause = torch.zeros(num_clauses, self.feature_dim)
        for i in range(num_clauses):
            x_clause[i, i % self.feature_dim] = 1.0

        # Build edges
        var_ids = [e[0] for e in factor_graph.edges]
        clause_ids = [e[1] for e in factor_graph.edges]
        var_clause_edge_index = torch.tensor([var_ids, clause_ids], dtype=torch.long)
        edge_polarity = torch.tensor(factor_graph.polarities, dtype=torch.float).unsqueeze(-1)

        if self.use_join_graph:
            # Tree decomposition
            try:
                td = decompose_cnf(cnf)
                join_graph = build_join_graph(factor_graph, td.bags, td.tree_edges)

                cluster_var_ids = join_graph.cluster_var_ids
                cluster_clause_ids = join_graph.cluster_clause_ids

                if join_graph.cluster_edges:
                    cluster_src = [e[0] for e in join_graph.cluster_edges]
                    cluster_dst = [e[1] for e in join_graph.cluster_edges]
                    cluster_edge_index = torch.tensor(
                        [cluster_src + cluster_dst, cluster_dst + cluster_src],
                        dtype=torch.long
                    )
                else:
                    cluster_edge_index = torch.zeros((2, 0), dtype=torch.long)

                shared_vars = join_graph.shared_vars
            except Exception as e:
                logger.warning(f"Tree decomposition failed, using single cluster: {e}")
                # Fallback: single cluster with all nodes
                cluster_var_ids = [list(range(num_vars))]
                cluster_clause_ids = [list(range(num_clauses))]
                cluster_edge_index = torch.zeros((2, 0), dtype=torch.long)
                shared_vars = []
        else:
            # No join-graph, single cluster
            cluster_var_ids = [list(range(num_vars))]
            cluster_clause_ids = [list(range(num_clauses))]
            cluster_edge_index = torch.zeros((2, 0), dtype=torch.long)
            shared_vars = []

        data = Data(
            x_var=x_var,
            x_clause=x_clause,
            var_clause_edge_index=var_clause_edge_index,
            edge_polarity=edge_polarity,
            cluster_edge_index=cluster_edge_index,
            cluster_var_ids=cluster_var_ids,
            cluster_clause_ids=cluster_clause_ids,
            shared_vars=shared_vars,
            num_variables=num_vars,
            num_clauses=num_clauses,
            y=torch.tensor([[label]], dtype=torch.float),
        )

        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_fn(batch):
    """Custom collate that returns single items (batch_size=1)."""
    return batch[0]


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_rmse = 0
    count = 0

    for batch in dataloader:
        # Move to device
        batch = batch.to(device)

        optimizer.zero_grad()

        # Forward
        logZ_pred = model(batch)
        logZ_true = batch.y

        # RMSE loss
        mse = F.mse_loss(logZ_pred, logZ_true)
        loss = torch.sqrt(mse + 1e-8)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rmse += loss.item()
        count += 1

    return total_loss / count, total_rmse / count


def evaluate(model, dataloader, device):
    """Evaluate model on dataset."""
    model.eval()
    total_mse = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            logZ_pred = model(batch)
            logZ_true = batch.y

            mse = F.mse_loss(logZ_pred, logZ_true, reduction='sum')
            total_mse += mse.item()
            count += logZ_true.numel()

    rmse = math.sqrt(total_mse / count)
    return rmse


def run_experiment(
    cnf_dir: str = "data/satlib/raw/rnd3sat",
    labels_file: str = "data/satlib/processed/labels.json",
    use_attention: bool = False,
    num_epochs: int = 50,
    lr: float = 0.001,
    seed: int = 42,
):
    """Run full experiment."""
    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = CNFDataset(cnf_dir, labels_file, use_join_graph=True)

    if len(dataset) == 0:
        logger.error("No data loaded!")
        return

    # Split dataset (60/20/20)
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)

    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]
    test_data = [dataset[i] for i in test_indices]

    logger.info(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Create model
    model = AttnJGNN(
        feature_dim=64,
        num_iterations=5,
        initial_heads=4,
        max_heads=8,
        head_increase_interval=1000,
        constraint_gamma=1.0,
        mlp_hidden_dim=64,
        use_constraint_aware=not use_attention,  # Disable for no-attention variant
        use_attention=use_attention,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: use_attention={use_attention}, params={param_count}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_val_rmse = float('inf')
    best_state = None

    logger.info(f"Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        train_loss, train_rmse = train_epoch(model, train_loader, optimizer, device)
        val_rmse = evaluate(model, val_loader, device)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}")

    # Load best model and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)

    test_rmse = evaluate(model, test_loader, device)

    logger.info("=" * 60)
    logger.info(f"EXPERIMENT RESULTS (use_attention={use_attention})")
    logger.info("=" * 60)
    logger.info(f"Best Val RMSE: {best_val_rmse:.4f}")
    logger.info(f"Test RMSE: {test_rmse:.4f}")
    logger.info("=" * 60)

    return test_rmse


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Attn-JGNN experiment")
    parser.add_argument("--cnf-dir", default="data/satlib/raw/rnd3sat")
    parser.add_argument("--labels-file", default="data/satlib/processed/labels.json")
    parser.add_argument("--use-attention", action="store_true", help="Use attention (default: no attention)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_experiment(
        cnf_dir=args.cnf_dir,
        labels_file=args.labels_file,
        use_attention=args.use_attention,
        num_epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
