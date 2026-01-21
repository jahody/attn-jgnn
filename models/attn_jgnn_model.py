"""
Attn-JGNN: Attention Enhanced Join-Graph Neural Networks for #SAT.

This module implements the Attn-JGNN model architecture consisting of:
- IntraClusterGAT (GAT1): Variable-clause message passing within clusters
- InterClusterGAT (GAT2): Cross-cluster message passing via shared variables
- AttnJGNN: Main model combining hierarchical attention with MLP readout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple
import math


class IntraClusterGAT(nn.Module):
    """
    GAT1: Intra-cluster attention for variable-clause message passing.

    Performs attention-based message passing between variables and clauses
    within each cluster of the join-graph.
    """

    def __init__(
        self,
        in_dim: int = 64,
        out_dim: int = 64,
        num_heads: int = 4,
        negative_slope: float = 0.2,
        use_constraint_aware: bool = True,
        constraint_gamma: float = 1.0,
        use_attention: bool = True,  # Set to False for no-attention variant
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.use_constraint_aware = use_constraint_aware
        self.constraint_gamma = constraint_gamma
        self.use_attention = use_attention

        self.head_dim = out_dim // num_heads

        # Query, Key, Value projections
        self.W_Q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_K = nn.Linear(in_dim, out_dim, bias=False)
        self.W_V = nn.Linear(in_dim, out_dim, bias=False)

        # Learnable head weights for dynamic attention
        self.head_weights = nn.Parameter(torch.ones(num_heads))

        # Output projection
        self.out_proj = nn.Linear(out_dim, out_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_var: Tensor,
        x_clause: Tensor,
        var_clause_edge_index: Tensor,
        edge_polarity: Tensor,
        cluster_var_ids: List[List[int]],
        cluster_clause_ids: List[List[int]],
        satisfaction_scores: Optional[Tensor] = None,
        active_heads: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for intra-cluster attention.

        Args:
            x_var: Variable features [num_vars, in_dim]
            x_clause: Clause features [num_clauses, in_dim]
            var_clause_edge_index: Edges [2, num_edges] (var_idx, clause_idx)
            edge_polarity: Polarity for each edge [num_edges, 1]
            cluster_var_ids: Variable indices per cluster
            cluster_clause_ids: Clause indices per cluster
            satisfaction_scores: Optional clause satisfaction scores [num_clauses]
            active_heads: Number of active attention heads (for dynamic mechanism)

        Returns:
            Updated (x_var, x_clause) features
        """
        num_vars = x_var.size(0)
        num_clauses = x_clause.size(0)
        device = x_var.device

        if active_heads is None:
            active_heads = self.num_heads

        # Concatenate all node features for unified processing
        x_all = torch.cat([x_var, x_clause], dim=0)  # [num_vars + num_clauses, in_dim]

        # Compute Q, K, V
        Q = self.W_Q(x_all)  # [N, out_dim]
        K = self.W_K(x_all)
        V = self.W_V(x_all)

        # Initialize output
        out = torch.zeros_like(x_all)
        count = torch.zeros(x_all.size(0), 1, device=device)

        # Process each cluster
        for cluster_idx, (var_ids, clause_ids) in enumerate(zip(cluster_var_ids, cluster_clause_ids)):
            if not var_ids or not clause_ids:
                continue

            var_ids_t = torch.tensor(var_ids, device=device, dtype=torch.long)
            clause_ids_t = torch.tensor(clause_ids, device=device, dtype=torch.long) + num_vars

            # Get all node indices in this cluster
            cluster_nodes = torch.cat([var_ids_t, clause_ids_t])

            # Extract Q, K, V for cluster nodes
            Q_cluster = Q[cluster_nodes]  # [cluster_size, out_dim]
            K_cluster = K[cluster_nodes]
            V_cluster = V[cluster_nodes]

            # Compute attention scores
            if self.use_attention:
                # α = LeakyReLU(Q @ K.T / sqrt(d))
                scale = math.sqrt(self.out_dim)
                attn_scores = torch.matmul(Q_cluster, K_cluster.T) / scale  # [cluster_size, cluster_size]

                # Add constraint-aware bias for clause nodes
                if self.use_constraint_aware and satisfaction_scores is not None:
                    clause_local_ids = clause_ids_t - num_vars
                    if len(clause_local_ids) > 0:
                        s_cluster = satisfaction_scores[clause_local_ids]  # [num_clauses_in_cluster]
                        # Add bias to columns corresponding to clauses
                        num_vars_in_cluster = len(var_ids)
                        for i, s in enumerate(s_cluster):
                            attn_scores[:, num_vars_in_cluster + i] += self.constraint_gamma * s

                # Apply LeakyReLU and softmax
                attn_scores = F.leaky_relu(attn_scores, negative_slope=self.negative_slope)
                attn_weights = F.softmax(attn_scores, dim=-1)
            else:
                # No attention: use uniform weights (IJGP-style message passing)
                cluster_size = len(cluster_nodes)
                attn_weights = torch.ones(cluster_size, cluster_size, device=device) / cluster_size

            # Aggregate: h_new = attn @ V
            h_new = torch.matmul(attn_weights, V_cluster)

            # Apply head weighting (simplified - use mean of active head weights)
            head_weight = self.head_weights[:active_heads].mean()
            h_new = h_new * head_weight

            # Accumulate output
            out[cluster_nodes] += h_new
            count[cluster_nodes] += 1

        # Average over clusters (nodes may appear in multiple clusters)
        count = count.clamp(min=1)
        out = out / count

        # Output projection
        out = self.out_proj(out)

        # Split back into variable and clause features
        x_var_new = out[:num_vars]
        x_clause_new = out[num_vars:]

        # Residual connection
        x_var_out = x_var + x_var_new
        x_clause_out = x_clause + x_clause_new

        return x_var_out, x_clause_out


class InterClusterGAT(nn.Module):
    """
    GAT2: Inter-cluster attention for cross-cluster message passing.

    Performs attention-based message passing between clusters through
    their shared variables.
    """

    def __init__(
        self,
        in_dim: int = 64,
        out_dim: int = 64,
        num_heads: int = 4,
        negative_slope: float = 0.2,
        use_attention: bool = True,  # Set to False for no-attention variant
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.use_attention = use_attention

        # Cluster-level projections
        self.W_Q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_K = nn.Linear(in_dim, out_dim, bias=False)
        self.W_V = nn.Linear(in_dim, out_dim, bias=False)

        # Learnable head weights
        self.head_weights = nn.Parameter(torch.ones(num_heads))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)

    def forward(
        self,
        x_var: Tensor,
        cluster_var_ids: List[List[int]],
        cluster_edge_index: Tensor,
        shared_vars: List[List[int]],
        active_heads: Optional[int] = None,
    ) -> Tensor:
        """
        Forward pass for inter-cluster attention.

        Args:
            x_var: Variable features [num_vars, in_dim]
            cluster_var_ids: Variable indices per cluster
            cluster_edge_index: Edges between clusters [2, num_cluster_edges]
            shared_vars: Shared variable indices for each cluster edge
            active_heads: Number of active attention heads

        Returns:
            Updated x_var features
        """
        if active_heads is None:
            active_heads = self.num_heads

        device = x_var.device
        num_clusters = len(cluster_var_ids)

        if cluster_edge_index.size(1) == 0:
            return x_var

        # Compute cluster representations (mean of variable features)
        cluster_feats = []
        for var_ids in cluster_var_ids:
            if var_ids:
                var_ids_t = torch.tensor(var_ids, device=device, dtype=torch.long)
                cluster_feat = x_var[var_ids_t].mean(dim=0)
            else:
                cluster_feat = torch.zeros(self.in_dim, device=device)
            cluster_feats.append(cluster_feat)
        cluster_feats = torch.stack(cluster_feats)  # [num_clusters, in_dim]

        # Compute Q, K, V for clusters
        Q = self.W_Q(cluster_feats)
        K = self.W_K(cluster_feats)
        V = self.W_V(cluster_feats)

        # Initialize output as copy of input
        x_var_out = x_var.clone()

        # Process each cluster edge
        num_edges = cluster_edge_index.size(1)
        for edge_idx in range(num_edges):
            c1 = cluster_edge_index[0, edge_idx].item()
            c2 = cluster_edge_index[1, edge_idx].item()
            shared = shared_vars[edge_idx % len(shared_vars)] if shared_vars else []

            if not shared:
                continue

            # Compute attention between clusters
            if self.use_attention:
                # α_inter = LeakyReLU(Q_c1 @ K_c2 / sqrt(d))
                scale = math.sqrt(self.out_dim)
                attn_score = torch.dot(Q[c1], K[c2]) / scale
                attn_score = F.leaky_relu(attn_score, negative_slope=self.negative_slope)
                attn_weight = torch.sigmoid(attn_score)  # Normalize to [0, 1]

                # Apply head weighting
                head_weight = self.head_weights[:active_heads].mean()
                attn_weight = attn_weight * head_weight
            else:
                # No attention: use uniform weight
                attn_weight = torch.tensor(0.5, device=device)

            # Update shared variables: h_x = h_x^(C1) + α_inter * W_V * h_x^(C2)
            shared_t = torch.tensor(shared, device=device, dtype=torch.long)
            update = attn_weight * V[c2].unsqueeze(0)  # [1, out_dim]
            x_var_out[shared_t] = x_var_out[shared_t] + update

        return x_var_out


class AttnJGNN(nn.Module):
    """
    Main Attn-JGNN model for #SAT (Model Counting).

    Combines hierarchical attention (GAT1 + GAT2) with MLP readout
    to predict log(model_count) for CNF formulas.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        num_iterations: int = 5,
        initial_heads: int = 4,
        max_heads: int = 8,
        head_increase_interval: int = 1000,
        constraint_gamma: float = 1.0,
        mlp_hidden_dim: int = 64,
        use_constraint_aware: bool = True,
        use_attention: bool = True,  # Set to False for no-attention variant
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        self.initial_heads = initial_heads
        self.max_heads = max_heads
        self.head_increase_interval = head_increase_interval
        self.constraint_gamma = constraint_gamma
        self.use_constraint_aware = use_constraint_aware
        self.use_attention = use_attention

        # Current number of active heads (updated during training)
        self.register_buffer('current_heads', torch.tensor(initial_heads))
        self.register_buffer('global_step', torch.tensor(0))

        # GAT layers
        self.gat1 = IntraClusterGAT(
            in_dim=feature_dim,
            out_dim=feature_dim,
            num_heads=max_heads,  # Allocate max, use subset
            use_constraint_aware=use_constraint_aware,
            constraint_gamma=constraint_gamma,
            use_attention=use_attention,
        )

        self.gat2 = InterClusterGAT(
            in_dim=feature_dim,
            out_dim=feature_dim,
            num_heads=max_heads,
            use_attention=use_attention,
        )

        # MLP for Bethe free energy approximation
        # Input: [H(b_C), sum (d_v - 1) * H(b_v)] -> dim 2
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

        # Pooling layer output activation
        self.pool_activation = nn.Tanh()

        # Store intermediate features for loss computation
        self.h_var = None
        self.satisfaction_scores = None

    def update_num_heads(self, global_step: int):
        """Update the number of active attention heads based on training step."""
        new_heads = min(
            self.max_heads,
            self.initial_heads + global_step // self.head_increase_interval
        )
        self.current_heads.fill_(new_heads)
        self.global_step.fill_(global_step)

    def compute_satisfaction_scores(
        self,
        x_var: Tensor,
        var_clause_edge_index: Tensor,
        edge_polarity: Tensor,
        num_clauses: int,
    ) -> Tensor:
        """
        Compute clause satisfaction scores for constraint-aware attention.

        Args:
            x_var: Variable features [num_vars, d]
            var_clause_edge_index: Edge indices [2, num_edges]
            edge_polarity: Polarity for each edge [num_edges, 1]
            num_clauses: Number of clauses

        Returns:
            s: Satisfaction scores [num_clauses]
        """
        device = x_var.device

        # Get assignment probabilities from variable features (mean + sigmoid)
        b = torch.sigmoid(x_var.mean(dim=-1))  # [num_vars]

        # Compute satisfaction score for each clause
        s = torch.zeros(num_clauses, device=device)
        clause_counts = torch.zeros(num_clauses, device=device)

        var_ids = var_clause_edge_index[0]
        clause_ids = var_clause_edge_index[1]
        polarities = edge_polarity.squeeze(-1)

        # s_i = sigmoid(sum_{x_j in φ_i} (2*b_j - 1) * polarity(x_j, φ_i))
        contributions = (2 * b[var_ids] - 1) * polarities

        # Scatter add to accumulate per clause
        s.scatter_add_(0, clause_ids, contributions)
        clause_counts.scatter_add_(0, clause_ids, torch.ones_like(contributions))

        # Apply sigmoid
        s = torch.sigmoid(s)

        return s

    def forward(self, batch) -> Tensor:
        """
        Forward pass of Attn-JGNN.

        Args:
            batch: PyG Data object with join-graph representation

        Returns:
            logZ: Predicted log model count [batch_size, 1]
        """
        # Extract batch data
        x_var = batch.x_var
        x_clause = batch.x_clause
        var_clause_edge_index = batch.var_clause_edge_index
        edge_polarity = batch.edge_polarity
        cluster_edge_index = batch.cluster_edge_index
        cluster_var_ids = batch.cluster_var_ids
        cluster_clause_ids = batch.cluster_clause_ids
        shared_vars = batch.shared_vars
        num_clauses = batch.num_clauses

        # Flatten batched lists if necessary
        # PyG batching might wrap lists from different graphs into a list of lists
        # We want a single flat list of clusters/edges for the entire batch
        if cluster_var_ids and isinstance(cluster_var_ids[0], list) and len(cluster_var_ids[0]) > 0 and isinstance(cluster_var_ids[0][0], list):
            cluster_var_ids = [c for graph_clusters in cluster_var_ids for c in graph_clusters]
            cluster_clause_ids = [c for graph_clusters in cluster_clause_ids for c in graph_clusters]
        
        if shared_vars and isinstance(shared_vars[0], list) and len(shared_vars[0]) > 0 and isinstance(shared_vars[0][0], list):
            shared_vars = [s for graph_shared in shared_vars for s in graph_shared]

        # Handle list attributes for single graph (ensure consistent List[List[int]] structure)
        # This handles the case where it might have been flattened too much or loaded oddly (edge case)
        if not isinstance(cluster_var_ids[0], list):
            cluster_var_ids = [cluster_var_ids]
            cluster_clause_ids = [cluster_clause_ids]

        active_heads = self.current_heads.item()

        # Iterative message passing
        h_var = x_var
        h_clause = x_clause

        for t in range(self.num_iterations):
            # Compute satisfaction scores for constraint awareness
            s = self.compute_satisfaction_scores(
                h_var, var_clause_edge_index, edge_polarity, num_clauses
            )

            # GAT1: Intra-cluster attention
            h_var, h_clause = self.gat1(
                h_var, h_clause,
                var_clause_edge_index, edge_polarity,
                cluster_var_ids, cluster_clause_ids,
                satisfaction_scores=s,
                active_heads=active_heads,
            )

            # GAT2: Inter-cluster attention
            h_var = self.gat2(
                h_var,
                cluster_var_ids,
                cluster_edge_index,
                shared_vars,
                active_heads=active_heads,
            )

        # Store for loss computation
        self.h_var = h_var
        self.satisfaction_scores = self.compute_satisfaction_scores(
            h_var, var_clause_edge_index, edge_polarity, num_clauses
        )

        # Compute cluster features for Bethe free energy approximation
        device = h_var.device
        cluster_feats = []

        for var_ids, clause_ids in zip(cluster_var_ids, cluster_clause_ids):
            # H(b_C_α) - cluster joint entropy approximation
            if var_ids:
                var_ids_t = torch.tensor(var_ids, device=device, dtype=torch.long)
                h_cluster_vars = h_var[var_ids_t].mean(dim=0)
            else:
                h_cluster_vars = torch.zeros(self.feature_dim, device=device)

            if clause_ids:
                clause_ids_t = torch.tensor(clause_ids, device=device, dtype=torch.long)
                h_cluster_clauses = h_clause[clause_ids_t].mean(dim=0)
            else:
                h_cluster_clauses = torch.zeros(self.feature_dim, device=device)

            # Concatenate variable and clause representations
            h_cluster = torch.cat([h_cluster_vars, h_cluster_clauses])
            cluster_feats.append(h_cluster)

        # Global pooling over clusters
        if cluster_feats:
            h_G = torch.stack(cluster_feats).mean(dim=0)  # [feature_dim * 2]
        else:
            h_G = torch.zeros(self.feature_dim * 2, device=device)

        # Apply pooling activation
        h_G = self.pool_activation(h_G)

        # MLP prediction: logZ = -F_Bethe_Join = -MLP(h_G)
        F_bethe = self.mlp(h_G.unsqueeze(0))  # [1, 1]
        logZ = -F_bethe

        return logZ


def create_attn_jgnn(
    feature_dim: int = 64,
    num_iterations: int = 5,
    initial_heads: int = 4,
    max_heads: int = 8,
    head_increase_interval: int = 1000,
    constraint_gamma: float = 1.0,
    mlp_hidden_dim: int = 64,
    use_constraint_aware: bool = True,
    use_attention: bool = True,
) -> AttnJGNN:
    """Factory function to create AttnJGNN model."""
    return AttnJGNN(
        feature_dim=feature_dim,
        num_iterations=num_iterations,
        initial_heads=initial_heads,
        max_heads=max_heads,
        head_increase_interval=head_increase_interval,
        constraint_gamma=constraint_gamma,
        mlp_hidden_dim=mlp_hidden_dim,
        use_constraint_aware=use_constraint_aware,
        use_attention=use_attention,
    )
