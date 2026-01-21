"""
Constraint-aware loss functions for Attn-JGNN.

The constraint-aware mechanism explicitly guides the model to preferentially
satisfy clause constraints in the CNF formula.
"""

import torch
from torch import Tensor


def compute_satisfaction_scores(
    var_probs: Tensor,
    var_clause_edge_index: Tensor,
    polarities: Tensor,
    num_clauses: int,
) -> Tensor:
    """
    Compute clause satisfaction scores.

    For each clause φ_i, compute:
        s_i = sigmoid(sum_{x_j in φ_i} (2*b_j(x_j) - 1) * polarity(x_j, φ_i))

    Args:
        var_probs: Variable assignment probabilities [num_vars]
        var_clause_edge_index: Edge indices [2, num_edges]
        polarities: Polarity for each edge [num_edges]
        num_clauses: Number of clauses

    Returns:
        s: Satisfaction scores [num_clauses] in range (0, 1)
    """
    device = var_probs.device

    var_ids = var_clause_edge_index[0]
    clause_ids = var_clause_edge_index[1]

    # Ensure polarities is 1D
    if polarities.dim() > 1:
        polarities = polarities.squeeze(-1)

    # Compute contributions: (2*b_j - 1) * polarity_j
    contributions = (2 * var_probs[var_ids] - 1) * polarities

    # Accumulate per clause
    s = torch.zeros(num_clauses, device=device)
    s.scatter_add_(0, clause_ids, contributions)

    # Apply sigmoid to get satisfaction score in (0, 1)
    s = torch.sigmoid(s)

    return s


def constraint_aware_loss(
    satisfaction_scores: Tensor,
    delta: float = 0.1,
    reduction: str = 'mean',
) -> Tensor:
    """
    Constraint-aware regularization loss.

    L_cons = -δ * sum_{i=1}^{m} ln(s_i)

    This loss encourages the model to satisfy clause constraints by penalizing
    low satisfaction scores.

    Args:
        satisfaction_scores: Tensor [num_clauses] of s_i values in (0, 1)
        delta: Coefficient for constraint awareness (δ)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        L_cons: Scalar loss value (or per-clause if reduction='none')
    """
    # Add small epsilon for numerical stability
    log_s = torch.log(satisfaction_scores + 1e-8)

    # L_cons = -δ * sum ln(s_i)
    if reduction == 'none':
        return -delta * log_s
    elif reduction == 'sum':
        return -delta * log_s.sum()
    else:  # mean
        return -delta * log_s.mean()


def combined_loss(
    logZ_pred: Tensor,
    logZ_true: Tensor,
    satisfaction_scores: Tensor,
    constraint_delta: float = 0.1,
) -> tuple:
    """
    Combined loss for Attn-JGNN training.

    L_total = L_RMSE + L_cons

    Args:
        logZ_pred: Predicted log model count [batch_size, 1]
        logZ_true: Ground truth log model count [batch_size, 1]
        satisfaction_scores: Clause satisfaction scores [num_clauses]
        constraint_delta: Coefficient for constraint-aware loss

    Returns:
        Tuple of (total_loss, rmse_loss, cons_loss)
    """
    # RMSE loss
    mse = torch.nn.functional.mse_loss(logZ_pred, logZ_true)
    rmse_loss = torch.sqrt(mse + 1e-8)

    # Constraint-aware loss
    cons_loss = constraint_aware_loss(satisfaction_scores, delta=constraint_delta)

    # Total loss
    total_loss = rmse_loss + cons_loss

    return total_loss, rmse_loss, cons_loss
