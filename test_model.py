"""
Test script to verify Attn-JGNN implementation works correctly.
"""

import torch
import sys


def create_dummy_batch():
    """Create a dummy batch for testing the model."""
    from torch_geometric.data import Data

    # Simulate a small CNF formula with 10 variables, 5 clauses, 3 clusters
    num_vars = 10
    num_clauses = 5
    num_clusters = 3
    feature_dim = 64

    # Variable features (zeros as in paper)
    x_var = torch.zeros(num_vars, feature_dim)

    # Clause features (self-identifying)
    x_clause = torch.zeros(num_clauses, feature_dim)
    for i in range(num_clauses):
        x_clause[i, i % feature_dim] = 1.0

    # Create bipartite edges (variable -> clause)
    # Each clause has ~3 variables (3-SAT)
    edges = [
        (0, 0), (1, 0), (2, 0),  # clause 0
        (1, 1), (3, 1), (4, 1),  # clause 1
        (2, 2), (4, 2), (5, 2),  # clause 2
        (5, 3), (6, 3), (7, 3),  # clause 3
        (7, 4), (8, 4), (9, 4),  # clause 4
    ]
    var_ids = [e[0] for e in edges]
    clause_ids = [e[1] for e in edges]
    var_clause_edge_index = torch.tensor([var_ids, clause_ids], dtype=torch.long)

    # Polarities (random +1/-1)
    edge_polarity = torch.tensor([1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1], dtype=torch.float).unsqueeze(-1)

    # Cluster information (from tree decomposition)
    # Cluster 0: vars 0,1,2,3 / clauses 0,1
    # Cluster 1: vars 2,3,4,5 / clauses 1,2
    # Cluster 2: vars 5,6,7,8,9 / clauses 3,4
    cluster_var_ids = [[0, 1, 2, 3], [2, 3, 4, 5], [5, 6, 7, 8, 9]]
    cluster_clause_ids = [[0, 1], [1, 2], [3, 4]]

    # Cluster edges (tree structure)
    cluster_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).T  # [2, 2]

    # Shared variables between clusters
    shared_vars = [[2, 3], [5]]  # Between (0,1) and (1,2)

    # Ground truth label (log model count)
    y = torch.tensor([[10.5]], dtype=torch.float)

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
        num_clusters=num_clusters,
        y=y,
    )

    return data


def test_model_forward():
    """Test that the model forward pass works."""
    print("=" * 60)
    print("Testing Attn-JGNN Model Forward Pass")
    print("=" * 60)

    from models.attn_jgnn_model import AttnJGNN

    # Create model
    model = AttnJGNN(
        feature_dim=64,
        num_iterations=5,
        initial_heads=4,
        max_heads=8,
        head_increase_interval=1000,
        constraint_gamma=1.0,
        mlp_hidden_dim=64,
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dummy batch
    batch = create_dummy_batch()
    print(f"Dummy batch created:")
    print(f"  - Variables: {batch.num_variables}")
    print(f"  - Clauses: {batch.num_clauses}")
    print(f"  - Clusters: {batch.num_clusters}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logZ = model(batch)

    print(f"\nForward pass successful!")
    print(f"  - Output shape: {logZ.shape}")
    print(f"  - Predicted logZ: {logZ.item():.4f}")
    print(f"  - Ground truth logZ: {batch.y.item():.4f}")

    return True


def test_wrapper():
    """Test the Lightning wrapper."""
    print("\n" + "=" * 60)
    print("Testing PyTorch Lightning Wrapper")
    print("=" * 60)

    try:
        import pytorch_lightning as pl
    except ImportError:
        print("pytorch_lightning not installed, skipping wrapper test")
        return True  # Skip but don't fail

    from models.wrapper import AttnJGNNWrapper
    from omegaconf import DictConfig

    # Create config
    config = DictConfig({
        'feature_dim': 64,
        'num_iterations': 5,
        'initial_heads': 4,
        'max_heads': 8,
        'head_increase_interval': 1000,
        'constraint_gamma': 1.0,
        'mlp_hidden_dim': 64,
        'use_constraint_aware': True,
        'constraint_delta': 0.1,
        'lr': 0.001,
        'weight_decay': 0.0,
        'use_scheduler': True,
        'scheduler_factor': 0.5,
        'scheduler_patience': 10,
    })

    # Create wrapper
    wrapper = AttnJGNNWrapper(config=config)
    print(f"Wrapper created")

    # Create dummy batch
    batch = create_dummy_batch()

    # Training step
    wrapper.train()
    loss = wrapper.training_step(batch, 0)
    print(f"Training step loss: {loss.item():.4f}")

    # Validation step
    wrapper.eval()
    with torch.no_grad():
        rmse = wrapper.validation_step(batch, 0)
    print(f"Validation RMSE: {rmse.item():.4f}")

    # Configure optimizers
    opt_config = wrapper.configure_optimizers()
    print(f"Optimizer configured: {type(opt_config['optimizer']).__name__}")
    print(f"Scheduler configured: {type(opt_config['lr_scheduler']['scheduler']).__name__}")

    return True


def test_loss_functions():
    """Test loss functions."""
    print("\n" + "=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)

    from losses.constraint_loss import compute_satisfaction_scores, constraint_aware_loss, combined_loss

    # Create dummy data
    num_vars = 10
    num_clauses = 5

    var_probs = torch.sigmoid(torch.randn(num_vars))
    edges = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (3, 1), (4, 1),
        (2, 2), (4, 2), (5, 2),
        (5, 3), (6, 3), (7, 3),
        (7, 4), (8, 4), (9, 4),
    ]
    var_clause_edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
    polarities = torch.tensor([1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1], dtype=torch.float)

    # Compute satisfaction scores
    s = compute_satisfaction_scores(var_probs, var_clause_edge_index, polarities, num_clauses)
    print(f"Satisfaction scores: {s}")
    print(f"  - Shape: {s.shape}")
    print(f"  - Range: [{s.min():.4f}, {s.max():.4f}]")

    # Constraint-aware loss
    cons_loss = constraint_aware_loss(s, delta=0.1)
    print(f"Constraint-aware loss: {cons_loss.item():.4f}")

    # Combined loss
    logZ_pred = torch.tensor([[10.0]])
    logZ_true = torch.tensor([[10.5]])
    total, rmse, cons = combined_loss(logZ_pred, logZ_true, s, constraint_delta=0.1)
    print(f"Combined loss - Total: {total.item():.4f}, RMSE: {rmse.item():.4f}, Cons: {cons.item():.4f}")

    return True


def test_gat_layers():
    """Test individual GAT layers."""
    print("\n" + "=" * 60)
    print("Testing GAT Layers")
    print("=" * 60)

    from models.attn_jgnn_model import IntraClusterGAT, InterClusterGAT

    feature_dim = 64
    num_vars = 10
    num_clauses = 5

    # Create features
    x_var = torch.randn(num_vars, feature_dim)
    x_clause = torch.randn(num_clauses, feature_dim)

    # Create edges
    edges = [(0, 0), (1, 0), (2, 0), (1, 1), (3, 1), (4, 1)]
    var_clause_edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
    edge_polarity = torch.ones(len(edges), 1)

    # Cluster info
    cluster_var_ids = [[0, 1, 2, 3], [2, 3, 4, 5, 6, 7, 8, 9]]
    cluster_clause_ids = [[0, 1], [2, 3, 4]]

    # Test IntraClusterGAT
    gat1 = IntraClusterGAT(feature_dim, feature_dim, num_heads=4)
    x_var_out, x_clause_out = gat1(
        x_var, x_clause,
        var_clause_edge_index, edge_polarity,
        cluster_var_ids, cluster_clause_ids,
    )
    print(f"IntraClusterGAT output shapes: var={x_var_out.shape}, clause={x_clause_out.shape}")

    # Test InterClusterGAT
    gat2 = InterClusterGAT(feature_dim, feature_dim, num_heads=4)
    cluster_edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    shared_vars = [[2, 3]]
    x_var_out2 = gat2(x_var_out, cluster_var_ids, cluster_edge_index, shared_vars)
    print(f"InterClusterGAT output shape: {x_var_out2.shape}")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Attn-JGNN Implementation Tests")
    print("=" * 60 + "\n")

    tests = [
        ("GAT Layers", test_gat_layers),
        ("Loss Functions", test_loss_functions),
        ("Model Forward", test_model_forward),
        ("Lightning Wrapper", test_wrapper),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    all_passed = True
    for name, success, error in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {name}: {status}")
        if error:
            print(f"    Error: {error}")
        if not success:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)

    return all_passed


if __name__ == '__main__':
    main()
