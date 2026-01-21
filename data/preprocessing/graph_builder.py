"""
Graph Builder for Factor Graphs and Join-Graphs.

This module constructs:
1. Factor graphs: Bipartite graphs between variables and clauses
2. Join-graphs: Cluster-based graphs from tree decomposition
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import torch
from torch_geometric.data import Data

from .cnf_parser import CNF


@dataclass
class FactorGraph:
    """
    Factor graph representation of a CNF formula.

    A factor graph is a bipartite graph where:
    - Variable nodes represent Boolean variables
    - Clause (factor) nodes represent clauses/constraints
    - Edges connect variables to clauses they appear in

    Attributes:
        num_variables: Number of variable nodes
        num_clauses: Number of clause (factor) nodes
        edges: List of (variable_idx, clause_idx) edges
        polarities: List of polarities (+1 or -1) for each edge
    """
    num_variables: int
    num_clauses: int
    edges: List[Tuple[int, int]]  # (var_idx, clause_idx) - 0-indexed
    polarities: List[int]  # +1 for positive literal, -1 for negative


@dataclass
class JoinGraph:
    """
    Join-graph representation from tree decomposition.

    A join-graph consists of clusters (bags) from tree decomposition,
    connected based on shared variables.

    Attributes:
        clusters: List of clusters, each containing variable and clause indices
        cluster_var_ids: Variable indices in each cluster
        cluster_clause_ids: Clause indices in each cluster
        cluster_edges: Edges between clusters
        shared_vars: Shared variables for each cluster edge
        factor_graph: The underlying factor graph
    """
    clusters: List[Set[int]]  # Node IDs (vars and clauses) per cluster
    cluster_var_ids: List[List[int]]  # Variable IDs per cluster
    cluster_clause_ids: List[List[int]]  # Clause IDs per cluster
    cluster_edges: List[Tuple[int, int]]  # Edges between clusters
    shared_vars: List[List[int]]  # Shared variables for each edge
    factor_graph: FactorGraph


def build_factor_graph(cnf: CNF) -> FactorGraph:
    """
    Build a factor graph from a CNF formula.

    Creates a bipartite graph between variables (0 to num_vars-1) and
    clauses (0 to num_clauses-1).

    Args:
        cnf: Parsed CNF formula

    Returns:
        FactorGraph object representing the bipartite structure
    """
    edges = []
    polarities = []

    for clause_idx, clause in enumerate(cnf.clauses):
        for literal in clause:
            var_idx = abs(literal) - 1  # Convert to 0-indexed
            polarity = 1 if literal > 0 else -1
            edges.append((var_idx, clause_idx))
            polarities.append(polarity)

    return FactorGraph(
        num_variables=cnf.num_variables,
        num_clauses=cnf.num_clauses,
        edges=edges,
        polarities=polarities
    )


def factor_graph_to_pyg(
    factor_graph: FactorGraph,
    feature_dim: int = 64,
    label: Optional[float] = None
) -> Data:
    """
    Convert a factor graph to a PyTorch Geometric Data object.

    The graph uses a bipartite structure where:
    - Variable nodes: indices 0 to num_variables-1
    - Clause nodes: indices num_variables to num_variables+num_clauses-1

    Args:
        factor_graph: FactorGraph object
        feature_dim: Dimension of node features (default 64 as per paper)
        label: Optional label (log model count)

    Returns:
        PyTorch Geometric Data object
    """
    num_vars = factor_graph.num_variables
    num_clauses = factor_graph.num_clauses

    # Initialize node features
    # Variable nodes: learnable features (initialized to zeros, will be learned)
    x_var = torch.zeros(num_vars, feature_dim)

    # Clause nodes: self-identifying features (initialized distinctly)
    # Use small random initialization to break symmetry
    x_clause = torch.zeros(num_clauses, feature_dim)
    # Initialize with small values based on clause index for identification
    for i in range(num_clauses):
        x_clause[i, i % feature_dim] = 1.0

    # Build edge index (variable -> clause edges in bipartite graph)
    # For PyG, we create bidirectional edges
    edge_src = []
    edge_dst = []
    edge_attr = []

    for (var_idx, clause_idx), polarity in zip(factor_graph.edges, factor_graph.polarities):
        # Variable to clause edge
        edge_src.append(var_idx)
        edge_dst.append(num_vars + clause_idx)  # Offset clause indices
        edge_attr.append(polarity)

        # Clause to variable edge (bidirectional)
        edge_src.append(num_vars + clause_idx)
        edge_dst.append(var_idx)
        edge_attr.append(polarity)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)

    # Concatenate node features
    x = torch.cat([x_var, x_clause], dim=0)

    # Create node type tensor (0 for variables, 1 for clauses)
    node_type = torch.cat([
        torch.zeros(num_vars, dtype=torch.long),
        torch.ones(num_clauses, dtype=torch.long)
    ])

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=node_type,
        num_variables=num_vars,
        num_clauses=num_clauses,
    )

    # Also store separate features for convenience
    data.x_var = x_var
    data.x_clause = x_clause

    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)

    return data


def build_join_graph(
    factor_graph: FactorGraph,
    tree_decomposition: List[Set[int]],
    tree_edges: List[Tuple[int, int]]
) -> JoinGraph:
    """
    Build a join-graph from a factor graph and tree decomposition.

    Args:
        factor_graph: The factor graph representation
        tree_decomposition: List of clusters (bags) from tree decomposition,
                           each containing variable indices (0-indexed)
        tree_edges: Edges between clusters in the tree decomposition

    Returns:
        JoinGraph object
    """
    num_vars = factor_graph.num_variables

    # Map clauses to clusters based on which cluster contains all their variables
    clause_to_cluster = {}
    for clause_idx in range(factor_graph.num_clauses):
        # Find variables in this clause
        clause_vars = set()
        for (var_idx, c_idx), _ in zip(factor_graph.edges, factor_graph.polarities):
            if c_idx == clause_idx:
                clause_vars.add(var_idx)

        # Find a cluster that contains all these variables
        for cluster_idx, cluster_vars in enumerate(tree_decomposition):
            if clause_vars.issubset(cluster_vars):
                clause_to_cluster[clause_idx] = cluster_idx
                break

    # Build cluster contents
    clusters = []
    cluster_var_ids = []
    cluster_clause_ids = []

    for cluster_idx, cluster_vars in enumerate(tree_decomposition):
        # Variables in this cluster (as node IDs: 0 to num_vars-1)
        var_ids = sorted(list(cluster_vars))

        # Clauses assigned to this cluster
        clause_ids = [c_idx for c_idx, assigned_cluster
                      in clause_to_cluster.items()
                      if assigned_cluster == cluster_idx]

        # All node IDs (vars + offset clauses)
        node_ids = set(var_ids) | set(num_vars + c for c in clause_ids)

        clusters.append(node_ids)
        cluster_var_ids.append(var_ids)
        cluster_clause_ids.append(clause_ids)

    # Compute shared variables for each edge
    shared_vars = []
    for (c1, c2) in tree_edges:
        shared = set(cluster_var_ids[c1]) & set(cluster_var_ids[c2])
        shared_vars.append(sorted(list(shared)))

    return JoinGraph(
        clusters=clusters,
        cluster_var_ids=cluster_var_ids,
        cluster_clause_ids=cluster_clause_ids,
        cluster_edges=tree_edges,
        shared_vars=shared_vars,
        factor_graph=factor_graph
    )


def join_graph_to_pyg(
    join_graph: JoinGraph,
    feature_dim: int = 64,
    label: Optional[float] = None
) -> Data:
    """
    Convert a join-graph to a PyTorch Geometric Data object.

    This creates a hierarchical representation with:
    - Original factor graph structure
    - Cluster-level structure from join-graph

    Args:
        join_graph: JoinGraph object
        feature_dim: Dimension of node features (default 64)
        label: Optional label (log model count)

    Returns:
        PyTorch Geometric Data object with both factor graph and join-graph info
    """
    fg = join_graph.factor_graph
    num_vars = fg.num_variables
    num_clauses = fg.num_clauses

    # Initialize node features (same as factor graph)
    x_var = torch.zeros(num_vars, feature_dim)
    x_clause = torch.zeros(num_clauses, feature_dim)
    for i in range(num_clauses):
        x_clause[i, i % feature_dim] = 1.0

    # Build factor graph edges
    var_clause_src = []
    var_clause_dst = []
    edge_polarity = []

    for (var_idx, clause_idx), polarity in zip(fg.edges, fg.polarities):
        var_clause_src.append(var_idx)
        var_clause_dst.append(clause_idx)
        edge_polarity.append(polarity)

    var_clause_edge_index = torch.tensor(
        [var_clause_src, var_clause_dst], dtype=torch.long
    )
    edge_polarity = torch.tensor(edge_polarity, dtype=torch.float).unsqueeze(1)

    # Build cluster edge index
    if join_graph.cluster_edges:
        cluster_src = [e[0] for e in join_graph.cluster_edges]
        cluster_dst = [e[1] for e in join_graph.cluster_edges]
        # Make bidirectional
        cluster_edge_index = torch.tensor(
            [cluster_src + cluster_dst, cluster_dst + cluster_src],
            dtype=torch.long
        )
    else:
        cluster_edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(
        x_var=x_var,
        x_clause=x_clause,
        var_clause_edge_index=var_clause_edge_index,
        edge_polarity=edge_polarity,
        cluster_edge_index=cluster_edge_index,
        num_variables=num_vars,
        num_clauses=num_clauses,
        num_clusters=len(join_graph.clusters),
    )

    # Store cluster information as Python lists (PyG supports this)
    data.cluster_var_ids = join_graph.cluster_var_ids
    data.cluster_clause_ids = join_graph.cluster_clause_ids
    data.shared_vars = join_graph.shared_vars

    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)

    return data


def cnf_to_factor_graph_data(
    cnf: CNF,
    feature_dim: int = 64,
    label: Optional[float] = None
) -> Data:
    """
    Convenience function to convert CNF directly to PyG Data.

    Args:
        cnf: Parsed CNF formula
        feature_dim: Dimension of node features
        label: Optional label (log model count)

    Returns:
        PyTorch Geometric Data object
    """
    factor_graph = build_factor_graph(cnf)
    return factor_graph_to_pyg(factor_graph, feature_dim, label)
