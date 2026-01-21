"""
Data module for Attn-JGNN: Attention Enhanced Join-Graph Neural Networks.

This module provides data loading, preprocessing, and benchmark datasets
for #SAT (model counting) problems.

Submodules:
- datasets: PyTorch Geometric dataset classes (BIRD, SATLIB)
- preprocessing: CNF parsing, graph building, tree decomposition
- solvers: Interface to exact #SAT solvers (DSharp)
"""

from .datasets import BIRDDataset, SATLIBDataset
from .preprocessing import (
    parse_dimacs,
    CNF,
    build_factor_graph,
    factor_graph_to_pyg,
    build_join_graph,
    join_graph_to_pyg,
    cnf_to_factor_graph_data,
    decompose_cnf,
    TreeDecomposition,
)
from .solvers import compute_model_count

__all__ = [
    # Datasets
    'BIRDDataset',
    'SATLIBDataset',
    # CNF parsing
    'parse_dimacs',
    'CNF',
    # Graph building
    'build_factor_graph',
    'factor_graph_to_pyg',
    'build_join_graph',
    'join_graph_to_pyg',
    'cnf_to_factor_graph_data',
    # Tree decomposition
    'decompose_cnf',
    'TreeDecomposition',
    # Solvers
    'compute_model_count',
]
