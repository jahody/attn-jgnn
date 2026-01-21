"""
Preprocessing utilities for CNF formulas.

Provides:
- CNF parsing (DIMACS format)
- Factor graph construction
- Join-graph construction from tree decomposition
- Tree decomposition interface (FlowCutter)
"""

from .cnf_parser import CNF, parse_dimacs, parse_dimacs_string, write_dimacs
from .graph_builder import (
    FactorGraph,
    JoinGraph,
    build_factor_graph,
    factor_graph_to_pyg,
    build_join_graph,
    join_graph_to_pyg,
    cnf_to_factor_graph_data,
)
from .tree_decomposition import (
    TreeDecomposition,
    decompose_cnf,
    decompose_with_flowcutter,
    verify_tree_decomposition,
)

__all__ = [
    # CNF parsing
    'CNF',
    'parse_dimacs',
    'parse_dimacs_string',
    'write_dimacs',
    # Graph building
    'FactorGraph',
    'JoinGraph',
    'build_factor_graph',
    'factor_graph_to_pyg',
    'build_join_graph',
    'join_graph_to_pyg',
    'cnf_to_factor_graph_data',
    # Tree decomposition
    'TreeDecomposition',
    'decompose_cnf',
    'decompose_with_flowcutter',
    'verify_tree_decomposition',
]
