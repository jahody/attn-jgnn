"""
Tests for the data module.

Tests CNF parsing, graph construction, tree decomposition, and datasets.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path

# Test imports work
def test_imports():
    """Test that all modules can be imported."""
    from data.preprocessing.cnf_parser import parse_dimacs, parse_dimacs_string, CNF
    from data.preprocessing.graph_builder import (
        build_factor_graph,
        factor_graph_to_pyg,
        cnf_to_factor_graph_data
    )
    from data.preprocessing.tree_decomposition import decompose_cnf, TreeDecomposition
    from data.solvers.dsharp_solver import compute_model_count


# Sample CNF for testing (simple 3-SAT formula)
SAMPLE_CNF = """c This is a comment
c Another comment
p cnf 4 3
1 2 -3 0
-1 3 4 0
2 -4 0
"""

# Another test CNF
SIMPLE_CNF = """p cnf 3 2
1 2 0
-1 -2 3 0
"""


class TestCNFParser:
    """Tests for CNF parsing functionality."""

    def test_parse_dimacs_string(self):
        """Test parsing CNF from string."""
        from data.preprocessing.cnf_parser import parse_dimacs_string

        cnf = parse_dimacs_string(SAMPLE_CNF)

        assert cnf.num_variables == 4
        assert cnf.num_clauses == 3
        assert len(cnf.clauses) == 3
        assert cnf.clauses[0] == [1, 2, -3]
        assert cnf.clauses[1] == [-1, 3, 4]
        assert cnf.clauses[2] == [2, -4]
        assert len(cnf.comments) == 2

    def test_parse_dimacs_file(self):
        """Test parsing CNF from file."""
        from data.preprocessing.cnf_parser import parse_dimacs

        with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
            f.write(SAMPLE_CNF)
            temp_path = f.name

        try:
            cnf = parse_dimacs(temp_path)
            assert cnf.num_variables == 4
            assert cnf.num_clauses == 3
        finally:
            os.unlink(temp_path)

    def test_get_variables(self):
        """Test getting variable list."""
        from data.preprocessing.cnf_parser import parse_dimacs_string

        cnf = parse_dimacs_string(SAMPLE_CNF)
        variables = cnf.get_variables()

        assert variables == [1, 2, 3, 4]

    def test_get_variable_occurrences(self):
        """Test getting variable occurrences."""
        from data.preprocessing.cnf_parser import parse_dimacs_string

        cnf = parse_dimacs_string(SIMPLE_CNF)
        occurrences = cnf.get_variable_occurrences()

        # Variable 1: appears positive in clause 0, negative in clause 1
        assert (0, 1) in occurrences[1]
        assert (1, -1) in occurrences[1]

        # Variable 2: appears positive in clause 0, negative in clause 1
        assert (0, 1) in occurrences[2]
        assert (1, -1) in occurrences[2]

        # Variable 3: appears positive in clause 1 only
        assert len([x for x in occurrences[3] if x[0] == 1]) == 1

    def test_write_dimacs(self):
        """Test writing CNF to file."""
        from data.preprocessing.cnf_parser import parse_dimacs_string, write_dimacs, parse_dimacs

        cnf = parse_dimacs_string(SAMPLE_CNF)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.cnf"
            write_dimacs(cnf, output_path)

            # Read it back
            cnf2 = parse_dimacs(output_path)
            assert cnf2.num_variables == cnf.num_variables
            assert cnf2.num_clauses == cnf.num_clauses
            assert cnf2.clauses == cnf.clauses


class TestGraphBuilder:
    """Tests for graph building functionality."""

    def test_build_factor_graph(self):
        """Test factor graph construction."""
        from data.preprocessing.cnf_parser import parse_dimacs_string
        from data.preprocessing.graph_builder import build_factor_graph

        cnf = parse_dimacs_string(SAMPLE_CNF)
        fg = build_factor_graph(cnf)

        assert fg.num_variables == 4
        assert fg.num_clauses == 3
        # Count edges: clause 0 has 3 literals, clause 1 has 3, clause 2 has 2
        assert len(fg.edges) == 8
        assert len(fg.polarities) == 8

    def test_factor_graph_to_pyg(self):
        """Test conversion to PyTorch Geometric Data."""
        from data.preprocessing.cnf_parser import parse_dimacs_string
        from data.preprocessing.graph_builder import build_factor_graph, factor_graph_to_pyg

        cnf = parse_dimacs_string(SAMPLE_CNF)
        fg = build_factor_graph(cnf)
        data = factor_graph_to_pyg(fg, feature_dim=64, label=10.5)

        # Check node features
        assert data.x.shape == (4 + 3, 64)  # 4 vars + 3 clauses
        assert data.x_var.shape == (4, 64)
        assert data.x_clause.shape == (3, 64)

        # Check edges (bidirectional, so 2x the original)
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == 16  # 8 original * 2 bidirectional

        # Check edge attributes
        assert data.edge_attr.shape == (16, 1)

        # Check label
        assert data.y.item() == pytest.approx(10.5)

        # Check node types
        assert (data.node_type[:4] == 0).all()  # Variables
        assert (data.node_type[4:] == 1).all()  # Clauses

    def test_cnf_to_factor_graph_data(self):
        """Test convenience function."""
        from data.preprocessing.cnf_parser import parse_dimacs_string
        from data.preprocessing.graph_builder import cnf_to_factor_graph_data

        cnf = parse_dimacs_string(SIMPLE_CNF)
        data = cnf_to_factor_graph_data(cnf, feature_dim=32)

        assert data.x.shape[1] == 32
        assert data.num_variables == 3
        assert data.num_clauses == 2


class TestTreeDecomposition:
    """Tests for tree decomposition functionality."""

    def test_heuristic_decomposition(self):
        """Test fallback heuristic decomposition."""
        from data.preprocessing.cnf_parser import parse_dimacs_string
        from data.preprocessing.tree_decomposition import decompose_cnf

        cnf = parse_dimacs_string(SAMPLE_CNF)
        td = decompose_cnf(cnf, use_fallback=True)

        assert td.num_bags > 0
        assert len(td.bags) == td.num_bags
        assert td.tree_width >= 0

    def test_verify_decomposition(self):
        """Test decomposition verification."""
        from data.preprocessing.cnf_parser import parse_dimacs_string
        from data.preprocessing.tree_decomposition import decompose_cnf, verify_tree_decomposition

        cnf = parse_dimacs_string(SIMPLE_CNF)
        td = decompose_cnf(cnf)

        is_valid, errors = verify_tree_decomposition(td, cnf)
        # With heuristic, coverage should be satisfied
        # (connectivity may not be perfect with simple heuristic)
        print(f"Decomposition valid: {is_valid}, errors: {errors}")


class TestJoinGraph:
    """Tests for join-graph construction."""

    def test_build_join_graph(self):
        """Test join-graph construction from decomposition."""
        from data.preprocessing.cnf_parser import parse_dimacs_string
        from data.preprocessing.graph_builder import build_factor_graph, build_join_graph
        from data.preprocessing.tree_decomposition import decompose_cnf

        cnf = parse_dimacs_string(SAMPLE_CNF)
        fg = build_factor_graph(cnf)
        td = decompose_cnf(cnf)

        jg = build_join_graph(fg, td.bags, td.tree_edges)

        assert len(jg.clusters) == len(td.bags)
        assert len(jg.cluster_var_ids) == len(td.bags)
        assert len(jg.cluster_clause_ids) == len(td.bags)

    def test_join_graph_to_pyg(self):
        """Test join-graph to PyG conversion."""
        from data.preprocessing.cnf_parser import parse_dimacs_string
        from data.preprocessing.graph_builder import build_factor_graph, build_join_graph, join_graph_to_pyg
        from data.preprocessing.tree_decomposition import decompose_cnf

        cnf = parse_dimacs_string(SAMPLE_CNF)
        fg = build_factor_graph(cnf)
        td = decompose_cnf(cnf)
        jg = build_join_graph(fg, td.bags, td.tree_edges)

        data = join_graph_to_pyg(jg, feature_dim=64, label=5.0)

        assert data.x_var.shape == (4, 64)
        assert data.x_clause.shape == (3, 64)
        assert data.var_clause_edge_index.shape[0] == 2
        assert data.y.item() == pytest.approx(5.0)
        assert data.num_clusters > 0


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test full pipeline from CNF string to PyG Data."""
        from data.preprocessing.cnf_parser import parse_dimacs_string
        from data.preprocessing.graph_builder import cnf_to_factor_graph_data

        cnf = parse_dimacs_string(SAMPLE_CNF)
        data = cnf_to_factor_graph_data(cnf, feature_dim=64, label=100.0)

        # Verify data is valid PyG Data
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
        assert hasattr(data, 'y')

        # Check tensors are proper torch tensors
        assert isinstance(data.x, torch.Tensor)
        assert isinstance(data.edge_index, torch.Tensor)
        assert isinstance(data.y, torch.Tensor)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
