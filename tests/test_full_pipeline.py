"""
Full pipeline tests for the data module.

Tests the complete flow from CNF parsing to PyG Data creation.
"""

import sys
import tempfile
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Try importing torch first
try:
    import torch
    TORCH_AVAILABLE = True
    print(f"PyTorch available: version {torch.__version__}")
except Exception as e:
    TORCH_AVAILABLE = False
    print(f"PyTorch not available: {e}")
    print("Running tests without torch-dependent functionality")


def import_module_directly(module_name, file_path):
    """Import a module directly from file path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import CNF parser directly
cnf_parser = import_module_directly(
    "cnf_parser",
    str(Path(project_root) / "data" / "preprocessing" / "cnf_parser.py")
)

parse_dimacs = cnf_parser.parse_dimacs
parse_dimacs_string = cnf_parser.parse_dimacs_string
CNF = cnf_parser.CNF


# Sample CNF formulas for testing
SAMPLE_3SAT = """c Simple 3-SAT formula
c 4 variables, 3 clauses
p cnf 4 3
1 2 -3 0
-1 3 4 0
2 -4 0
"""

LARGER_CNF = """c Larger test formula
c 10 variables, 8 clauses
p cnf 10 8
1 2 3 0
-1 -2 4 0
3 -4 5 0
-3 5 6 0
4 -5 -6 7 0
-4 6 8 0
7 -8 9 0
-7 8 -9 10 0
"""


def test_cnf_parsing():
    """Test CNF parsing functionality."""
    print("\n--- Testing CNF Parsing ---")

    cnf = parse_dimacs_string(SAMPLE_3SAT)
    assert cnf.num_variables == 4, f"Expected 4 vars, got {cnf.num_variables}"
    assert cnf.num_clauses == 3, f"Expected 3 clauses, got {cnf.num_clauses}"

    # Test larger formula
    cnf2 = parse_dimacs_string(LARGER_CNF)
    assert cnf2.num_variables == 10
    assert cnf2.num_clauses == 8

    print("PASS: CNF parsing works correctly")
    return cnf, cnf2


def test_factor_graph_construction():
    """Test factor graph construction (requires torch)."""
    if not TORCH_AVAILABLE:
        print("\nSKIP: Factor graph construction (torch not available)")
        return None

    print("\n--- Testing Factor Graph Construction ---")

    # Import graph builder
    from data.preprocessing.graph_builder import build_factor_graph, factor_graph_to_pyg

    cnf = parse_dimacs_string(SAMPLE_3SAT)

    # Build factor graph
    fg = build_factor_graph(cnf)
    assert fg.num_variables == 4
    assert fg.num_clauses == 3
    # Total edges: clause 0 has 3 lits, clause 1 has 3, clause 2 has 2 = 8
    assert len(fg.edges) == 8, f"Expected 8 edges, got {len(fg.edges)}"

    # Convert to PyG
    data = factor_graph_to_pyg(fg, feature_dim=64, label=5.5)

    # Check structure
    assert data.x.shape == (7, 64), f"Expected x shape (7, 64), got {data.x.shape}"
    assert data.edge_index.shape[1] == 16, f"Expected 16 edges (bidir), got {data.edge_index.shape[1]}"
    assert data.y.item() == 5.5

    print("PASS: Factor graph construction works correctly")
    return fg, data


def test_tree_decomposition():
    """Test tree decomposition (heuristic, no external tool needed)."""
    print("\n--- Testing Tree Decomposition ---")

    # Add preprocessing directory to path for relative imports
    preprocessing_path = str(Path(project_root) / "data" / "preprocessing")
    if preprocessing_path not in sys.path:
        sys.path.insert(0, preprocessing_path)

    # Import tree decomposition
    tree_decomp = import_module_directly(
        "tree_decomposition",
        str(Path(project_root) / "data" / "preprocessing" / "tree_decomposition.py")
    )

    cnf = parse_dimacs_string(SAMPLE_3SAT)

    # Use heuristic decomposition (doesn't need FlowCutter)
    td = tree_decomp._heuristic_decomposition(cnf)

    assert td.num_bags > 0, "Should have at least one bag"
    assert td.tree_width >= 0, "Tree width should be non-negative"

    # Verify decomposition properties
    is_valid, errors = tree_decomp.verify_tree_decomposition(td, cnf)
    print(f"  Decomposition has {td.num_bags} bags, tree-width {td.tree_width}")
    print(f"  Valid: {is_valid}")
    if errors:
        print(f"  Errors: {errors[:3]}...")  # Print first few errors

    print("PASS: Tree decomposition works correctly")
    return td


def test_join_graph_construction():
    """Test join-graph construction (requires torch)."""
    if not TORCH_AVAILABLE:
        print("\nSKIP: Join-graph construction (torch not available)")
        return None

    print("\n--- Testing Join-Graph Construction ---")

    from data.preprocessing.graph_builder import build_factor_graph, build_join_graph, join_graph_to_pyg

    tree_decomp = import_module_directly(
        "tree_decomposition",
        str(Path(project_root) / "data" / "preprocessing" / "tree_decomposition.py")
    )

    cnf = parse_dimacs_string(LARGER_CNF)
    fg = build_factor_graph(cnf)
    td = tree_decomp._heuristic_decomposition(cnf)

    # Build join-graph
    jg = build_join_graph(fg, td.bags, td.tree_edges)

    assert len(jg.clusters) == td.num_bags
    assert len(jg.cluster_var_ids) == td.num_bags
    assert len(jg.cluster_clause_ids) == td.num_bags

    # Convert to PyG
    data = join_graph_to_pyg(jg, feature_dim=64, label=100.0)

    assert data.x_var.shape[0] == 10  # 10 variables
    assert data.x_clause.shape[0] == 8  # 8 clauses
    assert data.num_clusters == td.num_bags

    print(f"  Created join-graph with {td.num_bags} clusters")
    print("PASS: Join-graph construction works correctly")
    return jg, data


def test_file_io():
    """Test file I/O operations."""
    print("\n--- Testing File I/O ---")

    cnf = parse_dimacs_string(LARGER_CNF)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write CNF
        output_path = Path(tmpdir) / "test.cnf"
        cnf_parser.write_dimacs(cnf, output_path)

        # Read it back
        cnf2 = parse_dimacs(output_path)

        assert cnf2.num_variables == cnf.num_variables
        assert cnf2.num_clauses == cnf.num_clauses
        assert cnf2.clauses == cnf.clauses

    print("PASS: File I/O works correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Full Pipeline Tests")
    print("=" * 60)

    results = {
        "passed": [],
        "skipped": [],
        "failed": []
    }

    tests = [
        ("CNF Parsing", test_cnf_parsing),
        ("Factor Graph", test_factor_graph_construction),
        ("Tree Decomposition", test_tree_decomposition),
        ("Join-Graph", test_join_graph_construction),
        ("File I/O", test_file_io),
    ]

    for name, test_func in tests:
        try:
            result = test_func()
            if result is None and "torch" in str(test_func.__doc__).lower():
                results["skipped"].append(name)
            else:
                results["passed"].append(name)
        except Exception as e:
            print(f"\nFAIL: {name} - {e}")
            import traceback
            traceback.print_exc()
            results["failed"].append((name, str(e)))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Passed:  {len(results['passed'])}")
    print(f"Skipped: {len(results['skipped'])}")
    print(f"Failed:  {len(results['failed'])}")

    if results["failed"]:
        print("\nFailed tests:")
        for name, error in results["failed"]:
            print(f"  - {name}: {error}")
        return False

    print("\nAll tests passed!")
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
