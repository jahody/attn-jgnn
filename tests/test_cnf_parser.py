"""
Simple tests for CNF parser that don't require torch.
"""

import sys
import tempfile
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Import directly from the module file to avoid importing torch
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cnf_parser",
    str(Path(project_root) / "data" / "preprocessing" / "cnf_parser.py")
)
cnf_parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnf_parser)

parse_dimacs = cnf_parser.parse_dimacs
parse_dimacs_string = cnf_parser.parse_dimacs_string
CNF = cnf_parser.CNF
write_dimacs = cnf_parser.write_dimacs

# Sample CNF for testing
SAMPLE_CNF = """c This is a comment
c Another comment
p cnf 4 3
1 2 -3 0
-1 3 4 0
2 -4 0
"""

SIMPLE_CNF = """p cnf 3 2
1 2 0
-1 -2 3 0
"""


def test_parse_dimacs_string():
    """Test parsing CNF from string."""
    cnf = parse_dimacs_string(SAMPLE_CNF)

    assert cnf.num_variables == 4, f"Expected 4 variables, got {cnf.num_variables}"
    assert cnf.num_clauses == 3, f"Expected 3 clauses, got {cnf.num_clauses}"
    assert len(cnf.clauses) == 3
    assert cnf.clauses[0] == [1, 2, -3], f"Clause 0 mismatch: {cnf.clauses[0]}"
    assert cnf.clauses[1] == [-1, 3, 4], f"Clause 1 mismatch: {cnf.clauses[1]}"
    assert cnf.clauses[2] == [2, -4], f"Clause 2 mismatch: {cnf.clauses[2]}"
    assert len(cnf.comments) == 2
    print("PASS: parse_dimacs_string works correctly")


def test_parse_dimacs_file():
    """Test parsing CNF from file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
        f.write(SAMPLE_CNF)
        temp_path = f.name

    try:
        cnf = parse_dimacs(temp_path)
        assert cnf.num_variables == 4
        assert cnf.num_clauses == 3
        print("PASS: parse_dimacs (file) works correctly")
    finally:
        os.unlink(temp_path)


def test_get_variables():
    """Test getting variable list."""
    cnf = parse_dimacs_string(SAMPLE_CNF)
    variables = cnf.get_variables()

    assert variables == [1, 2, 3, 4], f"Expected [1,2,3,4], got {variables}"
    print("PASS: get_variables works correctly")


def test_get_variable_occurrences():
    """Test getting variable occurrences."""
    cnf = parse_dimacs_string(SIMPLE_CNF)
    occurrences = cnf.get_variable_occurrences()

    # Variable 1: appears positive in clause 0, negative in clause 1
    assert (0, 1) in occurrences[1], "Variable 1 should appear positive in clause 0"
    assert (1, -1) in occurrences[1], "Variable 1 should appear negative in clause 1"

    # Variable 2: appears positive in clause 0, negative in clause 1
    assert (0, 1) in occurrences[2], "Variable 2 should appear positive in clause 0"
    assert (1, -1) in occurrences[2], "Variable 2 should appear negative in clause 1"

    # Variable 3: appears positive in clause 1 only
    v3_in_clause1 = [x for x in occurrences[3] if x[0] == 1]
    assert len(v3_in_clause1) == 1, "Variable 3 should appear once in clause 1"
    print("PASS: get_variable_occurrences works correctly")


def test_write_dimacs():
    """Test writing CNF to file."""
    cnf = parse_dimacs_string(SAMPLE_CNF)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.cnf"
        write_dimacs(cnf, output_path)

        # Read it back
        cnf2 = parse_dimacs(output_path)
        assert cnf2.num_variables == cnf.num_variables
        assert cnf2.num_clauses == cnf.num_clauses
        assert cnf2.clauses == cnf.clauses
        print("PASS: write_dimacs works correctly")


def test_get_clause_variables():
    """Test getting clause variables with polarities."""
    cnf = parse_dimacs_string(SIMPLE_CNF)
    clause_vars = cnf.get_clause_variables()

    assert len(clause_vars) == 2
    # Clause 0: 1 2 -> [(1, +1), (2, +1)]
    assert (1, 1) in clause_vars[0], "Clause 0 should contain (1, +1)"
    assert (2, 1) in clause_vars[0], "Clause 0 should contain (2, +1)"
    # Clause 1: -1 -2 3 -> [(1, -1), (2, -1), (3, +1)]
    assert (1, -1) in clause_vars[1], "Clause 1 should contain (1, -1)"
    assert (2, -1) in clause_vars[1], "Clause 1 should contain (2, -1)"
    assert (3, 1) in clause_vars[1], "Clause 1 should contain (3, +1)"
    print("PASS: get_clause_variables works correctly")


def run_all_tests():
    """Run all tests."""
    print("Running CNF Parser Tests...")
    print("=" * 50)

    test_parse_dimacs_string()
    test_parse_dimacs_file()
    test_get_variables()
    test_get_variable_occurrences()
    test_write_dimacs()
    test_get_clause_variables()

    print("=" * 50)
    print("All CNF parser tests passed!")


if __name__ == '__main__':
    run_all_tests()
