"""
CNF Parser for DIMACS format files.

This module provides functionality to parse CNF (Conjunctive Normal Form)
formulas in DIMACS format, which is the standard format for SAT instances.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
import re


@dataclass
class CNF:
    """
    Represents a CNF formula parsed from DIMACS format.

    Attributes:
        num_variables: Number of variables in the formula
        num_clauses: Number of clauses in the formula
        clauses: List of clauses, where each clause is a list of literals
                 (positive int = positive literal, negative int = negative literal)
        comments: Optional list of comment lines from the file
    """
    num_variables: int
    num_clauses: int
    clauses: List[List[int]]
    comments: List[str] = field(default_factory=list)

    def get_variables(self) -> List[int]:
        """Return list of all variable indices (1 to num_variables)."""
        return list(range(1, self.num_variables + 1))

    def get_variable_occurrences(self) -> dict:
        """
        Return dict mapping each variable to list of (clause_idx, polarity) pairs.

        Returns:
            Dictionary where keys are variable indices and values are lists of
            tuples (clause_index, polarity) where polarity is +1 or -1.
        """
        occurrences = {v: [] for v in range(1, self.num_variables + 1)}
        for clause_idx, clause in enumerate(self.clauses):
            for literal in clause:
                var = abs(literal)
                polarity = 1 if literal > 0 else -1
                if var in occurrences:
                    occurrences[var].append((clause_idx, polarity))
        return occurrences

    def get_clause_variables(self) -> List[List[Tuple[int, int]]]:
        """
        Return list where each element is a list of (variable, polarity) pairs for that clause.

        Returns:
            List of lists of (variable_index, polarity) tuples.
        """
        result = []
        for clause in self.clauses:
            clause_vars = [(abs(lit), 1 if lit > 0 else -1) for lit in clause]
            result.append(clause_vars)
        return result


def parse_dimacs(filepath: str | Path) -> CNF:
    """
    Parse a DIMACS CNF format file.

    DIMACS CNF format:
    - Lines starting with 'c' are comments
    - Problem line: 'p cnf <num_vars> <num_clauses>'
    - Clause lines: space-separated literals ending with 0

    Args:
        filepath: Path to the DIMACS CNF file

    Returns:
        CNF object containing the parsed formula

    Raises:
        ValueError: If the file format is invalid
        FileNotFoundError: If the file does not exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"CNF file not found: {filepath}")

    comments = []
    num_variables = None
    num_clauses = None
    clauses = []
    current_clause = []

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Comment line
            if line.startswith('c'):
                comments.append(line[1:].strip())
                continue

            # Problem line
            if line.startswith('p'):
                match = re.match(r'p\s+cnf\s+(\d+)\s+(\d+)', line)
                if not match:
                    raise ValueError(
                        f"Invalid problem line at line {line_num}: {line}"
                    )
                num_variables = int(match.group(1))
                num_clauses = int(match.group(2))
                continue

            # Clause line - contains literals ending with 0
            try:
                literals = [int(x) for x in line.split()]
            except ValueError as e:
                raise ValueError(
                    f"Invalid literal at line {line_num}: {line}"
                ) from e

            for lit in literals:
                if lit == 0:
                    # End of clause
                    if current_clause:
                        clauses.append(current_clause)
                        current_clause = []
                else:
                    current_clause.append(lit)

    # Handle case where last clause doesn't end with 0 (some files do this)
    if current_clause:
        clauses.append(current_clause)

    # Validate
    if num_variables is None or num_clauses is None:
        raise ValueError("Missing problem line (p cnf ...)")

    # Some files have different clause counts than declared - use actual count
    actual_num_clauses = len(clauses)
    if actual_num_clauses != num_clauses:
        # This is common in some benchmarks, so we just use the actual count
        pass

    return CNF(
        num_variables=num_variables,
        num_clauses=actual_num_clauses,
        clauses=clauses,
        comments=comments
    )


def parse_dimacs_string(content: str) -> CNF:
    """
    Parse a DIMACS CNF format string.

    Args:
        content: String containing DIMACS CNF format data

    Returns:
        CNF object containing the parsed formula
    """
    comments = []
    num_variables = None
    num_clauses = None
    clauses = []
    current_clause = []

    for line_num, line in enumerate(content.strip().split('\n'), 1):
        line = line.strip()

        if not line:
            continue

        if line.startswith('c'):
            comments.append(line[1:].strip())
            continue

        if line.startswith('p'):
            match = re.match(r'p\s+cnf\s+(\d+)\s+(\d+)', line)
            if not match:
                raise ValueError(f"Invalid problem line at line {line_num}: {line}")
            num_variables = int(match.group(1))
            num_clauses = int(match.group(2))
            continue

        try:
            literals = [int(x) for x in line.split()]
        except ValueError as e:
            raise ValueError(f"Invalid literal at line {line_num}: {line}") from e

        for lit in literals:
            if lit == 0:
                if current_clause:
                    clauses.append(current_clause)
                    current_clause = []
            else:
                current_clause.append(lit)

    if current_clause:
        clauses.append(current_clause)

    if num_variables is None or num_clauses is None:
        raise ValueError("Missing problem line (p cnf ...)")

    return CNF(
        num_variables=num_variables,
        num_clauses=len(clauses),
        clauses=clauses,
        comments=comments
    )


def write_dimacs(cnf: CNF, filepath: str | Path, comments: Optional[List[str]] = None) -> None:
    """
    Write a CNF formula to DIMACS format.

    Args:
        cnf: CNF object to write
        filepath: Output file path
        comments: Optional additional comments to include
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        # Write comments
        all_comments = (comments or []) + cnf.comments
        for comment in all_comments:
            f.write(f"c {comment}\n")

        # Write problem line
        f.write(f"p cnf {cnf.num_variables} {cnf.num_clauses}\n")

        # Write clauses
        for clause in cnf.clauses:
            f.write(' '.join(map(str, clause)) + ' 0\n')
