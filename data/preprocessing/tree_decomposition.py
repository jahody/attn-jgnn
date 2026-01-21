"""
Tree Decomposition Interface for FlowCutter.

This module provides an interface to the FlowCutter tree decomposition tool
for constructing join-graphs with controlled tree-width.
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass
import logging

try:
    from .cnf_parser import CNF
except ImportError:
    # Support direct import for testing
    from cnf_parser import CNF

logger = logging.getLogger(__name__)


@dataclass
class TreeDecomposition:
    """
    Tree decomposition result.

    Attributes:
        bags: List of bags (clusters), each containing variable indices (0-indexed)
        tree_edges: Edges between bags forming a tree structure
        tree_width: Width of the decomposition (max bag size - 1)
        num_bags: Number of bags in the decomposition
    """
    bags: List[Set[int]]
    tree_edges: List[Tuple[int, int]]
    tree_width: int
    num_bags: int


def _build_primal_graph(cnf: CNF) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Build the primal graph of a CNF formula.

    The primal graph has variables as nodes, with edges between
    variables that appear together in at least one clause.

    Args:
        cnf: Parsed CNF formula

    Returns:
        Tuple of (num_nodes, edge_list)
    """
    edges = set()
    for clause in cnf.clauses:
        vars_in_clause = [abs(lit) - 1 for lit in clause]  # 0-indexed
        for i, v1 in enumerate(vars_in_clause):
            for v2 in vars_in_clause[i + 1:]:
                if v1 != v2:
                    edge = (min(v1, v2), max(v1, v2))
                    edges.add(edge)

    return cnf.num_variables, list(edges)


def _write_graph_file(num_nodes: int, edges: List[Tuple[int, int]], filepath: Path) -> None:
    """
    Write graph in PACE format for FlowCutter.

    PACE format:
    - First line: p tw <num_nodes> <num_edges>
    - Edge lines: <node1> <node2> (1-indexed)
    """
    with open(filepath, 'w') as f:
        f.write(f"p tw {num_nodes} {len(edges)}\n")
        for v1, v2 in edges:
            f.write(f"{v1 + 1} {v2 + 1}\n")  # Convert to 1-indexed


def _parse_tree_decomposition(filepath: Path) -> TreeDecomposition:
    """
    Parse tree decomposition output in td format.

    td format:
    - First line: s td <num_bags> <tree_width+1> <num_nodes>
    - Bag lines: b <bag_id> <node1> <node2> ...
    - Edge lines: <bag1> <bag2>
    """
    bags = {}
    tree_edges = []
    tree_width = 0
    num_bags = 0

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue

            parts = line.split()
            if parts[0] == 's':
                # Solution line: s td num_bags width num_nodes
                num_bags = int(parts[2])
                tree_width = int(parts[3]) - 1  # width = max_bag_size - 1
            elif parts[0] == 'b':
                # Bag line: b bag_id node1 node2 ...
                bag_id = int(parts[1]) - 1  # Convert to 0-indexed
                nodes = set(int(n) - 1 for n in parts[2:])  # Convert to 0-indexed
                bags[bag_id] = nodes
            else:
                # Edge line: bag1 bag2
                try:
                    b1, b2 = int(parts[0]) - 1, int(parts[1]) - 1
                    tree_edges.append((b1, b2))
                except (ValueError, IndexError):
                    continue

    # Convert bags dict to list
    bag_list = [bags.get(i, set()) for i in range(num_bags)]

    return TreeDecomposition(
        bags=bag_list,
        tree_edges=tree_edges,
        tree_width=tree_width,
        num_bags=num_bags
    )


def decompose_with_flowcutter(
    cnf: CNF,
    target_width: Optional[int] = None,
    flowcutter_path: Optional[str] = None,
    timeout: int = 300
) -> TreeDecomposition:
    """
    Compute tree decomposition using FlowCutter.

    Args:
        cnf: Parsed CNF formula
        target_width: Target tree-width (FlowCutter will try to achieve this)
        flowcutter_path: Path to FlowCutter executable (default: 'flow_cutter_pace17')
        timeout: Timeout in seconds

    Returns:
        TreeDecomposition object

    Raises:
        RuntimeError: If FlowCutter fails or is not available
        TimeoutError: If decomposition times out
    """
    if flowcutter_path is None:
        flowcutter_path = 'flow_cutter_pace17'

    # Build primal graph
    num_nodes, edges = _build_primal_graph(cnf)

    # Handle edge case of no edges (each variable independent)
    if not edges:
        # Each variable in its own bag, arbitrary tree structure
        bags = [set([i]) for i in range(num_nodes)]
        tree_edges = [(i, i + 1) for i in range(num_nodes - 1)] if num_nodes > 1 else []
        return TreeDecomposition(
            bags=bags,
            tree_edges=tree_edges,
            tree_width=0,
            num_bags=num_nodes
        )

    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_file = Path(tmpdir) / "graph.gr"
        td_file = Path(tmpdir) / "decomposition.td"

        # Write graph file
        _write_graph_file(num_nodes, edges, graph_file)

        try:
            # Run FlowCutter
            cmd = [flowcutter_path]
            if target_width is not None:
                cmd.extend(['-w', str(target_width)])

            with open(graph_file, 'r') as infile, open(td_file, 'w') as outfile:
                process = subprocess.run(
                    cmd,
                    stdin=infile,
                    stdout=outfile,
                    stderr=subprocess.PIPE,
                    timeout=timeout
                )

            if process.returncode != 0:
                logger.warning(
                    f"FlowCutter returned non-zero exit code: {process.returncode}"
                )

            # Parse result
            return _parse_tree_decomposition(td_file)

        except FileNotFoundError:
            logger.warning(
                f"FlowCutter not found at '{flowcutter_path}'. "
                "Using fallback heuristic decomposition."
            )
            return _heuristic_decomposition(cnf)
        except subprocess.TimeoutExpired:
            logger.warning(
                f"FlowCutter timed out after {timeout}s. "
                "Using fallback heuristic decomposition."
            )
            return _heuristic_decomposition(cnf)


def _heuristic_decomposition(cnf: CNF, max_bag_size: int = 20) -> TreeDecomposition:
    """
    Fallback heuristic tree decomposition when FlowCutter is unavailable.

    Uses a simple min-degree elimination heuristic.

    Args:
        cnf: Parsed CNF formula
        max_bag_size: Maximum allowed bag size

    Returns:
        TreeDecomposition object (may not be optimal)
    """
    num_vars = cnf.num_variables

    # Build adjacency list for primal graph
    adj = {i: set() for i in range(num_vars)}
    for clause in cnf.clauses:
        vars_in_clause = [abs(lit) - 1 for lit in clause]
        for i, v1 in enumerate(vars_in_clause):
            for v2 in vars_in_clause[i + 1:]:
                if v1 != v2:
                    adj[v1].add(v2)
                    adj[v2].add(v1)

    # Min-degree elimination ordering
    remaining = set(range(num_vars))
    bags = []
    elimination_order = []

    while remaining:
        # Find vertex with minimum degree
        min_degree = float('inf')
        min_vertex = None
        for v in remaining:
            degree = len(adj[v] & remaining)
            if degree < min_degree:
                min_degree = degree
                min_vertex = v

        if min_vertex is None:
            break

        # Create bag: vertex + neighbors
        neighbors = adj[min_vertex] & remaining
        bag = {min_vertex} | neighbors

        # Limit bag size
        if len(bag) > max_bag_size:
            bag = {min_vertex} | set(list(neighbors)[:max_bag_size - 1])

        bags.append(bag)
        elimination_order.append(min_vertex)

        # Add fill-in edges (make neighbors clique)
        neighbor_list = list(neighbors)
        for i, n1 in enumerate(neighbor_list):
            for n2 in neighbor_list[i + 1:]:
                adj[n1].add(n2)
                adj[n2].add(n1)

        remaining.remove(min_vertex)

    # Build tree structure (simple chain for now)
    tree_edges = [(i, i + 1) for i in range(len(bags) - 1)] if len(bags) > 1 else []

    # Merge small consecutive bags to reduce number of bags
    merged_bags = []
    current_bag = set()
    for bag in bags:
        if len(current_bag | bag) <= max_bag_size:
            current_bag = current_bag | bag
        else:
            if current_bag:
                merged_bags.append(current_bag)
            current_bag = bag
    if current_bag:
        merged_bags.append(current_bag)

    tree_width = max(len(bag) - 1 for bag in merged_bags) if merged_bags else 0
    tree_edges = [(i, i + 1) for i in range(len(merged_bags) - 1)] if len(merged_bags) > 1 else []

    return TreeDecomposition(
        bags=merged_bags,
        tree_edges=tree_edges,
        tree_width=tree_width,
        num_bags=len(merged_bags)
    )


def decompose_cnf(
    cnf: CNF,
    target_width: Optional[int] = None,
    flowcutter_path: Optional[str] = None,
    timeout: int = 300,
    use_fallback: bool = True
) -> TreeDecomposition:
    """
    Main entry point for tree decomposition.

    Attempts to use FlowCutter, falls back to heuristic if unavailable.

    Args:
        cnf: Parsed CNF formula
        target_width: Target tree-width
        flowcutter_path: Path to FlowCutter executable
        timeout: Timeout in seconds
        use_fallback: Whether to use heuristic fallback

    Returns:
        TreeDecomposition object
    """
    try:
        return decompose_with_flowcutter(cnf, target_width, flowcutter_path, timeout)
    except Exception as e:
        if use_fallback:
            logger.warning(f"FlowCutter failed: {e}. Using heuristic decomposition.")
            return _heuristic_decomposition(cnf)
        raise


def verify_tree_decomposition(td: TreeDecomposition, cnf: CNF) -> Tuple[bool, List[str]]:
    """
    Verify that a tree decomposition satisfies required properties.

    Properties checked:
    1. Coverage: Each clause's variables are contained in at least one bag
    2. Connectivity: For any variable, bags containing it form a connected subtree

    Args:
        td: Tree decomposition to verify
        cnf: Original CNF formula

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Check coverage
    for clause_idx, clause in enumerate(cnf.clauses):
        clause_vars = set(abs(lit) - 1 for lit in clause)
        covered = False
        for bag in td.bags:
            if clause_vars.issubset(bag):
                covered = True
                break
        if not covered:
            errors.append(f"Clause {clause_idx} not covered: vars {clause_vars}")

    # Check connectivity (running intersection property)
    # Build adjacency for tree
    adj = {i: set() for i in range(len(td.bags))}
    for b1, b2 in td.tree_edges:
        adj[b1].add(b2)
        adj[b2].add(b1)

    # For each variable, check that bags containing it form connected subtree
    for var in range(cnf.num_variables):
        bags_with_var = [i for i, bag in enumerate(td.bags) if var in bag]
        if len(bags_with_var) <= 1:
            continue

        # BFS from first bag to check connectivity within bags containing var
        visited = {bags_with_var[0]}
        queue = [bags_with_var[0]]
        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor not in visited and neighbor in bags_with_var:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if len(visited) != len(bags_with_var):
            errors.append(
                f"Variable {var} violates connectivity: "
                f"bags {bags_with_var}, connected {list(visited)}"
            )

    return len(errors) == 0, errors
