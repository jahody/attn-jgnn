"""
Generate ground truth labels (model counts) for CNF instances.

Uses pycosat for small instances (enumeration-based) or DSharp for larger ones.
"""

import os
import json
import math
import logging
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_dimacs(filepath: Path) -> tuple:
    """Parse DIMACS CNF file."""
    num_vars = 0
    num_clauses = 0
    clauses = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p'):
                parts = line.split()
                num_vars = int(parts[2])
                num_clauses = int(parts[3])
            else:
                # Parse clause
                literals = list(map(int, line.split()))
                if literals and literals[-1] == 0:
                    literals = literals[:-1]
                if literals:
                    clauses.append(literals)

    return num_vars, clauses


def count_models_pycosat(filepath: Path, max_count: int = 1000000) -> Optional[int]:
    """
    Count models using pycosat (AllSAT enumeration).

    Only suitable for small instances (<50 variables typically).
    """
    try:
        import pycosat
    except ImportError:
        logger.error("pycosat not available")
        return None

    num_vars, clauses = parse_dimacs(filepath)

    if num_vars > 50:
        logger.warning(f"Instance too large for enumeration: {num_vars} vars")
        return None

    # Count satisfying assignments
    count = 0
    for solution in pycosat.itersolve(clauses):
        count += 1
        if count >= max_count:
            logger.warning(f"Reached max count {max_count} for {filepath}")
            break

    return count


def count_models_dpll(filepath: Path, max_count: int = 1000000) -> Optional[int]:
    """
    Simple DPLL-based model counter for small instances.
    """
    num_vars, clauses = parse_dimacs(filepath)

    if num_vars > 30:
        logger.warning(f"Instance too large for DPLL: {num_vars} vars")
        return None

    # Convert to set of frozensets for faster operations
    clause_set = [set(c) for c in clauses]

    def unit_propagate(clauses, assignment):
        """Apply unit propagation."""
        changed = True
        while changed:
            changed = False
            for clause in clauses:
                if len(clause) == 1:
                    lit = next(iter(clause))
                    if lit in assignment:
                        continue
                    if -lit in assignment:
                        return None, None  # Conflict
                    assignment.add(lit)
                    changed = True
                    # Simplify
                    new_clauses = []
                    for c in clauses:
                        if lit in c:
                            continue  # Clause satisfied
                        new_c = c - {-lit}
                        if not new_c:
                            return None, None  # Empty clause (UNSAT)
                        new_clauses.append(new_c)
                    clauses = new_clauses
                    break
        return clauses, assignment

    def dpll_count(clauses, assignment, count_ref):
        """Recursive DPLL with model counting."""
        if count_ref[0] >= max_count:
            return

        # Unit propagation
        clauses, assignment = unit_propagate([set(c) for c in clauses], set(assignment))
        if clauses is None:
            return  # Conflict

        if not clauses:
            count_ref[0] += 1
            return

        # Find unassigned variable
        assigned_vars = {abs(lit) for lit in assignment}
        all_vars = set()
        for c in clauses:
            all_vars.update(abs(lit) for lit in c)

        unassigned = all_vars - assigned_vars
        if not unassigned:
            count_ref[0] += 1
            return

        var = min(unassigned)  # Choose variable

        # Try var = True
        new_clauses = []
        for c in clauses:
            if var in c:
                continue
            new_c = set(c) - {-var}
            if not new_c:
                pass  # Will backtrack
            else:
                new_clauses.append(new_c)
        if new_clauses or not any(var in c or -var in c for c in clauses):
            dpll_count(new_clauses, assignment | {var}, count_ref)

        # Try var = False
        new_clauses = []
        for c in clauses:
            if -var in c:
                continue
            new_c = set(c) - {var}
            if not new_c:
                pass
            else:
                new_clauses.append(new_c)
        if new_clauses or not any(var in c or -var in c for c in clauses):
            dpll_count(new_clauses, assignment | {-var}, count_ref)

    count_ref = [0]
    dpll_count(clause_set, set(), count_ref)
    return count_ref[0]


def count_models_sharpsat(filepath: Path, timeout: int = 60) -> Optional[int]:
    """Try using sharpSAT if available."""
    import subprocess

    try:
        result = subprocess.run(
            ['sharpSAT', str(filepath)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        # Parse output for model count
        for line in result.stdout.split('\n'):
            if 'solutions' in line.lower() or line.strip().isdigit():
                try:
                    count = int(line.strip().split()[-1])
                    return count
                except:
                    pass
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def generate_labels(
    data_dir: str,
    output_file: str,
    method: str = "auto",
    timeout: int = 60,
    max_instances: Optional[int] = None
) -> Dict[str, float]:
    """
    Generate labels for all CNF files in a directory.

    Args:
        data_dir: Directory containing CNF files
        output_file: Output JSON file for labels
        method: "pycosat", "dpll", "sharpsat", or "auto"
        timeout: Timeout per instance
        max_instances: Maximum number of instances to process

    Returns:
        Dictionary mapping filename to log(model_count)
    """
    data_path = Path(data_dir)
    cnf_files = list(data_path.rglob("*.cnf"))

    if max_instances:
        cnf_files = cnf_files[:max_instances]

    logger.info(f"Processing {len(cnf_files)} CNF files from {data_dir}")

    labels = {}
    failed = 0

    for cnf_file in tqdm(cnf_files, desc="Generating labels"):
        try:
            num_vars, _ = parse_dimacs(cnf_file)

            count = None

            if method == "auto":
                # Choose method based on size
                if num_vars <= 30:
                    count = count_models_pycosat(cnf_file)
                elif num_vars <= 50:
                    count = count_models_pycosat(cnf_file)
                else:
                    count = count_models_sharpsat(cnf_file, timeout)
            elif method == "pycosat":
                count = count_models_pycosat(cnf_file)
            elif method == "dpll":
                count = count_models_dpll(cnf_file)
            elif method == "sharpsat":
                count = count_models_sharpsat(cnf_file, timeout)

            if count is not None and count > 0:
                labels[cnf_file.name] = math.log(count)
                logger.debug(f"{cnf_file.name}: count={count}, log={math.log(count):.4f}")
            elif count == 0:
                labels[cnf_file.name] = float('-inf')
                logger.debug(f"{cnf_file.name}: UNSAT")
            else:
                failed += 1
                logger.warning(f"Failed to count models for {cnf_file.name}")

        except Exception as e:
            failed += 1
            logger.error(f"Error processing {cnf_file}: {e}")

    # Save labels
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)

    logger.info(f"Generated {len(labels)} labels, {failed} failed")
    logger.info(f"Labels saved to {output_file}")

    return labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate model count labels")
    parser.add_argument("--data-dir", default="data/satlib/raw/rnd3sat", help="Input directory")
    parser.add_argument("--output", default="data/satlib/processed/labels.json", help="Output labels file")
    parser.add_argument("--method", default="auto", choices=["auto", "pycosat", "dpll", "sharpsat"])
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per instance")
    parser.add_argument("--max-instances", type=int, default=None, help="Max instances to process")

    args = parser.parse_args()

    generate_labels(
        args.data_dir,
        args.output,
        args.method,
        args.timeout,
        args.max_instances
    )
