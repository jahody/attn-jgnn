"""
DSharp Solver Interface for #SAT (Model Counting).

This module provides an interface to the DSharp exact #SAT solver
for generating ground truth labels (exact model counts).
"""

import subprocess
import tempfile
import os
import math
import re
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_model_count(
    cnf_file: str | Path,
    timeout: int = 5000,
    dsharp_path: Optional[str] = None
) -> Optional[float]:
    """
    Compute the exact model count using DSharp solver.

    Args:
        cnf_file: Path to CNF file in DIMACS format
        timeout: Timeout in seconds (default 5000 as per paper)
        dsharp_path: Path to DSharp executable (default: 'dsharp')

    Returns:
        log(Z) where Z is the exact model count, or None if solver times out
        or fails

    Note:
        The paper uses log of model count as the label, not the raw count.
        This is because model counts can be astronomically large (>2^1000).
    """
    if dsharp_path is None:
        dsharp_path = 'dsharp'

    cnf_file = Path(cnf_file)
    if not cnf_file.exists():
        raise FileNotFoundError(f"CNF file not found: {cnf_file}")

    try:
        # Run DSharp
        cmd = [dsharp_path, '-count', str(cnf_file)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            logger.warning(f"DSharp failed with return code {result.returncode}")
            logger.debug(f"stderr: {result.stderr}")
            return None

        # Parse output to extract model count
        count = _parse_dsharp_output(result.stdout)
        if count is None:
            return None

        # Return log of model count
        if count == 0:
            return float('-inf')  # UNSAT

        return math.log(count)

    except subprocess.TimeoutExpired:
        logger.info(f"DSharp timed out after {timeout}s for {cnf_file}")
        return None
    except FileNotFoundError:
        logger.error(f"DSharp not found at '{dsharp_path}'")
        raise RuntimeError(
            f"DSharp solver not found. Please install DSharp and ensure "
            f"'{dsharp_path}' is in your PATH."
        )
    except Exception as e:
        logger.error(f"Error running DSharp: {e}")
        return None


def _parse_dsharp_output(output: str) -> Optional[int]:
    """
    Parse DSharp output to extract the model count.

    DSharp output format varies slightly, but typically includes a line like:
    "# solutions: <count>" or "Model Count: <count>"

    Args:
        output: stdout from DSharp

    Returns:
        Model count as integer, or None if parsing fails
    """
    # Try different patterns that DSharp might use
    patterns = [
        r'#\s*solutions\s*[:=]\s*(\d+)',
        r'Model\s+Count\s*[:=]\s*(\d+)',
        r'Number\s+of\s+solutions\s*[:=]\s*(\d+)',
        r'mc\s*[:=]\s*(\d+)',
        r'count\s*[:=]\s*(\d+)',
        r'^(\d+)\s*$',  # Just a number on its own line
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue

    # Try to find any large integer that might be the count
    # (last resort - look for the largest number in output)
    numbers = re.findall(r'\b(\d{2,})\b', output)
    if numbers:
        # Return the largest number found (likely the model count)
        return max(int(n) for n in numbers)

    logger.warning("Could not parse model count from DSharp output")
    logger.debug(f"DSharp output: {output[:500]}")
    return None


def compute_model_count_from_string(
    cnf_string: str,
    timeout: int = 5000,
    dsharp_path: Optional[str] = None
) -> Optional[float]:
    """
    Compute model count for a CNF given as a string.

    Args:
        cnf_string: CNF formula in DIMACS format
        timeout: Timeout in seconds
        dsharp_path: Path to DSharp executable

    Returns:
        log(Z) where Z is the exact model count, or None if solver fails
    """
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.cnf', delete=False
    ) as f:
        f.write(cnf_string)
        temp_path = f.name

    try:
        return compute_model_count(temp_path, timeout, dsharp_path)
    finally:
        os.unlink(temp_path)


def batch_compute_model_counts(
    cnf_files: list,
    timeout: int = 5000,
    dsharp_path: Optional[str] = None,
    n_jobs: int = 1
) -> list:
    """
    Compute model counts for multiple CNF files.

    Args:
        cnf_files: List of paths to CNF files
        timeout: Timeout per file in seconds
        dsharp_path: Path to DSharp executable
        n_jobs: Number of parallel jobs (1 = sequential)

    Returns:
        List of log(Z) values (None for failed/timed out instances)
    """
    if n_jobs == 1:
        results = []
        for cnf_file in cnf_files:
            result = compute_model_count(cnf_file, timeout, dsharp_path)
            results.append(result)
        return results
    else:
        # Parallel execution using multiprocessing
        from multiprocessing import Pool
        from functools import partial

        func = partial(compute_model_count, timeout=timeout, dsharp_path=dsharp_path)
        with Pool(n_jobs) as pool:
            results = pool.map(func, cnf_files)
        return results


def is_dsharp_available(dsharp_path: Optional[str] = None) -> bool:
    """
    Check if DSharp solver is available.

    Args:
        dsharp_path: Path to DSharp executable

    Returns:
        True if DSharp is available and working
    """
    if dsharp_path is None:
        dsharp_path = 'dsharp'

    try:
        result = subprocess.run(
            [dsharp_path, '--help'],
            capture_output=True,
            timeout=10
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_solver_info(dsharp_path: Optional[str] = None) -> dict:
    """
    Get information about the DSharp solver.

    Args:
        dsharp_path: Path to DSharp executable

    Returns:
        Dictionary with solver information
    """
    if dsharp_path is None:
        dsharp_path = 'dsharp'

    info = {
        'name': 'DSharp',
        'type': 'exact #SAT solver',
        'available': False,
        'path': dsharp_path,
        'version': 'unknown'
    }

    try:
        result = subprocess.run(
            [dsharp_path, '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        info['available'] = True

        # Try to extract version
        version_match = re.search(r'version\s*([\d.]+)', result.stdout, re.IGNORECASE)
        if version_match:
            info['version'] = version_match.group(1)

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return info
