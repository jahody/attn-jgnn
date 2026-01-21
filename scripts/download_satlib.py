"""
Download SATLIB RND3SAT benchmark data.

SATLIB contains uniform random 3-SAT instances. For the paper evaluation,
we need satisfiable instances with variables in range 100-600.

Available sizes in SATLIB:
- uf20, uf50, uf75, uf100, uf125, uf150, uf175, uf200, uf225, uf250
"""

import os
import requests
import tarfile
import zipfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SATLIB base URLs
SATLIB_BASE = "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT"

# Available uniform random 3-SAT datasets (satisfiable)
# Format: (name, url_suffix, approx_num_instances)
UF_DATASETS = [
    ("uf100-430", "uf100-430.tar.gz", 1000),  # 100 vars, 430 clauses
    ("uf125-538", "uf125-538.tar.gz", 100),   # 125 vars
    ("uf150-645", "uf150-645.tar.gz", 100),   # 150 vars
    ("uf175-753", "uf175-753.tar.gz", 100),   # 175 vars
    ("uf200-860", "uf200-860.tar.gz", 100),   # 200 vars
    ("uf225-960", "uf225-960.tar.gz", 100),   # 225 vars
    ("uf250-1065", "uf250-1065.tar.gz", 100), # 250 vars
]


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file from URL to destination path."""
    try:
        logger.info(f"Downloading {url}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    """Extract tar.gz or zip archive."""
    try:
        if archive_path.suffix == '.gz' or str(archive_path).endswith('.tar.gz'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        elif archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            logger.error(f"Unknown archive format: {archive_path}")
            return False

        logger.info(f"Extracted to {extract_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract {archive_path}: {e}")
        return False


def download_satlib_rnd3sat(output_dir: str = "data/satlib/raw/rnd3sat"):
    """Download SATLIB RND3SAT benchmark data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    temp_dir = output_path / "_temp"
    temp_dir.mkdir(exist_ok=True)

    total_files = 0

    for name, url_suffix, expected_count in UF_DATASETS:
        url = f"{SATLIB_BASE}/{url_suffix}"
        archive_path = temp_dir / url_suffix

        # Download
        if not archive_path.exists():
            if not download_file(url, archive_path):
                # Try alternative URL format
                alt_url = f"https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/{url_suffix}"
                if not download_file(alt_url, archive_path):
                    continue

        # Extract
        extract_dir = output_path / name
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            if extract_archive(archive_path, extract_dir):
                # Count CNF files
                cnf_files = list(extract_dir.rglob("*.cnf"))
                logger.info(f"{name}: {len(cnf_files)} CNF files")
                total_files += len(cnf_files)

    # Cleanup temp
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    logger.info(f"Total: {total_files} CNF files downloaded")
    return total_files


def generate_synthetic_rnd3sat(
    output_dir: str = "data/satlib/raw/rnd3sat",
    num_instances: int = 500,
    var_range: tuple = (100, 250),
    clause_ratio: float = 4.3,  # Phase transition ratio for 3-SAT
    seed: int = 42
):
    """
    Generate synthetic random 3-SAT instances.

    Uses the phase transition ratio (~4.3 clauses per variable) where
    random 3-SAT is typically satisfiable but hard.
    """
    import random
    random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {num_instances} synthetic 3-SAT instances")

    for i in range(num_instances):
        # Random number of variables in range
        num_vars = random.randint(var_range[0], var_range[1])
        num_clauses = int(num_vars * clause_ratio)

        # Generate random 3-SAT clauses
        clauses = []
        for _ in range(num_clauses):
            # Pick 3 distinct variables
            vars_in_clause = random.sample(range(1, num_vars + 1), 3)
            # Random polarity
            clause = [v if random.random() > 0.5 else -v for v in vars_in_clause]
            clauses.append(clause)

        # Write DIMACS CNF file
        filename = output_path / f"synth_{num_vars}_{i:04d}.cnf"
        with open(filename, 'w') as f:
            f.write(f"c Synthetic random 3-SAT instance\n")
            f.write(f"c Variables: {num_vars}, Clauses: {num_clauses}\n")
            f.write(f"p cnf {num_vars} {num_clauses}\n")
            for clause in clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")

    logger.info(f"Generated {num_instances} instances in {output_path}")
    return num_instances


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download or generate SATLIB data")
    parser.add_argument("--output", default="data/satlib/raw/rnd3sat", help="Output directory")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data instead of downloading")
    parser.add_argument("--num-instances", type=int, default=500, help="Number of synthetic instances")
    parser.add_argument("--min-vars", type=int, default=100, help="Minimum variables")
    parser.add_argument("--max-vars", type=int, default=250, help="Maximum variables")

    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_rnd3sat(
            args.output,
            args.num_instances,
            (args.min_vars, args.max_vars)
        )
    else:
        # Try downloading, fall back to synthetic if fails
        count = download_satlib_rnd3sat(args.output)
        if count == 0:
            logger.warning("Download failed, generating synthetic data instead")
            generate_synthetic_rnd3sat(args.output, args.num_instances)
