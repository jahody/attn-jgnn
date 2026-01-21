"""
BIRD Benchmark Dataset for #SAT (Model Counting).

BIRD (BIg Bench for LaRge-scale Database Grounded Text-to-SQL Evaluation)
contains CNF formulas from real-world model counting applications.

Categories:
- DQMR networks
- Grid networks
- Bit-blasted versions of SMTLIB benchmarks
- ISCAS89 combinatorial circuits
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Callable, Union
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm

from ..preprocessing.cnf_parser import parse_dimacs, CNF
from ..preprocessing.graph_builder import (
    build_factor_graph,
    factor_graph_to_pyg,
    build_join_graph,
    join_graph_to_pyg
)
from ..preprocessing.tree_decomposition import decompose_cnf, TreeDecomposition
from ..solvers.dsharp_solver import compute_model_count

logger = logging.getLogger(__name__)


class BIRDDataset(Dataset):
    """
    PyTorch Geometric Dataset for BIRD benchmark.

    This dataset:
    - Loads CNF files from BIRD benchmark directory
    - Parses DIMACS CNF format
    - Converts CNF to factor graph representation
    - Optionally applies tree decomposition for join-graph structure
    - Returns PyTorch Geometric Data objects

    Args:
        root: Root directory where the dataset is stored
        categories: List of categories to include (default: all 8 categories)
        split: Dataset split ('train' or 'test')
        split_ratio: Dictionary with 'train' and 'test' ratios (default: 0.7/0.3)
        use_join_graph: Whether to use join-graph representation (default: False)
        feature_dim: Dimension of node features (default: 64)
        transform: Optional transform to apply to data
        pre_transform: Optional pre-transform to apply
        pre_filter: Optional filter function
        force_reload: Whether to force reprocessing
        label_file: Optional path to pre-computed labels file
    """

    # Default categories from BIRD benchmark
    CATEGORIES = [
        'dqmr',       # DQMR networks
        'grid',       # Grid networks
        'smtlib',     # Bit-blasted SMTLIB benchmarks
        'iscas89',    # ISCAS89 combinatorial circuits
        'blasted_case',  # Additional categories that may exist
        'blockmap',
        'logistics',
        'plan_recognition'
    ]

    def __init__(
        self,
        root: str,
        categories: Optional[List[str]] = None,
        split: str = 'train',
        split_ratio: Optional[Dict[str, float]] = None,
        use_join_graph: bool = False,
        feature_dim: int = 64,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        label_file: Optional[str] = None,
    ):
        self.categories = categories or self.CATEGORIES
        self.split = split
        self.split_ratio = split_ratio or {'train': 0.7, 'test': 0.3}
        self.use_join_graph = use_join_graph
        self.feature_dim = feature_dim
        self.label_file = label_file
        self._labels_cache = None

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data info
        self._load_data_info()

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw CNF files."""
        files = []
        raw_path = Path(self.raw_dir)
        if raw_path.exists():
            for category in self.categories:
                category_path = raw_path / category
                if category_path.exists():
                    for cnf_file in category_path.glob('*.cnf'):
                        files.append(str(cnf_file.relative_to(raw_path)))
        return files if files else ['placeholder.cnf']

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.split}_data.pt', f'{self.split}_info.json']

    def _load_labels(self) -> Dict[str, float]:
        """Load pre-computed labels from file."""
        if self._labels_cache is not None:
            return self._labels_cache

        if self.label_file and os.path.exists(self.label_file):
            with open(self.label_file, 'r') as f:
                self._labels_cache = json.load(f)
            return self._labels_cache

        # Try default location
        default_label_file = os.path.join(self.processed_dir, 'labels.json')
        if os.path.exists(default_label_file):
            with open(default_label_file, 'r') as f:
                self._labels_cache = json.load(f)
            return self._labels_cache

        return {}

    def _load_data_info(self):
        """Load information about processed data."""
        info_path = os.path.join(self.processed_dir, f'{self.split}_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self._data_info = json.load(f)
        else:
            self._data_info = {'num_samples': 0, 'files': []}

    def download(self):
        """Download BIRD benchmark data."""
        # BIRD benchmark should be downloaded manually
        # Following NSNet experimental settings
        logger.info(
            "BIRD benchmark data should be downloaded manually.\n"
            "Please download from the BIRD benchmark repository and place "
            f"the CNF files in: {self.raw_dir}"
        )

    def process(self):
        """Process raw CNF files into PyG Data objects."""
        raw_path = Path(self.raw_dir)
        processed_path = Path(self.processed_dir)
        processed_path.mkdir(parents=True, exist_ok=True)

        labels = self._load_labels()

        # Collect all CNF files by category
        all_files = []
        for category in self.categories:
            category_path = raw_path / category
            if category_path.exists():
                cnf_files = list(category_path.glob('*.cnf'))
                for cnf_file in cnf_files:
                    all_files.append((category, cnf_file))

        if not all_files:
            logger.warning(f"No CNF files found in {raw_path}")
            self._save_empty_dataset()
            return

        # Split files into train/test
        n_total = len(all_files)
        n_train = int(n_total * self.split_ratio['train'])

        # Use deterministic splitting (sorted by filename)
        all_files.sort(key=lambda x: str(x[1]))

        if self.split == 'train':
            files_to_process = all_files[:n_train]
        else:
            files_to_process = all_files[n_train:]

        # Process files
        data_list = []
        file_list = []

        for category, cnf_file in tqdm(files_to_process, desc=f'Processing {self.split}'):
            try:
                data = self._process_single_file(cnf_file, category, labels)
                if data is not None:
                    if self.pre_filter is None or self.pre_filter(data):
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)
                        data_list.append(data)
                        file_list.append(str(cnf_file.relative_to(raw_path)))
            except Exception as e:
                logger.warning(f"Failed to process {cnf_file}: {e}")

        # Save processed data
        if data_list:
            torch.save(data_list, processed_path / f'{self.split}_data.pt')

        # Save info
        info = {
            'num_samples': len(data_list),
            'files': file_list,
            'categories': self.categories,
            'split': self.split,
            'use_join_graph': self.use_join_graph,
            'feature_dim': self.feature_dim
        }
        with open(processed_path / f'{self.split}_info.json', 'w') as f:
            json.dump(info, f, indent=2)

        self._data_info = info

    def _process_single_file(
        self,
        cnf_file: Path,
        category: str,
        labels: Dict[str, float]
    ) -> Optional[Data]:
        """Process a single CNF file into a PyG Data object."""
        # Parse CNF
        cnf = parse_dimacs(cnf_file)

        # Get label
        file_key = str(cnf_file.name)
        label = labels.get(file_key)

        # Build factor graph
        factor_graph = build_factor_graph(cnf)

        if self.use_join_graph:
            # Get tree decomposition
            td = decompose_cnf(cnf)
            join_graph = build_join_graph(factor_graph, td.bags, td.tree_edges)
            data = join_graph_to_pyg(join_graph, self.feature_dim, label)
        else:
            data = factor_graph_to_pyg(factor_graph, self.feature_dim, label)

        # Add metadata
        data.category = category
        data.filename = cnf_file.name
        data.num_variables_cnf = cnf.num_variables
        data.num_clauses_cnf = cnf.num_clauses

        return data

    def _save_empty_dataset(self):
        """Save empty dataset when no files are found."""
        processed_path = Path(self.processed_dir)
        processed_path.mkdir(parents=True, exist_ok=True)

        torch.save([], processed_path / f'{self.split}_data.pt')
        info = {
            'num_samples': 0,
            'files': [],
            'categories': self.categories,
            'split': self.split
        }
        with open(processed_path / f'{self.split}_info.json', 'w') as f:
            json.dump(info, f, indent=2)

        self._data_info = info

    def len(self) -> int:
        return self._data_info.get('num_samples', 0)

    def get(self, idx: int) -> Data:
        data_path = os.path.join(self.processed_dir, f'{self.split}_data.pt')
        data_list = torch.load(data_path)
        return data_list[idx]


def generate_bird_labels(
    data_dir: str,
    output_file: str,
    categories: Optional[List[str]] = None,
    timeout: int = 5000,
    dsharp_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Generate labels for BIRD benchmark using DSharp solver.

    Args:
        data_dir: Directory containing BIRD CNF files
        output_file: Path to save labels JSON
        categories: List of categories to process
        timeout: Timeout per instance in seconds
        dsharp_path: Path to DSharp executable

    Returns:
        Dictionary mapping filename to log(model_count)
    """
    if categories is None:
        categories = BIRDDataset.CATEGORIES

    data_path = Path(data_dir)
    labels = {}

    for category in categories:
        category_path = data_path / category
        if not category_path.exists():
            continue

        cnf_files = list(category_path.glob('*.cnf'))
        logger.info(f"Processing {len(cnf_files)} files from {category}")

        for cnf_file in tqdm(cnf_files, desc=category):
            try:
                result = compute_model_count(cnf_file, timeout, dsharp_path)
                if result is not None:
                    labels[cnf_file.name] = result
                else:
                    logger.info(f"Skipping {cnf_file.name} (timeout or failure)")
            except Exception as e:
                logger.warning(f"Error processing {cnf_file}: {e}")

    # Save labels
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)

    logger.info(f"Generated labels for {len(labels)} instances")
    return labels
