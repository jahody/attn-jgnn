"""
SATLIB Benchmark Dataset for #SAT (Model Counting).

SATLIB is an open-source dataset containing CNF formulas from various distributions.

Categories used in the paper:
1. RND3SAT - Uniform random 3-SAT on phase transition region
2. BMS - Backbone-minimal random 3-SAT
3. CBS - Random 3-SAT with controlled backbone size
4. GCP - "Flat" graph coloring
5. SW-GCP - "Morphed" graph coloring

Total: 46,200 SAT instances
Variables: 100-600 per instance
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Callable, Tuple
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import random

from ..preprocessing.cnf_parser import parse_dimacs, CNF
from ..preprocessing.graph_builder import (
    build_factor_graph,
    factor_graph_to_pyg,
    build_join_graph,
    join_graph_to_pyg
)
from ..preprocessing.tree_decomposition import decompose_cnf
from ..solvers.dsharp_solver import compute_model_count

logger = logging.getLogger(__name__)


class SATLIBDataset(Dataset):
    """
    PyTorch Geometric Dataset for SATLIB benchmark.

    This dataset:
    - Loads CNF files from SATLIB benchmark directory
    - Parses DIMACS CNF format
    - Converts CNF to factor graph representation
    - Optionally applies tree decomposition for join-graph structure
    - Returns PyTorch Geometric Data objects

    Args:
        root: Root directory where the dataset is stored
        categories: List of categories to include (default: all 5 categories)
        split: Dataset split ('train', 'val', or 'test')
        split_ratio: Dictionary with 'train', 'val', 'test' ratios
        use_join_graph: Whether to use join-graph representation
        feature_dim: Dimension of node features (default: 64)
        variable_range: Dict with 'min' and 'max' variable counts to filter
        min_instances_per_category: Minimum instances required per category
        transform: Optional transform to apply to data
        pre_transform: Optional pre-transform to apply
        pre_filter: Optional filter function
        seed: Random seed for reproducible splits
        label_file: Optional path to pre-computed labels file
    """

    # Categories from SATLIB used in the paper
    CATEGORIES = [
        'rnd3sat',   # Uniform random 3-SAT phase transition
        'bms',       # Backbone-minimal random 3-SAT
        'cbs',       # Random 3-SAT controlled backbone size
        'gcp',       # Flat graph coloring
        'sw_gcp'     # Morphed graph coloring
    ]

    # Alternative category names that might be used in SATLIB
    CATEGORY_ALIASES = {
        'rnd3sat': ['uf', 'uuf', 'random3sat', 'uniform'],
        'bms': ['bms', 'backbone_minimal'],
        'cbs': ['cbs', 'controlled_backbone'],
        'gcp': ['gcp', 'flat', 'graph_coloring'],
        'sw_gcp': ['sw_gcp', 'morphed', 'sw-gcp', 'swgcp']
    }

    def __init__(
        self,
        root: str,
        categories: Optional[List[str]] = None,
        split: str = 'train',
        split_ratio: Optional[Dict[str, float]] = None,
        use_join_graph: bool = False,
        feature_dim: int = 64,
        variable_range: Optional[Dict[str, int]] = None,
        min_instances_per_category: int = 100,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        seed: int = 42,
        label_file: Optional[str] = None,
    ):
        self.categories = categories or self.CATEGORIES
        self.split = split
        self.split_ratio = split_ratio or {'train': 0.6, 'val': 0.2, 'test': 0.2}
        self.use_join_graph = use_join_graph
        self.feature_dim = feature_dim
        self.variable_range = variable_range or {'min': 100, 'max': 600}
        self.min_instances_per_category = min_instances_per_category
        self.seed = seed
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
                # Try category name and aliases
                possible_names = [category] + self.CATEGORY_ALIASES.get(category, [])
                for name in possible_names:
                    category_path = raw_path / name
                    if category_path.exists():
                        for cnf_file in category_path.glob('**/*.cnf'):
                            files.append(str(cnf_file.relative_to(raw_path)))
                        break
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
        """Download SATLIB benchmark data."""
        logger.info(
            "SATLIB benchmark data should be downloaded from:\n"
            "https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html\n"
            f"Please place the CNF files in: {self.raw_dir}\n"
            "Organize by category: rnd3sat/, bms/, cbs/, gcp/, sw_gcp/"
        )

    def _find_category_path(self, category: str) -> Optional[Path]:
        """Find the actual path for a category, checking aliases."""
        raw_path = Path(self.raw_dir)
        possible_names = [category] + self.CATEGORY_ALIASES.get(category, [])
        for name in possible_names:
            category_path = raw_path / name
            if category_path.exists():
                return category_path
        return None

    def _filter_by_variable_count(self, cnf_file: Path) -> bool:
        """Check if a CNF file has variable count within the allowed range."""
        try:
            cnf = parse_dimacs(cnf_file)
            return (self.variable_range['min'] <= cnf.num_variables
                    <= self.variable_range['max'])
        except Exception:
            return False

    def process(self):
        """Process raw CNF files into PyG Data objects."""
        raw_path = Path(self.raw_dir)
        processed_path = Path(self.processed_dir)
        processed_path.mkdir(parents=True, exist_ok=True)

        labels = self._load_labels()

        # Collect all CNF files by category
        all_files = []
        category_counts = {}

        for category in self.categories:
            category_path = self._find_category_path(category)
            if category_path is None:
                logger.warning(f"Category {category} not found in {raw_path}")
                continue

            cnf_files = list(category_path.glob('**/*.cnf'))

            # Filter by variable count
            valid_files = []
            for cnf_file in cnf_files:
                if self._filter_by_variable_count(cnf_file):
                    valid_files.append((category, cnf_file))

            if len(valid_files) >= self.min_instances_per_category:
                all_files.extend(valid_files)
                category_counts[category] = len(valid_files)
                logger.info(f"Category {category}: {len(valid_files)} valid instances")
            else:
                logger.warning(
                    f"Category {category} has only {len(valid_files)} instances "
                    f"(minimum: {self.min_instances_per_category})"
                )

        if not all_files:
            logger.warning(f"No valid CNF files found in {raw_path}")
            self._save_empty_dataset()
            return

        # Shuffle with seed for reproducible splits
        random.seed(self.seed)
        random.shuffle(all_files)

        # Split files into train/val/test
        n_total = len(all_files)
        n_train = int(n_total * self.split_ratio['train'])
        n_val = int(n_total * self.split_ratio['val'])

        if self.split == 'train':
            files_to_process = all_files[:n_train]
        elif self.split == 'val':
            files_to_process = all_files[n_train:n_train + n_val]
        else:  # test
            files_to_process = all_files[n_train + n_val:]

        logger.info(f"Processing {len(files_to_process)} files for {self.split} split")

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
            'category_counts': category_counts,
            'split': self.split,
            'split_ratio': self.split_ratio,
            'use_join_graph': self.use_join_graph,
            'feature_dim': self.feature_dim,
            'variable_range': self.variable_range,
            'seed': self.seed
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
        data_list = torch.load(data_path, weights_only=False)
        return data_list[idx]

    def get_category_stats(self) -> Dict[str, int]:
        """Get statistics about instances per category."""
        return self._data_info.get('category_counts', {})


def generate_satlib_labels(
    data_dir: str,
    output_file: str,
    categories: Optional[List[str]] = None,
    variable_range: Optional[Dict[str, int]] = None,
    timeout: int = 5000,
    dsharp_path: Optional[str] = None,
    n_jobs: int = 1
) -> Dict[str, float]:
    """
    Generate labels for SATLIB benchmark using DSharp solver.

    Args:
        data_dir: Directory containing SATLIB CNF files
        output_file: Path to save labels JSON
        categories: List of categories to process
        variable_range: Variable count filter (min/max)
        timeout: Timeout per instance in seconds
        dsharp_path: Path to DSharp executable
        n_jobs: Number of parallel jobs

    Returns:
        Dictionary mapping filename to log(model_count)
    """
    if categories is None:
        categories = SATLIBDataset.CATEGORIES
    if variable_range is None:
        variable_range = {'min': 100, 'max': 600}

    data_path = Path(data_dir)
    labels = {}

    for category in categories:
        # Find category path
        possible_names = [category] + SATLIBDataset.CATEGORY_ALIASES.get(category, [])
        category_path = None
        for name in possible_names:
            p = data_path / name
            if p.exists():
                category_path = p
                break

        if category_path is None:
            logger.warning(f"Category {category} not found")
            continue

        cnf_files = list(category_path.glob('**/*.cnf'))
        logger.info(f"Processing {len(cnf_files)} files from {category}")

        for cnf_file in tqdm(cnf_files, desc=category):
            try:
                # Check variable count
                cnf = parse_dimacs(cnf_file)
                if not (variable_range['min'] <= cnf.num_variables
                        <= variable_range['max']):
                    continue

                result = compute_model_count(cnf_file, timeout, dsharp_path)
                if result is not None:
                    labels[cnf_file.name] = result
                else:
                    logger.debug(f"Skipping {cnf_file.name} (timeout or failure)")
            except Exception as e:
                logger.warning(f"Error processing {cnf_file}: {e}")

    # Save labels
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)

    logger.info(f"Generated labels for {len(labels)} instances")
    return labels
