# BENCHMARK DATA SPECIFICATION
## Attn-JGNN: Attention Enhanced Join-Graph Neural Networks

---

**AGENT SCOPE:** You are implementing ONLY the data/benchmark component. Do not create or modify anything related to models, training, evaluation, or results.

**Repository uses Hydra for configuration. Structure:**
- `conf/` for yaml configs (create if missing)
- `data/` for dataset classes, data generators, data processing, label solvers (create if missing)
- Other folders (`models/`, `train.py`, `evaluate.py`, `utils/`) will exist but ARE NOT YOUR CONCERN - do not create or reference them.

**If graph data is involved, use PyTorch Geometric.**

---

## SECTION 1 - BENCHMARK OVERVIEW

### BENCHMARK_1: BIRD Benchmark

**What it is:** BIRD (BIg Bench for LaRge-scale Database Grounded Text-to-SQL Evaluation) is used in this paper as a benchmark for #SAT (propositional model counting) problems. The BIRD benchmark contains CNF formulas from real-world model counting applications.

**Task type:** #SAT (Model Counting) - counting the number of satisfying assignments for Boolean CNF formulas.

**Data source:** Downloaded from existing benchmark repository.

**Categories:** Eight categories arising from:
- DQMR networks
- Grid networks
- Bit-blasted versions of SMTLIB benchmarks
- ISCAS89 combinatorial circuits

**Scale:** Each category has 20 to 150 CNF formulas. Contains large-sized formulas with more than 10,000 variables and clauses.

### BENCHMARK_2: SATLIB Benchmark

**What it is:** SATLIB is an open-source dataset containing a broad range of CNF formulas collected from various distributions.

**Task type:** #SAT (Model Counting) - counting the number of satisfying assignments for Boolean CNF formulas.

**Data source:** Downloaded from SATLIB benchmark repository.

**Categories (5 used in paper):**
1. **RND3SAT** - Uniform random 3-SAT on phase transition region
2. **BMS** - Backbone-minimal random 3-SAT
3. **CBS** - Random 3-SAT with controlled backbone size
4. **GCP** - "Flat" graph coloring
5. **SW-GCP** - "Morphed" graph coloring

**Scale:**
- Total: 46,200 SAT instances
- Number of variables: ranging from 100 to 600
- Only satisfiable instances with at least 100 instances per distribution are used

---

## SECTION 2 - DATA SOURCE OR GENERATION

### BIRD Benchmark

**Source:** BIRD benchmark as referenced in [31] - Soos and Meel, "BIRD: engineering an efficient CNF-XOR SAT solver and its applications to approximate model counting", AAAI 2019.

**Format:** CNF (Conjunctive Normal Form) formulas in DIMACS format.

**Download/Access:** Follow NSNet experimental settings - use same subset of BIRD benchmark as NSNet paper.

**Data characteristics:**
- 8 categories of CNF formulas
- 20-150 formulas per category
- Large formulas: >10,000 variables and clauses
- Challenges generalization due to small dataset size

### SATLIB Benchmark

**Source:** SATLIB - open-source benchmark repository for SAT problems.

**Selection criteria:**
- Distributions with at least 100 satisfiable instances
- 5 categories selected (RND3SAT, BMS, CBS, GCP, SW-GCP)

**Format:** CNF formulas in DIMACS format.

**Data characteristics:**
- Total instances: 46,200
- Variables per instance: 100-600
- All instances are satisfiable

---

## SECTION 3 - LABEL GENERATION

### Solver Used: DSharp

**Solver type:** Exact #SAT solver (decision-DNNF compiler)

**Solver name:** DSharp [24] - Christian J. Muise et al., "Dsharp: Fast d-DNNF compilation with sharpSAT", Canadian AI 2012.

**What is being solved:**
- Input: CNF formula
- Output: Exact model count (number of satisfying assignments)

**Solver parameters:**
- **Time limit:** 5,000 seconds per instance
- **Tolerance/precision:** Exact counting (no approximation)

**Label processing:**
- Raw output: Model count Z (integer, potentially very large)
- Transformed label: log(Z) - logarithm of model count
- Instances where DSharp fails to finish within time limit are **discarded**

**Ground truth format:** log Z (log of the exact model count)

---

## SECTION 4 - IMPLEMENTATION INSTRUCTIONS

### 4.1 Dataset Loader

Create file `data/datasets/bird_dataset.py` containing:
- Class `BIRDDataset` that:
  - Loads CNF files from BIRD benchmark directory
  - Parses DIMACS CNF format
  - Converts CNF to factor graph representation
  - Applies tree decomposition using FlowCutter
  - Constructs join-graph structure
  - Returns PyTorch Geometric Data objects

Create file `data/datasets/satlib_dataset.py` containing:
- Class `SATLIBDataset` that:
  - Loads CNF files from SATLIB benchmark directory
  - Parses DIMACS CNF format
  - Converts CNF to factor graph representation
  - Applies tree decomposition using FlowCutter
  - Constructs join-graph structure
  - Returns PyTorch Geometric Data objects

### 4.2 CNF Parser

Create file `data/preprocessing/cnf_parser.py` containing:
- Function `parse_dimacs(filepath)` that:
  - Reads DIMACS CNF format files
  - Extracts: number of variables, number of clauses, clause list
  - Returns structured CNF representation
  - Handles comments and problem line

### 4.3 Graph Construction

Create file `data/preprocessing/graph_builder.py` containing:

**Factor Graph Construction:**
- Function `build_factor_graph(cnf)` that:
  - Creates bipartite graph between variables and clauses
  - Edge exists between variable x_i and clause C_j if x_i appears in C_j
  - Node types: variable nodes and clause nodes
  - Edge attributes: polarity of literal (+1 for positive, -1 for negative)

**Join-Graph Construction:**
- Function `build_join_graph(factor_graph, tree_width)` that:
  - Uses external tree decomposition tool FlowCutter
  - Constructs clusters {C_1, C_2, ..., C_k}
  - Each cluster contains variables and clauses forming local substructure
  - Shared variables between clusters form edge labels
  - Returns join-graph with cluster structure

### 4.4 Tree Decomposition Interface

Create file `data/preprocessing/tree_decomposition.py` containing:
- Function `decompose_with_flowcutter(graph, target_width)` that:
  - Interfaces with FlowCutter tree decomposition tool
  - Controls tree-width of decomposition manually
  - Returns tree decomposition with clusters
  - Properties ensured:
    - **Coverage:** Each factor included in at least one cluster
    - **Connectivity:** For any two clusters sharing a variable, path exists connecting them with all clusters on path containing the variable

### 4.5 Label Solver

Create file `data/solvers/dsharp_solver.py` containing:
- Function `compute_model_count(cnf_file, timeout=5000)` that:
  - Calls DSharp exact #SAT solver
  - Input: path to CNF file in DIMACS format
  - Timeout: 5000 seconds
  - Returns: log(Z) where Z is exact model count
  - Returns None if solver times out (instance discarded)

### 4.6 Config Files

Create file `conf/data/bird.yaml`:
```yaml
name: bird
data_dir: ${paths.data_dir}/bird
categories:
  - dqmr
  - grid
  - smtlib
  - iscas89
  # (full list of 8 categories)
split_ratio:
  train: 0.7
  test: 0.3
label_solver:
  name: dsharp
  timeout: 5000  # seconds
tree_decomposition:
  tool: flowcutter
  # tree_width controlled manually per instance
```

Create file `conf/data/satlib.yaml`:
```yaml
name: satlib
data_dir: ${paths.data_dir}/satlib
categories:
  - rnd3sat    # Uniform random 3-SAT phase transition
  - bms        # Backbone-minimal random 3-SAT
  - cbs        # Random 3-SAT controlled backbone size
  - gcp        # Flat graph coloring
  - sw_gcp     # Morphed graph coloring
min_instances_per_category: 100
variable_range:
  min: 100
  max: 600
total_instances: 46200
split_ratio:
  train: 0.6
  val: 0.2
  test: 0.2
label_solver:
  name: dsharp
  timeout: 5000  # seconds
tree_decomposition:
  tool: flowcutter
```

---

## SECTION 5 - DATA STRUCTURE

### Single Data Sample (PyTorch Geometric Data Object)

**For Factor Graph representation:**
```
Data(
    x_var: Tensor[num_variables, d]     # Variable node features (initialized)
    x_clause: Tensor[num_clauses, d]    # Clause node features (self-identifying)
    edge_index: Tensor[2, num_edges]    # Bipartite edges (var-clause connections)
    edge_attr: Tensor[num_edges, 1]     # Polarity: +1 (positive) or -1 (negative)
    y: Tensor[1]                        # Label: log(model_count)
)
```

**For Join-Graph representation (after tree decomposition):**
```
Data(
    # Cluster-level data
    cluster_node_ids: List[List[int]]   # Variable/clause IDs per cluster
    cluster_var_ids: List[List[int]]    # Variable IDs per cluster
    cluster_clause_ids: List[List[int]] # Clause IDs per cluster

    # Inter-cluster edges
    cluster_edge_index: Tensor[2, num_cluster_edges]
    shared_vars: List[List[int]]        # Shared variables for each cluster edge

    # Original factor graph data
    x_var: Tensor[num_variables, d]
    x_clause: Tensor[num_clauses, d]
    var_clause_edge_index: Tensor[2, E]
    edge_polarity: Tensor[E, 1]

    # Label
    y: Tensor[1]                        # log(Z) - log model count
)
```

**Feature dimensions:**
- d = 64 (feature dimension as per paper)
- num_variables: varies (100-600 for SATLIB, can exceed 10,000 for BIRD)
- num_clauses: varies (can exceed 10,000 for BIRD)

**Node feature initialization:**
- Variable nodes: h_v (learned/initialized features)
- Clause nodes: h_phi (self-identifying node features)

**Label:**
- Type: Regression target (continuous)
- Format: log(Z) where Z is exact model count
- Range: Can exceed log(e^1000) = 1000 for large instances

---

## SECTION 6 - SPLITS

### BIRD Benchmark
- **Train/Test split:** 70% / 30%
- **Validation:** Not explicitly mentioned (no separate validation set)
- **Split method:** Per category (each category split independently)
- **Random seed:** Not specified in paper

### SATLIB Benchmark
- **Train/Val/Test split:** 60% / 20% / 20%
- **Total instances:** 46,200
  - Train: ~27,720 instances
  - Validation: ~9,240 instances
  - Test: ~9,240 instances
- **Split method:** Not explicitly specified (likely random)
- **Random seed:** Not specified in paper

---

## SECTION 7 - VERIFICATION CHECKLIST

### BIRD Benchmark
- [ ] 8 categories loaded
- [ ] 20-150 CNF formulas per category
- [ ] Train/test split: 70%/30%
- [ ] Instances with DSharp timeout (>5000s) discarded
- [ ] Labels are log(model_count)

### SATLIB Benchmark
- [ ] 5 categories loaded: RND3SAT, BMS, CBS, GCP, SW-GCP
- [ ] Total instances: 46,200
- [ ] Variable count range: 100-600
- [ ] All instances satisfiable
- [ ] Minimum 100 instances per category
- [ ] Train/val/test split: 60%/20%/20%
- [ ] Instances with DSharp timeout (>5000s) discarded
- [ ] Labels are log(model_count)

### Graph Construction
- [ ] Factor graph: bipartite between variables and clauses
- [ ] Edge polarity correctly encoded (+1/-1)
- [ ] Tree decomposition produces valid join-graph
- [ ] Cluster coverage property satisfied
- [ ] Cluster connectivity property satisfied

### Expected RMSE values (from paper, for reference):
| Method | RND3SAT | BMS | CBS | GCP | SW-GCP |
|--------|---------|-----|-----|-----|--------|
| Attn-JGNN | 1.15 | 1.66 | 1.20 | 1.96 | 0.96 |

---

## SECTION 8 - MISSING INFORMATION

The following data-related parameters are **NOT specified** in the paper:

1. **Random seed** for train/val/test splits - not mentioned
2. **Exact list of 8 BIRD categories** - only 4 types mentioned (DQMR, grid, SMTLIB, ISCAS89), full list not provided
3. **FlowCutter parameters** - tree-width is "controlled manually" but specific values per instance/category not given
4. **Node feature initialization method** - h_v and h_phi initialization not detailed
5. **BIRD benchmark exact download URL/version** - referenced as "same subset as NSNet"
6. **SATLIB benchmark exact download URL/version** - referenced as open-source but no direct link
7. **DSharp version** - not specified
8. **FlowCutter version** - not specified
9. **Instance filtering criteria** - beyond satisfiability and solver timeout, unclear if other filters applied
10. **Batch size** for data loading - not specified in data section
11. **Data augmentation** - not mentioned (likely none)
12. **Stratification** in splits - not mentioned whether splits are stratified by category

---

## SECTION 9 - EXTERNAL DEPENDENCIES

### Required External Tools

1. **FlowCutter** - Tree decomposition tool
   - Purpose: Decompose factor graph into join-graph with controlled tree-width
   - Installation: Requires separate installation
   - Interface: Command-line or library binding needed

2. **DSharp** - Exact #SAT solver
   - Purpose: Generate ground truth labels (exact model counts)
   - Reference: [24] in paper
   - Installation: Requires separate installation
   - Time limit: 5000 seconds per instance

### Python Dependencies
- PyTorch
- PyTorch Geometric
- Hydra (for configuration)
- Standard: numpy, scipy

---

## SECTION 10 - FILE STRUCTURE SUMMARY

```
conf/
  data/
    bird.yaml
    satlib.yaml

data/
  datasets/
    bird_dataset.py       # BIRD benchmark dataset class
    satlib_dataset.py     # SATLIB benchmark dataset class

  preprocessing/
    cnf_parser.py         # DIMACS CNF format parser
    graph_builder.py      # Factor graph and join-graph construction
    tree_decomposition.py # FlowCutter interface for tree decomposition

  solvers/
    dsharp_solver.py      # DSharp interface for label generation
```

---

**END OF SPECIFICATION**
