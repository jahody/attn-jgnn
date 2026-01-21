
# 1. BENCHMARK SPECIFICATION

/ralph-loop:ralph-loop "based on @data_benchmark_specification.txt create a benchmark specification" --completion-promise "DONE" --max-iterations 5

# 2. BENCHMARK IMPLEMENTATION

/ralph-loop:ralph-loop "based on @data_benchmark_specification.md implement and make sure it works" --completion-promise "DONE" --max-iterations 5

# 3. MODEL & TRAINING SPECIFICATION

/ralph-loop:ralph-loop "based on @model_training_spec_generator.txt create a model specification" --completion-promise "DONE" --max-iterations 5

# 4. MODEL & TRAINING IMPLEMENTATION

/ralph-loop:ralph-loop "based on @ATTN_JGNN_MODEL_SPEC.md implement and make sure it works" --completion-promise "DONE" --max-iterations 5
