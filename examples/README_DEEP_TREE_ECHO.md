# Deep Tree Echo Example

## Overview

This example demonstrates how to use the Deep Tree Echo model in Yggdrasil Decision Forests. Deep Tree Echo is an advanced decision forest algorithm that implements identity hypergraph integration for enhanced echo-based predictions.

## Running the Example

To run the Deep Tree Echo example:

```bash
python examples/deep_tree_echo_example.py
```

## What the Example Demonstrates

The example shows:

1. **Creating a synthetic dataset** - Demonstrates data preparation
2. **Initializing a Deep Tree Echo learner** - Shows how to configure hyperparameters
3. **Training the model** - Trains on the sample data
4. **Accessing Deep Tree Echo features** - Shows echo depth and hypergraph configuration
5. **Making predictions** - Uses the trained model for inference
6. **Evaluating the model** - Computes accuracy metrics

## Key Parameters

### DeepTreeEchoLearner Parameters

- **`label`** (str, required): Name of the target column
- **`task`** (Task): Type of ML task (CLASSIFICATION, REGRESSION)
- **`num_trees`** (int): Number of trees in the forest
- **`echo_depth`** (int): Maximum depth for echo operations
- **`enable_hypergraph`** (bool): Enable identity hypergraph integration
- **`random_seed`** (int): Random seed for reproducibility

## Deep Tree Echo Specific Features

The Deep Tree Echo model provides specialized methods:

### `echo_depth()`
Returns the echo depth configuration of the model:
```python
depth = model.echo_depth()
```

### `identity_hypergraph_config()`
Returns the hypergraph configuration:
```python
config = model.identity_hypergraph_config()
print(f"Integration mode: {config['integration_mode']}")
```

## Use Cases

Deep Tree Echo is particularly suited for:

- **Complex Pattern Recognition**: Deep hierarchical pattern detection
- **Identity-Preserving Transformations**: Maintaining data identity across transformations
- **Hierarchical Data Structures**: Data with natural echo relationships
- **High-Dimensional Data**: Leveraging hypergraph structures for complex feature interactions

## Next Steps

For more detailed information, see:
- Main documentation: `documentation/DEEP_TREE_ECHO.md`
- API reference: YDF documentation for `DeepTreeEchoLearner` and `DeepTreeEchoModel`

## Requirements

This example requires:
- Yggdrasil Decision Forests (YDF) installed
- pandas (for data manipulation)
- Python 3.7+

Install YDF with:
```bash
pip install ydf -U
```

## License

Copyright 2022 Google LLC. Licensed under the Apache License, Version 2.0.
