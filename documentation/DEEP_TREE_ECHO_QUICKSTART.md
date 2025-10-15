# Deep Tree Echo - Quick Start Guide

## 5-Minute Quick Start

Get started with Deep Tree Echo in just a few minutes!

### Step 1: Install YDF

```bash
pip install ydf -U
```

### Step 2: Import and Prepare Data

```python
import ydf
import pandas as pd

# Load your dataset
data = pd.read_csv("your_data.csv")

# Or create a simple example dataset
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': ['a', 'b', 'a', 'b', 'a'],
    'target': [0, 1, 0, 1, 0]
})
```

### Step 3: Train a Model

```python
# Create a Deep Tree Echo learner
learner = ydf.DeepTreeEchoLearner(
    label="target",
    task=ydf.Task.CLASSIFICATION
)

# Train the model
model = learner.train(data)
```

### Step 4: Make Predictions

```python
# Predict on new data
predictions = model.predict(data)
print(predictions)

# Evaluate the model
evaluation = model.evaluate(data)
print(f"Accuracy: {evaluation.accuracy}")
```

### Step 5: Explore Echo Features

```python
# Check echo depth
print(f"Echo Depth: {model.echo_depth()}")

# View hypergraph configuration
config = model.identity_hypergraph_config()
print(f"Hypergraph Config: {config}")
```

## Common Use Cases

### Classification Task

```python
import ydf
import pandas as pd

# Binary classification example
learner = ydf.DeepTreeEchoLearner(
    label="is_fraud",
    task=ydf.Task.CLASSIFICATION,
    num_trees=100,
    echo_depth=16
)

model = learner.train(training_data)
predictions = model.predict(test_data)
```

### Regression Task

```python
import ydf

# Regression example
learner = ydf.DeepTreeEchoLearner(
    label="price",
    task=ydf.Task.REGRESSION,
    num_trees=200,
    echo_depth=12
)

model = learner.train(training_data)
predicted_prices = model.predict(test_data)
```

### With Validation Dataset

```python
import ydf

learner = ydf.DeepTreeEchoLearner(
    label="target",
    num_trees=150
)

# Train with validation data (optional for Deep Tree Echo)
model = learner.train(train_data, valid=validation_data)
```

### Custom Hyperparameters

```python
import ydf

# Advanced configuration
learner = ydf.DeepTreeEchoLearner(
    label="outcome",
    task=ydf.Task.CLASSIFICATION,
    num_trees=300,
    echo_depth=20,
    enable_hypergraph=True,
    max_depth=16,
    min_examples=10,
    random_seed=42
)

model = learner.train(data)
```

## Configuration Tips

### Echo Depth

The `echo_depth` parameter controls the maximum depth for echo operations:

```python
# Shallow echo (faster, less complex)
learner = ydf.DeepTreeEchoLearner(label="target", echo_depth=8)

# Default echo
learner = ydf.DeepTreeEchoLearner(label="target", echo_depth=16)

# Deep echo (slower, more complex patterns)
learner = ydf.DeepTreeEchoLearner(label="target", echo_depth=24)
```

### Number of Trees

Balance between accuracy and speed:

```python
# Fast training (fewer trees)
learner = ydf.DeepTreeEchoLearner(label="target", num_trees=50)

# Balanced (default)
learner = ydf.DeepTreeEchoLearner(label="target", num_trees=300)

# High accuracy (more trees)
learner = ydf.DeepTreeEchoLearner(label="target", num_trees=1000)
```

### Hypergraph Integration

Enable or disable identity hypergraph integration:

```python
# With hypergraph (default, recommended)
learner = ydf.DeepTreeEchoLearner(
    label="target",
    enable_hypergraph=True
)

# Without hypergraph (standard Random Forest behavior)
learner = ydf.DeepTreeEchoLearner(
    label="target",
    enable_hypergraph=False
)
```

## Working with the Model

### Model Information

```python
# Get model details
print(f"Number of trees: {model.num_trees()}")
print(f"Number of nodes: {model.num_nodes()}")
print(f"Echo depth: {model.echo_depth()}")

# View hypergraph configuration
config = model.identity_hypergraph_config()
print(f"Hypergraph enabled: {config['enabled']}")
print(f"Integration mode: {config['integration_mode']}")
```

### Predictions

```python
# Single prediction
single_example = pd.DataFrame({
    'feature1': [5.5],
    'feature2': ['a']
})
prediction = model.predict(single_example)

# Batch predictions
predictions = model.predict(test_data)

# Get probabilities (classification only)
probabilities = model.predict(test_data)  # Returns class probabilities
```

### Evaluation

```python
# Evaluate on test set
evaluation = model.evaluate(test_data)

# Access metrics
print(f"Accuracy: {evaluation.accuracy}")
print(f"Loss: {evaluation.loss}")

# For classification
if hasattr(evaluation, 'confusion_matrix'):
    print(evaluation.confusion_matrix)
```

### Model Inspection

```python
# View model description
print(model.describe())

# Get feature importance
print(model.variable_importances())

# Plot a tree (inherited from DecisionForestModel)
model.plot_tree(tree_idx=0, max_depth=3)
```

### Save and Load

```python
# Save model
model.save("/path/to/model")

# Load model later
loaded_model = ydf.load_model("/path/to/model")

# Use loaded model
predictions = loaded_model.predict(new_data)
```

## Integration with Other Tools

### With Pandas

```python
import pandas as pd
import ydf

# Load from CSV
df = pd.read_csv("data.csv")

# Train
learner = ydf.DeepTreeEchoLearner(label="target")
model = learner.train(df)

# Predict and add to DataFrame
df['predictions'] = model.predict(df)
```

### With NumPy

```python
import numpy as np
import pandas as pd
import ydf

# Convert NumPy to DataFrame
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
df['target'] = y

# Train
learner = ydf.DeepTreeEchoLearner(label="target")
model = learner.train(df)
```

### With Scikit-learn

```python
from sklearn.model_selection import train_test_split
import ydf

# Split data using sklearn
train_df, test_df = train_test_split(data, test_size=0.2)

# Train with YDF
learner = ydf.DeepTreeEchoLearner(label="target")
model = learner.train(train_df)

# Evaluate
evaluation = model.evaluate(test_df)
```

## Troubleshooting

### Issue: Import Error

```python
# Error: cannot import name 'DeepTreeEchoLearner'
# Solution: Make sure YDF is up to date
pip install ydf -U --force-reinstall
```

### Issue: Memory Error

```python
# If dealing with large datasets
learner = ydf.DeepTreeEchoLearner(
    label="target",
    num_trees=100,  # Reduce number of trees
    max_depth=12,   # Limit tree depth
)
```

### Issue: Slow Training

```python
# Speed up training
learner = ydf.DeepTreeEchoLearner(
    label="target",
    num_trees=100,        # Fewer trees
    num_threads=8,        # Use more threads
    echo_depth=8,         # Reduce echo depth
)
```

## Next Steps

- Read the [comprehensive documentation](DEEP_TREE_ECHO.md)
- Explore the [architecture details](DEEP_TREE_ECHO_ARCHITECTURE.md)
- Try the [example script](../examples/deep_tree_echo_example.py)
- Review the [implementation summary](../IMPLEMENTATION_SUMMARY.md)

## Getting Help

If you encounter issues:

1. Check the [YDF documentation](https://ydf.readthedocs.io/)
2. Review the [FAQ](DEEP_TREE_ECHO.md#common-questions)
3. Look at the [example code](../examples/deep_tree_echo_example.py)
4. File an issue on the GitHub repository

## Complete Example

Here's a complete, runnable example:

```python
#!/usr/bin/env python3
"""Complete Deep Tree Echo example."""

import ydf
import pandas as pd

# 1. Prepare data
data = pd.DataFrame({
    'age': [25, 45, 35, 50, 23, 38, 42, 29],
    'income': [50000, 80000, 60000, 90000, 45000, 70000, 75000, 55000],
    'education': ['BS', 'MS', 'BS', 'PhD', 'HS', 'MS', 'BS', 'MS'],
    'purchased': [0, 1, 0, 1, 0, 1, 1, 0]
})

# 2. Create learner
learner = ydf.DeepTreeEchoLearner(
    label="purchased",
    task=ydf.Task.CLASSIFICATION,
    num_trees=100,
    echo_depth=16,
    enable_hypergraph=True,
    random_seed=42
)

# 3. Train model
print("Training model...")
model = learner.train(data)

# 4. Model info
print(f"\nModel trained with {model.num_trees()} trees")
print(f"Echo depth: {model.echo_depth()}")
print(f"Hypergraph config: {model.identity_hypergraph_config()}")

# 5. Evaluate
evaluation = model.evaluate(data)
print(f"\nAccuracy: {evaluation.accuracy:.2%}")

# 6. Predict on new data
new_data = pd.DataFrame({
    'age': [30, 55],
    'income': [65000, 95000],
    'education': ['MS', 'PhD']
})

predictions = model.predict(new_data)
print(f"\nPredictions: {predictions}")

# 7. Save model
model.save("/tmp/deep_tree_echo_model")
print("\nModel saved successfully!")
```

Happy modeling with Deep Tree Echo! 🌳
