#!/usr/bin/env python3
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example usage of Deep Tree Echo model in Yggdrasil Decision Forests.

This example demonstrates how to train and use a Deep Tree Echo model,
which extends traditional Random Forest with identity hypergraph integration
for enhanced echo-based predictions.
"""

import pandas as pd
import ydf


def main():
  """Main function demonstrating Deep Tree Echo usage."""
  
  print("Deep Tree Echo Example")
  print("=" * 60)
  
  # Create a simple synthetic dataset for demonstration
  # In practice, you would load your own dataset
  data = pd.DataFrame({
      'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
      'feature2': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
      'feature3': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
      'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
  })
  
  print("\nDataset shape:", data.shape)
  print("\nFirst few rows:")
  print(data.head())
  
  # Create a Deep Tree Echo learner
  print("\n" + "=" * 60)
  print("Training Deep Tree Echo Model")
  print("=" * 60)
  
  learner = ydf.DeepTreeEchoLearner(
      label="label",
      task=ydf.Task.CLASSIFICATION,
      num_trees=50,
      echo_depth=8,
      enable_hypergraph=True,
      random_seed=42
  )
  
  # Train the model
  model = learner.train(data)
  
  print("\nModel trained successfully!")
  print(f"Number of trees: {model.num_trees()}")
  print(f"Number of nodes: {model.num_nodes()}")
  
  # Get Deep Tree Echo specific information
  print("\n" + "=" * 60)
  print("Deep Tree Echo Specific Features")
  print("=" * 60)
  
  print(f"Echo depth: {model.echo_depth()}")
  print(f"Identity hypergraph config: {model.identity_hypergraph_config()}")
  
  # Make predictions
  print("\n" + "=" * 60)
  print("Making Predictions")
  print("=" * 60)
  
  predictions = model.predict(data)
  print(f"Predictions: {predictions}")
  
  # Evaluate the model
  evaluation = model.evaluate(data)
  print(f"\nAccuracy: {evaluation.accuracy}")
  
  print("\n" + "=" * 60)
  print("Example completed successfully!")
  print("=" * 60)


if __name__ == "__main__":
  main()
