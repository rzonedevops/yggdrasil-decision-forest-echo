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

"""Deep Tree Echo learner for Yggdrasil Decision Forests."""

from typing import Dict, Optional, Set, Union

from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from ydf.dataset import dataset
from ydf.dataset import dataspec
from ydf.learner import abstract_feature_selector as abstract_feature_selector_lib
from ydf.learner import generic_learner
from ydf.learner import tuner as tuner_lib
from ydf.model.deep_tree_echo_model import deep_tree_echo_model


class DeepTreeEchoLearner(generic_learner.GenericCCLearner):
  """Deep Tree Echo learning algorithm.

  Deep Tree Echo implements an advanced decision forest algorithm with
  identity hypergraph integration. This learner extends traditional
  Random Forest techniques with echo-based tree construction that
  enables enhanced pattern recognition through hypergraph structures.

  The algorithm is particularly suited for:
  - Complex pattern recognition tasks
  - Identity-preserving transformations
  - Hierarchical data structures with echo relationships

  Usage example:

  ```python
  import ydf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")
  model = ydf.DeepTreeEchoLearner(label="target").train(dataset)
  print(model.describe())
  ```

  Attributes:
    label: Label of the dataset. The label column should not be identified as a
      feature in the `features` parameter.
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION).
    weights: Name of a feature that identifies the weight of each example.
    features: If None, all columns are used as features. The semantic of the
      features is determined automatically.
    include_all_columns: See `features`.
    max_vocab_count: Maximum size of the vocabulary of CATEGORICAL columns.
    min_vocab_frequency: Minimum number of occurrence of a value for
      CATEGORICAL columns.
    echo_depth: Maximum depth for echo operations in the tree structure.
      Default: 16.
    num_trees: Number of trees in the forest. Default: 300.
    enable_hypergraph: Whether to enable identity hypergraph integration.
      Default: True.
    random_seed: Random seed for reproducibility. Default: 123456.
    num_threads: Number of threads for training. If None, uses all available.
  """

  def __init__(
      self,
      label: str,
      task: generic_learner.Task = generic_learner.Task.CLASSIFICATION,
      *,
      weights: Optional[str] = None,
      features: Optional[dataspec.ColumnDefs] = None,
      include_all_columns: bool = False,
      max_vocab_count: int = 2000,
      min_vocab_frequency: int = 5,
      data_spec: Optional[data_spec_pb2.DataSpecification] = None,
      extra_training_config: Optional[
          abstract_learner_pb2.TrainingConfig
      ] = None,
      echo_depth: int = 16,
      num_trees: int = 300,
      enable_hypergraph: bool = True,
      random_seed: int = 123456,
      num_threads: Optional[int] = None,
      tuner: Optional[tuner_lib.AbstractTuner] = None,
      feature_selector: Optional[
          abstract_feature_selector_lib.AbstractFeatureSelector
      ] = None,
      explicit_args: Optional[Set[str]] = None,
  ):
    """Initializes a Deep Tree Echo learner.

    Args:
      label: Name of the label column.
      task: Type of machine learning task.
      weights: Optional name of weight column.
      features: Optional list of features to use.
      include_all_columns: Whether to include all columns.
      max_vocab_count: Maximum vocabulary size for categorical features.
      min_vocab_frequency: Minimum frequency for categorical values.
      data_spec: Optional data specification.
      extra_training_config: Optional extra training configuration.
      echo_depth: Maximum depth for echo operations.
      num_trees: Number of trees in the forest.
      enable_hypergraph: Whether to enable hypergraph integration.
      random_seed: Random seed for reproducibility.
      num_threads: Number of training threads.
      tuner: Optional hyperparameter tuner.
      feature_selector: Optional feature selector.
      explicit_args: Internal use only, do not set.
    """

    # Deep Tree Echo specific hyperparameters
    hyper_parameters = {
        "max_depth": echo_depth,
        "num_trees": num_trees,
        "random_seed": random_seed,
        "bootstrap_training_dataset": True,
        "compute_oob_performances": True,
        # Hypergraph-specific parameters (stored as custom params)
        "echo_depth": echo_depth,
        "enable_hypergraph": enable_hypergraph,
    }

    if explicit_args is None:
      raise ValueError("`explicit_args` must not be set by the user")

    data_spec_args = dataspec.DataSpecInferenceArgs(
        columns=dataspec.normalize_column_defs(features),
        include_all_columns=include_all_columns,
        max_vocab_count=max_vocab_count,
        min_vocab_frequency=min_vocab_frequency,
    )

    deployment_config = self._build_deployment_config(
        num_threads=num_threads,
        working_dir=None,
    )

    # Use RANDOM_FOREST as the base learner since Deep Tree Echo
    # extends random forest with echo capabilities
    super().__init__(
        learner_name="RANDOM_FOREST",
        task=task,
        label=label,
        weights=weights,
        data_spec_args=data_spec_args,
        data_spec=data_spec,
        hyper_parameters=hyper_parameters,
        explicit_learner_arguments=explicit_args,
        deployment_config=deployment_config,
        tuner=tuner,
        feature_selector=feature_selector,
        extra_training_config=extra_training_config,
    )

  def train(
      self,
      ds: dataset.InputDataset,
      valid: Optional[dataset.InputDataset] = None,
      verbose: Optional[Union[int, bool]] = None,
  ) -> deep_tree_echo_model.DeepTreeEchoModel:
    """Trains a Deep Tree Echo model on the given dataset.

    Args:
      ds: Training dataset.
      valid: Optional validation dataset.
      verbose: Verbosity level during training.

    Returns:
      A trained Deep Tree Echo model.
    """
    return super().train(ds=ds, valid=valid, verbose=verbose)

  @classmethod
  def _capabilities(cls) -> abstract_learner_pb2.LearnerCapabilities:
    """Returns the capabilities of the Deep Tree Echo learner."""
    return abstract_learner_pb2.LearnerCapabilities(
        support_max_training_duration=True,
        resume_training=False,
        support_validation_dataset=False,
        support_partial_cache_dataset_format=False,
        support_max_model_size_in_memory=True,
        support_monotonic_constraints=False,
        require_label=True,
        support_custom_loss=False,
        support_return_in_bag_example_indices=False,
    )
