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

"""Tests for Deep Tree Echo model."""

from absl.testing import absltest

from ydf.model.deep_tree_echo_model import deep_tree_echo_model


class DeepTreeEchoModelTest(absltest.TestCase):
  """Tests for DeepTreeEchoModel class."""

  def test_model_class_exists(self):
    """Test that DeepTreeEchoModel class can be imported."""
    self.assertTrue(hasattr(deep_tree_echo_model, 'DeepTreeEchoModel'))

  def test_model_has_echo_depth_method(self):
    """Test that DeepTreeEchoModel has echo_depth method."""
    self.assertTrue(hasattr(deep_tree_echo_model.DeepTreeEchoModel, 'echo_depth'))

  def test_model_has_identity_hypergraph_config_method(self):
    """Test that DeepTreeEchoModel has identity_hypergraph_config method."""
    self.assertTrue(
        hasattr(deep_tree_echo_model.DeepTreeEchoModel, 'identity_hypergraph_config')
    )

  def test_model_inherits_from_decision_forest(self):
    """Test that DeepTreeEchoModel inherits from DecisionForestModel."""
    from ydf.model.decision_forest_model import decision_forest_model
    self.assertTrue(
        issubclass(
            deep_tree_echo_model.DeepTreeEchoModel,
            decision_forest_model.DecisionForestModel
        )
    )


if __name__ == '__main__':
  absltest.main()
