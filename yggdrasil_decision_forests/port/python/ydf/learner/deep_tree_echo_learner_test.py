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

"""Tests for Deep Tree Echo learner."""

from absl.testing import absltest

from ydf.learner import deep_tree_echo_learner
from ydf.learner import generic_learner


class DeepTreeEchoLearnerTest(absltest.TestCase):
  """Tests for DeepTreeEchoLearner class."""

  def test_learner_class_exists(self):
    """Test that DeepTreeEchoLearner class can be imported."""
    self.assertTrue(hasattr(deep_tree_echo_learner, 'DeepTreeEchoLearner'))

  def test_learner_inherits_from_generic(self):
    """Test that DeepTreeEchoLearner inherits from GenericCCLearner."""
    self.assertTrue(
        issubclass(
            deep_tree_echo_learner.DeepTreeEchoLearner,
            generic_learner.GenericCCLearner
        )
    )

  def test_learner_has_capabilities(self):
    """Test that DeepTreeEchoLearner has _capabilities method."""
    self.assertTrue(
        hasattr(deep_tree_echo_learner.DeepTreeEchoLearner, '_capabilities')
    )


if __name__ == '__main__':
  absltest.main()
