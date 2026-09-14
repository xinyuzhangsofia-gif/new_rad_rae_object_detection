"""Architecture compatibility checks for the Controlled Split package."""

import unittest

from data import splits
from data.split import controlled
from data.split.controlled import (
    apply_train_control_split_indices,
    prepare_controlled_train_data,
)
from data.split.controlled import generation, matching, reporting, runtime


class ControlledSplitArchitectureTests(unittest.TestCase):
    def test_public_api_is_forwarded_from_responsibility_modules(self):
        self.assertIs(
            prepare_controlled_train_data,
            generation.prepare_controlled_train_data,
        )
        self.assertIs(
            apply_train_control_split_indices,
            runtime.apply_train_control_split_indices,
        )
        self.assertIs(
            controlled.prepare_controlled_train_data,
            generation.prepare_controlled_train_data,
        )
        self.assertIs(
            controlled.apply_train_control_split_indices,
            runtime.apply_train_control_split_indices,
        )

    def test_historical_data_splits_facade_forwards_canonical_helpers(self):
        self.assertIs(splits._run_trial, matching._run_trial)
        self.assertIs(splits._summarize_frames, matching._summarize_frames)
        self.assertIs(
            splits._build_override_frames,
            generation._build_override_frames,
        )
        self.assertIs(splits._comparison_text, reporting._comparison_text)
        self.assertIs(splits._request_signature, reporting._request_signature)

    def test_controlled_import_resolves_to_package_boundary(self):
        self.assertEqual(controlled.__file__.rsplit("/", 1)[-1], "__init__.py")


if __name__ == "__main__":
    unittest.main()
