"""Architecture compatibility checks for the Controlled Split package."""

import unittest

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

    def test_private_helpers_stay_in_responsibility_modules(self):
        for private_name in (
            "_run_trial",
            "_summarize_frames",
            "_build_override_frames",
            "_comparison_text",
            "_request_signature",
        ):
            self.assertFalse(hasattr(controlled, private_name))
        self.assertTrue(callable(matching._run_trial))
        self.assertTrue(callable(matching._summarize_frames))
        self.assertTrue(callable(generation._build_override_frames))
        self.assertTrue(callable(reporting._comparison_text))
        self.assertTrue(callable(reporting._request_signature))

    def test_controlled_import_resolves_to_package_boundary(self):
        self.assertEqual(controlled.__file__.rsplit("/", 1)[-1], "__init__.py")


if __name__ == "__main__":
    unittest.main()
