import unittest
from types import SimpleNamespace

from training_utils.configuration import (
    apply_domain_shift_training_configuration,
    apply_test_sequence_weather_configuration,
    infer_test_weather_group,
)


def domain_shift_args(branch="source", control_enabled=False):
    return SimpleNamespace(
        split_mode="sequence",
        domain_shift_train_branch=branch,
        shared_train_sequences=(9,),
        source_train_sequences=(1,),
        target_train_sequences=(22,),
        target_test_sequences=(13,),
        train_control_split_enabled=control_enabled,
    )


class DomainShiftTrainingConfigurationTests(unittest.TestCase):
    def test_source_branch_uses_shared_plus_source(self):
        args = apply_domain_shift_training_configuration(
            domain_shift_args(branch="source")
        )

        self.assertTrue(args.domain_shift_experiment_enabled)
        self.assertEqual(args.train_sequences, (9, 1))
        self.assertEqual(args.val_sequences, (13,))
        self.assertEqual(args.controled_sequences, (1,))
        self.assertEqual(args.reference_sequences, (22,))

    def test_target_branch_uses_shared_plus_target(self):
        args = apply_domain_shift_training_configuration(
            domain_shift_args(branch="target")
        )

        self.assertEqual(args.train_sequences, (9, 22))
        self.assertEqual(args.val_sequences, (13,))
        self.assertEqual(args.controled_sequences, (1,))
        self.assertEqual(args.reference_sequences, (22,))

    def test_controlled_split_is_source_branch_only(self):
        with self.assertRaisesRegex(ValueError, "only be used"):
            apply_domain_shift_training_configuration(
                domain_shift_args(branch="target", control_enabled=True)
            )

    def test_domain_sequence_groups_must_not_overlap(self):
        args = domain_shift_args()
        args.target_test_sequences = (13, 22)

        with self.assertRaisesRegex(ValueError, "overlap"):
            apply_domain_shift_training_configuration(args)

    def test_test_weather_is_inferred_from_sequence_information(self):
        args = apply_domain_shift_training_configuration(
            domain_shift_args(branch="source")
        )
        args.sequence_information_path = "sequence_information.csv"
        args = apply_test_sequence_weather_configuration(args)

        self.assertEqual(args.weather_group, "overcast")
        self.assertEqual(args.test_sequence_weather, {13: "overcast"})
        self.assertEqual(args.weather_group_source, "sequence_information.csv")

    def test_heavysnow_weather_uses_existing_folder_spelling(self):
        weather_group, weather_by_sequence, _ = infer_test_weather_group(
            (46, 47),
            "sequence_information.csv",
        )

        self.assertEqual(weather_group, "heavy_snow")
        self.assertEqual(
            weather_by_sequence,
            {46: "heavysnow", 47: "heavysnow"},
        )

    def test_mixed_test_weather_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same weather"):
            infer_test_weather_group(
                (13, 46),
                "sequence_information.csv",
            )


if __name__ == "__main__":
    unittest.main()
