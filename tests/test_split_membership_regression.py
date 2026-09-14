"""Golden exact-membership tests captured before the Step 9 extraction."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from data.dataloader import (
    apply_train_control_split_indices,
    build_exact_frame_manifest_indices,
    build_file_split_indices,
    build_order_split_indices,
    build_random_split_indices,
    build_sequence_split_indices,
    build_sequence_tail_split_indices,
    get_dataset_sequences_for_split,
    normalize_sequence_list,
    read_split_file,
    split_line_to_sequence_and_frame_names,
    unique_sequences,
)
from data.splits import (
    _bin_key,
    _build_override_frames,
    _build_population,
    _normalize_sequence_parts,
    _pair_sequence_parts,
    _run_trial,
    _select_sequence_part,
    _summarize_frames,
)
from training_utils.configuration import (
    SUPPORTED_TRAINING_SPLIT_MODES,
    validate_training_split_mode,
)


class DummyRadarDataset:
    def __init__(self, frame_names):
        self.frame_names = list(frame_names)


class DummySequenceDataset:
    def __init__(self, sequence, frame_names):
        self.sequence = int(sequence)
        self.radar_dataset = DummyRadarDataset(frame_names)

    def __len__(self):
        return len(self.radar_dataset.frame_names)


class DummyMultiSequenceDataset:
    def __init__(self, specifications=None):
        if specifications is None:
            specifications = (
                (1, ("s1f0", "s1f1", "s1f2", "s1f3", "s1f4")),
                (2, ("s2f0", "s2f1", "s2f2", "s2f3")),
                (9, ("s9f0", "s9f1", "s9f2", "s9f3", "s9f4", "s9f5", "s9f6")),
            )
        self.sequence_datasets = [
            DummySequenceDataset(sequence, frame_names)
            for sequence, frame_names in specifications
        ]
        self._ranges = []
        start = 0
        for dataset in self.sequence_datasets:
            end = start + len(dataset)
            self._ranges.append({
                "sequence": dataset.sequence,
                "start": start,
                "end": end,
            })
            start = end

    def get_sequence_ranges(self):
        return list(self._ranges)

    def identity(self, global_index):
        for dataset, sequence_range in zip(self.sequence_datasets, self._ranges):
            if sequence_range["start"] <= global_index < sequence_range["end"]:
                local_index = global_index - sequence_range["start"]
                return dataset.sequence, dataset.radar_dataset.frame_names[local_index]
        raise IndexError(global_index)


class StandardSplitMembershipGoldenTests(unittest.TestCase):
    def setUp(self):
        self.dataset = DummyMultiSequenceDataset()

    def assert_membership(self, actual, expected_indices, expected_identities):
        self.assertEqual(actual, expected_indices)
        self.assertEqual(
            [self.dataset.identity(index) for index in actual],
            expected_identities,
        )

    def test_training_and_low_level_mode_distinction(self):
        self.assertEqual(
            SUPPORTED_TRAINING_SPLIT_MODES,
            ("random", "file", "sequence", "sequence_tail"),
        )
        for mode in SUPPORTED_TRAINING_SPLIT_MODES:
            self.assertEqual(validate_training_split_mode(mode), mode)
        for low_level_only_or_alias in ("order", "sequence-tail"):
            with self.subTest(mode=low_level_only_or_alias), self.assertRaisesRegex(
                ValueError,
                "split_mode must be one of",
            ):
                validate_training_split_mode(low_level_only_or_alias)

    def test_random_split_seed_42_exact_membership_and_order(self):
        train, val = build_random_split_indices(self.dataset, 0.6, 42, None)
        self.assert_membership(
            train,
            [2, 4, 3, 7, 8, 14, 9, 10, 11],
            [
                (1, "s1f2"), (1, "s1f4"), (1, "s1f3"),
                (2, "s2f2"), (2, "s2f3"),
                (9, "s9f5"), (9, "s9f0"), (9, "s9f1"), (9, "s9f2"),
            ],
        )
        self.assert_membership(
            val,
            [0, 1, 5, 6, 12, 13, 15],
            [
                (1, "s1f0"), (1, "s1f1"),
                (2, "s2f0"), (2, "s2f1"),
                (9, "s9f3"), (9, "s9f4"), (9, "s9f6"),
            ],
        )

    def test_random_split_seed_7_and_global_limit_are_exact(self):
        self.assertEqual(
            build_random_split_indices(self.dataset, 0.6, 7, None),
            (
                [0, 1, 3, 8, 6, 10, 14, 12, 9],
                [2, 4, 5, 7, 11, 13, 15],
            ),
        )
        self.assertEqual(
            build_random_split_indices(self.dataset, 0.6, 42, 8),
            ([2, 4, 3, 5], [0, 1, 7, 6]),
        )

    def test_order_split_preserves_per_sequence_cutoff_and_order(self):
        self.assertEqual(
            build_order_split_indices(self.dataset, 0.6, None),
            (
                [0, 1, 2, 5, 6, 9, 10, 11, 12],
                [3, 4, 7, 8, 13, 14, 15],
            ),
        )

    def test_sequence_tail_odd_lengths_boundary_and_limit_are_exact(self):
        self.assertEqual(
            build_sequence_tail_split_indices(
                self.dataset,
                val_ratio=0.3,
                boundary_drop_frames=1,
            ),
            ([0, 1, 5, 9, 10, 11], [3, 4, 7, 8, 13, 14, 15]),
        )
        self.assertEqual(
            build_sequence_tail_split_indices(
                self.dataset,
                val_ratio=0.3,
                boundary_drop_frames=0,
                limit_samples=8,
            ),
            ([0, 1, 2, 5, 6], [3, 4, 7]),
        )

    def test_sequence_split_full_first_last_string_ids_and_order(self):
        self.assertEqual(
            build_sequence_split_indices(
                self.dataset, (1, 9), (2,), None,
            ),
            ([0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15], [5, 6, 7, 8]),
        )
        self.assertEqual(
            build_sequence_split_indices(
                self.dataset,
                (1, 9),
                (2,),
                None,
                train_sequence_half_selection={1: "first", 9: "last"},
                train_sequence_half_ratio=0.5,
            ),
            ([0, 1, 2, 12, 13, 14, 15], [5, 6, 7, 8]),
        )
        self.assertEqual(
            build_sequence_split_indices(
                self.dataset,
                (1, 9),
                (2,),
                None,
                train_sequence_half_selection={"1": "last"},
                train_sequence_half_ratio=0.5,
            ),
            ([2, 3, 4, 9, 10, 11, 12, 13, 14, 15], [5, 6, 7, 8]),
        )
        self.assertEqual(
            build_sequence_split_indices(
                self.dataset,
                (9, 1),
                (2,),
                3,
                train_sequence_half_selection={9: "first"},
                train_sequence_half_ratio=0.5,
            ),
            ([9, 10, 11], [5, 6, 7]),
        )

    def test_sequence_normalization_keeps_duplicates_and_unique_order_contract(self):
        self.assertEqual(normalize_sequence_list("1,3-5,3"), (1, 3, 4, 5, 3))
        self.assertEqual(unique_sequences((2, 1, 2), (3, 1)), (2, 1, 3))
        cfg = type("Cfg", (), {"sequences": (8, 7)})()
        self.assertEqual(
            get_dataset_sequences_for_split(cfg, "random"),
            (8, 7),
        )
        self.assertEqual(
            get_dataset_sequences_for_split(cfg, "sequence", (2, 1, 2), (3, 1)),
            (2, 1, 3),
        )
        self.assertEqual(
            get_dataset_sequences_for_split(cfg, "sequence_tail", (3, 9), None),
            (3, 9),
        )

    def test_sequence_split_error_contracts(self):
        with self.assertRaisesRegex(ValueError, r"overlap=\[2\]"):
            build_sequence_split_indices(self.dataset, (1, 2), (2,), None)
        with self.assertRaisesRegex(ValueError, r"missing_train=\[99\]"):
            build_sequence_split_indices(self.dataset, (99,), (2,), None)
        with self.assertRaisesRegex(ValueError, "values must be 'first' or 'last'"):
            build_sequence_split_indices(
                self.dataset, (1,), (2,), None,
                train_sequence_half_selection={1: "middle"},
            )
        with self.assertRaisesRegex(ValueError, "val_ratio must be between"):
            build_sequence_tail_split_indices(self.dataset, val_ratio=1.0)


class ManifestMembershipGoldenTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = DummyMultiSequenceDataset((
            (2, ("b0", "b1", "b2")),
            (1, ("a0", "a1", "a2", "a3")),
        ))

    def test_split_line_formats_comments_and_errors(self):
        self.assertIsNone(split_line_to_sequence_and_frame_names("  # comment"))
        self.assertIsNone(split_line_to_sequence_and_frame_names(""))
        self.assertEqual(
            split_line_to_sequence_and_frame_names("1,00033_00001.txt"),
            (1, ["00033", "00033_00001"]),
        )
        self.assertEqual(
            split_line_to_sequence_and_frame_names("2, b1.txt, ignored"),
            (2, ["b1"]),
        )
        with self.assertRaisesRegex(ValueError, "Invalid split line"):
            split_line_to_sequence_and_frame_names("1")
        # Preserve the historical os.path.splitext edge case: a dotfile-like
        # token is accepted verbatim rather than treated as an empty stem.
        self.assertEqual(
            split_line_to_sequence_and_frame_names("1,.txt"),
            (1, [".txt"]),
        )

    def test_file_split_grouping_duplicates_missing_and_limit_are_exact(self):
        (self.root / "train.txt").write_text(
            "# comment\n1,a2.txt\n2,b1_old.txt\n1,a2.txt\n"
            "2,missing.txt\n3,outside.txt\n",
            encoding="utf-8",
        )
        (self.root / "test.txt").write_text(
            "2,b2.txt\n1,a0.txt\n",
            encoding="utf-8",
        )
        parsed = read_split_file(self.root / "train.txt", (2, 1))
        self.assertEqual(
            parsed,
            {1: [["a2"], ["a2"]], 2: [["b1", "b1_old"], ["missing"]]},
        )
        output = io.StringIO()
        with redirect_stdout(output):
            train, val = build_file_split_indices(
                self.dataset,
                self.root,
                (2, 1),
                None,
            )
        self.assertEqual((train, val), ([5, 1], [3, 2]))
        self.assertIn("skipped 1 samples", output.getvalue())
        self.assertEqual(
            build_file_split_indices(self.dataset, self.root, (2, 1), 1),
            ([5], [3]),
        )

    def test_exact_manifest_preserves_rows_and_rejects_duplicates_or_empty(self):
        manifest = self.root / "manifest.txt"
        manifest.write_text(
            "2,b2.txt\n1,a1_old.txt\n# comment\n2,b0.txt\n",
            encoding="utf-8",
        )
        self.assertEqual(
            build_exact_frame_manifest_indices(self.dataset, manifest),
            [2, 4, 0],
        )
        manifest.write_text("2,b2.txt\n2,b2_old.txt\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicates sequence 2"):
            build_exact_frame_manifest_indices(self.dataset, manifest)
        manifest.write_text("# only comments\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest is empty"):
            build_exact_frame_manifest_indices(self.dataset, manifest)

    def test_missing_file_split_error_contract(self):
        with self.assertRaisesRegex(FileNotFoundError, "Training split file not found"):
            build_file_split_indices(self.dataset, self.root / "missing", (1, 2), None)

    def test_control_manifest_filters_only_named_sequences_without_reordering(self):
        control_dir = self.root / "control"
        control_dir.mkdir()
        (control_dir / "train.txt").write_text(
            "2,b2.txt\n2,b0.txt\n",
            encoding="utf-8",
        )
        with redirect_stdout(io.StringIO()):
            actual = apply_train_control_split_indices(
                self.dataset,
                [3, 0, 1, 4, 2, 5],
                control_dir,
            )
        self.assertEqual(actual, [3, 0, 4, 2, 5])


def _frame_info(file_idx, labels):
    key = _bin_key("Sedan", 0.0, 20.0)
    return {
        "file_idx": file_idx,
        "frame_name": f"f{file_idx}",
        "category_object_labels": {key: list(labels)},
        "category_counts": {key: len(labels)},
        "all_target_object_labels": list(labels),
        "outside_bin_object_labels": [],
        "total_boxes_in_bins": len(labels),
    }


class ControlledSplitGoldenTests(unittest.TestCase):
    def setUp(self):
        self.source = [
            _frame_info(0, [10, 11, 12]),
            _frame_info(1, [20, 21, 22]),
            _frame_info(2, [30, 31, 32]),
        ]
        self.reference = [_frame_info(0, [100, 101, 102, 103])]

    def run_trial(self, seed):
        return _run_trial(
            self.source,
            _summarize_frames(self.reference),
            _build_population(self.source),
            seed=seed,
            range_m_bins=((0.0, 20.0),),
            total_bbox_tolerance_ratio=0.0,
        )

    def test_seeded_control_selection_is_exact_and_repeatable(self):
        expected = {
            42: {0: [10, 11, 12], 1: [22], 2: []},
            43: {0: [10, 11], 1: [21, 22], 2: []},
            44: {0: [11], 1: [20], 2: [30, 31]},
        }
        for seed, expected_membership in expected.items():
            with self.subTest(seed=seed):
                trial = self.run_trial(seed)
                actual = {
                    frame_idx: sorted(labels)
                    for frame_idx, labels in trial["keep_by_frame"].items()
                }
                self.assertEqual(actual, expected_membership)
                repeated = self.run_trial(seed)
                self.assertEqual(trial["keep_by_frame"], repeated["keep_by_frame"])

    def test_ignore_override_membership_and_counts_are_exact(self):
        trial = self.run_trial(42)
        overrides, counts = _build_override_frames(
            self.source,
            trial["keep_by_frame"],
        )
        self.assertEqual(overrides, {
            "f1": {"ignore_object_labels": [20, 21]},
            "f2": {"ignore_object_labels": [30, 31, 32]},
        })
        self.assertEqual(counts, {
            "f0": {"total_kept": 3, "category_counts": {"sedan_range_m_0_20": 3}},
            "f1": {"total_kept": 1, "category_counts": {"sedan_range_m_0_20": 1}},
            "f2": {"total_kept": 0, "category_counts": {"sedan_range_m_0_20": 0}},
        })

    def test_sequence_part_full_first_last_odd_and_complementary_contract(self):
        frames = list(range(5))
        self.assertEqual(_select_sequence_part(frames, "full", 0.4), frames)
        self.assertEqual(_select_sequence_part(frames, "first", 0.4), [0, 1])
        self.assertEqual(_select_sequence_part(frames, "last", 0.4), [3, 4])
        first = _select_sequence_part(frames, "first", 0.5, complementary=True)
        last = _select_sequence_part(frames, "last", 0.5, complementary=True)
        self.assertEqual((first, last), ([0, 1, 2], [3, 4]))
        self.assertEqual(first + last, frames)

    def test_sequence_part_normalization_keeps_duplicates_and_validates(self):
        self.assertEqual(
            _normalize_sequence_parts([9, (9, "first"), [12, "LAST"]], "parts"),
            ((9, "full"), (9, "first"), (12, "last")),
        )
        with self.assertRaisesRegex(ValueError, "Invalid parts part"):
            _normalize_sequence_parts([(9, "middle")], "parts")

    def test_controlled_reference_pairing_order_and_repeat_rule(self):
        args = type("Args", (), {
            "controlled_sequence_parts": ((9, "first"), (9, "last")),
            "reference_sequence_parts": ((12, "full"),),
        })()
        self.assertEqual(
            _pair_sequence_parts(args),
            ((9, "first", 12, "full"), (9, "last", 12, "full")),
        )


if __name__ == "__main__":
    unittest.main()
