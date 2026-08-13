import unittest

from controlled_sequences import (
    _build_population,
    _run_trial,
    _select_sequence_part,
    _summarize_frames,
    _bin_key,
    _category_keys,
    _request_signature,
)


def frame_info(file_idx, labels_by_category, outside=()):
    category_counts = {
        key: len(labels)
        for key, labels in labels_by_category.items()
    }
    all_labels = [
        int(label)
        for labels in labels_by_category.values()
        for label in labels
    ]
    all_labels.extend(int(label) for label in outside)
    return {
        "file_idx": file_idx,
        "frame_name": f"frame_{file_idx:03d}",
        "category_object_labels": {
            key: list(labels)
            for key, labels in labels_by_category.items()
        },
        "category_counts": category_counts,
        "all_target_object_labels": sorted(all_labels),
        "outside_bin_object_labels": list(outside),
        "total_boxes_in_bins": sum(category_counts.values()),
    }


class ControlledSequenceTests(unittest.TestCase):
    def test_default_control_categories_are_sedan_only(self):
        keys = _category_keys(((0.0, 20.0), (20.0, 40.0)))

        self.assertEqual(len(keys), 2)
        self.assertTrue(all(key.startswith("sedan_") for key in keys))
        self.assertFalse(any(key.startswith("bus_or_truck_") for key in keys))

    def test_control_class_names_are_part_of_split_signature(self):
        common = {
            "schema_version": 7,
            "pairs": [],
            "box_coordinate_mode": "cartesian",
            "range_m_bins": [],
            "window_position": "last",
            "seed": 42,
            "num_trials": 300,
            "total_bbox_tolerance_ratio": 0.05,
        }
        sedan_signature = _request_signature({
            **common,
            "control_class_names": ["Sedan"],
        })
        mixed_signature = _request_signature({
            **common,
            "control_class_names": ["Sedan", "Bus or Truck"],
        })

        self.assertNotEqual(sedan_signature, mixed_signature)

    def test_complementary_first_and_last_parts_are_disjoint_and_complete(self):
        frame_infos = list(range(5))
        first = _select_sequence_part(
            frame_infos,
            "first",
            ratio=0.5,
            complementary=True,
        )
        last = _select_sequence_part(
            frame_infos,
            "last",
            ratio=0.5,
            complementary=True,
        )

        self.assertEqual(first, [0, 1, 2])
        self.assertEqual(last, [3, 4])
        self.assertFalse(set(first) & set(last))
        self.assertEqual(first + last, frame_infos)

    def test_close_total_keeps_all_boxes_even_with_range_mismatch(self):
        near = _bin_key("Sedan", 0.0, 20.0)
        far = _bin_key("Sedan", 80.0, 120.0)
        source = [frame_info(0, {near: [1, 2, 3]})]
        reference = [frame_info(0, {far: [10, 11, 12]})]

        trial = _run_trial(
            source,
            _summarize_frames(reference),
            _build_population(source),
            seed=42,
            total_bbox_tolerance_ratio=0.05,
        )

        self.assertTrue(trial["keep_all"])
        self.assertEqual(trial["after_summary"]["selected_target_objects"], 3)
        self.assertEqual(trial["keep_by_frame"][0], {1, 2, 3})
        self.assertEqual(trial["score"][0], 0)

    def test_zero_tolerance_masks_even_one_excess_sedan(self):
        near = _bin_key("Sedan", 0.0, 20.0)
        source = [frame_info(0, {near: [1, 2, 3]})]
        reference = [frame_info(0, {near: [10, 11]})]

        trial = _run_trial(
            source,
            _summarize_frames(reference),
            _build_population(source),
            seed=42,
            total_bbox_tolerance_ratio=0.0,
        )

        self.assertFalse(trial["keep_all"])
        self.assertEqual(trial["total_bbox_tolerance"], 0)
        self.assertEqual(trial["after_summary"]["selected_target_objects"], 2)

    def test_source_shortage_keeps_all_sedans(self):
        near = _bin_key("Sedan", 0.0, 20.0)
        source = [frame_info(0, {near: [1, 2]})]
        reference = [frame_info(0, {near: [10, 11, 12]})]

        trial = _run_trial(
            source,
            _summarize_frames(reference),
            _build_population(source),
            seed=42,
            total_bbox_tolerance_ratio=0.0,
        )

        self.assertTrue(trial["keep_all"])
        self.assertEqual(trial["after_summary"]["selected_target_objects"], 2)

    def test_large_total_difference_controls_count_before_range(self):
        near = _bin_key("Sedan", 0.0, 20.0)
        far = _bin_key("Sedan", 80.0, 120.0)
        source = [frame_info(0, {near: list(range(10))})]
        reference = [frame_info(0, {far: list(range(100, 105))})]

        trial = _run_trial(
            source,
            _summarize_frames(reference),
            _build_population(source),
            seed=42,
            total_bbox_tolerance_ratio=0.05,
        )

        self.assertFalse(trial["keep_all"])
        self.assertEqual(trial["after_summary"]["selected_target_objects"], 5)
        self.assertEqual(len(trial["keep_by_frame"][0]), 5)

    def test_excess_sedans_are_masked_to_reference_distance_counts(self):
        near = _bin_key("Sedan", 0.0, 20.0)
        far = _bin_key("Sedan", 80.0, 120.0)
        source = [
            frame_info(
                0,
                {
                    near: list(range(10)),
                    far: list(range(10, 20)),
                },
            )
        ]
        reference = [
            frame_info(
                0,
                {
                    near: [100, 101],
                    far: [102, 103, 104, 105],
                },
            )
        ]

        trial = _run_trial(
            source,
            _summarize_frames(reference),
            _build_population(source),
            seed=42,
            total_bbox_tolerance_ratio=0.0,
        )

        self.assertEqual(trial["after_summary"]["selected_target_objects"], 6)
        self.assertEqual(trial["after_summary"]["category_totals"][near], 2)
        self.assertEqual(trial["after_summary"]["category_totals"][far], 4)

    def test_distance_bins_prioritize_0_to_80_before_80_to_120(self):
        near = _bin_key("Sedan", 0.0, 20.0)
        middle = _bin_key("Sedan", 20.0, 40.0)
        far = _bin_key("Sedan", 80.0, 120.0)
        source = [
            frame_info(
                0,
                {
                    near: list(range(8)),
                    middle: list(range(10, 16)),
                    far: [],
                },
            )
        ]
        reference = [
            frame_info(
                0,
                {
                    near: [100],
                    middle: [101],
                    far: list(range(102, 112)),
                },
            )
        ]

        trial = _run_trial(
            source,
            _summarize_frames(reference),
            _build_population(source),
            seed=42,
            range_m_bins=(
                (0.0, 20.0),
                (20.0, 40.0),
                (40.0, 60.0),
                (60.0, 80.0),
                (80.0, 120.0),
            ),
            total_bbox_tolerance_ratio=0.0,
        )

        # The source has no far boxes.  Extra boxes needed to reach the target
        # total are retained strictly from near to far.
        self.assertEqual(
            trial["after_summary"]["category_totals"][near],
            8,
        )
        self.assertEqual(
            trial["after_summary"]["category_totals"][middle],
            4,
        )
        self.assertEqual(
            trial["after_summary"]["category_totals"][far],
            0,
        )

    def test_nearest_remaining_boxes_are_kept_first(self):
        near = _bin_key("Sedan", 0.0, 20.0)
        middle = _bin_key("Sedan", 20.0, 40.0)
        far = _bin_key("Sedan", 80.0, 120.0)
        source = [
            frame_info(
                0,
                {
                    near: list(range(500)),
                    middle: list(range(500, 800)),
                    far: [],
                },
            )
        ]
        reference = [
            frame_info(
                0,
                {
                    near: list(range(1000, 1100)),
                    middle: list(range(1100, 1200)),
                    far: list(range(1200, 1600)),
                },
            )
        ]

        trial = _run_trial(
            source,
            _summarize_frames(reference),
            _build_population(source),
            seed=42,
            range_m_bins=(
                (0.0, 20.0),
                (20.0, 40.0),
                (40.0, 60.0),
                (60.0, 80.0),
                (80.0, 120.0),
            ),
            total_bbox_tolerance_ratio=0.0,
        )

        # The initial per-bin reservation keeps 100 near and 100 middle boxes.
        # The remaining 400 boxes required by the target total all come from
        # the nearest bin before any extra middle-range boxes are retained.
        self.assertEqual(
            trial["after_summary"]["category_totals"][near],
            500,
        )
        self.assertEqual(
            trial["after_summary"]["category_totals"][middle],
            100,
        )
        self.assertEqual(
            trial["after_summary"]["selected_target_objects"],
            600,
        )


if __name__ == "__main__":
    unittest.main()
