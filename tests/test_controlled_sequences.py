import unittest

from controlled_sequences import (
    _build_population,
    _run_trial,
    _summarize_frames,
    _bin_key,
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

        # The source has no far boxes.  The required extra boxes are therefore
        # taken from the 0-80 m group before considering the far group.
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


if __name__ == "__main__":
    unittest.main()
