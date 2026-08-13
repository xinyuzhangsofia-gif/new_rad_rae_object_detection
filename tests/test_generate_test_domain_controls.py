import math
import unittest

import numpy as np

from scripts.generate_test_domain_controls import (
    allocate_keep_counts,
    build_override,
    select_continuous_window,
    select_kept_objects,
    summarize_frames,
    validate_override_and_frame_counts,
    discover_control_definitions,
)


BINS = (
    {"tag": "q1", "lower_m": 0.0, "upper_m": 10.0, "bbox_count": 1},
    {"tag": "q2", "lower_m": 10.0, "upper_m": 20.0, "bbox_count": 1},
    {"tag": "q3", "lower_m": 20.0, "upper_m": 30.0, "bbox_count": 1},
    {"tag": "q4", "lower_m": 30.0, "upper_m": math.inf, "bbox_count": 1},
)


def make_frame(file_idx, distances):
    return {
        "sequence": 18,
        "file_idx": file_idx,
        "frame_name": f"{file_idx:05d}",
        "eligible_objects": [
            {
                "object_label": label,
                "distance_m": float(distance),
                "metric_box": np.array(
                    [distance, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
                    dtype=np.float64,
                ),
            }
            for label, distance in enumerate(distances)
        ],
    }


class TestTestDomainControlMath(unittest.TestCase):
    def test_discovers_all_five_weather_groups_and_17_test_sets(self):
        definitions = discover_control_definitions()
        self.assertEqual(len(definitions), 17)
        self.assertEqual(
            {item["weather_group"] for item in definitions},
            {"heavy_snow", "light_snow", "overcast", "rain", "sleet"},
        )
        self.assertIn(
            (46, 47),
            {tuple(item["target_sequences"]) for item in definitions},
        )

    def test_allocate_keeps_all_when_source_is_short(self):
        self.assertEqual(
            allocate_keep_counts([1, 2, 3, 4], [5, 5, 5, 5]),
            [1, 2, 3, 4],
        )

    def test_allocate_exact_total_and_uses_surplus_bins(self):
        allocation = allocate_keep_counts([10, 0, 10, 0], [2, 2, 2, 2])
        self.assertEqual(sum(allocation), 8)
        self.assertEqual(allocation[1], 0)
        self.assertEqual(allocation[3], 0)
        self.assertTrue(allocation[0] <= 10)
        self.assertTrue(allocation[2] <= 10)

    def test_continuous_window_chooses_closest_distribution(self):
        frames = [
            make_frame(0, [2.0, 2.0]),
            make_frame(1, [2.0, 2.0]),
            make_frame(2, [12.0]),
            make_frame(3, [22.0]),
        ]
        target_summary = {
            "frame_count": 2,
            "eligible_bbox_count": 2,
            "bin_counts": {"q1": 0, "q2": 1, "q3": 1, "q4": 0},
            "empty_frame_count": 0,
            "frame_bbox_histogram": {"1": 2},
        }
        selected, start, reason = select_continuous_window(
            frames, target_summary, BINS, seed=42
        )
        self.assertEqual(start, 2)
        self.assertEqual([frame["file_idx"] for frame in selected], [2, 3])
        self.assertEqual(reason, "closest_continuous_window")

    def test_override_count_and_per_frame_counts_are_exact(self):
        frames = [
            make_frame(0, [2.0, 3.0, 12.0]),
            make_frame(1, [22.0, 32.0, 33.0]),
        ]
        target_summary = {
            "frame_count": 2,
            "eligible_bbox_count": 4,
            "bin_counts": {"q1": 1, "q2": 1, "q3": 1, "q4": 1},
            "empty_frame_count": 0,
            "frame_bbox_histogram": {"2": 2},
        }
        kept, _audit = select_kept_objects(
            frames, target_summary, BINS, seed=42, num_trials=10
        )
        override, masked = build_override(18, frames, kept)
        per_frame, validated = validate_override_and_frame_counts(
            frames, kept, override
        )
        after = summarize_frames(frames, BINS, kept_keys=kept)
        self.assertEqual(after["eligible_bbox_count"], 4)
        self.assertEqual(masked, 2)
        self.assertEqual(validated, 2)
        self.assertEqual(
            sum(item["masked_post_fov"] for item in per_frame.values()), 2
        )

    def test_negative_eligible_label_is_mandatory_when_target_can_keep_it(self):
        frame = make_frame(0, [2.0])
        frame["eligible_objects"][0]["object_label"] = -1
        target_summary = {
            "frame_count": 1,
            "eligible_bbox_count": 1,
            "bin_counts": {"q1": 1, "q2": 0, "q3": 0, "q4": 0},
            "empty_frame_count": 0,
            "frame_bbox_histogram": {"1": 1},
        }
        kept, audit = select_kept_objects(
            [frame], target_summary, BINS, seed=42, num_trials=1
        )
        self.assertIn((0, -1), kept)
        self.assertEqual(audit["mandatory_unmaskable_counts"], [1, 0, 0, 0])

    def test_unmaskable_count_above_target_fails_closed(self):
        frame = make_frame(0, [2.0])
        frame["eligible_objects"][0]["object_label"] = -1
        target_summary = {
            "frame_count": 1,
            "eligible_bbox_count": 0,
            "bin_counts": {"q1": 0, "q2": 0, "q3": 0, "q4": 0},
            "empty_frame_count": 1,
            "frame_bbox_histogram": {"0": 1},
        }
        with self.assertRaisesRegex(RuntimeError, "Unmaskable"):
            select_kept_objects(
                [frame], target_summary, BINS, seed=42, num_trials=1
            )


if __name__ == "__main__":
    unittest.main()
