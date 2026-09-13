import unittest

from data.dataloader import (
    build_sequence_tail_split_indices,
    get_dataset_sequences_for_split,
)


class DummyDataset:
    def __init__(self, ranges):
        self.ranges = ranges

    def get_sequence_ranges(self):
        return self.ranges


class ChronologicalSplitTest(unittest.TestCase):
    def test_tail_split_is_independent_and_drops_boundary_frames(self):
        dataset = DummyDataset([
            {"sequence": 3, "start": 0, "end": 100},
            {"sequence": 9, "start": 100, "end": 250},
        ])

        train_indices, val_indices = build_sequence_tail_split_indices(
            full_dataset=dataset,
            val_ratio=0.1,
            boundary_drop_frames=30,
        )

        self.assertEqual(train_indices[:2], [0, 1])
        self.assertEqual(train_indices[58:60], [58, 59])
        self.assertEqual(val_indices[:2], [90, 91])
        self.assertEqual(val_indices[-2:], [248, 249])
        self.assertEqual(len(train_indices), 60 + 105)
        self.assertEqual(len(val_indices), 10 + 15)
        self.assertTrue(set(train_indices).isdisjoint(val_indices))
        used_indices = set(train_indices) | set(val_indices)
        self.assertTrue(set(range(60, 90)).isdisjoint(used_indices))
        self.assertTrue(set(range(205, 235)).isdisjoint(used_indices))

    def test_sequence_tail_uses_train_sequences_only(self):
        sequences = get_dataset_sequences_for_split(
            cfg=type("Cfg", (), {"sequences": (1, 2, 3)})(),
            split_mode="sequence_tail",
            train_sequences=(3, 9),
            val_sequences=None,
        )

        self.assertEqual(sequences, (3, 9))


if __name__ == "__main__":
    unittest.main()
