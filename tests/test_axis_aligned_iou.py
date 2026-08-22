import unittest

import numpy as np

from eval.kitti_eval.axis_aligned_iou import rotate_iou_gpu_eval


class AxisAlignedIouTest(unittest.TestCase):
    def test_yaw_is_ignored(self):
        boxes = np.array([[0.0, 0.0, 4.0, 2.0, 0.0]], dtype=np.float64)
        queries = np.array(
            [[0.0, 0.0, 4.0, 2.0, np.pi / 2.0]],
            dtype=np.float64,
        )
        overlaps = rotate_iou_gpu_eval(boxes, queries)
        self.assertAlmostEqual(float(overlaps[0, 0]), 1.0)

    def test_iou_and_raw_intersection(self):
        boxes = np.array([[0.0, 0.0, 2.0, 2.0, 0.0]], dtype=np.float32)
        queries = np.array([[1.0, 0.0, 2.0, 2.0, 1.2]], dtype=np.float32)
        iou = rotate_iou_gpu_eval(boxes, queries, criterion=-1)
        intersection = rotate_iou_gpu_eval(boxes, queries, criterion=2)
        self.assertAlmostEqual(float(intersection[0, 0]), 2.0)
        self.assertAlmostEqual(float(iou[0, 0]), 2.0 / 6.0, places=6)


if __name__ == "__main__":
    unittest.main()
