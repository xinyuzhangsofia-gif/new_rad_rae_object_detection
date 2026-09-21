import unittest
from types import SimpleNamespace

import torch

from training.configuration import resolve_centerpoint_gwd_loss_weight
from training.losses import cartesian_centerpoint_detection_loss


class Model7LossSemanticsTests(unittest.TestCase):
    def test_gwd_weight_uses_the_canonical_default(self):
        args = SimpleNamespace()
        resolve_centerpoint_gwd_loss_weight(args)
        self.assertEqual(args.centerpoint_gwd_loss_weight, 2.0)

    def test_gwd_weight_is_normalized_to_float(self):
        args = SimpleNamespace(centerpoint_gwd_loss_weight="3.0")
        resolve_centerpoint_gwd_loss_weight(args)
        self.assertEqual(args.centerpoint_gwd_loss_weight, 3.0)

    def test_model7_outputs_do_not_report_inactive_quality_or_giou_loss(self):
        outputs = {
            "cls_logits": torch.zeros(1, 1, 4, 4),
            "center_offset": torch.zeros(1, 2, 4, 4),
            "center_height": torch.zeros(1, 1, 4, 4),
            "size": torch.zeros(1, 3, 4, 4),
            "yaw": torch.cat(
                (torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4)),
                dim=1,
            ),
        }
        total_loss, loss_dict = cartesian_centerpoint_detection_loss(
            outputs=outputs,
            gt_boxes_raw_list=[torch.tensor([[100.0, 53.0, 18.0, 3.0, 3.0, 3.0, 0.0]])],
            gt_metric_boxes_list=[torch.tensor([[40.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])],
            gt_labels_list=[torch.tensor([0])],
            scope_modes=["full"],
            full_rae_shapes=[(256, 107, 37)],
            gwd_loss_weight=2.0,
            num_classes=1,
        )
        self.assertIn("gwd_loss", loss_dict)
        self.assertNotIn("giou_loss", loss_dict)
        self.assertNotIn("quality_loss", loss_dict)
        self.assertAlmostEqual(
            total_loss.item(),
            loss_dict["box_loss"] + loss_dict["cls_loss"],
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
