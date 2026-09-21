"""Golden numerical regression tests for the training loss contracts.

The constants in this module were captured from the monolithic
``training.losses`` implementation before the Step 8 extraction.  Keep
them literal: deriving expected values through loss helpers would make these
tests unable to detect numerical drift during responsibility-only moves.
"""

import math
import unittest

import torch

from training.configuration import resolve_loss_mode
from training.losses import (
    cartesian_centerpoint_detection_loss,
    gaussian_wasserstein_distance_batch,
    heatmap_focal_loss,
    radenet_detection_loss,
    yolox_detection_loss,
)
from training.losses import centerpoint, gwd, radenet, targets, yolox


RTOL = 1e-6
ATOL = 1e-7


def _leaf(values, shape):
    return torch.tensor(values, dtype=torch.float32).reshape(shape).requires_grad_()


def _assert_float(actual, expected):
    torch.testing.assert_close(
        torch.tensor(actual),
        torch.tensor(expected),
        rtol=RTOL,
        atol=ATOL,
    )


def _assert_metrics(actual, expected):
    assert set(actual) == set(expected)
    for key, expected_value in expected.items():
        if isinstance(expected_value, int):
            assert actual[key] == expected_value
        else:
            _assert_float(actual[key], expected_value)


def test_loss_mode_resolver_keeps_supported_family_selection():
    assert resolve_loss_mode("model5", "polar", "auto") == "centerpoint"
    assert resolve_loss_mode("model7", "cartesian", "radenet") == "radenet"
    assert resolve_loss_mode("model7", "cartesian", "centerpoint") == "centerpoint"
    assert resolve_loss_mode("model12", "cartesian", "auto") == "radenet"
    assert resolve_loss_mode("model14", "cartesian", "auto") == "yolox"


def test_historical_loss_imports_are_canonical_reexports():
    assert cartesian_centerpoint_detection_loss is centerpoint.cartesian_centerpoint_detection_loss
    assert radenet_detection_loss is radenet.radenet_detection_loss
    assert yolox_detection_loss is yolox.yolox_detection_loss
    assert gaussian_wasserstein_distance_batch is gwd.gaussian_wasserstein_distance_batch


def test_heatmap_background_and_ignore_mask_weighting_golden():
    logits = torch.tensor([[[[-1.0, 0.5], [1.5, -0.2]]]])
    targets = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]])
    full = heatmap_focal_loss(logits, targets)
    explicit_full = heatmap_focal_loss(logits, targets, torch.ones_like(targets))
    ignored = heatmap_focal_loss(
        logits,
        targets,
        torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]]),
    )
    _assert_float(full.item(), 1.348716139793396)
    _assert_float(explicit_full.item(), 1.348716139793396)
    _assert_float(ignored.item(), 0.21144402027130127)


def _radenet_outputs(num_classes):
    return {
        "heatmap": torch.linspace(0.08, 0.86, num_classes * 9)
        .reshape(1, num_classes, 3, 3).requires_grad_(),
        "regression": torch.linspace(-0.45, 1.25, 72)
        .reshape(1, 8, 3, 3).requires_grad_(),
    }


def test_radenet_uses_exact_metric_gt_and_has_finite_gradients():
    outputs = _radenet_outputs(2)
    loss, metrics = radenet_detection_loss(
        outputs=outputs,
        gt_boxes_raw_list=[torch.tensor([[128.0, 53.0, 18.0, 10.0, 4.0, 3.0, 0.4]])],
        gt_metric_boxes_list=[torch.tensor([[51.0, -1.0, 1.0, 4.2, 1.8, 1.6, 0.4]])],
        gt_labels_list=[torch.tensor([1])],
        scope_modes=["full"],
        full_rae_shapes=[(256, 107, 37)],
        num_classes=2,
        gaussian_sigma=1.5,
    )
    assert torch.isfinite(loss)
    assert math.isfinite(metrics["total_loss"])
    loss.backward()
    assert torch.isfinite(outputs["heatmap"].grad).all()
    assert torch.isfinite(outputs["regression"].grad).all()


def test_radenet_empty_target_golden():
    outputs = _radenet_outputs(1)
    loss, metrics = radenet_detection_loss(
        outputs=outputs,
        gt_boxes_raw_list=[torch.empty((0, 7))],
        gt_metric_boxes_list=[torch.empty((0, 7))],
        gt_labels_list=[torch.empty((0,), dtype=torch.long)],
        scope_modes=["full"],
        full_rae_shapes=[(256, 107, 37)],
        num_classes=1,
        gaussian_sigma=1.5,
    )
    assert math.isfinite(loss.item())
    _assert_float(loss.item(), 2.0)
    _assert_metrics(metrics, {
        "total_loss": 0.3643825352191925,
        "box_loss": 0.0,
        "cls_loss": 0.3643825352191925,
        "heatmap_loss": 0.3643825352191925,
        "gwd_loss": 0.0,
        "l1_loss": 0.0,
        "ignore_pixels": 0,
    })
    loss.backward()
    torch.testing.assert_close(
        outputs["heatmap"].grad.flatten(),
        torch.tensor([
            0.012378671206533909, 0.0656665712594986,
            0.17148113250732422, 0.3465857207775116,
            0.6181403398513794, 1.034300446510315,
            1.6921138763427734, 2.829960346221924,
            5.284172058105469,
        ]),
        rtol=RTOL,
        atol=ATOL,
    )
    assert outputs["regression"].grad is None


def _cartesian_outputs():
    return {
        "cls_logits": torch.linspace(-0.8, 0.9, 18)
        .reshape(1, 2, 3, 3).requires_grad_(),
        "center_offset": torch.linspace(-0.4, 0.4, 18)
        .reshape(1, 2, 3, 3).requires_grad_(),
        "center_height": torch.linspace(-0.3, 0.5, 9)
        .reshape(1, 1, 3, 3).requires_grad_(),
        "size": torch.linspace(0.7, 2.1, 27)
        .reshape(1, 3, 3, 3).requires_grad_(),
        "yaw": torch.linspace(-0.6, 0.8, 18)
        .reshape(1, 2, 3, 3).requires_grad_(),
    }


def test_cartesian_centerpoint_forward_and_gradient_golden():
    outputs = _cartesian_outputs()
    loss, metrics = cartesian_centerpoint_detection_loss(
        outputs=outputs,
        gt_boxes_raw_list=[torch.tensor([[128.0, 53.0, 18.0, 10.0, 4.0, 3.0, 0.4]])],
        gt_metric_boxes_list=[torch.tensor([[20.0, 1.5, 0.8, 4.2, 1.9, 1.6, 0.35]])],
        gt_labels_list=[torch.tensor([1])],
        scope_modes=["full"],
        full_rae_shapes=[(256, 107, 37)],
        box_loss_weight=1.2,
        cls_loss_weight=0.8,
        gwd_loss_weight=1.7,
        heatmap_radius=1,
        num_classes=2,
    )
    _assert_float(loss.item(), 29.73999786376953)
    _assert_metrics(metrics, {
        "total_loss": 29.73999786376953,
        "box_loss": 22.61220932006836,
        "cls_loss": 3.256680727005005,
        "heatmap_loss": 3.256680727005005,
        "offset_loss": 19.54751205444336,
        "height_loss": 0.24500003457069397,
        "size_loss": 0.9833727478981018,
        "yaw_loss": 0.1782882660627365,
        "gwd_loss": 0.9753143191337585,
        "num_center_targets": 1,
        "ignore_pixels": 0,
    })
    loss.backward()
    torch.testing.assert_close(
        outputs["cls_logits"].grad.flatten(),
        torch.tensor([
            0.06321541219949722, 0.07668377459049225,
            0.09233919531106949, 0.1103496178984642,
            0.13084915280342102, 0.1539267599582672,
            0.17961552739143372, 0.20788387954235077,
            0.23862943053245544, 0.25231221318244934,
            0.1714780479669571, 0.3191149830818176,
            0.2134133279323578, -0.11034958064556122,
            0.2574242055416107, 0.4645068943500519,
            0.3014712929725647, 0.5364430546760559,
        ]),
        rtol=RTOL,
        atol=ATOL,
    )
    selected = {
        "center_offset": ([4, 13], [0.6012414693832397, -0.6000412106513977]),
        "center_height": ([4], [-0.8400000929832458]),
        "size": ([4, 13, 22], [-0.4000147581100464, -0.20001541078090668, 0.11384611576795578]),
        "yaw": ([4, 13], [-0.7334954738616943, -0.42175984382629395]),
    }
    for key, (indices, expected) in selected.items():
        torch.testing.assert_close(
            outputs[key].grad.flatten()[torch.tensor(indices)],
            torch.tensor(expected),
            rtol=RTOL,
            atol=ATOL,
        )


def test_gwd_formula_and_stability_edges_golden():
    pred = torch.tensor([
        [1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.6, 0.8],
        [2.0, -1.0, 0.0, 3.0, 1.5, 1.0, -0.3, 0.9539392],
        [0.0, 0.0, 0.0, 1e-5, 2e-5, 1.0, 0.0, 1.0],
    ])
    target = torch.tensor([
        [1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.6, 0.8],
        [3.0, 1.0, 0.0, 2.5, 2.0, 1.0, 0.2, 0.9797959],
        [0.1, 0.0, 0.0, 2e-5, 1e-5, 1.0, 0.01, 0.99995],
    ])
    similarity, loss = gaussian_wasserstein_distance_batch(pred, target)
    torch.testing.assert_close(
        similarity,
        torch.tensor([0.606060266494751, 0.2542790472507477, 0.5714350938796997]),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        loss,
        torch.tensor([0.393939733505249, 0.7457209825515747, 0.4285649061203003]),
        rtol=RTOL,
        atol=ATOL,
    )
    assert torch.isfinite(loss).all()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
def test_loss_helpers_preserve_cuda_device():
    boxes = torch.tensor(
        [[1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.6, 0.8]],
        device="cuda",
    )
    similarity, loss = gaussian_wasserstein_distance_batch(boxes, boxes)
    assert similarity.device.type == "cuda"
    assert loss.device.type == "cuda"


def load_tests(loader, tests, pattern):
    """Expose the function-style golden cases to unittest discovery."""
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
