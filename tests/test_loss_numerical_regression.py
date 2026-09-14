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
    build_centerpoint_targets,
    cartesian_centerpoint_detection_loss,
    centerpoint_detection_loss,
    gaussian_wasserstein_distance_batch,
    heatmap_focal_loss,
    radenet_detection_loss,
    yolox_detection_loss,
)
from training.losses import centerpoint, gwd, radenet, targets, yolox
from training.losses.matching import simota_assign
from training import yolox_utils


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


def _polar_centerpoint_outputs(include_quality=True, qfl=False):
    outputs = {
        "cls_logits": _leaf(
            [-0.7, 0.2, 1.1, -1.3, 0.5, -0.4, 0.8, -0.2],
            (1, 2, 2, 2),
        ),
        "center_offset": _leaf(
            [-0.3, 0.4, -0.1, 0.2, 0.6, -0.5, 0.3, -0.2],
            (1, 2, 2, 2),
        ),
        "center_height": _leaf([-0.4, 0.1, 0.7, -0.8], (1, 1, 2, 2)),
        "size": _leaf(
            [-0.2, 0.3, 0.9, -0.6, 0.1, -0.7, 0.5, 0.2,
             -0.5, 0.8, -0.1, 0.4],
            (1, 3, 2, 2),
        ),
        "yaw": _leaf(
            [0.2, -0.5, 0.8, 0.3, 0.9, 0.4, -0.2, 0.7],
            (1, 2, 2, 2),
        ),
    }
    if include_quality:
        outputs["quality_logits"] = _leaf(
            [-0.6, 0.1, 0.7, -0.2],
            (1, 1, 2, 2),
        )
    if qfl:
        # The historical model11 tensor contract uses this key as a mode marker;
        # cls_logits contains the differentiable tensor used by QFL.
        outputs["qfl_cls_logits"] = torch.tensor(1.0)
    return outputs


def _centerpoint_target():
    return (
        [torch.tensor([[0.62, 0.28, 0.45, 0.22, 0.16, 0.12, 0.73]])],
        [torch.tensor([1])],
    )


def test_loss_mode_resolver_keeps_supported_family_selection():
    assert resolve_loss_mode("model5", "polar", "auto") == "centerpoint"
    assert resolve_loss_mode("model7", "cartesian", "radenet") == "radenet"
    assert resolve_loss_mode("model7", "cartesian", "centerpoint") == "centerpoint"
    assert resolve_loss_mode("model12", "polar", "auto") == "yolox"
    assert resolve_loss_mode("model14", "polar", "auto") == "yolox"


def test_historical_loss_imports_are_canonical_reexports():
    assert centerpoint_detection_loss is centerpoint.centerpoint_detection_loss
    assert cartesian_centerpoint_detection_loss is centerpoint.cartesian_centerpoint_detection_loss
    assert radenet_detection_loss is radenet.radenet_detection_loss
    assert yolox_detection_loss is yolox.yolox_detection_loss
    assert gaussian_wasserstein_distance_batch is gwd.gaussian_wasserstein_distance_batch
    assert build_centerpoint_targets is targets.build_centerpoint_targets
    assert yolox_utils.simota_assign is simota_assign


def test_centerpoint_quality_forward_and_gradient_golden():
    outputs = _polar_centerpoint_outputs()
    boxes, labels = _centerpoint_target()
    loss, metrics = centerpoint_detection_loss(
        outputs=outputs,
        gt_boxes_list=boxes,
        gt_labels_list=labels,
        scope_modes=["full"],
        full_rae_shapes=[(256, 107, 37)],
        heatmap_radius=1,
        num_classes=2,
        box_loss_weight=1.3,
        cls_loss_weight=0.7,
        gwd_loss_weight=1.8,
        quality_loss_weight=0.35,
    )
    assert loss.dtype == torch.float32
    _assert_float(loss.item(), 4.9563188552856445)
    _assert_metrics(metrics, {
        "total_loss": 4.9563188552856445,
        "box_loss": 2.7396538257598877,
        "cls_loss": 1.4687728881835938,
        "heatmap_loss": 1.4687728881835938,
        "offset_loss": 0.12473164498806,
        "height_loss": 0.21818774938583374,
        "size_loss": 0.4361431896686554,
        "yaw_loss": 0.1949203759431839,
        "gwd_loss": 0.9809283018112183,
        "quality_loss": 1.0475071668624878,
        "num_center_targets": 1,
        "ignore_pixels": 0,
    })
    expected_weighted = (
        1.3 * metrics["box_loss"]
        + 0.7 * metrics["cls_loss"]
        + 0.35 * metrics["quality_loss"]
    )
    _assert_float(loss.item(), expected_weighted)

    loss.backward()
    expected_gradients = {
        "cls_logits": [
            0.06709830462932587, 0.2684266269207001,
            0.5686566829681396, 0.019037669524550438,
            0.20587334036827087, 0.10633262246847153,
            -0.05531349033117294, 0.08785022795200348,
        ],
        "center_offset": [
            0.0, 0.0, 0.17098107933998108, 0.0,
            0.0, 0.0, 0.15922679007053375, 0.0,
        ],
        "center_height": [0.0, 0.0, 0.28733959794044495, 0.0],
        "size": [
            0.0, 0.0, 0.09504211694002151, 0.0,
            0.0, 0.0, 0.11434737592935562, 0.0,
            0.0, 0.0, 0.10806295275688171, 0.0,
        ],
        "yaw": [
            0.0, 0.0, -0.231778085231781, 0.0,
            0.0, 0.0, -0.9271122217178345, 0.0,
        ],
        "quality_logits": [0.0, 0.0, 0.20602628588676453, 0.0],
    }
    for key, expected in expected_gradients.items():
        torch.testing.assert_close(
            outputs[key].grad.flatten(),
            torch.tensor(expected),
            rtol=RTOL,
            atol=ATOL,
        )


def test_centerpoint_qfl_mode_golden():
    outputs = _polar_centerpoint_outputs(include_quality=False, qfl=True)
    boxes, labels = _centerpoint_target()
    loss, metrics = centerpoint_detection_loss(
        outputs=outputs,
        gt_boxes_list=boxes,
        gt_labels_list=labels,
        scope_modes=["full"],
        full_rae_shapes=[(256, 107, 37)],
        heatmap_radius=1,
        num_classes=2,
    )
    _assert_float(loss.item(), 5.007413864135742)
    _assert_metrics(metrics, {
        "total_loss": 5.007413864135742,
        "box_loss": 2.9358396530151367,
        "cls_loss": 2.0715739727020264,
        "heatmap_loss": 2.0715739727020264,
        "offset_loss": 0.12473164498806,
        "height_loss": 0.21818774938583374,
        "size_loss": 0.4361431896686554,
        "yaw_loss": 0.1949203759431839,
        "gwd_loss": 0.9809283018112183,
        "num_center_targets": 1,
        "ignore_pixels": 0,
    })
    loss.backward()
    torch.testing.assert_close(
        outputs["cls_logits"].grad.flatten(),
        torch.tensor([
            0.09585472196340561, 0.38346660137176514,
            0.8123665452003479, 0.027196675539016724,
            0.5261518359184265, 0.16356146335601807,
            0.5166860222816467, 0.22451940178871155,
        ]),
        rtol=RTOL,
        atol=ATOL,
    )


def test_centerpoint_empty_target_golden():
    outputs = _polar_centerpoint_outputs(include_quality=False)
    outputs["cls_logits"] = _leaf([-0.7, 0.2, 1.1, -1.3], (1, 1, 2, 2))
    outputs["objectness_logits"] = _leaf(
        [-0.6, 0.1, 0.7, -0.2],
        (1, 1, 2, 2),
    )
    loss, metrics = centerpoint_detection_loss(
        outputs=outputs,
        gt_boxes_list=[torch.empty((0, 7))],
        gt_labels_list=[torch.empty((0,), dtype=torch.long)],
        heatmap_radius=1,
        num_classes=1,
        quality_loss_weight=0.25,
    )
    assert math.isfinite(loss.item())
    _assert_metrics(metrics, {
        "total_loss": 1.2578542232513428,
        "box_loss": 0.0,
        "cls_loss": 1.0776536464691162,
        "heatmap_loss": 1.0776536464691162,
        "offset_loss": 0.0,
        "height_loss": 0.0,
        "size_loss": 0.0,
        "yaw_loss": 0.0,
        "gwd_loss": 0.0,
        "quality_loss": 0.720802366733551,
        "num_center_targets": 0,
        "ignore_pixels": 0,
    })
    loss.backward()
    torch.testing.assert_close(
        outputs["cls_logits"].grad.flatten(),
        torch.tensor([
            0.09585472196340561, 0.38346660137176514,
            0.8123666644096375, 0.027196675539016724,
        ]),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        outputs["objectness_logits"].grad.flatten(),
        torch.tensor([
            0.02214648202061653, 0.03281119838356972,
            0.04176173359155655, 0.028135376051068306,
        ]),
        rtol=RTOL,
        atol=ATOL,
    )


def test_centerpoint_target_layout_and_multiclass_order_golden():
    cls_logits = torch.zeros(1, 3, 4, 5)
    reg_reference = torch.zeros(1, 2, 4, 5)
    heatmap, targets, mask = build_centerpoint_targets(
        gt_boxes=[torch.tensor([
            [0.51, 0.39, 0.4, 0.2, 0.3, 0.1, 0.75],
            [0.1, 0.9, 0.6, 0.1, 0.2, 0.3, 0.25],
        ])],
        gt_labels=[torch.tensor([2, 0])],
        cls_logits=cls_logits,
        num_classes=3,
        radius=1,
        reg_reference=reg_reference,
    )
    assert heatmap.shape == (1, 3, 4, 5)
    assert torch.nonzero(heatmap == 1.0).tolist() == [[0, 0, 0, 4], [0, 2, 2, 1]]
    assert torch.nonzero(mask).tolist() == [[0, 0, 0, 4], [0, 0, 2, 1]]
    torch.testing.assert_close(
        targets["center_offset"][0, :, 2, 1],
        torch.tensor([0.039999961853027344, 0.9499999284744263]),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        targets["yaw"][0, :, 2, 1],
        torch.tensor([1.0, 7.549790126404332e-08]),
        rtol=RTOL,
        atol=ATOL,
    )
    assert targets["label"][0, 2, 1].item() == 2


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


def test_radenet_forward_and_gradient_golden():
    outputs = _radenet_outputs(2)
    loss, metrics = radenet_detection_loss(
        outputs=outputs,
        gt_boxes_raw_list=[torch.tensor([[128.0, 53.0, 18.0, 10.0, 4.0, 3.0, 0.4]])],
        gt_labels_list=[torch.tensor([1])],
        scope_modes=["full"],
        full_rae_shapes=[(256, 107, 37)],
        num_classes=2,
        gaussian_sigma=1.5,
    )
    # RADE-Net deliberately backpropagates a detached-mean-normalized scalar,
    # while reporting the unnormalized component sum as total_loss.
    _assert_float(loss.item(), 4.0)
    _assert_metrics(metrics, {
        "total_loss": 2.3373708724975586,
        "box_loss": 1.913644790649414,
        "cls_loss": 0.4237259328365326,
        "heatmap_loss": 0.4237259328365326,
        "gwd_loss": 0.7628260254859924,
        "l1_loss": 1.1508188247680664,
        "ignore_pixels": 0,
    })
    loss.backward()
    torch.testing.assert_close(
        outputs["heatmap"].grad.flatten(),
        torch.tensor([
            0.09580513089895248, 0.2454458475112915,
            0.4737139344215393, 0.7900905609130859,
            1.2060519456863403, 1.735671043395996,
            2.3964591026306152, 3.210571765899658,
            4.206583023071289, 0.08988296985626221,
            0.010891178622841835, 0.1448187232017517,
            0.017355697229504585, -1.924095630645752,
            0.02776041440665722, 0.3750106990337372,
            0.04692002385854721, 0.6779479384422302,
        ]),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        outputs["regression"].grad.flatten()[
            torch.tensor([4, 13, 22, 31, 40, 49, 58, 67])
        ],
        torch.tensor([
            -0.08044329285621643, -0.019055236130952835,
            0.008337605744600296, -0.17130473256111145,
            -0.15714403986930847, -0.10861831158399582,
            0.022749759256839752, -0.01908838376402855,
        ]),
        rtol=RTOL,
        atol=ATOL,
    )


def test_radenet_empty_target_golden():
    outputs = _radenet_outputs(1)
    loss, metrics = radenet_detection_loss(
        outputs=outputs,
        gt_boxes_raw_list=[torch.empty((0, 7))],
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


def _yolox_outputs():
    return {
        "cls_logits": _leaf(
            [1.2, -0.4, 0.1, -1.0, -0.7, 0.8, -0.2, 1.1],
            (1, 2, 2, 2),
        ),
        "objectness_logits": _leaf([0.9, -0.5, 0.3, 0.7], (1, 1, 2, 2)),
        "center_offset": _leaf(
            [0.15, -0.2, 0.1, -0.1, 0.05, 0.2, -0.15, 0.1],
            (1, 2, 2, 2),
        ),
        "center_height": _leaf([-0.2, 0.4, 0.1, -0.3], (1, 1, 2, 2)),
        "size": _leaf(
            [-0.6, -0.4, -0.5, -0.3, -0.7, -0.5,
             -0.4, -0.6, -0.2, 0.1, -0.1, 0.2],
            (1, 3, 2, 2),
        ),
        "yaw": _leaf(
            [0.3, -0.2, 0.4, -0.1, 0.8, 0.6, -0.5, 0.9],
            (1, 2, 2, 2),
        ),
    }


def test_yolox_multi_target_forward_and_gradient_golden():
    outputs = _yolox_outputs()
    loss, metrics = yolox_detection_loss(
        outputs=outputs,
        gt_boxes_list=[torch.tensor([
            [0.075, 0.025, 0.45, 0.28, 0.24, 0.42, 0.56],
            [0.55, 0.45, 0.52, 0.32, 0.27, 0.48, 0.61],
        ])],
        gt_labels_list=[torch.tensor([0, 1])],
        num_classes=2,
        box_loss_weight=4.2,
        obj_loss_weight=0.9,
        cls_loss_weight=1.1,
        l1_loss_weight=0.8,
    )
    _assert_float(loss.item(), 4.094449520111084)
    _assert_metrics(metrics, {
        "total_loss": 4.094449520111084,
        "box_loss": 0.4197852611541748,
        "cls_loss": 1.0555232763290405,
        "obj_loss": 1.0363860130310059,
        "l1_loss": 0.29691094160079956,
        "gwd_loss": 0.4197852611541748,
        "num_center_targets": 2,
        "ignore_pixels": 0,
    })
    expected_weighted = (
        4.2 * metrics["box_loss"]
        + 0.9 * metrics["obj_loss"]
        + 1.1 * metrics["cls_loss"]
        + 0.8 * metrics["l1_loss"]
    )
    _assert_float(loss.item(), expected_weighted)
    loss.backward()
    expected_gradients = {
        "cls_logits": [
            -0.09866093844175339, 0.0, 0.0, 0.14791779220104218,
            0.18249672651290894, 0.0, 0.0, 0.25337857007980347,
        ],
        "objectness_logits": [
            -0.1300727277994156, 0.1698932945728302,
            0.2584991157054901, -0.14931552112102509,
        ],
        "center_offset": [
            0.0, 0.0, 0.0, -0.41964343190193176,
            0.0, 0.0, 0.0, 0.41964343190193176,
        ],
        "center_height": [
            0.09900663793087006, 0.0, 0.0, -0.09778332710266113,
        ],
        "size": [
            -0.16823995113372803, 0.0, 0.0, 0.17905430495738983,
            0.1778150051832199, 0.0, 0.0, 0.103411003947258,
            0.09900663793087006, 0.0, 0.0, 0.09900663048028946,
        ],
        "yaw": [
            -0.07135279476642609, 0.0, 0.0, -0.07565765827894211,
            0.026757298037409782, 0.0, 0.0, -0.00840640626847744,
        ],
    }
    for key, expected in expected_gradients.items():
        torch.testing.assert_close(
            outputs[key].grad.flatten(),
            torch.tensor(expected),
            rtol=RTOL,
            atol=ATOL,
        )


def test_yolox_empty_target_golden():
    outputs = _yolox_outputs()
    loss, metrics = yolox_detection_loss(
        outputs=outputs,
        gt_boxes_list=[torch.empty((0, 7))],
        gt_labels_list=[torch.empty((0,), dtype=torch.long)],
        num_classes=2,
    )
    assert math.isfinite(loss.item())
    _assert_metrics(metrics, {
        "total_loss": 3.67277193069458,
        "box_loss": 0.0,
        "cls_loss": 0.0,
        "obj_loss": 3.67277193069458,
        "l1_loss": 0.0,
        "gwd_loss": 0.0,
        "num_center_targets": 0,
        "ignore_pixels": 0,
    })
    loss.backward()
    torch.testing.assert_close(
        outputs["objectness_logits"].grad.flatten(),
        torch.tensor([
            0.7109494805335999, 0.3775406777858734,
            0.5744425058364868, 0.6681877374649048,
        ]),
        rtol=RTOL,
        atol=ATOL,
    )
    for key in ("cls_logits", "center_offset", "center_height", "size", "yaw"):
        assert outputs[key].grad is None


def test_simota_assignment_order_and_edge_cases_golden():
    pred_boxes = torch.tensor([
        [0.25, 0.25, 0.5, 0.3, 0.3, 0.2, 0.5],
        [0.75, 0.25, 0.5, 0.3, 0.3, 0.2, 0.5],
        [0.25, 0.75, 0.5, 0.3, 0.3, 0.2, 0.5],
        [0.75, 0.75, 0.5, 0.3, 0.3, 0.2, 0.5],
    ])
    cls_logits = torch.tensor([
        [2.0, -1.0], [-0.5, 1.5], [1.0, 0.2], [-1.0, 2.0],
    ])
    objectness = torch.tensor([1.0, 0.8, 0.3, 1.2])
    grid_centers = pred_boxes[:, :2]

    def assign(boxes, labels):
        return simota_assign(
            pred_boxes, cls_logits, objectness, grid_centers,
            boxes, labels, 2, 2, 2,
        )

    empty = assign(torch.empty((0, 7)), torch.empty((0,), dtype=torch.long))
    assert [value.tolist() for value in empty] == [[], [], [], []]

    one = assign(
        torch.tensor([[0.26, 0.24, 0.5, 0.3, 0.3, 0.2, 0.5]]),
        torch.tensor([0]),
    )
    assert [value.tolist() for value in one[:3]] == [[0], [0], [0]]
    torch.testing.assert_close(one[3], torch.tensor([0.876945972442627]))

    multi = assign(
        torch.tensor([
            [0.26, 0.24, 0.5, 0.3, 0.3, 0.2, 0.5],
            [0.74, 0.76, 0.5, 0.3, 0.3, 0.2, 0.5],
        ]),
        torch.tensor([0, 1]),
    )
    assert [value.tolist() for value in multi[:3]] == [[0, 3], [0, 1], [0, 1]]
    torch.testing.assert_close(
        multi[3],
        torch.tensor([0.876945972442627, 0.8769460916519165]),
    )

    more_gt = assign(
        torch.tensor([
            [0.26, 0.24, 0.5, 0.3, 0.3, 0.2, 0.5],
            [0.74, 0.76, 0.5, 0.3, 0.3, 0.2, 0.5],
            [0.72, 0.22, 0.5, 0.3, 0.3, 0.2, 0.5],
            [0.24, 0.74, 0.5, 0.3, 0.3, 0.2, 0.5],
            [0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.5],
        ]),
        torch.tensor([0, 1, 1, 0, 1]),
    )
    assert [value.tolist() for value in more_gt[:3]] == [
        [0, 1, 2, 3], [0, 2, 3, 1], [0, 1, 0, 1],
    ]


def test_simota_deterministic_tie_golden():
    pred_boxes = torch.tensor([
        [0.5, 0.5, 0.5, 0.4, 0.4, 0.2, 0.5],
        [0.5, 0.5, 0.5, 0.4, 0.4, 0.2, 0.5],
    ])
    result = simota_assign(
        pred_boxes=pred_boxes,
        cls_logits=torch.zeros(2, 2),
        objectness_logits=torch.zeros(2),
        grid_centers=torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        gt_boxes=pred_boxes[:1],
        gt_labels=torch.tensor([0]),
        num_classes=2,
        height=1,
        width=2,
        candidate_topk=2,
    )
    assert [value.tolist() for value in result[:3]] == [[0], [0], [0]]
    torch.testing.assert_close(result[3], torch.tensor([0.9999937415122986]))


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
