"""Axis-aligned BEV overlap backend for the KITTI-style evaluator.

The input layout matches ``rotate_iou_gpu_eval``: ``[x, y, width, length,
yaw]``.  Yaw is intentionally ignored.  This backend is therefore not the
official rotated-IoU metric; it exists for explicitly requested, faster
axis-aligned Cartesian evaluation.
"""

import numpy as np


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """Return axis-aligned overlap while preserving the legacy call shape."""
    del device_id
    boxes = np.asarray(boxes)
    query_boxes = np.asarray(query_boxes)
    output_dtype = np.result_type(boxes.dtype, query_boxes.dtype)
    overlaps = np.zeros(
        (int(boxes.shape[0]), int(query_boxes.shape[0])),
        dtype=output_dtype,
    )
    if boxes.shape[0] == 0 or query_boxes.shape[0] == 0:
        return overlaps

    query_width = np.maximum(query_boxes[:, 2], 0.0)
    query_length = np.maximum(query_boxes[:, 3], 0.0)
    query_x_min = query_boxes[:, 0] - query_width * 0.5
    query_x_max = query_boxes[:, 0] + query_width * 0.5
    query_y_min = query_boxes[:, 1] - query_length * 0.5
    query_y_max = query_boxes[:, 1] + query_length * 0.5
    query_area = query_width * query_length

    for box_index, box in enumerate(boxes):
        width = max(float(box[2]), 0.0)
        length = max(float(box[3]), 0.0)
        box_x_min = float(box[0]) - width * 0.5
        box_x_max = float(box[0]) + width * 0.5
        box_y_min = float(box[1]) - length * 0.5
        box_y_max = float(box[1]) + length * 0.5

        intersection_width = np.maximum(
            np.minimum(box_x_max, query_x_max)
            - np.maximum(box_x_min, query_x_min),
            0.0,
        )
        intersection_length = np.maximum(
            np.minimum(box_y_max, query_y_max)
            - np.maximum(box_y_min, query_y_min),
            0.0,
        )
        intersection = intersection_width * intersection_length
        box_area = width * length

        if criterion == -1:
            denominator = box_area + query_area - intersection
        elif criterion == 0:
            denominator = np.full_like(intersection, box_area)
        elif criterion == 1:
            denominator = query_area
        else:
            overlaps[box_index] = intersection
            continue

        overlaps[box_index] = np.divide(
            intersection,
            denominator,
            out=np.zeros_like(intersection),
            where=denominator > 0.0,
        )

    return overlaps
