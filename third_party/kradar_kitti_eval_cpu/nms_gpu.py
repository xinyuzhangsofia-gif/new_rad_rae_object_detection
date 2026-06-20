import math

import numpy as np


def rbbox_to_corners(rbbox):
    angle = float(rbbox[4])
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = float(rbbox[0])
    center_y = float(rbbox[1])
    x_d = float(rbbox[2])
    y_d = float(rbbox[3])

    local = [
        (-x_d / 2.0, -y_d / 2.0),
        (-x_d / 2.0, y_d / 2.0),
        (x_d / 2.0, y_d / 2.0),
        (x_d / 2.0, -y_d / 2.0),
    ]
    return [
        (
            a_cos * x + a_sin * y + center_x,
            -a_sin * x + a_cos * y + center_y,
        )
        for x, y in local
    ]


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def polygon_orientation(points):
    signed_area = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        signed_area += x1 * y2 - x2 * y1
    return 1.0 if signed_area >= 0.0 else -1.0


def inside_clip_edge(point, edge_start, edge_end, orientation):
    cross = (
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    )
    return cross * orientation >= -1e-7


def line_intersection(p1, p2, q1, q2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-12:
        return p2
    px = (
        ((x1 * y2 - y1 * x2) * (x3 - x4))
        - ((x1 - x2) * (x3 * y4 - y3 * x4))
    ) / denominator
    py = (
        ((x1 * y2 - y1 * x2) * (y3 - y4))
        - ((y1 - y2) * (x3 * y4 - y3 * x4))
    ) / denominator
    return px, py


def polygon_clip(subject_polygon, clip_polygon):
    if len(subject_polygon) == 0 or len(clip_polygon) == 0:
        return []

    output = subject_polygon
    orientation = polygon_orientation(clip_polygon)
    for edge_idx in range(len(clip_polygon)):
        edge_start = clip_polygon[edge_idx]
        edge_end = clip_polygon[(edge_idx + 1) % len(clip_polygon)]
        input_polygon = output
        output = []
        if len(input_polygon) == 0:
            break

        previous = input_polygon[-1]
        previous_inside = inside_clip_edge(
            previous,
            edge_start,
            edge_end,
            orientation,
        )
        for current in input_polygon:
            current_inside = inside_clip_edge(
                current,
                edge_start,
                edge_end,
                orientation,
            )
            if current_inside:
                if not previous_inside:
                    output.append(line_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(line_intersection(previous, current, edge_start, edge_end))
            previous = current
            previous_inside = current_inside
    return output


def rotate_intersection(box, query_box):
    polygon = rbbox_to_corners(box)
    query_polygon = rbbox_to_corners(query_box)
    return polygon_area(polygon_clip(polygon, query_polygon))


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    box_dtype = boxes.dtype
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    ious = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float32)

    for box_idx, box in enumerate(boxes):
        area = max(float(box[2] * box[3]), 0.0)
        for query_idx, query_box in enumerate(query_boxes):
            query_area = max(float(query_box[2] * query_box[3]), 0.0)
            intersection = rotate_intersection(box, query_box)
            if criterion == -1:
                denominator = area + query_area - intersection
            elif criterion == 0:
                denominator = area
            elif criterion == 1:
                denominator = query_area
            else:
                ious[box_idx, query_idx] = intersection
                continue
            ious[box_idx, query_idx] = 0.0 if denominator <= 0.0 else intersection / denominator

    return ious.astype(box_dtype)
