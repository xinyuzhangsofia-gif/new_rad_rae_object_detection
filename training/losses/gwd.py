"""Gaussian Wasserstein box conversion and distance loss."""

import torch

def _box_to_gaussian_batch(boxes):
    x, y, _, length, width, _, yaw_sin, yaw_cos = torch.unbind(boxes, dim=-1)
    width_half = width / 2.0
    length_half = length / 2.0
    cos_sq = yaw_cos ** 2
    sin_sq = yaw_sin ** 2
    cos_sin = yaw_cos * yaw_sin

    sigma_11 = width_half * cos_sq + length_half * sin_sq
    sigma_12 = (width_half - length_half) * cos_sin
    sigma_21 = sigma_12
    sigma_22 = width_half * sin_sq + length_half * cos_sq

    sigma = torch.stack(
        [
            torch.stack([sigma_11, sigma_12], dim=-1),
            torch.stack([sigma_21, sigma_22], dim=-1),
        ],
        dim=-2,
    )
    mu = torch.stack([x, y], dim=-1)
    return mu, sigma


def _matrix_sqrt_batch(matrix):
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=1e-12))
    eigen_diag = torch.diag_embed(sqrt_eigenvalues)
    return eigenvectors @ eigen_diag @ eigenvectors.transpose(-1, -2)


def gaussian_wasserstein_distance_batch(pred_boxes, gt_boxes, tau=1.65):
    pred_mu, pred_sigma = _box_to_gaussian_batch(pred_boxes)
    gt_mu, gt_sigma = _box_to_gaussian_batch(gt_boxes)

    first_term = (pred_mu - gt_mu).pow(2).sum(dim=-1)
    pred_sigma_sq = pred_sigma @ pred_sigma
    gt_sigma_sq = gt_sigma @ gt_sigma
    intermediate = (-2.0) * _matrix_sqrt_batch(pred_sigma @ gt_sigma_sq @ pred_sigma)
    matrix_add = pred_sigma_sq + gt_sigma_sq + intermediate
    second_term = torch.diagonal(matrix_add, dim1=-2, dim2=-1).sum(dim=-1)
    gwd = first_term + second_term
    iou_like = 1.0 / (tau + torch.sqrt(torch.clamp(gwd, min=1e-12)))
    return iou_like, 1.0 - iou_like

