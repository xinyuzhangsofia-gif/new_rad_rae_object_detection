"""Dataset classes that pair Cartesian annotations with RAD/RAE tensors.

File I/O and geometry live in :mod:`data.labels` and :mod:`data.geometry`.
This module is responsible only for sample pairing and annotation policy.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from configs.coordinates import BOX_COORDINATE_CARTESIAN, require_cartesian_data
from .coordinates import SCOPE_FULL, crop_rad_rae_to_scope, validate_scope_mode
from .geometry import (
    build_cartesian_box_tensors,
    filter_objects_to_scope,
    prepare_cartesian_objects,
)
from .ignore_overrides import (
    load_object_ignore_override_map,
    validate_object_ignore_overrides,
)
from .labels import load_cartesian_gt


CLASS_NAMES = {0: "Sedan", 1: "Bus or Truck"}
CLASS_TO_IDX = {
    class_name: class_id for class_id, class_name in CLASS_NAMES.items()
}


class KRadarRADRAEDataset(Dataset):
    """Load paired preprocessed RAD and RAE ``.npy`` tensors."""

    def __init__(self, rad_root_dir, sequence, scope_mode=SCOPE_FULL):
        self.scope_mode = validate_scope_mode(scope_mode)
        self.sequence = sequence
        self.sequence_dir = os.path.join(rad_root_dir, str(sequence))
        self.rad_dir = os.path.join(self.sequence_dir, "rad")
        self.rae_dir = os.path.join(self.sequence_dir, "rae")

        rad_files_by_name = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in glob.glob(os.path.join(self.rad_dir, "*.npy"))
        }
        rae_files_by_name = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in glob.glob(os.path.join(self.rae_dir, "*.npy"))
        }
        self.frame_names = sorted(set(rad_files_by_name) & set(rae_files_by_name))
        if not self.frame_names:
            raise ValueError(
                f"No matching rad/rae npy files found in {self.rad_dir} "
                f"and {self.rae_dir}"
            )

        self.rad_files = [rad_files_by_name[name] for name in self.frame_names]
        self.rae_files = [rae_files_by_name[name] for name in self.frame_names]

    def __len__(self):
        return len(self.rad_files)

    def _load_one_dataset_idx(self, dataset_idx):
        if dataset_idx < 0 or dataset_idx >= len(self):
            raise IndexError(
                f"dataset_idx {dataset_idx} out of range for "
                f"{len(self)} radar frames"
            )

        rad_file = self.rad_files[dataset_idx]
        rae_file = self.rae_files[dataset_idx]
        rad = np.load(rad_file)
        rae = np.load(rae_file)
        full_rae_shape = tuple(rae.shape)
        rad, rae = crop_rad_rae_to_scope(
            rad=rad,
            rae=rae,
            scope_mode=self.scope_mode,
        )
        return {
            "rad": rad,
            "rae": rae,
            "full_rae_shape": full_rae_shape,
            "file_idx": dataset_idx,
            "gt_frame_idx": dataset_idx + 1,
            "frame_name": self.frame_names[dataset_idx],
            "rad_file": rad_file,
            "rae_file": rae_file,
        }

    def __getitem__(self, index):
        return self._load_one_dataset_idx(index)


class KRadarGTDetectionDataset(Dataset):
    """Pair RAD/RAE tensors with radar-aligned Cartesian annotations."""

    def __init__(
            self,
            radar_dataset,
            class_to_idx=None,
            sequence=None,
            ignore_unmapped_classes=True,
            ignore_class_names=None,
            gt_object_ignore_override_path=None,
            scope_mode=SCOPE_FULL,
            box_coordinate_mode=BOX_COORDINATE_CARTESIAN,
            cartesian_gt_root=None,
            ignore_object_label_minus_one=False,
            ignore_out_of_scope_gt=True,
            strict_object_ignore_override=False,
        ):
        super().__init__()
        self.radar_dataset = radar_dataset
        self.scope_mode = validate_scope_mode(scope_mode)
        self.box_coordinate_mode = require_cartesian_data(box_coordinate_mode)
        self.sequence = sequence
        if self.sequence is None:
            self.sequence = getattr(radar_dataset, "sequence", None)
        if self.sequence is None:
            raise ValueError("KRadarGTDetectionDataset requires a sequence.")

        self.ignore_object_label_minus_one = bool(ignore_object_label_minus_one)
        self.ignore_out_of_scope_gt = bool(ignore_out_of_scope_gt)
        self.num_invalid_object_labels_ignored = 0
        # Kept because this name is present in existing checkpoints and logs.
        self.num_invalid_cartesian_object_labels_ignored = 0

        key_mode, gt_mapping = load_cartesian_gt(
            self.sequence, cartesian_gt_root
        )
        self.gt_by_file_idx = gt_mapping if key_mode == "file_idx" else None
        self.gt_by_frame_name = gt_mapping if key_mode == "frame_name" else None
        if self.ignore_object_label_minus_one:
            self._remove_invalid_object_labels()

        selected_classes = CLASS_TO_IDX if class_to_idx is None else class_to_idx
        self.class_to_idx = {
            class_name: int(class_id)
            for class_name, class_id in selected_classes.items()
        }
        self.ignore_unmapped_classes = ignore_unmapped_classes
        self.ignore_class_names = set(ignore_class_names or ())

        (
            self.object_ignore_override_map,
            self.object_ignore_override_summary,
        ) = load_object_ignore_override_map(
            override_path=gt_object_ignore_override_path,
            sequence=self.sequence,
            frame_names=getattr(self.radar_dataset, "frame_names", ()),
        )
        if strict_object_ignore_override and gt_object_ignore_override_path is not None:
            validate_object_ignore_overrides(
                override_map=self.object_ignore_override_map,
                summary=self.object_ignore_override_summary,
                class_to_idx=self.class_to_idx,
                sequence=self.sequence,
                objects_for_file_idx=self._raw_objects_for_file_idx,
            )

    def __len__(self):
        return len(self.radar_dataset)

    def _raw_objects_for_file_idx(self, file_idx, frame_name=None):
        if self.gt_by_file_idx is not None:
            return self.gt_by_file_idx.get(int(file_idx), [])
        if frame_name is None:
            frame_name = self.radar_dataset.frame_names[int(file_idx)]
        return self.gt_by_frame_name.get(frame_name, [])

    def _remove_invalid_object_labels(self):
        """Remove GT rows whose object identifier is ``-1`` when requested."""
        mapping = (
            self.gt_by_file_idx
            if self.gt_by_file_idx is not None
            else self.gt_by_frame_name
        )
        if mapping is None:
            return

        removed = 0
        for frame_key, objects in list(mapping.items()):
            kept = [obj for obj in objects if int(obj["object_label"]) != -1]
            removed += len(objects) - len(kept)
            mapping[frame_key] = kept
        self.num_invalid_object_labels_ignored = removed
        self.num_invalid_cartesian_object_labels_ignored = removed

    def _partition_objects(self, objects, override_labels):
        """Separate positive, generic-ignore, and explicit-ignore objects."""
        positive = []
        ignored = []
        explicit_ignored = []
        generic_ignored = []
        for obj in objects:
            class_name = obj["cls"]
            if int(obj["object_label"]) in override_labels:
                ignored.append(obj)
                explicit_ignored.append(obj)
            elif class_name in self.class_to_idx or not self.ignore_unmapped_classes:
                positive.append(obj)
            elif class_name in self.ignore_class_names:
                ignored.append(obj)
                generic_ignored.append(obj)
        return positive, ignored, explicit_ignored, generic_ignored

    def __getitem__(self, index):
        radar_data = self.radar_dataset[index]
        file_idx = radar_data["file_idx"]
        frame_name = radar_data["frame_name"]
        full_rae_shape = radar_data["full_rae_shape"]

        raw_objects = self._raw_objects_for_file_idx(file_idx, frame_name)
        all_objects = prepare_cartesian_objects(raw_objects, full_rae_shape)
        object_groups = self._partition_objects(
            all_objects,
            self.object_ignore_override_map.get(file_idx, set()),
        )
        objects, ignored, explicit_ignored, generic_ignored = object_groups

        if self.ignore_out_of_scope_gt:
            filtered_groups = tuple(
                filter_objects_to_scope(group, self.scope_mode)
                for group in object_groups
            )
        else:
            filtered_groups = tuple(list(group) for group in object_groups)
        objects_in_fov, ignored_in_fov, explicit_in_fov, generic_in_fov = (
            filtered_groups
        )

        gt_boxes_raw, gt_metric_boxes = build_cartesian_box_tensors(
            objects_in_fov, self.scope_mode, full_rae_shape
        )
        gt_ignore_boxes_raw, gt_ignore_metric_boxes = (
            build_cartesian_box_tensors(
                ignored_in_fov, self.scope_mode, full_rae_shape
            )
        )
        evaluable_explicit = [
            obj for obj in explicit_in_fov if obj["cls"] in self.class_to_idx
        ]
        (
            gt_override_ignore_boxes_raw,
            gt_override_ignore_metric_boxes,
        ) = build_cartesian_box_tensors(
            evaluable_explicit, self.scope_mode, full_rae_shape
        )

        gt_labels = torch.tensor(
            [self.class_to_idx[obj["cls"]] for obj in objects_in_fov],
            dtype=torch.long,
        )
        gt_override_ignore_labels = torch.tensor(
            [self.class_to_idx[obj["cls"]] for obj in evaluable_explicit],
            dtype=torch.long,
        )

        return {
            "rad": torch.from_numpy(radar_data["rad"]).float(),
            "rae": torch.from_numpy(radar_data["rae"]).float(),
            "gt_boxes_raw": gt_boxes_raw,
            "gt_metric_boxes": gt_metric_boxes,
            "gt_ignore_boxes_raw": gt_ignore_boxes_raw,
            "gt_ignore_metric_boxes": gt_ignore_metric_boxes,
            "gt_ignore_class_names": tuple(
                str(obj["cls"]) for obj in ignored_in_fov
            ),
            "gt_override_ignore_boxes_raw": gt_override_ignore_boxes_raw,
            "gt_override_ignore_metric_boxes": gt_override_ignore_metric_boxes,
            "gt_override_ignore_labels": gt_override_ignore_labels,
            "gt_labels": gt_labels,
            "gt_frame_idx": radar_data["gt_frame_idx"],
            "file_idx": file_idx,
            "frame_name": frame_name,
            "sequence": self.sequence,
            "sequence_id": f"{self.sequence}_{file_idx}",
            "rad_file": radar_data["rad_file"],
            "rae_file": radar_data["rae_file"],
            "scope_mode": self.scope_mode,
            "box_coordinate_mode": self.box_coordinate_mode,
            "full_rae_shape": full_rae_shape,
            "num_gt_before_fov": len(objects),
            "num_gt_after_fov": len(objects_in_fov),
            "num_ignore_before_fov": len(ignored),
            "num_ignore_after_fov": len(ignored_in_fov),
            "num_override_ignored": len(explicit_ignored),
            "num_override_ignored_after_fov": len(explicit_in_fov),
            "num_generic_ignored_after_fov": len(generic_in_fov),
            "num_invalid_cartesian_object_labels_ignored": int(
                self.num_invalid_cartesian_object_labels_ignored
            ),
            "num_invalid_object_labels_ignored": int(
                self.num_invalid_object_labels_ignored
            ),
            "ignore_object_label_minus_one": self.ignore_object_label_minus_one,
            "ignore_out_of_scope_gt": self.ignore_out_of_scope_gt,
        }


class KRadarMultiSequenceGTDetectionDataset(Dataset):
    """Present multiple per-sequence detection datasets as one dataset."""

    def __init__(self, sequence_datasets):
        super().__init__()
        if not sequence_datasets:
            raise ValueError("sequence_datasets must not be empty")
        self.sequence_datasets = list(sequence_datasets)
        self.cumulative_sizes = []
        total = 0
        for dataset in self.sequence_datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _resolve_index(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"index {index} out of range for {len(self)} samples")

        dataset_idx = 0
        while index >= self.cumulative_sizes[dataset_idx]:
            dataset_idx += 1
        previous_size = (
            0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        )
        return dataset_idx, index - previous_size

    def __getitem__(self, index):
        dataset_idx, sample_idx = self._resolve_index(index)
        return self.sequence_datasets[dataset_idx][sample_idx]

    def get_sequence_ranges(self):
        ranges = []
        start = 0
        for dataset, end in zip(self.sequence_datasets, self.cumulative_sizes):
            ranges.append({
                "sequence": getattr(dataset, "sequence", None),
                "start": start,
                "end": end,
                "length": end - start,
            })
            start = end
        return ranges
