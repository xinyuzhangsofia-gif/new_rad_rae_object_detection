from torch.utils.data import Dataset
import math
import torch
import numpy as np
from scipy.io import loadmat
import glob
import os
from cfg_model import (
    AZIMUTH_AXIS,
    cartesian_to_rae,
    ELEVATION_AXIS,
    RANGE_AXIS,
    SCOPE_FULL,
    crop_rad_rae_to_scope,
    global_rae_boxes_to_local_scope,
    is_rae_center_in_gt_scope,
    normalize_rae_boxes_for_scope,
    validate_scope_mode,
)
from coordinate_modes import (
    BOX_COORDINATE_CARTESIAN,
    BOX_COORDINATE_POLAR,
    validate_box_coordinate_mode,
)
from object_ignore_overrides import load_object_ignore_override_map
from training_utils.radenet_utils import (
    metric_boxes_to_raw_local_rae,
    raw_local_rae_boxes_to_metric_boxes,
)
from zxy_label_utils import (
    read_cartesian_gt_txt,
    read_gt_txt,
    read_kradar_revised_label_dir,
)


CLASS_NAMES = {
    0: "Sedan",
    1: "Bus or Truck",
}

CLASS_TO_IDX = {
    class_name: class_id
    for class_id, class_name in CLASS_NAMES.items()
}


class KRadarDataset(Dataset):  #used for past sensor visualization
    def __init__(self, radar_folder):
        self.files = sorted(glob.glob(os.path.join(radar_folder, "*.mat")))

        self.idx_to_file ={}
        for f in self.files:
            fname = os.path.basename(f)
            tesseract_idx = fname.split('_')[1].split('.')[0]
            self.idx_to_file[tesseract_idx] = f

    def __len__(self):
        return len(self.files)
    
    def _drea2rea(self, drea: np.ndarray) -> np.ndarray:
        return np.mean(drea, axis=0)  

    def _drea2rad(self, drea: np.ndarray) -> np.ndarray:
        return np.mean(drea, axis=2)  
    
    def _drea2aed(self, drea: np.ndarray) -> np.ndarray:
        return np.mean(drea, axis=1)
    
    def _rea2ra(self,rea:np.ndarray):
        return np.sum(rea,axis=1)
    
    def _rea2re(self,rea:np.array):
        return np.sum(rea, axis=2)

    
    def _load_one_file(self, file_path):
        drea = loadmat(file_path)['arrDREA']  
        drea = np.asarray(drea)
        rea = self._drea2rea(drea)

        return {
            "rea": rea,
            "rad": self._drea2rad(drea),
            "aed": self._drea2aed(drea),
            "ra_map":self._rea2ra(rea),
            "re_map":self._rea2re(rea)
        }
    
    def __getitem__(self, idx):
        return self._load_one_file(self.files[idx])
    
    def get_by_tesseract_idx(self,tesseract_idx):
        if tesseract_idx not in self.idx_to_file:
            raise KeyError(f"tesseract_idx {tesseract_idx} not found in dataset")
        file_path=self.idx_to_file[tesseract_idx]
        return self._load_one_file(file_path)


class KRadarGTDetectionDataset(Dataset):
    def __init__(
            self,
            radar_dataset,
            gt_txt_path=None,
            class_to_idx=None,
            sequence=None,
            ignore_unmapped_classes=True,
            ignore_class_names=None,
            gt_object_ignore_override_path=None,
            scope_mode=SCOPE_FULL,
            box_coordinate_mode=BOX_COORDINATE_POLAR,
            cartesian_gt_root=None,
            ignore_object_label_minus_one=False,
            ignore_out_of_scope_gt=True,
            strict_object_ignore_override=False,
            ):
        super().__init__()
        self.radar_dataset = radar_dataset
        self.scope_mode = validate_scope_mode(scope_mode)
        self.box_coordinate_mode = validate_box_coordinate_mode(
            box_coordinate_mode
        )
        self.sequence = sequence
        self.ignore_object_label_minus_one = bool(
            ignore_object_label_minus_one
        )
        self.ignore_out_of_scope_gt = bool(ignore_out_of_scope_gt)
        if self.sequence is None:
            self.sequence = getattr(radar_dataset, "sequence", None)
        if self.sequence is None:
            raise ValueError("KRadarGTDetectionDataset requires a sequence.")

        self.gt_by_file_idx = None
        self.gt_by_frame_name = None
        self.num_invalid_object_labels_ignored = 0
        # Keep the old metadata name for compatibility with existing logs.
        self.num_invalid_cartesian_object_labels_ignored = 0
        if self.box_coordinate_mode == BOX_COORDINATE_POLAR:
            if gt_txt_path in (None, ""):
                raise ValueError("Polar mode requires gt_txt_path.")
            self.gt_by_file_idx = read_gt_txt(gt_txt_path)
        else:
            if cartesian_gt_root in (None, ""):
                raise ValueError(
                    "Cartesian mode requires cartesian_gt_root."
                )
            cartesian_gt_txt_path = os.path.join(
                str(cartesian_gt_root),
                str(int(self.sequence)),
                "gt",
                "gt.txt",
            )
            if os.path.isfile(cartesian_gt_txt_path):
                # Flat Cartesian GT has the same one-based radar-frame
                # indexing convention as Polar gt.txt.
                self.gt_by_file_idx = read_cartesian_gt_txt(
                    cartesian_gt_txt_path
                )
            else:
                # Backward-compatible fallback for the official per-frame
                # revised-label root.
                self.gt_by_frame_name = read_kradar_revised_label_dir(
                    label_root=cartesian_gt_root,
                    sequence=self.sequence,
                    radar_visibility_tokens=("R", "LR"),
                )

        if self.ignore_object_label_minus_one:
            self._remove_invalid_object_labels()

        self.class_to_idx = None
        if class_to_idx is not None:
            self.class_to_idx = {
                class_name: int(class_id)
                for class_name, class_id in class_to_idx.items()
            }
        self.ignore_unmapped_classes = ignore_unmapped_classes
        self.ignore_class_names = set(ignore_class_names or [])
        if self.class_to_idx is None:
            self.class_to_idx = CLASS_TO_IDX.copy()
        (
            self.object_ignore_override_map,
            self.object_ignore_override_summary,
        ) = load_object_ignore_override_map(
            override_path=gt_object_ignore_override_path,
            sequence=self.sequence,
            frame_names=getattr(self.radar_dataset, "frame_names", ()),
        )
        if strict_object_ignore_override and gt_object_ignore_override_path is not None:
            self._validate_object_ignore_overrides()

    def __len__(self):
        return len(self.radar_dataset)

    def _validate_object_ignore_overrides(self):
        missing_frames = tuple(
            self.object_ignore_override_summary.get("missing_frame_names", ())
        )
        if missing_frames:
            raise ValueError(
                "Evaluation object-ignore override contains frames absent from "
                f"sequence {self.sequence}: {missing_frames[:10]}"
            )

        validated = 0
        frame_names = tuple(getattr(self.radar_dataset, "frame_names", ()))
        for file_idx, requested_labels in self.object_ignore_override_map.items():
            if self.gt_by_file_idx is not None:
                frame_objects = self.gt_by_file_idx.get(int(file_idx), [])
            else:
                frame_name = frame_names[int(file_idx)]
                frame_objects = self.gt_by_frame_name.get(frame_name, [])
            by_label = {}
            for obj in frame_objects:
                by_label.setdefault(int(obj["object_label"]), []).append(obj)
            for object_label in requested_labels:
                matches = by_label.get(int(object_label), [])
                if len(matches) != 1:
                    raise ValueError(
                        "Evaluation object-ignore override must identify exactly "
                        "one existing GT object: "
                        f"sequence={self.sequence}, file_idx={file_idx}, "
                        f"object_label={object_label}, matches={len(matches)}"
                    )
                class_name = str(matches[0]["cls"])
                if class_name not in self.class_to_idx:
                    raise ValueError(
                        "Evaluation object-ignore override may contain only "
                        "evaluable target classes: "
                        f"sequence={self.sequence}, file_idx={file_idx}, "
                        f"object_label={object_label}, class={class_name!r}"
                    )
                validated += 1
        expected = int(
            self.object_ignore_override_summary.get("object_override_count", 0)
        )
        if validated != expected:
            raise ValueError(
                "Evaluation object-ignore override validation count mismatch: "
                f"expected={expected}, validated={validated}."
            )
        self.object_ignore_override_summary["strict_validation"] = True
        self.object_ignore_override_summary["validated_object_count"] = validated

    def __getitem__(self, index):
        radar_data = self.radar_dataset[index]
        file_idx = radar_data["file_idx"]
        gt_frame_idx = radar_data["gt_frame_idx"]
        frame_name = radar_data["frame_name"]
        if self.box_coordinate_mode == BOX_COORDINATE_POLAR:
            all_objects = self.gt_by_file_idx.get(file_idx, [])
        elif self.gt_by_file_idx is not None:
            all_objects = self._prepare_cartesian_objects(
                objects=self.gt_by_file_idx.get(file_idx, []),
                full_rae_shape=radar_data["full_rae_shape"],
            )
        else:
            all_objects = self._prepare_cartesian_objects(
                objects=self.gt_by_frame_name.get(frame_name, []),
                full_rae_shape=radar_data["full_rae_shape"],
            )
        override_ignore_object_labels = self.object_ignore_override_map.get(
            file_idx,
            set(),
        )
        objects = []
        ignore_objects = []
        override_ignore_objects = []
        generic_ignore_objects = []
        num_override_ignored = 0
        for obj in all_objects:
            cls = obj["cls"]
            is_override_ignored = (
                int(obj["object_label"]) in override_ignore_object_labels
            )
            if is_override_ignored:
                ignore_objects.append(obj)
                override_ignore_objects.append(obj)
                num_override_ignored += 1
                continue

            if cls in self.class_to_idx or not self.ignore_unmapped_classes:
                objects.append(obj)
                continue

            if cls in self.ignore_class_names and cls not in self.class_to_idx:
                ignore_objects.append(obj)
                generic_ignore_objects.append(obj)

        rad = torch.from_numpy(radar_data["rad"]).float()
        rae = torch.from_numpy(radar_data["rae"]).float()
        full_rae_shape = radar_data["full_rae_shape"]
        if self.ignore_out_of_scope_gt:
            objects_in_fov = [
                obj for obj in objects
                if self._object_overlaps_rae_fov(obj, full_rae_shape)
                and self._object_center_in_scope(obj)
            ]
            ignore_objects_in_fov = [
                obj for obj in ignore_objects
                if self._object_overlaps_rae_fov(obj, full_rae_shape)
                and self._object_center_in_scope(obj)
            ]
            override_ignore_objects_in_fov = [
                obj for obj in override_ignore_objects
                if self._object_overlaps_rae_fov(obj, full_rae_shape)
                and self._object_center_in_scope(obj)
            ]
            generic_ignore_objects_in_fov = [
                obj for obj in generic_ignore_objects
                if self._object_overlaps_rae_fov(obj, full_rae_shape)
                and self._object_center_in_scope(obj)
            ]
        else:
            # Keep all parsed GT rows, including boxes outside the radar RAE
            # tensor or the selected narrow scope.  The target builder may
            # clamp their heatmap location to the tensor boundary.
            objects_in_fov = list(objects)
            ignore_objects_in_fov = list(ignore_objects)
            override_ignore_objects_in_fov = list(override_ignore_objects)
            generic_ignore_objects_in_fov = list(generic_ignore_objects)

        gt_boxes, gt_boxes_raw, gt_metric_boxes = self._build_box_tensors(
            objects_in_fov,
            full_rae_shape,
        )
        (
            gt_ignore_boxes,
            gt_ignore_boxes_raw,
            gt_ignore_metric_boxes,
        ) = self._build_box_tensors(
            ignore_objects_in_fov,
            full_rae_shape,
        )
        # Keep explicit control-mask objects independently addressable. The
        # legacy gt_ignore_* tensors above intentionally remain the union so
        # existing training loss masking is unchanged.
        override_evaluable_objects_in_fov = [
            obj for obj in override_ignore_objects_in_fov
            if obj["cls"] in self.class_to_idx
        ]
        (
            gt_override_ignore_boxes,
            gt_override_ignore_boxes_raw,
            gt_override_ignore_metric_boxes,
        ) = self._build_box_tensors(
            override_evaluable_objects_in_fov,
            full_rae_shape,
        )
        if override_evaluable_objects_in_fov:
            gt_override_ignore_labels = torch.tensor(
                [
                    self.class_to_idx[obj["cls"]]
                    for obj in override_evaluable_objects_in_fov
                ],
                dtype=torch.long,
            )
        else:
            gt_override_ignore_labels = torch.zeros((0,), dtype=torch.long)

        if len(objects_in_fov) > 0:
            gt_labels = torch.tensor(
                [self.class_to_idx[obj["cls"]] for obj in objects_in_fov],
                dtype=torch.long
            )
        else:
            gt_labels = torch.zeros((0,), dtype=torch.long)
        gt_ignore_class_names = tuple(
            str(obj["cls"])
            for obj in ignore_objects_in_fov
        )

        return {
            "rad": rad,
            "rae": rae,
            "gt_boxes": gt_boxes,
            "gt_boxes_raw": gt_boxes_raw,
            "gt_metric_boxes": gt_metric_boxes,
            "gt_ignore_boxes": gt_ignore_boxes,
            "gt_ignore_boxes_raw": gt_ignore_boxes_raw,
            "gt_ignore_metric_boxes": gt_ignore_metric_boxes,
            "gt_ignore_class_names": gt_ignore_class_names,
            "gt_override_ignore_boxes": gt_override_ignore_boxes,
            "gt_override_ignore_boxes_raw": gt_override_ignore_boxes_raw,
            "gt_override_ignore_metric_boxes": gt_override_ignore_metric_boxes,
            "gt_override_ignore_labels": gt_override_ignore_labels,
            "gt_labels": gt_labels,
            "gt_frame_idx": gt_frame_idx,
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
            "num_ignore_before_fov": len(ignore_objects),
            "num_ignore_after_fov": len(ignore_objects_in_fov),
            "num_override_ignored": int(num_override_ignored),
            "num_override_ignored_after_fov": int(
                len(override_ignore_objects_in_fov)
            ),
            "num_generic_ignored_after_fov": int(
                len(generic_ignore_objects_in_fov)
            ),
            "num_invalid_cartesian_object_labels_ignored": int(
                self.num_invalid_cartesian_object_labels_ignored
            ),
            "num_invalid_object_labels_ignored": int(
                self.num_invalid_object_labels_ignored
            ),
            "ignore_object_label_minus_one": bool(
                self.ignore_object_label_minus_one
            ),
            "ignore_out_of_scope_gt": bool(self.ignore_out_of_scope_gt),
        }

    def _remove_invalid_object_labels(self):
        """Optionally remove GT rows whose object index is ``-1``.

        This applies to both Polar and Cartesian GT. The default configuration
        keeps these rows; the caller must explicitly enable the filter.
        """

        if self.gt_by_file_idx is not None:
            mapping = self.gt_by_file_idx
        elif self.gt_by_frame_name is not None:
            mapping = self.gt_by_frame_name
        else:
            return

        removed = 0
        for frame_key, objects in list(mapping.items()):
            kept_objects = []
            for obj in objects:
                if int(obj["object_label"]) == -1:
                    removed += 1
                else:
                    kept_objects.append(obj)
            mapping[frame_key] = kept_objects

        self.num_invalid_object_labels_ignored = removed
        self.num_invalid_cartesian_object_labels_ignored = removed

    def _remove_invalid_cartesian_object_labels(self):
        """Backward-compatible alias for older callers."""
        self._remove_invalid_object_labels()

    def _build_box_tensors(self, objects, full_rae_shape):
        if len(objects) == 0:
            empty = torch.zeros((0, 7), dtype=torch.float32)
            return empty, empty, empty

        boxes_global = torch.stack([obj["box_rae"] for obj in objects], dim=0)
        boxes_raw = global_rae_boxes_to_local_scope(
            boxes=boxes_global,
            scope_mode=self.scope_mode,
            rae_shape=full_rae_shape,
        )
        boxes = self._normalize_boxes_rae(boxes_global, full_rae_shape)
        if self.box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
            metric_boxes = torch.stack(
                [obj["box_metric"] for obj in objects],
                dim=0,
            ).to(torch.float32)
        else:
            metric_boxes_with_yaw_vector = raw_local_rae_boxes_to_metric_boxes(
                raw_boxes=boxes_raw,
                scope_mode=self.scope_mode,
                full_rae_shape=full_rae_shape,
            )
            metric_boxes = torch.cat(
                [
                    metric_boxes_with_yaw_vector[:, :6],
                    torch.atan2(
                        metric_boxes_with_yaw_vector[:, 6:7],
                        metric_boxes_with_yaw_vector[:, 7:8],
                    ),
                ],
                dim=-1,
            )
        return boxes, boxes_raw, metric_boxes

    def _prepare_cartesian_objects(self, objects, full_rae_shape):
        if len(objects) == 0:
            return []

        metric_boxes = torch.stack(
            [obj["box_metric"] for obj in objects],
            dim=0,
        ).to(torch.float32)
        global_raw_boxes = metric_boxes_to_raw_local_rae(
            metric_boxes=metric_boxes,
            scope_mode=SCOPE_FULL,
            full_rae_shape=full_rae_shape,
            use_planar_center_range=True,
        )

        prepared = []
        for obj, box_rae in zip(objects, global_raw_boxes):
            yaw_rad = float(box_rae[6].item())
            prepared_obj = dict(obj)
            prepared_obj["box_rae"] = box_rae
            prepared_obj["raw"] = {
                "r_idx": float(box_rae[0].item()),
                "a_idx": float(box_rae[1].item()),
                "e_idx": float(box_rae[2].item()),
                "r_width": float(box_rae[3].item()),
                "a_width": float(box_rae[4].item()),
                "e_width": float(box_rae[5].item()),
                "yaw": math.degrees(yaw_rad),
                "yaw_rad": yaw_rad,
            }
            prepared.append(prepared_obj)
        return prepared

    def _object_overlaps_rae_fov(self, obj, rae_shape):
        """Return whether any part of a bbox is inside the full RAE tensor.

        The old implementation checked only the bbox center. A bbox whose
        center is outside but whose physical extent enters the radar view is
        still retained; only a completely out-of-view bbox is removed.
        """
        if self.box_coordinate_mode == BOX_COORDINATE_CARTESIAN:
            return self._cartesian_box_overlaps_rae_fov(obj)

        r_size, a_size, e_size = rae_shape
        raw = obj["raw"]
        r_half = abs(float(raw["r_width"])) / 2.0
        a_half = abs(float(raw["a_width"])) / 2.0
        e_half = abs(float(raw["e_width"])) / 2.0
        return (
            raw["r_idx"] + r_half >= 0
            and raw["r_idx"] - r_half <= r_size - 1
            and raw["a_idx"] + a_half >= 0
            and raw["a_idx"] - a_half <= a_size - 1
            and raw["e_idx"] + e_half >= 0
            and raw["e_idx"] - e_half <= e_size - 1
        )

    def _cartesian_box_overlaps_rae_fov(self, obj):
        metric_box = obj.get("box_metric")
        if metric_box is None:
            return False

        x, y, z, length, width, height, yaw = [
            float(value)
            for value in metric_box.detach().cpu().tolist()
        ]
        half_length = abs(length) / 2.0
        half_width = abs(width) / 2.0
        half_height = abs(height) / 2.0
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        corners = []
        for sign_x in (-1.0, 1.0):
            for sign_y in (-1.0, 1.0):
                for sign_z in (-1.0, 1.0):
                    local_x = sign_x * half_length
                    local_y = sign_y * half_width
                    corner_x = x + local_x * cos_yaw - local_y * sin_yaw
                    corner_y = y + local_x * sin_yaw + local_y * cos_yaw
                    corner_z = z + sign_z * half_height
                    corners.append((corner_x, corner_y, corner_z))

        rae = [cartesian_to_rae(*corner) for corner in corners]
        r_values = [value[0] for value in rae]
        e_values = [value[2] for value in rae]

        # Unwrap azimuth around the box-center angle so a box near +/-180
        # degrees does not falsely appear to cross the forward-facing FOV.
        center_azimuth = cartesian_to_rae(x, y, z)[1]
        a_values = []
        for _, azimuth, _ in rae:
            delta = (azimuth - center_azimuth + 180.0) % 360.0 - 180.0
            a_values.append(center_azimuth + delta)

        return (
            max(r_values) >= RANGE_AXIS.minimum
            and min(r_values) <= RANGE_AXIS.maximum
            and max(a_values) >= AZIMUTH_AXIS.minimum
            and min(a_values) <= AZIMUTH_AXIS.maximum
            and max(e_values) >= ELEVATION_AXIS.minimum
            and min(e_values) <= ELEVATION_AXIS.maximum
        )

    def _object_center_in_rae_fov(self, obj, rae_shape):
        """Backward-compatible center-only helper."""
        r_size, a_size, e_size = rae_shape
        raw = obj["raw"]
        return (
            0 <= raw["r_idx"] < r_size
            and 0 <= raw["a_idx"] < a_size
            and 0 <= raw["e_idx"] < e_size
        )

    def _object_center_in_scope(self, obj):
        if self.scope_mode == SCOPE_FULL:
            return True

        raw = obj["raw"]
        return is_rae_center_in_gt_scope(
            r_idx=raw["r_idx"],
            a_idx=raw["a_idx"],
            e_idx=raw["e_idx"],
        )

    def _normalize_boxes_rae(self, boxes, rae_shape):
        return normalize_rae_boxes_for_scope(
            boxes=boxes,
            scope_mode=self.scope_mode,
            rae_shape=rae_shape,
        )

class KRadarMultiSequenceGTDetectionDataset(Dataset):
    def __init__(self, sequence_datasets):
        super().__init__()
        if len(sequence_datasets) == 0:
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

        previous_size = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        sample_idx = index - previous_size
        return dataset_idx, sample_idx

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


def detection_collate(batch):
    return {
        "rad": torch.stack([item["rad"] for item in batch], dim=0),
        "rae": torch.stack([item["rae"] for item in batch], dim=0),
        "gt_boxes": [item["gt_boxes"] for item in batch],
        "gt_boxes_raw": [item["gt_boxes_raw"] for item in batch],
        "gt_metric_boxes": [item["gt_metric_boxes"] for item in batch],
        "gt_ignore_boxes": [item["gt_ignore_boxes"] for item in batch],
        "gt_ignore_boxes_raw": [item["gt_ignore_boxes_raw"] for item in batch],
        "gt_ignore_metric_boxes": [
            item["gt_ignore_metric_boxes"] for item in batch
        ],
        "gt_ignore_class_names": [item["gt_ignore_class_names"] for item in batch],
        "gt_override_ignore_boxes": [
            item["gt_override_ignore_boxes"] for item in batch
        ],
        "gt_override_ignore_boxes_raw": [
            item["gt_override_ignore_boxes_raw"] for item in batch
        ],
        "gt_override_ignore_metric_boxes": [
            item["gt_override_ignore_metric_boxes"] for item in batch
        ],
        "gt_override_ignore_labels": [
            item["gt_override_ignore_labels"] for item in batch
        ],
        "gt_labels": [item["gt_labels"] for item in batch],
        "gt_frame_idx": [item["gt_frame_idx"] for item in batch],
        "file_idx": [item["file_idx"] for item in batch],
        "frame_name": [item["frame_name"] for item in batch],
        "sequence": [item["sequence"] for item in batch],
        "sequence_id": [item["sequence_id"] for item in batch],
        "rad_file": [item["rad_file"] for item in batch],
        "rae_file": [item["rae_file"] for item in batch],
        "scope_mode": [item["scope_mode"] for item in batch],
        "box_coordinate_mode": [
            item["box_coordinate_mode"] for item in batch
        ],
        "full_rae_shape": [item["full_rae_shape"] for item in batch],
        "num_gt_before_fov": [item["num_gt_before_fov"] for item in batch],
        "num_gt_after_fov": [item["num_gt_after_fov"] for item in batch],
        "num_ignore_before_fov": [item["num_ignore_before_fov"] for item in batch],
        "num_ignore_after_fov": [item["num_ignore_after_fov"] for item in batch],
        "num_override_ignored": [item["num_override_ignored"] for item in batch],
        "num_override_ignored_after_fov": [
            item["num_override_ignored_after_fov"] for item in batch
        ],
    }


class KRadarRADRAEDataset(Dataset):
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

        shared_names = sorted(set(rad_files_by_name) & set(rae_files_by_name))
        if len(shared_names) == 0:
            raise ValueError(
                f"No matching rad/rae npy files found in {self.rad_dir} and {self.rae_dir}"
            )

        self.frame_names = shared_names
        self.rad_files = [rad_files_by_name[name] for name in self.frame_names]
        self.rae_files = [rae_files_by_name[name] for name in self.frame_names]

    def __len__(self):
        return len(self.rad_files)

    def _load_one_dataset_idx(self, dataset_idx):
        if dataset_idx < 0 or dataset_idx >= len(self):
            raise IndexError(
                f"dataset_idx {dataset_idx} out of range for {len(self)} radar frames"
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

    def __getitem__(self, idx):
        return self._load_one_dataset_idx(idx)
