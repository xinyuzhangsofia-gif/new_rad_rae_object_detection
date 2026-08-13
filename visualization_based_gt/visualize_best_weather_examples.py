"""Save one epoch-15 Camera + Radar example for each weather condition."""

from dataclasses import dataclass
import gc
from pathlib import Path

import cv2

import path_setup
from checkpoint_predictor import build_checkpoint_predictor
from info_label_reader import read_info_label
from radar_npy_reader import build_current_radar_dataset, get_current_radar_axes
from sensor_transformation import (
    load_lidar2radar_calib,
    transform_radar_boxes_to_lidar,
)
from visualization import (
    combine_camera_radar_frames,
    get_camera_frame,
    get_radar_frame,
)
from visualization_cfg import DataConfig, VISUALIZATION_DIR
from visualization_utils import get_label_dir, get_visualization_camera_dir
from zxy_data_path import (
    get_camera_calib_path,
    get_label_files,
)


PROJECT_ROOT = VISUALIZATION_DIR.parent
OUTPUT_DIR = VISUALIZATION_DIR / "generated" / "best_weather_epoch15"
MIN_GT_BOXES = 3
MIN_PREDICTION_BOXES = 1
MAX_CANDIDATES_PER_WEATHER = 30


@dataclass(frozen=True)
class BestExperiment:
    weather: str
    group: str
    seed: int
    target_sequences: tuple[int, ...]
    test_sequences: tuple[int, ...]
    bev_target_ap: float
    ap_3d_target: float
    checkpoint_path: Path


# The experiment tables identify the winning group. These paths are the
# target-domain epoch-15 run belonging to that group.
EPOCH15_CHECKPOINT_BY_WEATHER_GROUP = {
    ("heavy_snow", "group7"): (
        PROJECT_ROOT
        / "checkpoints/heavy_snow"
        / "0806_train_seq9_12_47_54_57-58_test_seq55-56"
        / "0807_epoch_015.pth"
    ),
    ("light_snow", "group4"): (
        PROJECT_ROOT
        / "checkpoints/light_snow"
        / "0802_train_seq9_12_first_42_43_49_test_seq48"
        / "0802_epoch_015.pth"
    ),
    ("overcast", "group5"): (
        PROJECT_ROOT
        / "checkpoints/overcast"
        / "0728_train_seq9_13_test_seq22"
        / "0728_epoch_015.pth"
    ),
    ("rain", "group2"): (
        PROJECT_ROOT
        / "checkpoints/rain"
        / "0804_train_seq9_23_25_test_seq24"
        / "0804_epoch_015.pth"
    ),
    ("sleet", "group4"): (
        PROJECT_ROOT
        / "checkpoints/sleet"
        / "0805_train_seq9_12_first_50_51_52_test_seq53"
        / "0805_epoch_015.pth"
    ),
}


def _parse_test_sequences(value):
    return tuple(int(part) for part in value.split(","))


def find_best_experiment(weather):
    """Return the row with maximum target-domain BEV AP."""
    experiment_path = PROJECT_ROOT / "experiments" / f"{weather}_experiments.txt"
    candidates = []
    for line in experiment_path.read_text().splitlines():
        fields = line.split()
        if (
            not fields
            or not fields[0].startswith("group")
            or not fields[0][len("group"):].isdigit()
            or len(fields) < 10
        ):
            continue
        if fields[8] == "-" or fields[9] == "-":
            continue
        candidates.append(
            {
                "group": fields[0],
                "seed": int(fields[1]),
                "target_sequences": _parse_test_sequences(fields[4]),
                "test_sequences": _parse_test_sequences(fields[5]),
                "bev_target_ap": float(fields[8]),
                "ap_3d_target": float(fields[9]),
            }
        )
    if not candidates:
        raise ValueError(f"No completed AP rows found in {experiment_path}")

    best = max(
        candidates,
        key=lambda row: (row["bev_target_ap"], row["ap_3d_target"]),
    )
    checkpoint_path = EPOCH15_CHECKPOINT_BY_WEATHER_GROUP.get(
        (weather, best["group"])
    )
    if checkpoint_path is None:
        raise KeyError(
            f"No epoch-15 checkpoint mapped for {weather} {best['group']}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    return BestExperiment(weather=weather, checkpoint_path=checkpoint_path, **best)


def _frame_candidates(sequences):
    """Return data-complete frames sorted by descending GT count."""
    candidates = []
    datasets = {}
    for sequence in sequences:
        cfg = DataConfig(sequence=sequence)
        label_dir = Path(get_label_dir(cfg))
        label_files = get_label_files(str(label_dir))
        camera_dir = get_visualization_camera_dir(cfg)
        dataset = build_current_radar_dataset(cfg)
        datasets[sequence] = dataset
        available_frame_names = set(dataset.frame_names)
        available_camera_indices = {
            path.stem.removeprefix("cam-front_")
            for path in Path(camera_dir).iterdir()
            if path.is_file() and path.name.startswith("cam-front_")
        }

        for frame_idx, label_filename in enumerate(label_files):
            label_path = label_dir / label_filename
            frame_info = read_info_label(label_path)
            frame_name = str(frame_info["tesseract_idx"])
            if frame_name not in available_frame_names:
                continue
            if frame_info["cam_front_idx"] not in available_camera_indices:
                continue
            candidates.append(
                (
                    len(frame_info["objects"]),
                    sequence,
                    frame_idx,
                    frame_name,
                    label_filename,
                )
            )

    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    return candidates, datasets


def _render_selected_frame(
    experiment,
    cfg,
    dataset,
    label_files,
    frame_idx,
    radar_data,
    prediction,
):
    rotation, translation = load_lidar2radar_calib(cfg.lidar2radar_calib_path)
    prediction_lidar_boxes = transform_radar_boxes_to_lidar(
        prediction["radar_boxes"],
        rotation,
        translation,
    )
    arr_range, arr_azimuth_deg, _ = get_current_radar_axes()
    label_dir = get_label_dir(cfg)
    camera_frame = get_camera_frame(
        label_dir=label_dir,
        label_files=label_files,
        camera_dir=get_visualization_camera_dir(cfg),
        path_calib=get_camera_calib_path(cfg),
        frame_idx=frame_idx,
        show_texts=cfg.show_texts,
        show_gt_texts=cfg.show_gt_texts,
        prediction_lidar_boxes=prediction_lidar_boxes,
        prediction_texts=prediction["texts"],
    )
    radar_frame = get_radar_frame(
        label_dir=label_dir,
        label_files=label_files,
        radar_dataset=dataset,
        arr_range=arr_range,
        arr_azimuth_deg=arr_azimuth_deg,
        R_l2r=rotation,
        T_l2r=translation,
        radar_mode=cfg.radar_mode,
        frame_idx=frame_idx,
        show_texts=cfg.show_texts,
        show_gt_texts=cfg.show_gt_texts,
        prediction_lidar_boxes=prediction_lidar_boxes,
        prediction_texts=prediction["texts"],
        radar_data=radar_data,
    )
    combined = combine_camera_radar_frames(camera_frame, radar_frame)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / (
        f"{experiment.weather}_{experiment.group}_epoch015_"
        f"seq{cfg.sequence}_frame{prediction['frame_name']}_gt_pred.png"
    )
    if not cv2.imwrite(str(output_path), combined):
        raise RuntimeError(f"Cannot save visualization: {output_path}")
    return output_path


def visualize_experiment(experiment):
    print(
        f"[{experiment.weather}] {experiment.group}, seed={experiment.seed}, "
        f"BEV_tgt={experiment.bev_target_ap:.4f}, "
        f"3D_tgt={experiment.ap_3d_target:.4f}",
        flush=True,
    )
    selection_split = "test"
    candidates, datasets = _frame_candidates(experiment.test_sequences)
    if not candidates or candidates[0][0] < MIN_GT_BOXES:
        selection_split = "target_train"
        print(
            f"[{experiment.weather}] test frames have fewer than "
            f"{MIN_GT_BOXES} GT boxes; using target-weather train sequences "
            f"{experiment.target_sequences}",
            flush=True,
        )
        candidates, datasets = _frame_candidates(experiment.target_sequences)
    if not candidates:
        raise RuntimeError(f"No data-complete frames for {experiment.weather}")

    base_cfg = DataConfig(
        prediction_checkpoint_path=str(experiment.checkpoint_path),
        prediction_device="cpu",
    )
    predictor = build_checkpoint_predictor(base_cfg)

    for candidate_index, candidate in enumerate(
        candidates[:MAX_CANDIDATES_PER_WEATHER],
        start=1,
    ):
        gt_count, sequence, frame_idx, frame_name, label_filename = candidate
        print(
            f"[{experiment.weather}] candidate {candidate_index}: "
            f"split={selection_split}, seq={sequence}, "
            f"frame={frame_name}, GT={gt_count}",
            flush=True,
        )
        if gt_count < MIN_GT_BOXES:
            break
        dataset = datasets[sequence]
        radar_data = dataset.get_by_tesseract_idx(frame_name)
        prediction = predictor.predict(radar_data)
        prediction_count = len(prediction["texts"])
        if prediction_count < MIN_PREDICTION_BOXES:
            continue

        cfg = DataConfig(
            sequence=sequence,
            prediction_checkpoint_path=str(experiment.checkpoint_path),
            prediction_device="cpu",
        )
        label_files = get_label_files(get_label_dir(cfg))
        output_path = _render_selected_frame(
            experiment=experiment,
            cfg=cfg,
            dataset=dataset,
            label_files=label_files,
            frame_idx=frame_idx,
            radar_data=radar_data,
            prediction=prediction,
        )
        result = {
            "weather": experiment.weather,
            "group": experiment.group,
            "seed": experiment.seed,
            "bev_target_ap": experiment.bev_target_ap,
            "ap_3d_target": experiment.ap_3d_target,
            "checkpoint": str(experiment.checkpoint_path),
            "data_split": selection_split,
            "sequence": sequence,
            "frame": frame_name,
            "label_file": label_filename,
            "gt_boxes": gt_count,
            "prediction_boxes": prediction_count,
            "output": str(output_path),
        }
        print(
            f"[{experiment.weather}] saved GT={gt_count}, "
            f"Pred={prediction_count}: {output_path}",
            flush=True,
        )
        del predictor
        gc.collect()
        return result

    raise RuntimeError(
        f"No {experiment.weather} frame met GT>={MIN_GT_BOXES}, "
        f"Pred>={MIN_PREDICTION_BOXES} within "
        f"{MAX_CANDIDATES_PER_WEATHER} candidates"
    )


def _write_summary(results):
    lines = [
        "Best weather epoch-15 visualization selections",
        "Selection metric: maximum BEV_tgt AP; tie-breaker: 3D_tgt AP",
        f"Frame rule: GT>={MIN_GT_BOXES}, Pred>={MIN_PREDICTION_BOXES}",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"[{result['weather']}]",
                f"group: {result['group']} (seed {result['seed']})",
                f"BEV_tgt AP: {result['bev_target_ap']:.4f}",
                f"3D_tgt AP: {result['ap_3d_target']:.4f}",
                f"checkpoint: {result['checkpoint']}",
                f"data split: {result['data_split']}",
                f"sequence/frame: {result['sequence']}/{result['frame']}",
                f"GT/Prediction boxes: {result['gt_boxes']}/{result['prediction_boxes']}",
                f"output: {result['output']}",
                "",
            ]
        )
    summary_path = OUTPUT_DIR / "selection_summary.txt"
    summary_path.write_text("\n".join(lines))
    return summary_path


def main():
    weathers = ("heavy_snow", "light_snow", "overcast", "rain", "sleet")
    experiments = [find_best_experiment(weather) for weather in weathers]
    results = [visualize_experiment(experiment) for experiment in experiments]
    summary_path = _write_summary(results)
    print(f"Saved selection summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
