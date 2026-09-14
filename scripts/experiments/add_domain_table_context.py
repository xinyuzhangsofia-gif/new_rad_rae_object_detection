from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.domain_shift_tables import add_comparison_table_context


OUTPUT_ROOT = PROJECT_ROOT / "evaluation_results"
SELECTION_METRICS = {
    "table1_best_bev": "official_bev_mAP_0.3",
    "table2_best_3d": "official_3d_mAP_0.3",
    "table3_best_overall": (
        "(official_bev_mAP_0.3 + official_3d_mAP_0.3) / 2"
    ),
}


def main() -> None:
    paths = sorted(OUTPUT_ROOT.glob("*/*/table*.txt"))
    for table_path in paths:
        record_paths = sorted(table_path.parent.joinpath("records").glob("*.json"))
        latest_record = {}
        if record_paths:
            latest_record = json.loads(
                record_paths[-1].read_text(encoding="utf-8")
            )
        metadata = latest_record.get("evaluation_metadata") or {}
        source_domain_labels = set()
        target_domain_labels = set()
        for record_path in record_paths:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            source_label = (record.get("source_domain") or {}).get("label")
            target_label = (record.get("target_domain") or {}).get("label")
            if source_label:
                source_domain_labels.add(str(source_label))
            if target_label:
                target_domain_labels.add(str(target_label))
        group_name = table_path.parent.name
        configuration_name = table_path.parents[1].name
        model_type = metadata.get("base_model_type") or configuration_name.split("_")[0]
        selection_name = table_path.stem
        context_lines = [
            f"model_type: {model_type}",
            f"model_variant: {metadata.get('model_type', 'unknown')}",
            f"model_configuration: {configuration_name}",
            f"evaluation_group: {group_name}",
            f"include_bus_as_target: {group_name == 'before'}",
            f"bus_ignored_during_evaluation: {group_name == 'after'}",
            f"train_sequences: {metadata.get('train_sequences', 'unknown')}",
            f"val_sequences: {metadata.get('val_sequences', 'unknown')}",
            f"train_control_split_enabled: {metadata.get('train_control_split_enabled', 'unknown')}",
            f"eval_scope: {metadata.get('eval_scope', 'unknown')}",
            f"ap_score_thresh: {metadata.get('ap_score_thresh', 'unknown')}",
            f"score_thresh: {metadata.get('score_thresh', 'unknown')}",
            f"source_evaluation_txt: {metadata.get('source_txt_path', 'unknown')}",
            "source_domain_details:",
            *[f"  - {label}" for label in sorted(source_domain_labels)],
            "target_domain_details:",
            *[f"  - {label}" for label in sorted(target_domain_labels)],
            f"table_selection: {selection_name}",
            f"selection_metric: {SELECTION_METRICS.get(selection_name, 'unknown')}",
            "selection_rule: one best epoch per source-target pair",
            "metric_pair: BEV@0.3/3D@0.3",
            "normal_only_target_rows: excluded",
        ]
        add_comparison_table_context(table_path, context_lines)
    print(f"context_added={len(paths)}")


if __name__ == "__main__":
    main()
