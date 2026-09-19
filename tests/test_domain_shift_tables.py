import csv
import json
import re
import tempfile
import unittest
from pathlib import Path

from data.sequence_metadata import (
    DEFAULT_SEQUENCE_INFO_PATH,
    DomainShiftTableError,
    load_sequence_information,
)
from eval.domain_shift_tables import (
    BEV_METRIC_KEY,
    DEFAULT_OUTPUT_DIR,
    DomainShiftTableError as EvaluationDomainShiftTableError,
    TABLE_CORNER_HEADER,
    THREED_METRIC_KEY,
    build_model_configuration,
    convert_legacy_csv_table,
    parse_evaluation_table_txt,
    update_domain_shift_tables,
)


def metric_results(*rows):
    return [
        {
            "epoch": epoch,
            BEV_METRIC_KEY: bev_ap,
            THREED_METRIC_KEY: threed_ap,
        }
        for epoch, bev_ap, threed_ap in rows
    ]


def base_metadata(**overrides):
    metadata = {
        "model_type": "model7_64_128_train_1,5",
        "model_configuration_name": "model7_64_128_lr5e-05",
        "model_configuration": {"learning_rate": 5e-5},
        "train_sequences": (1, 5),
        "val_sequences": (13,),
        "include_bus_as_target": True,
        "train_control_split_enabled": False,
    }
    metadata.update(overrides)
    return metadata


def read_table(path):
    rows = []
    table_started = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or re.fullmatch(r"[-+| ]+", stripped):
            continue
        if not table_started:
            if TABLE_CORNER_HEADER not in stripped:
                continue
            table_started = True
        rows.append([cell.strip() for cell in line.split("|")])
    return rows


class DomainShiftTablesTest(unittest.TestCase):
    def test_moved_modules_preserve_default_paths_and_exception_identity(self):
        project_root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            DEFAULT_SEQUENCE_INFO_PATH,
            project_root / "sequence_information.csv",
        )
        self.assertEqual(DEFAULT_OUTPUT_DIR, project_root / "evaluation_results")
        self.assertIs(DomainShiftTableError, EvaluationDomainShiftTableError)

    def test_sequence_information_contains_all_image_rows(self):
        information = load_sequence_information()
        self.assertEqual(len(information), 58)
        self.assertEqual(information[1]["weather"], "normal")
        self.assertEqual(information[13]["weather"], "overcast")
        self.assertEqual(information[46]["weather"], "heavysnow")
        self.assertEqual(information[58]["object_count"], 926)

    def test_three_epoch_selections_and_before_after_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            results = metric_results(
                (1, 10.0, 8.0),
                (2, 12.0, 5.0),
                (3, 9.0, 11.0),
            )
            before = update_domain_shift_tables(
                results,
                base_metadata(),
                output_dir=output_dir,
            )
            empty_after_table = (
                output_dir
                / "model7_64_128_lr5e-05"
                / "after"
                / "table1_best_bev.txt"
            )
            self.assertTrue(empty_after_table.is_file())
            self.assertEqual(read_table(empty_after_table)[1][1], "")
            after = update_domain_shift_tables(
                results,
                base_metadata(include_bus_as_target=False),
                output_dir=output_dir,
            )

            self.assertEqual(before["evaluation_group"], "before")
            self.assertEqual(after["evaluation_group"], "after")
            self.assertEqual(before["selections"]["best_bev"]["epoch"], 2)
            self.assertEqual(before["selections"]["best_3d"]["epoch"], 3)
            self.assertEqual(before["selections"]["best_overall"]["epoch"], 3)
            before_bev_rows = read_table(before["table_paths"]["best_bev"])
            self.assertEqual(before_bev_rows[0][0], TABLE_CORNER_HEADER)
            self.assertTrue(before_bev_rows[0][1].endswith(" ->"))
            self.assertTrue(before_bev_rows[1][0].startswith("-> "))
            self.assertEqual(before_bev_rows[1][1], "12.00/5.00")
            table_text = Path(before["table_paths"]["best_bev"]).read_text()
            self.assertIn("evaluation_group: before", table_text)
            self.assertIn("table_selection: best_bev", table_text)
            self.assertIn("metric_pair: BEV@0.3/3D@0.3", table_text)
            self.assertIn("source_domains:\n  - Normal 1 ->", table_text)
            self.assertIn(
                "source_domain_details:\n  - Normal 1 (Sequences: 1, 5)",
                table_text,
            )
            self.assertNotIn("source_domains: Normal 1 (Sequences:", table_text)
            self.assertEqual(
                Path(before["table_paths"]["best_bev"]).parent.name,
                "before",
            )
            self.assertEqual(
                Path(after["table_paths"]["best_bev"]).parent.name,
                "after",
            )
            self.assertEqual(list(output_dir.rglob("*.xlsx")), [])

    def test_mixed_weather_uses_non_normal_name_and_preserves_all_weather(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            summary = update_domain_shift_tables(
                metric_results((1, 1.0, 2.0)),
                base_metadata(train_sequences=(10, 13)),
                output_dir=output_dir,
            )
            self.assertTrue(summary["source_domain"].startswith("Overcast 1"))
            self.assertIn(
                "Weather: Normal + Overcast",
                summary["source_domain"],
            )

            registry = json.loads(
                (output_dir / "domain_registry.json").read_text()
            )
            source_entries = [
                entry
                for entry in registry["domains"].values()
                if entry["role"] == "source"
            ]
            self.assertEqual(
                source_entries[0]["weather_conditions"],
                ["normal", "overcast"],
            )

    def test_normal_target_is_recorded_in_json_but_omitted_from_table(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            summary = update_domain_shift_tables(
                metric_results((1, 10.0, 5.0)),
                base_metadata(val_sequences=(9,)),
                output_dir=output_dir,
            )

            self.assertFalse(summary["target_recorded_in_table"])
            self.assertTrue(Path(summary["record_path"]).is_file())
            for table_path in summary["table_paths"].values():
                self.assertEqual(read_table(table_path), [[TABLE_CORNER_HEADER]])

    def test_mixed_normal_and_adverse_target_remains_in_table(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            summary = update_domain_shift_tables(
                metric_results((1, 10.0, 5.0)),
                base_metadata(val_sequences=(10, 13)),
                output_dir=Path(temporary_dir),
            )

            self.assertTrue(summary["target_recorded_in_table"])
            rows = read_table(summary["table_paths"]["best_bev"])
            self.assertEqual(rows[1][1], "10.00/5.00")

    def test_same_experiment_updates_cell_without_duplicate_row_or_column(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            first = update_domain_shift_tables(
                metric_results((1, 1.0, 2.0)),
                base_metadata(),
                output_dir=output_dir,
            )
            second = update_domain_shift_tables(
                metric_results((4, 4.0, 5.0)),
                base_metadata(),
                output_dir=output_dir,
            )
            rows = read_table(second["table_paths"]["best_bev"])
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(rows[0]), 2)
            self.assertEqual(rows[1][1], "4.00/5.00")
            self.assertEqual(first["record_path"], second["record_path"])

    def test_legacy_csv_is_converted_to_paper_text_layout(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "table.csv"
            text_path = Path(temporary_dir) / "table.txt"
            with csv_path.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.writer(output_file)
                writer.writerow(["Target domain", "Source A"])
                writer.writerow([
                    "Target A",
                    "epoch=1; BEV@0.3=1.0000; 3D@0.3=2.0000",
                ])

            convert_legacy_csv_table(csv_path, text_path)
            rows = read_table(text_path)
            self.assertEqual(rows[0][0], TABLE_CORNER_HEADER)
            self.assertEqual(rows[1][1], "1.00/2.00")

    def test_source_rows_layout_is_transposed_to_weather_table_layout(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "table.csv"
            text_path = Path(temporary_dir) / "table.txt"
            with csv_path.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.writer(output_file)
                writer.writerow([TABLE_CORNER_HEADER, "-> Target A"])
                writer.writerow(["Source A ->", "1.00/2.00"])

            convert_legacy_csv_table(csv_path, text_path)
            rows = read_table(text_path)
            self.assertEqual(rows[0], [TABLE_CORNER_HEADER, "Source A ->"])
            self.assertEqual(rows[1], ["-> Target A", "1.00/2.00"])

    def test_new_domains_extend_table_and_leave_other_cells_empty(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            first = update_domain_shift_tables(
                metric_results((1, 1.0, 2.0)),
                base_metadata(),
                output_dir=output_dir,
            )
            second = update_domain_shift_tables(
                metric_results((2, 3.0, 4.0)),
                base_metadata(train_sequences=(9, 10), val_sequences=(22,)),
                output_dir=output_dir,
            )
            rows = read_table(second["table_paths"]["best_bev"])
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(rows[0]), 3)
            self.assertEqual(rows[1][2], "")
            self.assertEqual(rows[2][1], "")
            self.assertNotEqual(first["source_domain"], second["source_domain"])

    def test_parse_evaluation_txt_uses_header_names_not_fixed_positions(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            table_path = Path(temporary_dir) / "evaluation.txt"
            table_path.write_text(
                "\n".join([
                    "model_type: model7",
                    "model_variant: model7_64_128_ig_train_1,5",
                    "train_sequences: (1, 5)",
                    "val_sequences: (13,)",
                    "include_bus_as_target: False",
                    "",
                    "best_epoch_bev@0.3: epoch 2 (bev@0.3=12.0000)",
                    "best_epoch_3d@0.3: epoch 3 (3d@0.3=11.0000)",
                    "",
                    "epoch val_loss val_box bev@0.3 bev@0.5 3d@0.3 3d@0.5",
                    "------------------------------------------------------------",
                    "1 4.0 1.0 10.0 2.0 8.0 1.0",
                    "2 3.0 1.0 12.0 3.0 5.0 1.0",
                    "3 2.0 1.0 9.0 4.0 11.0 2.0",
                ]) + "\n",
                encoding="utf-8",
            )
            results, metadata = parse_evaluation_table_txt(table_path)
            self.assertEqual(len(results), 3)
            self.assertEqual(results[1][BEV_METRIC_KEY], 12.0)
            self.assertEqual(results[2][THREED_METRIC_KEY], 11.0)
            self.assertFalse(metadata["include_bus_as_target"])
            self.assertEqual(metadata["train_sequences"], (1, 5))

    def test_batch_seed_and_active_loss_settings_create_separate_configurations(self):
        common = {
            "lr": 5e-5,
            "batch_size": 8,
            "seed": 42,
            "heatmap_radius": 3,
            "centerpoint_gwd_loss_weight": 2.0,
            "quality_loss_weight": 0.25,
        }
        names = set()
        for overrides in (
            {},
            {"batch_size": 16},
            {"seed": 43},
            {"centerpoint_gwd_loss_weight": 3.0},
        ):
            checkpoint_config = {**common, **overrides}
            name, _ = build_model_configuration(
                model_type="model7",
                model_variant_name="model7_64_128_train_1,5",
                checkpoint_config=checkpoint_config,
            )
            names.add(name)
        self.assertEqual(len(names), 4)
        self.assertTrue(any("_bs8_seed42_loss_hm3_gwd2" in name for name in names))

    def test_model7_configuration_ignores_inactive_quality_weight(self):
        common = {
            "lr": 5e-5,
            "batch_size": 8,
            "seed": 42,
            "heatmap_radius": 3,
            "centerpoint_gwd_loss_weight": 2.0,
        }
        first_name, first_config = build_model_configuration(
            model_type="model7",
            checkpoint_config={**common, "quality_loss_weight": 0.25},
        )
        second_name, second_config = build_model_configuration(
            model_type="model7",
            checkpoint_config={**common, "quality_loss_weight": 0.5},
        )
        self.assertEqual(first_name, second_name)
        self.assertFalse(first_config["quality_loss_active"])
        self.assertIsNone(first_config["quality_loss_weight"])
        self.assertEqual(first_config, second_config)

    def test_missing_gwd_config_is_named_unknown(self):
        common = {
            "lr": 5e-5,
            "batch_size": 8,
            "seed": 42,
            "heatmap_radius": 3,
            "quality_loss_weight": 0.25,
        }
        missing_name, missing_config = build_model_configuration(
            model_type="model7",
            checkpoint_config=common,
        )
        current_name, current_config = build_model_configuration(
            model_type="model7",
            checkpoint_config={
                **common,
                "centerpoint_gwd_loss_weight": 2.0,
            },
        )
        self.assertIn("loss_unknown", missing_name)
        self.assertIsNone(missing_config["centerpoint_gwd_loss_weight"])
        self.assertNotEqual(missing_name, current_name)
        self.assertEqual(current_config["centerpoint_gwd_loss_weight"], 2.0)


if __name__ == "__main__":
    unittest.main()
