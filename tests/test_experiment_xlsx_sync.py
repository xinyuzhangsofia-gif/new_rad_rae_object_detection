import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from scripts.sync_experiment_xlsx_to_txt import (
    normalize_workbook_layout,
    read_xlsx_matrix,
    render_xlsx_as_txt,
    sync_result_matrix_to_workbook,
    sync_workbook,
)
from training_utils.experiment_queue import (
    load_domain_shift_experiments,
    update_experiment_sheet_result,
)


class ExperimentXlsxSyncTests(unittest.TestCase):
    def _write_workbook(self, directory, malformed_insert=False):
        path = Path(directory) / "heavy_snow_experiments.xlsx"
        strings = (
            "group",
            "seed",
            "shared_seq",
            "source_seq",
            "target_seq",
            "test_seq",
            "BEV_src",
            "3D_src",
            "BEV_tgt",
            "3D_tgt",
            "TD_BEV",
            "TD_3D",
            "group1",
            "42",
            "9,12",
            "14,15,18,20",
            "58,56,54,55",
            "46,47",
            "-",
            "group2",
            "average(0)",
        )
        shared_items = "".join(
            f"<si><t>{value}</t></si>"
            for value in strings
        )
        header_cells = "".join(
            (
                f'<c r="{chr(ord("A") + index)}1" s="1" t="s">'
                f"<v>{index}</v></c>"
            )
            for index in range(12)
        )
        row_indices = (12, 13, 14, 15, 16, 17)
        data_cells = (
            '<c r="A2" s="2" t="s"><v>12</v></c>'
            + "".join(
                (
                    f'<c r="{chr(ord("B") + index)}2" s="3" t="s">'
                    f"<v>{shared_index}</v></c>"
                )
                for index, shared_index in enumerate(row_indices[1:])
            )
        )
        data_cells += '<c r="G2" s="5"><v>1.2</v></c>'
        data_cells += "".join(
            (
                f'<c r="{chr(ord("H") + index)}2" s="4" t="s">'
                "<v>18</v></c>"
            )
            for index in range(5)
        )
        average_row_number = 5 if malformed_insert else 3
        average_cells = (
            f'<c r="A{average_row_number}" s="6" t="s"><v>20</v></c>'
            + "".join(
                (
                    f'<c r="{chr(ord("B") + index)}'
                    f'{average_row_number}" s="7" t="s"><v>18</v></c>'
                )
                for index in range(11)
            )
        )
        inserted_rows = ""
        if malformed_insert:
            inserted_rows = (
                '<row r="3"><c r="A3" t="s"><v>19</v></c></row>'
                '<row r="4"></row>'
            )

        with ZipFile(path, "w") as archive:
            archive.writestr(
                "xl/workbook.xml",
                (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    f'<workbook xmlns="{MAIN_NS}" '
                    f'xmlns:r="{OFFICE_REL_NS}">'
                    "<sheets><sheet name=\"heavy_snow\" sheetId=\"1\" "
                    "r:id=\"rId1\"/></sheets></workbook>"
                ),
            )
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    f'<Relationships xmlns="{PACKAGE_REL_NS}">'
                    '<Relationship Id="rId1" '
                    'Type="http://schemas.openxmlformats.org/'
                    'officeDocument/2006/relationships/worksheet" '
                    'Target="worksheets/sheet1.xml"/>'
                    "</Relationships>"
                ),
            )
            archive.writestr(
                "xl/sharedStrings.xml",
                (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    f'<sst xmlns="{MAIN_NS}" count="{len(strings)}" '
                    f'uniqueCount="{len(strings)}">{shared_items}</sst>'
                ),
            )
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    f'<worksheet xmlns="{MAIN_NS}">'
                    f'<dimension ref="A1:L{average_row_number}"/>'
                    "<sheetData>"
                    f'<row r="1">{header_cells}</row>'
                    f'<row r="2">{data_cells}</row>'
                    f"{inserted_rows}"
                    f'<row r="{average_row_number}">{average_cells}</row>'
                    "</sheetData>"
                    '<autoFilter ref="A1:L2"/>'
                    "</worksheet>"
                ),
            )
            archive.writestr(
                "xl/tables/table1.xml",
                (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    f'<table xmlns="{MAIN_NS}" id="1" name="Experiments" '
                    'displayName="Experiments" ref="A1:L2" '
                    + (
                        'totalsRowShown="1" headerRowCount="0">'
                        if malformed_insert
                        else 'totalsRowShown="0">'
                    )
                    + '<autoFilter ref="A1:L2"/>'
                    "<tableColumns count=\"12\">"
                    + "".join(
                        (
                            f'<tableColumn id="{index + 1}" '
                            f'name="{strings[index]}"/>'
                        )
                        for index in range(12)
                    )
                    + "</tableColumns></table>"
                ),
            )
        return path

    def test_sync_preserves_non_numeric_sequence_pair_order(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            workbook_path = self._write_workbook(temporary_dir)
            matrix = read_xlsx_matrix(workbook_path)
            txt_path, changed = sync_workbook(workbook_path)
            experiment = load_domain_shift_experiments(txt_path)[0]

        self.assertEqual(matrix[1][3], "14,15,18,20")
        self.assertEqual(matrix[1][4], "58,56,54,55")
        self.assertEqual(matrix[1][6], "1.2000")
        self.assertTrue(changed)
        self.assertEqual(
            experiment.source_sequences,
            (14, 15, 18, 20),
        )
        self.assertEqual(
            experiment.target_sequences,
            (58, 56, 54, 55),
        )

    def test_result_update_writes_txt_values_back_to_sibling_xlsx(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            workbook_path = self._write_workbook(temporary_dir)
            txt_path, _changed = sync_workbook(workbook_path)
            experiment = load_domain_shift_experiments(txt_path)[0]

            update_experiment_sheet_result(
                txt_path,
                experiment,
                "source",
                bev_ap=10.0,
                threed_ap=8.0,
            )
            update_experiment_sheet_result(
                txt_path,
                experiment,
                "target",
                bev_ap=13.0,
                threed_ap=9.5,
            )
            matrix = read_xlsx_matrix(workbook_path)
            rendered = render_xlsx_as_txt(workbook_path)
            txt_text = txt_path.read_text(encoding="utf-8")

        self.assertEqual(matrix[1][6:12], [
            "10.0000",
            "8.0000",
            "13.0000",
            "9.5000",
            "3.0000",
            "1.5000",
        ])
        self.assertEqual(matrix[-1][0], "average(1)")
        self.assertEqual(matrix[-1][10:12], ["3.0000", "1.5000"])
        self.assertEqual(rendered, txt_text)

    def test_result_matrix_sync_preserves_sequence_pair_order(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            workbook_path = self._write_workbook(temporary_dir)
            matrix = read_xlsx_matrix(workbook_path)
            matrix[1][6:12] = [
                "10.0000",
                "8.0000",
                "13.0000",
                "9.5000",
                "3.0000",
                "1.5000",
            ]
            matrix[-1][0] = "average(1)"
            matrix[-1][10:12] = ["3.0000", "1.5000"]

            changed = sync_result_matrix_to_workbook(
                workbook_path,
                matrix,
            )
            updated = read_xlsx_matrix(workbook_path)

        self.assertTrue(changed)
        self.assertEqual(updated[1][3], "14,15,18,20")
        self.assertEqual(updated[1][4], "58,56,54,55")
        self.assertEqual(updated[1][6:12], matrix[1][6:12])

    def test_repairs_inserted_row_styles_and_table_range(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            workbook_path = self._write_workbook(
                temporary_dir,
                malformed_insert=True,
            )
            changed = normalize_workbook_layout(
                workbook_path,
                remove_blank_rows=True,
            )
            matrix = read_xlsx_matrix(workbook_path)
            with ZipFile(workbook_path, "r") as archive:
                worksheet = ET.fromstring(
                    archive.read("xl/worksheets/sheet1.xml")
                )
                table = ET.fromstring(archive.read("xl/tables/table1.xml"))

        rows = worksheet.findall(f".//{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row")
        repaired_cells = {
            cell.get("r"): cell
            for cell in rows[2].findall(f"./{{{MAIN_NS}}}c")
        }
        self.assertTrue(changed)
        self.assertEqual(len(matrix), 4)
        self.assertEqual(matrix[2][0], "group2")
        self.assertEqual(matrix[2][1:], ["-"] * 11)
        self.assertEqual(repaired_cells["A3"].get("s"), "2")
        self.assertEqual(repaired_cells["B3"].get("s"), "4")
        self.assertEqual(
            worksheet.find(f"./{{{MAIN_NS}}}dimension").get("ref"),
            "A1:L4",
        )
        self.assertEqual(
            worksheet.find(f"./{{{MAIN_NS}}}autoFilter").get("ref"),
            "A1:L3",
        )
        self.assertEqual(table.get("ref"), "A1:L3")
        self.assertEqual(table.get("totalsRowShown"), "0")
        self.assertNotIn("headerRowCount", table.attrib)

    def test_blank_insert_gets_next_group_number_and_enters_table(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            workbook_path = self._write_workbook(
                temporary_dir,
                malformed_insert=True,
            )
            changed = normalize_workbook_layout(workbook_path)
            matrix = read_xlsx_matrix(workbook_path)
            with ZipFile(workbook_path, "r") as archive:
                worksheet = ET.fromstring(
                    archive.read("xl/worksheets/sheet1.xml")
                )
                table = ET.fromstring(archive.read("xl/tables/table1.xml"))

        rows = worksheet.findall(
            f".//{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row"
        )
        new_group_cells = {
            cell.get("r"): cell
            for cell in rows[3].findall(f"./{{{MAIN_NS}}}c")
        }
        self.assertTrue(changed)
        self.assertEqual(len(matrix), 5)
        self.assertEqual(matrix[3][0], "group3")
        self.assertEqual(matrix[3][1:], ["-"] * 11)
        self.assertEqual(new_group_cells["A4"].get("s"), "2")
        self.assertEqual(new_group_cells["B4"].get("s"), "4")
        self.assertEqual(
            worksheet.find(f"./{{{MAIN_NS}}}dimension").get("ref"),
            "A1:L5",
        )
        self.assertEqual(
            worksheet.find(f"./{{{MAIN_NS}}}autoFilter").get("ref"),
            "A1:L4",
        )
        self.assertEqual(table.get("ref"), "A1:L4")
        self.assertEqual(matrix[4][0], "average(0)")


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


if __name__ == "__main__":
    unittest.main()
