"""Normalize experiment XLSX layouts and synchronize them to aligned TXT."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import os
import posixpath
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import BadZipFile, ZipFile
from zipfile import ZIP_DEFLATED


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_utils.experiment_queue import _format_txt_experiment_matrix


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
X14AC_NS = "http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac"
REQUIRED_HEADERS = {
    "shared_seq",
    "source_seq",
    "target_seq",
    "test_seq",
}
FOUR_DECIMAL_HEADERS = {
    "bev_src",
    "3d_src",
    "bev_tgt",
    "3d_tgt",
    "td_bev",
    "td_3d",
    "bev_ap(source)",
    "3d_ap(source)",
    "bev_ap(target)",
    "3d_ap(target)",
}
RESULT_HEADERS = (
    "bev_src",
    "3d_src",
    "bev_tgt",
    "3d_tgt",
    "td_bev",
    "td_3d",
)
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
DEFAULT_PATTERN = "*_experiments.xlsx"
XML_SPACE = "http://www.w3.org/XML/1998/namespace"


def _normalized_header(value):
    return " ".join(str(value or "").strip().lower().split())


def _column_number(cell_reference):
    letters = []
    for character in str(cell_reference):
        if character.isalpha():
            letters.append(character.upper())
        else:
            break
    if not letters:
        raise ValueError(f"Invalid XLSX cell reference: {cell_reference!r}")
    result = 0
    for letter in letters:
        result = result * 26 + ord(letter) - ord("A") + 1
    return result


def _column_letters(column_number):
    if int(column_number) <= 0:
        raise ValueError(f"Invalid XLSX column number: {column_number!r}")
    result = []
    remaining = int(column_number)
    while remaining:
        remaining, remainder = divmod(remaining - 1, 26)
        result.append(chr(ord("A") + remainder))
    return "".join(reversed(result))


def _first_worksheet_path(archive):
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = workbook.find(f".//{{{MAIN_NS}}}sheets/{{{MAIN_NS}}}sheet")
    if sheet is None:
        raise ValueError("XLSX workbook contains no worksheet")

    relationship_id = sheet.get(f"{{{OFFICE_REL_NS}}}id")
    if not relationship_id:
        raise ValueError("First XLSX worksheet has no relationship id")

    relationships = ET.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    for relationship in relationships.findall(
        f".//{{{PACKAGE_REL_NS}}}Relationship"
    ):
        if relationship.get("Id") != relationship_id:
            continue
        target = str(relationship.get("Target") or "")
        if target.startswith("/"):
            return target.lstrip("/")
        return posixpath.normpath(posixpath.join("xl", target))
    raise ValueError(
        f"Cannot resolve first XLSX worksheet relationship {relationship_id!r}"
    )


def _shared_strings(archive):
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return ()
    return tuple(
        "".join(
            text_node.text or ""
            for text_node in item.findall(f".//{{{MAIN_NS}}}t")
        )
        for item in root.findall(f"./{{{MAIN_NS}}}si")
    )


def _cell_text(cell, shared_strings):
    cell_type = str(cell.get("t") or "")
    if cell_type == "inlineStr":
        return "".join(
            text_node.text or ""
            for text_node in cell.findall(
                f"./{{{MAIN_NS}}}is//{{{MAIN_NS}}}t"
            )
        )

    value_node = cell.find(f"./{{{MAIN_NS}}}v")
    value = "" if value_node is None else str(value_node.text or "")
    if cell_type == "s":
        if value == "":
            return ""
        index = int(value)
        if index < 0 or index >= len(shared_strings):
            raise ValueError(f"Invalid XLSX shared-string index: {index}")
        return shared_strings[index]
    if cell_type == "b":
        return "TRUE" if value == "1" else "FALSE"
    return value


def _set_inline_string(cell, value):
    for child in list(cell):
        cell.remove(child)
    cell.set("t", "inlineStr")
    inline_string = ET.SubElement(cell, f"{{{MAIN_NS}}}is")
    text_node = ET.SubElement(inline_string, f"{{{MAIN_NS}}}t")
    text = str(value)
    if text != text.strip():
        text_node.set(f"{{{XML_SPACE}}}space", "preserve")
    text_node.text = text


def _row_cells_by_column(row):
    result = {}
    for cell in row.findall(f"./{{{MAIN_NS}}}c"):
        reference = cell.get("r")
        if reference:
            result[_column_number(reference)] = cell
    return result


def _row_values(row, shared_strings):
    return {
        column_number: _cell_text(cell, shared_strings)
        for column_number, cell in _row_cells_by_column(row).items()
    }


def _style_id(cell, fallback="0"):
    if cell is None:
        return str(fallback)
    return str(cell.get("s") or fallback)


def _set_attribute(element, name, value):
    value = str(value)
    if element.get(name) == value:
        return False
    element.set(name, value)
    return True


def _remove_attribute(element, name):
    if name not in element.attrib:
        return False
    element.attrib.pop(name, None)
    return True


def _write_workbook_entries(workbook_path, entries):
    workbook_path = Path(workbook_path)
    original_mode = workbook_path.stat().st_mode & 0o777
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{workbook_path.name}.",
        suffix=".tmp",
        dir=workbook_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with ZipFile(
            temporary_path,
            "w",
            compression=ZIP_DEFLATED,
        ) as output:
            for info, data in entries:
                output.writestr(info, data)
        os.chmod(temporary_path, original_mode)
        os.replace(temporary_path, workbook_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def normalize_workbook_layout(workbook_path, remove_blank_rows=False):
    """Repair inserted experiment rows while preserving all entered values.

    A newly inserted group row receives ``group<max + 1>`` automatically.
    Existing unique group numbers are preserved, while accidental duplicate
    group numbers are reassigned to the next available number.
    """
    workbook_path = Path(workbook_path)
    with ZipFile(workbook_path, "r") as archive:
        worksheet_path = _first_worksheet_path(archive)
        shared_strings = _shared_strings(archive)
        entries = [
            (info, archive.read(info.filename))
            for info in archive.infolist()
        ]

    entry_index = {
        info.filename: index
        for index, (info, _data) in enumerate(entries)
    }
    worksheet_index = entry_index[worksheet_path]
    worksheet_info, worksheet_data = entries[worksheet_index]
    worksheet = ET.fromstring(worksheet_data)
    sheet_data = worksheet.find(f"./{{{MAIN_NS}}}sheetData")
    if sheet_data is None:
        raise ValueError(f"XLSX worksheet has no sheetData: {workbook_path}")

    rows = list(sheet_data.findall(f"./{{{MAIN_NS}}}row"))
    if not rows:
        raise ValueError(f"XLSX worksheet contains no rows: {workbook_path}")
    rows.sort(key=lambda row: int(row.get("r") or 0))
    header_row = rows[0]

    average_rows = []
    for row in rows[1:]:
        first_value = _row_values(row, shared_strings).get(1, "")
        if str(first_value).strip().lower().startswith("average"):
            average_rows.append(row)
    if len(average_rows) != 1:
        raise ValueError(
            "Experiment workbook must contain exactly one average row; "
            f"found {len(average_rows)} in {workbook_path}"
        )
    average_row = average_rows[0]

    def row_has_value(row):
        return any(
            str(value).strip() != ""
            for value in _row_values(row, shared_strings).values()
        )

    data_rows = [
        row
        for row in rows[1:]
        if row is not average_row
        and (not remove_blank_rows or row_has_value(row))
    ]
    if not data_rows:
        raise ValueError(
            f"Experiment workbook contains no group rows: {workbook_path}"
        )

    width = max(
        1,
        len(_row_cells_by_column(header_row)),
        max(
            (
                max(_row_cells_by_column(row), default=0)
                for row in rows
            ),
            default=0,
        ),
    )
    if width < len(REQUIRED_HEADERS):
        raise ValueError(
            f"Experiment workbook has only {width} columns: {workbook_path}"
        )

    styled_data_rows = [
        row
        for row in data_rows
        if _row_cells_by_column(row).get(1) is not None
        and _row_cells_by_column(row).get(1).get("s") is not None
    ]
    style_source_row = (
        styled_data_rows[0] if styled_data_rows else data_rows[0]
    )
    source_cells = _row_cells_by_column(style_source_row)
    group_label_style = _style_id(source_cells.get(1), "0")
    active_style = _style_id(source_cells.get(2), group_label_style)

    inactive_style = None
    metric_style = None
    for row in data_rows:
        values = _row_values(row, shared_strings)
        cells = _row_cells_by_column(row)
        for column_number, value in values.items():
            text = str(value).strip()
            if text == "-" and inactive_style is None:
                inactive_style = _style_id(
                    cells.get(column_number),
                    active_style,
                )
            if (
                column_number >= 7
                and text not in {"", "-"}
                and metric_style is None
            ):
                try:
                    Decimal(text)
                except InvalidOperation:
                    continue
                metric_style = _style_id(
                    cells.get(column_number),
                    active_style,
                )
    inactive_style = inactive_style or active_style
    metric_style = metric_style or active_style

    average_cells = _row_cells_by_column(average_row)
    average_label_style = _style_id(
        average_cells.get(1),
        group_label_style,
    )
    average_placeholder_style = _style_id(
        average_cells.get(2),
        inactive_style,
    )
    average_metric_style = next(
        (
            _style_id(average_cells[column_number], average_placeholder_style)
            for column_number in range(7, width + 1)
            if column_number in average_cells
            and str(
                _cell_text(
                    average_cells[column_number],
                    shared_strings,
                )
            ).strip()
            not in {"", "-"}
        ),
        metric_style,
    )

    group_number_pattern = re.compile(r"group\s*(\d+)", flags=re.IGNORECASE)
    existing_group_numbers = []
    for row in data_rows:
        group_value = str(
            _row_values(row, shared_strings).get(1, "") or ""
        ).strip()
        match = group_number_pattern.fullmatch(group_value)
        if match is not None:
            existing_group_numbers.append(int(match.group(1)))

    next_group_number = max(existing_group_numbers, default=0) + 1
    used_group_numbers = set()
    automatic_group_labels = {}
    for row in data_rows:
        group_value = str(
            _row_values(row, shared_strings).get(1, "") or ""
        ).strip()
        match = group_number_pattern.fullmatch(group_value)
        if (
            match is not None
            and int(match.group(1)) not in used_group_numbers
        ):
            used_group_numbers.add(int(match.group(1)))
            continue
        if group_value.lower() not in {"", "-", "none", "nan"} and match is None:
            # Preserve intentional custom experiment names.
            continue
        while next_group_number in used_group_numbers:
            next_group_number += 1
        automatic_group_labels[id(row)] = f"group{next_group_number}"
        used_group_numbers.add(next_group_number)
        next_group_number += 1

    changed = False

    def normalize_row(
        row,
        row_number,
        row_kind,
    ):
        nonlocal changed
        changed |= _set_attribute(row, "r", row_number)
        changed |= _set_attribute(row, "ht", "22")
        changed |= _set_attribute(row, "customHeight", "1")
        changed |= _set_attribute(row, "spans", f"1:{width}")

        cells = _row_cells_by_column(row)
        for column_number in list(cells):
            if column_number > width:
                row.remove(cells.pop(column_number))
                changed = True

        for column_number in range(1, width + 1):
            cell = cells.get(column_number)
            if cell is None:
                cell = ET.Element(f"{{{MAIN_NS}}}c")
                cells[column_number] = cell
                _set_inline_string(cell, "-")
                changed = True
            reference = f"{_column_letters(column_number)}{row_number}"
            changed |= _set_attribute(cell, "r", reference)

            value = str(_cell_text(cell, shared_strings) or "")
            automatic_group_label = (
                automatic_group_labels.get(id(row))
                if row_kind == "group" and column_number == 1
                else None
            )
            if automatic_group_label is not None:
                if value.strip() != automatic_group_label:
                    _set_inline_string(cell, automatic_group_label)
                    value = automatic_group_label
                    changed = True
            elif value.strip() == "":
                _set_inline_string(cell, "-")
                value = "-"
                changed = True

            if row_kind == "average":
                if column_number == 1:
                    target_style = average_label_style
                elif column_number >= 7 and value.strip() != "-":
                    target_style = average_metric_style
                else:
                    target_style = average_placeholder_style
            else:
                if column_number == 1:
                    target_style = group_label_style
                elif value.strip() == "-":
                    target_style = inactive_style
                elif column_number >= 7:
                    target_style = metric_style
                else:
                    target_style = active_style
            changed |= _set_attribute(cell, "s", target_style)

        for cell in list(row.findall(f"./{{{MAIN_NS}}}c")):
            row.remove(cell)
        for column_number in sorted(cells):
            row.append(cells[column_number])

    header_cells = _row_cells_by_column(header_row)
    changed |= _set_attribute(header_row, "r", "1")
    changed |= _set_attribute(header_row, "ht", "26")
    changed |= _set_attribute(header_row, "customHeight", "1")
    changed |= _set_attribute(header_row, "spans", f"1:{width}")
    for column_number, cell in header_cells.items():
        changed |= _set_attribute(
            cell,
            "r",
            f"{_column_letters(column_number)}1",
        )

    for index, row in enumerate(data_rows, start=2):
        normalize_row(row, index, "group")
    average_row_number = len(data_rows) + 2
    normalize_row(average_row, average_row_number, "average")

    desired_rows = [header_row] + data_rows + [average_row]
    if list(sheet_data) != desired_rows:
        for row in list(sheet_data):
            sheet_data.remove(row)
        for row in desired_rows:
            sheet_data.append(row)
        changed = True

    group_last_row = len(data_rows) + 1
    dimension_ref = f"A1:{_column_letters(width)}{average_row_number}"
    table_ref = f"A1:{_column_letters(width)}{group_last_row}"
    dimension = worksheet.find(f"./{{{MAIN_NS}}}dimension")
    if dimension is not None:
        changed |= _set_attribute(dimension, "ref", dimension_ref)
    worksheet_filter = worksheet.find(f"./{{{MAIN_NS}}}autoFilter")
    if worksheet_filter is not None:
        changed |= _set_attribute(worksheet_filter, "ref", table_ref)

    table_paths = sorted(
        path
        for path in entry_index
        if path.startswith("xl/tables/") and path.endswith(".xml")
    )
    ET.register_namespace("", MAIN_NS)
    ET.register_namespace("r", OFFICE_REL_NS)
    ET.register_namespace("mc", MC_NS)
    ET.register_namespace("x14ac", X14AC_NS)
    table_updates = {}
    for table_path in table_paths:
        table_info, table_data = entries[entry_index[table_path]]
        table = ET.fromstring(table_data)
        table_changed = False
        table_changed |= _set_attribute(table, "ref", table_ref)
        table_changed |= _set_attribute(table, "totalsRowShown", "0")
        table_changed |= _remove_attribute(table, "headerRowCount")
        # ExcelJS may leave revision prefixes in mc:Ignorable after removing
        # every revision element. Dropping that stale marker keeps the
        # rewritten table XML self-contained.
        table_changed |= _remove_attribute(
            table,
            f"{{{MC_NS}}}Ignorable",
        )
        table_filter = table.find(f"./{{{MAIN_NS}}}autoFilter")
        if table_filter is not None:
            table_changed |= _set_attribute(table_filter, "ref", table_ref)
        if table_changed:
            table_updates[table_path] = (
                table_info,
                ET.tostring(
                    table,
                    encoding="utf-8",
                    xml_declaration=True,
                ),
            )
            changed = True

    if not changed:
        return False

    entries[worksheet_index] = (
        worksheet_info,
        ET.tostring(
            worksheet,
            encoding="utf-8",
            xml_declaration=True,
        ),
    )
    for table_path, updated_entry in table_updates.items():
        entries[entry_index[table_path]] = updated_entry
    _write_workbook_entries(workbook_path, entries)
    return True


def read_xlsx_matrix(workbook_path):
    """Read the first worksheet as a rectangular text matrix."""
    workbook_path = Path(workbook_path)
    with ZipFile(workbook_path, "r") as archive:
        shared_strings = _shared_strings(archive)
        worksheet = ET.fromstring(
            archive.read(_first_worksheet_path(archive))
        )

    values_by_row = {}
    max_column = 0
    max_nonempty_row = 0
    fallback_row_number = 0
    for row in worksheet.findall(f".//{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row"):
        fallback_row_number += 1
        row_number = int(row.get("r") or fallback_row_number)
        row_values = values_by_row.setdefault(row_number, {})
        for cell in row.findall(f"./{{{MAIN_NS}}}c"):
            reference = cell.get("r")
            if not reference:
                continue
            column_number = _column_number(reference)
            value = _cell_text(cell, shared_strings)
            row_values[column_number] = value
            if value.strip() != "":
                max_column = max(max_column, column_number)
                max_nonempty_row = max(max_nonempty_row, row_number)

    if max_nonempty_row == 0 or max_column == 0:
        raise ValueError(f"XLSX worksheet is empty: {workbook_path}")

    matrix = [
        [
            values_by_row.get(row_number, {}).get(column_number, "")
            for column_number in range(1, max_column + 1)
        ]
        for row_number in range(1, max_nonempty_row + 1)
    ]
    normalized_headers = {
        _normalized_header(value)
        for value in matrix[0]
    }
    if not REQUIRED_HEADERS.issubset(normalized_headers):
        raise ValueError(
            f"XLSX experiment header is missing "
            f"{sorted(REQUIRED_HEADERS - normalized_headers)}: {workbook_path}"
        )
    for column_index, header in enumerate(matrix[0]):
        if _normalized_header(header) not in FOUR_DECIMAL_HEADERS:
            continue
        for row in matrix[1:]:
            text = str(row[column_index] or "").strip()
            if text.lower() in {"", "-", "none", "nan"}:
                continue
            try:
                row[column_index] = f"{Decimal(text):.4f}"
            except InvalidOperation:
                # Keep formulas or intentionally non-numeric notes unchanged.
                pass
    return matrix


def sync_result_matrix_to_workbook(workbook_path, matrix):
    """Copy result columns from an experiment matrix into its XLSX ledger.

    Sequence-plan columns remain owned by the workbook. Only BEV/3D/TD result
    columns and the ``average(n)`` label are written back.
    """
    workbook_path = Path(workbook_path)
    if not workbook_path.is_file():
        raise FileNotFoundError(
            f"Experiment workbook not found: {workbook_path}"
        )
    if not matrix or len(matrix) < 2:
        raise ValueError("Experiment result matrix is empty.")

    source_headers = [
        _normalized_header(value)
        for value in matrix[0]
    ]
    source_header_index = {
        header: index
        for index, header in enumerate(source_headers)
    }
    missing_source_headers = [
        header
        for header in ("group", "seed", *RESULT_HEADERS)
        if header not in source_header_index
    ]
    if missing_source_headers:
        raise ValueError(
            "Experiment result matrix is missing columns: "
            f"{missing_source_headers}"
        )

    source_rows = {}
    source_average_row = None
    group_source_index = source_header_index["group"]
    seed_source_index = source_header_index["seed"]
    for row in matrix[1:]:
        group = str(row[group_source_index] or "").strip()
        if group.lower().startswith("average"):
            source_average_row = row
            continue
        seed = str(row[seed_source_index] or "").strip()
        source_rows[(group, seed)] = row
    if source_average_row is None:
        raise ValueError(
            "Experiment result matrix has no average row."
        )

    with ZipFile(workbook_path, "r") as archive:
        worksheet_path = _first_worksheet_path(archive)
        shared_strings = _shared_strings(archive)
        entries = [
            (info, archive.read(info.filename))
            for info in archive.infolist()
        ]

    entry_index = {
        info.filename: index
        for index, (info, _data) in enumerate(entries)
    }
    worksheet_index = entry_index[worksheet_path]
    worksheet_info, worksheet_data = entries[worksheet_index]
    worksheet = ET.fromstring(worksheet_data)
    worksheet_rows = list(worksheet.findall(
        f".//{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row"
    ))
    if not worksheet_rows:
        raise ValueError(
            f"XLSX worksheet contains no rows: {workbook_path}"
        )
    worksheet_rows.sort(key=lambda row: int(row.get("r") or 0))

    workbook_header_cells = _row_cells_by_column(worksheet_rows[0])
    workbook_header_index = {
        _normalized_header(_cell_text(cell, shared_strings)): column_number
        for column_number, cell in workbook_header_cells.items()
    }
    missing_workbook_headers = [
        header
        for header in ("group", "seed", *RESULT_HEADERS)
        if header not in workbook_header_index
    ]
    if missing_workbook_headers:
        raise ValueError(
            "Experiment workbook is missing result columns: "
            f"{missing_workbook_headers}"
        )

    group_workbook_column = workbook_header_index["group"]
    seed_workbook_column = workbook_header_index["seed"]
    matched_source_keys = set()
    average_matched = False
    changed = False

    for worksheet_row in worksheet_rows[1:]:
        cells = _row_cells_by_column(worksheet_row)
        group_cell = cells.get(group_workbook_column)
        group = (
            ""
            if group_cell is None
            else str(_cell_text(group_cell, shared_strings) or "").strip()
        )
        if group.lower().startswith("average"):
            source_row = source_average_row
            average_matched = True
        else:
            seed_cell = cells.get(seed_workbook_column)
            seed = (
                ""
                if seed_cell is None
                else str(_cell_text(seed_cell, shared_strings) or "").strip()
            )
            source_row = source_rows.get((group, seed))
            if source_row is None:
                continue
            matched_source_keys.add((group, seed))

        row_number = int(worksheet_row.get("r") or 0)
        if row_number <= 0:
            raise ValueError(
                f"Invalid worksheet row number in {workbook_path}"
            )

        if group.lower().startswith("average"):
            desired_group = str(
                source_row[group_source_index] or ""
            ).strip() or "-"
            if group != desired_group:
                if group_cell is None:
                    group_cell = ET.SubElement(
                        worksheet_row,
                        f"{{{MAIN_NS}}}c",
                    )
                    group_cell.set(
                        "r",
                        f"{_column_letters(group_workbook_column)}{row_number}",
                    )
                    cells[group_workbook_column] = group_cell
                _set_inline_string(group_cell, desired_group)
                changed = True

        for header in RESULT_HEADERS:
            source_index = source_header_index[header]
            workbook_column = workbook_header_index[header]
            desired_value = str(source_row[source_index] or "").strip() or "-"
            cell = cells.get(workbook_column)
            current_value = (
                ""
                if cell is None
                else str(_cell_text(cell, shared_strings) or "").strip()
            )
            if current_value == desired_value:
                continue
            if cell is None:
                cell = ET.SubElement(
                    worksheet_row,
                    f"{{{MAIN_NS}}}c",
                )
                cell.set(
                    "r",
                    f"{_column_letters(workbook_column)}{row_number}",
                )
                cells[workbook_column] = cell
            _set_inline_string(cell, desired_value)
            changed = True

        if changed:
            for cell in list(worksheet_row.findall(f"./{{{MAIN_NS}}}c")):
                worksheet_row.remove(cell)
            for column_number in sorted(cells):
                worksheet_row.append(cells[column_number])

    unmatched_source_keys = set(source_rows) - matched_source_keys
    if unmatched_source_keys:
        raise RuntimeError(
            "Experiment TXT contains rows not present in the XLSX workbook: "
            f"{sorted(unmatched_source_keys)}"
        )
    if not average_matched:
        raise RuntimeError(
            f"Experiment workbook has no average row: {workbook_path}"
        )

    if changed:
        entries[worksheet_index] = (
            worksheet_info,
            ET.tostring(
                worksheet,
                encoding="utf-8",
                xml_declaration=True,
            ),
        )
        _write_workbook_entries(workbook_path, entries)
    layout_changed = normalize_workbook_layout(workbook_path)
    return bool(changed or layout_changed)


def render_xlsx_as_txt(workbook_path):
    return _format_txt_experiment_matrix(read_xlsx_matrix(workbook_path))


def _atomic_write_text(path, text):
    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            output_file.write(text)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def sync_workbook(workbook_path):
    """Write one workbook to its same-name TXT file; return whether it changed."""
    workbook_path = Path(workbook_path)
    txt_path = workbook_path.with_suffix(".txt")
    rendered = render_xlsx_as_txt(workbook_path)
    current = (
        txt_path.read_text(encoding="utf-8-sig")
        if txt_path.is_file()
        else None
    )
    if current == rendered:
        return txt_path, False
    _atomic_write_text(txt_path, rendered)
    return txt_path, True


def _workbooks(experiment_dir, pattern):
    return tuple(
        path
        for path in sorted(Path(experiment_dir).glob(pattern))
        if path.is_file() and not path.name.startswith("~$")
    )


def _file_stamp(path):
    stat = Path(path).stat()
    return stat.st_mtime_ns, stat.st_size


def _wait_until_stable(path, timeout_seconds=5.0):
    deadline = time.monotonic() + float(timeout_seconds)
    previous = None
    while time.monotonic() < deadline:
        current = _file_stamp(path)
        if current == previous:
            return current
        previous = current
        time.sleep(0.1)
    return _file_stamp(path)


def _sync_with_retry(workbook_path, attempts=20):
    last_error = None
    for _attempt in range(int(attempts)):
        try:
            _wait_until_stable(workbook_path)
            layout_changed = normalize_workbook_layout(workbook_path)
            output_path, txt_changed = sync_workbook(workbook_path)
            return output_path, bool(layout_changed or txt_changed)
        except (BadZipFile, ET.ParseError, EOFError, OSError) as error:
            last_error = error
            time.sleep(0.2)
    raise RuntimeError(
        f"XLSX remained unreadable after {attempts} attempts: {workbook_path}"
    ) from last_error


def _lock_watcher():
    digest = hashlib.sha256(str(PROJECT_ROOT).encode("utf-8")).hexdigest()[:12]
    lock_path = Path(tempfile.gettempdir()) / (
        f"mvrss_experiment_xlsx_sync_{digest}.lock"
    )
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None
    return lock_file


def watch(experiment_dir, pattern, poll_seconds):
    lock_file = _lock_watcher()
    if lock_file is None:
        print(
            "Experiment XLSX watcher already running for this workspace.",
            flush=True,
        )
        return 0

    print("Experiment XLSX watcher starting.", flush=True)
    known_stamps = {}
    try:
        for workbook_path in _workbooks(experiment_dir, pattern):
            stamp = _file_stamp(workbook_path)
            known_stamps[workbook_path] = stamp
            txt_path = workbook_path.with_suffix(".txt")
            if (
                not txt_path.exists()
                or workbook_path.stat().st_mtime_ns > txt_path.stat().st_mtime_ns
            ):
                output_path, changed = _sync_with_retry(workbook_path)
                status = "updated" if changed else "already matched"
                print(
                    f"[startup] {workbook_path.name} -> "
                    f"{output_path.name} ({status})",
                    flush=True,
                )
        print("Experiment XLSX watcher ready.", flush=True)

        while True:
            current_paths = set(_workbooks(experiment_dir, pattern))
            for missing_path in set(known_stamps) - current_paths:
                known_stamps.pop(missing_path, None)
            for workbook_path in current_paths:
                try:
                    stamp = _file_stamp(workbook_path)
                except FileNotFoundError:
                    continue
                previous_stamp = known_stamps.get(workbook_path)
                if previous_stamp == stamp:
                    continue
                try:
                    output_path, changed = _sync_with_retry(workbook_path)
                    stable_stamp = _file_stamp(workbook_path)
                    known_stamps[workbook_path] = stable_stamp
                    status = "updated" if changed else "already matched"
                    print(
                        f"[save] {workbook_path.name} -> "
                        f"{output_path.name} ({status})",
                        flush=True,
                    )
                except Exception as error:
                    known_stamps[workbook_path] = stamp
                    print(
                        f"[error] {workbook_path.name}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
            time.sleep(float(poll_seconds))
    except KeyboardInterrupt:
        print("Experiment XLSX watcher stopped.", flush=True)
        return 0
    finally:
        lock_file.close()


def check(experiment_dir, pattern):
    all_matched = True
    for workbook_path in _workbooks(experiment_dir, pattern):
        txt_path = workbook_path.with_suffix(".txt")
        expected = render_xlsx_as_txt(workbook_path)
        actual = (
            txt_path.read_text(encoding="utf-8-sig")
            if txt_path.is_file()
            else None
        )
        matched = actual == expected
        all_matched = all_matched and matched
        print(
            f"{workbook_path.name}: "
            f"{'MATCH' if matched else 'DIFFERENT'}",
            flush=True,
        )
    return 0 if all_matched else 1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize experiments/*_experiments.xlsx to same-name TXT files."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
    )
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.watch and args.check:
        raise ValueError("--watch and --check cannot be used together")
    if args.check:
        return check(args.experiment_dir, args.pattern)
    if args.watch:
        return watch(
            args.experiment_dir,
            args.pattern,
            args.poll_seconds,
        )

    for workbook_path in _workbooks(args.experiment_dir, args.pattern):
        output_path, changed = sync_workbook(workbook_path)
        status = "updated" if changed else "already matched"
        print(
            f"{workbook_path.name} -> {output_path.name} ({status})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
