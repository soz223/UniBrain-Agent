"""Import the reviewed workbook without changing it."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_rows(archive, sheet):
    strings = ["".join(node.itertext()) for node in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]
    rows = []
    for node in ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet}.xml")).findall(".//s:row", NS):
        row = {}
        for cell in node.findall("s:c", NS):
            value = cell.find("s:v", NS)
            if value is not None:
                row[re.sub(r"\d", "", cell.get("r"))] = strings[int(value.text)] if cell.get("t") == "s" else value.text
        rows.append(row)
    return rows


def convert(path):
    with ZipFile(path) as archive:
        matrix, details, pending = [read_rows(archive, sheet) for sheet in (1, 2, 3)]
    stages = {}
    for row in matrix:
        for column in ("C", "D", "E"):
            match = re.search(r"\bM\d{2}\b", row.get(column, ""))
            if match:
                stages.setdefault(match.group(), []).append(row["A"])
    fields = {"B": "publication", "C": "interface", "D": "prior_check", "E": "adaptation", "F": "limitations",
              "G": "prior_evidence", "H": "paper", "I": "code"}
    records = [{"id": row["A"], "stages": stages[row["A"]], **{name: row.get(col, "") for col, name in fields.items()}}
               for row in details if re.fullmatch(r"M\d{2}", row.get("A", ""))]
    excluded = [{"stage": row["A"], "publication": row["B"], "status": row["C"], "reason": row["D"], "paper": row.get("E")}
                for row in pending if row.get("D") and row.get("A") != "Stage"]
    return {"source": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": records, "pending": excluded}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(convert(args.workbook), indent=2, ensure_ascii=False), encoding="utf-8")
