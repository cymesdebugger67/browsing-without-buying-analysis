from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
EXPECTED_RAW_SHA256 = "B3055EE355F59134D851D32641183CB4A8B45DEF7124D2F50442A042F358E0D9"


def verify(root: Path = ROOT, expected_sha: str = EXPECTED_RAW_SHA256) -> dict:
    source = root / "data/raw/online_shoppers_intention.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest().upper()
    if digest != expected_sha:
        raise AssertionError("Source SHA-256 differs from the approved raw UCI CSV")
    output = root / "outputs"
    audit = json.loads((output / "audit/data_audit.json").read_text(encoding="utf-8"))
    model = json.loads((output / "modeling/metrics.json").read_text(encoding="utf-8"))
    if any([
        audit["sha256"] != digest, audit["sessions"] != 12330,
        audit["purchase_sessions"] != 1908, audit["columns"] != 18,
        audit["exact_duplicate_rows"] != 125,
        model["source_sessions"] != 12330, model["source_purchases"] != 1908,
        model["group_overlap"] != 0,
        model["training_sessions"] + model["test_sessions"] != 12330,
        not 0.05 <= model["train_cv_selected_threshold"] <= 0.50,
        model["selected_threshold_heldout_metrics"]["test_sessions"] != model["test_sessions"],
    ]):
        raise AssertionError("Known-source audit or model counts do not reconcile")
    key_charts = [
        "charts/conversion_by_visitor.png", "charts/conversion_by_month.png",
        "charts/conversion_by_page_depth.png", "deep_dive/charts/visitor_depth_heatmap.png",
        "deep_dive/charts/high_volume_traffic_codes.png", "deep_dive/charts/conversion_by_bounce_band.png",
    ]
    key_charts += [f"advanced_segmentation/charts/{name}" for name in (
        "purchase_share_donut.png", "visitor_mix_donut.png", "page_depth_donut.png",
        "region_intervals.png", "browser_intervals.png", "product_pages_by_outcome.png")]
    key_charts += [f"modeling/charts/{name}" for name in (
        "precision_recall.png", "roc_curves.png", "confusion_matrix.png",
        "confusion_matrix_cv_threshold.png", "calibration.png", "permutation_importance.png")]
    for name in key_charts:
        path = output / name
        if not path.is_file() or path.stat().st_size < 100:
            raise AssertionError(f"Missing or empty chart: {name}")
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    if len(notebooks) != 5:
        raise AssertionError("Expected exactly five notebooks")
    for rel in ["advanced_segmentation/summary.md", "modeling/model_report.md",
                "advanced_case_study.md", "portfolio_case_study.md"]:
        if not (output / rel).is_file():
            raise AssertionError(f"Missing report: {rel}")
    table_dirs = [output / "advanced_segmentation/tables", output / "modeling/tables"]
    if len(list(table_dirs[0].glob("*.csv"))) != 8 or len(list(table_dirs[1].glob("*.csv"))) != 5:
        raise AssertionError("Expected eight segment tables and five model tables")
    for path in table_dirs[0].glob("conversion_by_*.csv"):
        df = pd.read_csv(path)
        if (int(df["sessions"].sum()) != 12330 or int(df["purchases"].sum()) != 1908):
            raise AssertionError(f"Table does not reconcile: {path.name}")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if "Cyril James Lunzaga" not in readme:
        raise AssertionError("Owner attribution missing")
    links = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', readme)
    missing_links = [s for s in links if not s.startswith(("http:", "https:", "#", "mailto:"))
                     and not (root / s.split("#", 1)[0]).exists()]
    if missing_links:
        raise AssertionError("README links missing: " + ", ".join(missing_links))
    tracked = subprocess.check_output(["git", "ls-files", "--", "data/raw/*", ".venv/*"],
                                      cwd=root, text=True).splitlines()
    forbidden = [x for x in tracked if x != "data/raw/.gitkeep"]
    if forbidden:
        raise AssertionError("Sensitive/local files are tracked: " + ", ".join(forbidden))
    return {"source_sha256": digest, "sessions": audit["sessions"],
            "purchases": audit["purchase_sessions"], "charts": len(key_charts),
            "notebooks": len(notebooks), "advanced_tables": 13,
            "readme_links_checked": len(links)}


def main() -> None:
    summary = verify()
    print("PASS - Source hash and reference counts verified")
    print("PASS - Training and test totals reconcile; no exact feature-group overlap")
    print(f"PASS - {summary['charts']} charts, {summary['notebooks']} notebooks, {summary['advanced_tables']} advanced tables")
    print(f"PASS - {summary['readme_links_checked']} README links inspected; excluded raw data absent from Git index")
    print("PASS - REAL-DATA RELEASE ARTIFACT AUDIT")


if __name__ == "__main__":
    main()
