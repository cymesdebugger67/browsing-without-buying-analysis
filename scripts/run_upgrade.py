from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    "scripts.audit_sessions",
    "analysis",
    "scripts.conversion_deep_dive",
    "scripts.statistical_review",
    "scripts.advanced_segmentation",
    "scripts.model_purchase_intent",
    "scripts.build_portfolio_report",
    "scripts.build_upgrade_report",
]
REQUIRED_OUTPUTS = [
    "outputs/audit/data_audit.json", "outputs/findings.md",
    "outputs/deep_dive/summary.md", "outputs/statistical_review/statistical_summary.md",
    "outputs/advanced_segmentation/summary.md", "outputs/advanced_case_study.md",
    "outputs/modeling/metrics.json", "outputs/modeling/model_report.md",
    "outputs/modeling/charts/precision_recall.png", "outputs/modeling/charts/roc_curves.png",
    "outputs/modeling/charts/confusion_matrix.png", "outputs/modeling/charts/confusion_matrix_cv_threshold.png", "outputs/modeling/charts/calibration.png",
    "outputs/modeling/charts/permutation_importance.png",
    "outputs/advanced_segmentation/charts/purchase_share_donut.png",
    "outputs/advanced_segmentation/charts/visitor_mix_donut.png",
    "outputs/advanced_segmentation/charts/page_depth_donut.png",
    "outputs/advanced_segmentation/charts/region_intervals.png",
    "outputs/advanced_segmentation/charts/browser_intervals.png",
    "outputs/advanced_segmentation/charts/product_pages_by_outcome.png",
]


def run_process(args: list[str]) -> None:
    print("RUN", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def run_notebooks() -> None:
    import nbformat
    from nbclient import NotebookClient

    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if len(paths) != 5:
        raise AssertionError(f"Expected five notebooks, found {len(paths)}")
    for path in paths:
        print(f"NOTEBOOK {path.name}", flush=True)
        notebook = nbformat.read(str(path), as_version=4)
        nbformat.validate(notebook)
        NotebookClient(notebook, timeout=300, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
    print("PASS - Five notebooks validated and executed without saving outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the conversion analytics upgrade")
    parser.add_argument("--verify", action="store_true", help="Execute notebooks and tests after generating reports")
    args = parser.parse_args()
    source = ROOT / "data/raw/online_shoppers_intention.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Original raw dataset missing: {source}")
    for stage in STAGES:
        run_process([sys.executable, "-m", stage])
    missing = [item for item in REQUIRED_OUTPUTS if not (ROOT / item).is_file()]
    if missing:
        raise AssertionError("Missing expected outputs: " + ", ".join(missing))
    print(f"PASS - {len(STAGES)} analytical stages; {len(REQUIRED_OUTPUTS)} required report/figure files")
    if args.verify:
        (ROOT / ".venv").mkdir(exist_ok=True)
        run_process([sys.executable, "-m", "pytest", "tests", "-q", "--basetemp=.venv/pytest-tmp"])
        run_process([sys.executable, "-m", "pip", "check"])
        run_notebooks()
    print("PASS - Browsing Without Buying upgrade pipeline completed")


if __name__ == "__main__":
    main()
