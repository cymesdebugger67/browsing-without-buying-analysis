import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_portfolio_report import build_report


def make_results(root: Path):
    for directory in ["audit", "statistical_review", "deep_dive/tables"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    audit = {"sessions": 10, "purchase_sessions": 4, "missing_cells": 0,
             "exact_duplicate_rows": 1, "duplicate_sensitivity_pp": 1.1}
    (root / "audit/data_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    pd.DataFrame([{"new_conversion_pct": 50., "returning_conversion_pct": 40.,
                   "new_purchases": 2, "new_sessions": 4,
                   "returning_purchases": 2, "returning_sessions": 5,
                   "difference_percentage_points": 10.,
                   "bootstrap_95pct_lower_pp": 2., "bootstrap_95pct_upper_pp": 41.}]).to_csv(
                       root / "statistical_review/visitor_conversion_gap.csv", index=False)
    pd.DataFrame([{"variable": x, "sessions": 10, "cramers_v": v} for x, v in
                  [("page_depth", .2), ("VisitorType", .1), ("weekend_label", .02),
                   ("bounce_band", .15)]]).to_csv(root / "statistical_review/categorical_tests.csv", index=False)
    pd.DataFrame([{"variable": x, "rank_biserial_purchasing_vs_nonpurchasing": v} for x, v in
                  [("ProductRelated", .3), ("ProductRelated_Duration", .35),
                   ("BounceRates", -.2), ("ExitRates", -.4)]]).to_csv(
                       root / "statistical_review/numeric_tests.csv", index=False)
    pd.DataFrame([{"metric": "ProductRelated", "outcome": x, "median": y} for x, y in
                  [("Purchasing", 29), ("Non-purchasing", 16)]]).to_csv(
                       root / "deep_dive/tables/engagement_by_outcome.csv", index=False)
    pd.DataFrame([
        {"VisitorType": "New_Visitor", "page_depth": "21+ pages", "sessions": 4,
         "purchases": 2, "conversion_rate_pct": 50.},
        {"VisitorType": "Returning_Visitor", "page_depth": "21+ pages", "sessions": 6,
         "purchases": 2, "conversion_rate_pct": 33.33},
    ]).to_csv(root / "deep_dive/tables/conversion_by_visitor_depth.csv", index=False)


def test_portfolio_report_reads_actual_outputs(tmp_path):
    make_results(tmp_path)
    report = build_report(tmp_path)
    content = report.read_text(encoding="utf-8")
    assert "10 sessions" in content
    assert "40.00%" in content
    assert "+10.00 percentage points" in content
    assert "Cyril James Lunzaga" in content
    assert "charts/conversion_by_visitor.png" in content


def test_portfolio_report_rejects_nonreconciling_counts(tmp_path):
    make_results(tmp_path)
    path = tmp_path / "audit/data_audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    audit["sessions"] = 11
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="reconcile"):
        build_report(tmp_path)
