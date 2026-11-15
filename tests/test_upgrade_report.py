import json

import pandas as pd
import pytest

from scripts.build_upgrade_report import build, load_evidence


def create_sample_results(root):
    for folder in ("audit", "statistical_review", "advanced_segmentation/tables", "modeling/tables"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "audit/data_audit.json").write_text(json.dumps({
        "sessions": 20, "purchase_sessions": 6, "nonpurchase_sessions": 14,
        "conversion_rate_pct": 30, "columns": 18, "missing_cells": 0,
        "exact_duplicate_rows": 0, "duplicate_sensitivity_pp": 0, "sha256": "abc",
    }), encoding="utf-8")
    pd.DataFrame([{"new_conversion_pct": 40, "new_purchases": 4, "new_sessions": 10,
                   "returning_conversion_pct": 20, "returning_purchases": 2, "returning_sessions": 10,
                   "difference_percentage_points": 20,
                   "bootstrap_95pct_lower_pp": 1, "bootstrap_95pct_upper_pp": 39}]).to_csv(
                       root / "statistical_review/visitor_conversion_gap.csv", index=False)
    pd.DataFrame([{"variable": "page_depth", "cramers_v": 0.21}]).to_csv(root / "statistical_review/categorical_tests.csv", index=False)
    pd.DataFrame([{"variable": "ExitRates", "rank_biserial_purchasing_vs_nonpurchasing": -0.35}]).to_csv(
        root / "statistical_review/numeric_tests.csv", index=False)
    pd.DataFrame([{"Region": 1, "sessions": 20, "purchases": 6}]).to_csv(
        root / "advanced_segmentation/tables/conversion_by_region.csv", index=False)
    pd.DataFrame([
        {"page_depth": "1–5 pages", "sessions": 10, "purchases": 2, "conversion_rate_pct": 20.0},
        {"page_depth": "21+ pages", "sessions": 10, "purchases": 4, "conversion_rate_pct": 40.0},
    ]).to_csv(root / "advanced_segmentation/tables/conversion_by_product_depth.csv", index=False)
    pd.DataFrame([
        {"weekend_label": "Weekday", "sessions": 15, "purchases": 3, "conversion_rate_pct": 20.0},
        {"weekend_label": "Weekend", "sessions": 5, "purchases": 3, "conversion_rate_pct": 60.0},
    ]).to_csv(root / "advanced_segmentation/tables/conversion_by_weekend.csv", index=False)
    pd.DataFrame([
        {"special_day_band": "Zero proximity", "sessions": 12, "purchases": 4, "conversion_rate_pct": 100/3},
        {"special_day_band": "Nonzero proximity", "sessions": 8, "purchases": 2, "conversion_rate_pct": 25.0},
    ]).to_csv(root / "advanced_segmentation/tables/conversion_by_special_day.csv", index=False)
    models = [
        {"model": "Majority-prior baseline", "average_precision": .3, "roc_auc": .5, "precision": 0, "recall": 0, "brier_score": .2},
        {"model": "Logistic regression", "average_precision": .6, "roc_auc": .72, "precision": .6, "recall": .7, "brier_score": .1},
        {"model": "Random forest", "average_precision": .5, "roc_auc": .66, "precision": .5, "recall": .6, "brier_score": .14},
    ]
    pd.DataFrame(models).to_csv(root / "modeling/tables/model_comparison.csv", index=False)
    pd.DataFrame([{"feature_set": "Without PageValues"}, {"feature_set": "With PageValues"}]).to_csv(
        root / "modeling/tables/pagevalues_sensitivity.csv", index=False)
    (root / "modeling/metrics.json").write_text(json.dumps({
        "source_sessions": 20, "source_purchases": 6, "training_sessions": 15,
        "test_sessions": 5, "chosen_model": "Logistic regression",
        "selected_cv_average_precision": .59, "chosen_test_ap_interval_95": [.3, .7],
        "train_cv_selected_threshold": .25,
        "selected_threshold_heldout_metrics": {"precision": .75, "recall": .55, "f1": .635},
    }), encoding="utf-8")


def test_report_uses_evidence_and_author(tmp_path):
    out = tmp_path / "outputs"
    create_sample_results(out)
    readme, report = build(out, tmp_path / "README.md")
    text = readme.read_text(encoding="utf-8")
    assert "Cyril James Lunzaga" in text
    assert "20" in text and "0.600" in text
    assert "PageValues" in text
    assert "training-only out-of-fold F1" in text
    assert "0.250" in text
    assert "## Project objectives" in text
    assert "## Additional quantified patterns" in text
    assert "40.00%" in text and "20.00%" in text
    assert "60.00%" in text and "25.00%" in text
    assert "not a causal weekend effect" in text
    assert "## Implications, limitations and conclusion" in text
    assert report.is_file()
    case = report.read_text(encoding="utf-8")
    assert "## Additional descriptive comparisons" in case
    assert "(advanced_segmentation/tables/conversion_by_weekend.csv)" in case


def test_report_rejects_inconsistent_source_counts(tmp_path):
    out = tmp_path / "outputs"
    create_sample_results(out)
    audit_file = out / "audit/data_audit.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    audit["sessions"] = 21
    audit_file.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="reconcile"):
        load_evidence(out)
