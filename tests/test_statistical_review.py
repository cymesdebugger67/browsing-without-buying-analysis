from pathlib import Path

import pandas as pd
import pytest

from scripts.statistical_review import analyze, benjamini_hochberg, run


def example() -> pd.DataFrame:
    rows = []
    for i in range(80):
        rows.append({
            "Revenue": i % 4 == 0,
            "VisitorType": "New_Visitor" if i < 40 else "Returning_Visitor",
            "Weekend": i % 2 == 0,
            "ProductRelated": 3 + i % 30,
            "TrafficType": 1 + i % 3,
            "ProductRelated_Duration": 10.0 + i * 3,
            "BounceRates": (i % 7) / 100,
            "ExitRates": (i % 7) / 100 + 0.02,
        })
    return pd.DataFrame(rows)


def test_bh_adjustment_known_values():
    result = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert result == pytest.approx([0.04, 0.0533333333, 0.0533333333, 0.20])
    with pytest.raises(ValueError, match="P-values"):
        benjamini_hochberg([1.2])


def test_statistical_tables_and_gap():
    categorical, numeric, gap = analyze(example())
    assert len(categorical) == 4
    assert len(numeric) == 4
    assert (categorical.sessions == 80).all()
    assert (numeric.purchasing_sessions == 20).all()
    assert (numeric.nonpurchasing_sessions == 60).all()
    assert (categorical.cramers_v.between(0, 1)).all()
    assert (numeric.rank_biserial_purchasing_vs_nonpurchasing.abs() <= 1).all()
    assert (categorical.p_value_bh_8tests >= categorical.p_value).all()
    assert (numeric.p_value_bh_8tests >= numeric.p_value).all()
    assert gap["new_sessions"] + gap["returning_sessions"] == 80
    assert gap["bootstrap_95pct_lower_pp"] <= gap["bootstrap_95pct_upper_pp"]


def test_bootstrap_is_deterministic():
    _, _, a = analyze(example())
    _, _, b = analyze(example())
    assert a == b


def test_pipeline_writes_four_outputs(tmp_path: Path):
    source = tmp_path / "sessions.csv"
    example().to_csv(source, index=False)
    out = tmp_path / "statistical_review"
    result = run(source, out)
    assert result == {"sessions": 80, "categorical_tests": 4, "numeric_tests": 4, "outputs": 4}
    assert len(list(out.glob("*.csv"))) == 3
    report = (out / "statistical_summary.md").read_text(encoding="utf-8")
    assert "80" in report and "Cramér" in report


def test_invalid_outcome_fails():
    sample = example()
    sample["Revenue"] = sample["Revenue"].astype("object")
    sample.loc[0, "Revenue"] = "unknown"
    with pytest.raises(ValueError, match="Revenue"):
        analyze(sample)
