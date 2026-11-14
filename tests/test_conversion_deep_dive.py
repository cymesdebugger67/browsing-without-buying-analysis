from pathlib import Path

import pandas as pd
import pytest

from scripts.conversion_deep_dive import (
    BOUNCE_LABELS,
    DEPTH_LABELS,
    conversion_table,
    engagement_table,
    prepare_sessions,
    run,
)


def fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "Revenue": [False, False, True, True, False, True],
        "VisitorType": ["Returning_Visitor", "New_Visitor", "New_Visitor", "Returning_Visitor", "Other", "Returning_Visitor"],
        "Weekend": [False, True, False, True, False, True],
        "ProductRelated": [0, 5, 8, 35, 21, 1],
        "TrafficType": [1, 2, 1, 2, 3, 1],
        "ProductRelated_Duration": [0.0, 24.0, 150.0, 2300.0, 100.0, 80.0],
        "BounceRates": [0.0, 0.01, 0.04, 0.09, 0.0, 0.001],
        "ExitRates": [0.2, 0.06, 0.02, 0.01, 0.1, 0.01],
    })


def test_classification_and_reconciliation():
    sessions = prepare_sessions(fixture())
    assert sessions["page_depth"].astype(str).tolist() == [
        DEPTH_LABELS[0], DEPTH_LABELS[1], DEPTH_LABELS[2], DEPTH_LABELS[3], DEPTH_LABELS[3], DEPTH_LABELS[1]
    ]
    assert set(sessions["bounce_band"].astype(str)).issubset(set(BOUNCE_LABELS))
    result = conversion_table(sessions, ["VisitorType", "page_depth"])
    assert result["sessions"].sum() == 6
    assert result["purchases"].sum() == 3
    assert result["nonpurchase_sessions"].sum() == 3
    assert result["interpret_with_caution"].all()


def test_identical_rows_are_retained():
    sample = fixture()
    combined = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)
    data = prepare_sessions(combined)
    assert len(data) == 7
    assert data["purchased"].sum() == 3


def test_invalid_outcomes_rejected():
    df = fixture()
    df["Revenue"] = df["Revenue"].astype("object")
    df.loc[0, "Revenue"] = "unknown"
    with pytest.raises(ValueError, match="Revenue"):
        prepare_sessions(df)


def test_invalid_rates_rejected():
    df = fixture()
    df.loc[0, "BounceRates"] = 1.2
    with pytest.raises(ValueError, match="between 0 and 1"):
        prepare_sessions(df)


def test_metric_summaries_use_full_data():
    df = prepare_sessions(fixture())
    stats = engagement_table(df)
    assert len(stats) == 8
    product_rows = stats.loc[stats.metric.eq("ProductRelated")].set_index("outcome")
    assert product_rows.at["Purchasing", "median"] == 8
    assert product_rows.at["Non-purchasing", "median"] == 5


def test_pipeline_writes_outputs(tmp_path: Path):
    source = tmp_path / "sample.csv"
    fixture().to_csv(source, index=False)
    output = tmp_path / "deep_dive"
    result = run(source, output)
    assert result == {"sessions": 6, "purchases": 3, "tables": 5, "charts": 3}
    assert len(list((output / "tables").glob("*.csv"))) == 5
    assert len(list((output / "charts").glob("*.png"))) == 3
    assert "50.00%" in (output / "summary.md").read_text(encoding="utf-8")
