from pathlib import Path

import pandas as pd
import pytest

from scripts.advanced_segmentation import prepare, run, summarize, wilson_interval


def sample() -> pd.DataFrame:
    return pd.DataFrame({
        "Revenue": [True, False, True, False, False, True],
        "VisitorType": ["New_Visitor", "Returning_Visitor", "New_Visitor", "Other", "Returning_Visitor", "Returning_Visitor"],
        "Weekend": [False, True, True, False, False, True],
        "ProductRelated": [2, 8, 30, 0, 15, 25],
        "TrafficType": [1, 2, 3, 4, 1, 3],
        "ProductRelated_Duration": [20, 80, 210, 0, 25, 155],
        "BounceRates": [0, 0.02, 0.01, 0.08, 0.03, 0.01],
        "ExitRates": [0.01, 0.06, 0.01, 0.09, 0.04, 0.02],
        "Region": [1, 2, 2, 1, 1, 3],
        "Browser": [1, 1, 2, 2, 3, 3],
        "OperatingSystems": [2, 1, 2, 1, 2, 3],
        "SpecialDay": [0, 0, 0.5, 0, 0.8, 0],
    })


def test_wilson_interval_properties():
    low, high = wilson_interval(50, 100)
    assert low < 50 < high
    assert high - 50 == pytest.approx(50 - low)
    assert wilson_interval(0, 100)[0] >= 0
    assert wilson_interval(100, 100)[1] <= 100.000001
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


def test_segments_reconcile_and_flag_small_groups():
    df = prepare(sample())
    table = summarize(df, "Region")
    assert table["sessions"].sum() == 6
    assert table["purchases"].sum() == 3
    assert table["small_sample_flag"].all()
    assert table["conversion_rate_pct"].between(0, 100).all()
    assert table["wilson_95_lower_pct"].le(table["conversion_rate_pct"]).all()


def test_invalid_specialday_is_rejected():
    df = sample()
    df.loc[0, "SpecialDay"] = 1.5
    with pytest.raises(ValueError, match="SpecialDay"):
        prepare(df)


def test_advanced_pipeline_creates_six_figures(tmp_path: Path):
    source = tmp_path / "samples.csv"
    sample().to_csv(source, index=False)
    output = tmp_path / "advanced"
    result = run(source, output)
    assert result == {"sessions": 6, "purchases": 3, "tables": 8, "charts": 6}
    assert len(list((output / "tables").glob("*.csv"))) == 8
    assert len(list((output / "charts").glob("*.png"))) == 6
