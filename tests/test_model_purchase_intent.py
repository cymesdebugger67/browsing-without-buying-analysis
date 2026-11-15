from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.model_purchase_intent import (
    FEATURES, bootstrap_ap, evaluate, load_model_data, run, split_sessions, select_training_threshold,
)


def example(n: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(13)
    pages = rng.integers(1, 85, n)
    rates = rng.uniform(0.002, 0.17, n)
    revenue = rng.random(n) < np.clip(0.09 + 0.0035 * pages - rates, 0.03, 0.5)
    return pd.DataFrame({
        "Administrative": rng.integers(0, 9, n),
        "Administrative_Duration": rng.uniform(0, 100, n),
        "Informational": rng.integers(0, 5, n),
        "Informational_Duration": rng.uniform(0, 80, n),
        "ProductRelated": pages,
        "ProductRelated_Duration": pages * rng.uniform(3, 25, n),
        "BounceRates": rates,
        "ExitRates": np.minimum(1, rates + rng.uniform(0, 0.05, n)),
        "PageValues": rng.uniform(0, 5, n) * revenue,
        "SpecialDay": rng.choice([0, 0.2, 0.8], n),
        "Month": rng.choice(["Feb", "May", "Nov"], n),
        "OperatingSystems": rng.integers(1, 4, n),
        "Browser": rng.integers(1, 5, n),
        "Region": rng.integers(1, 5, n),
        "TrafficType": rng.integers(1, 6, n),
        "VisitorType": rng.choice(["New_Visitor", "Returning_Visitor"], n),
        "Weekend": rng.choice([True, False], n),
        "Revenue": revenue,
    })


def test_page_values_excluded_and_duplicates_grouped():
    data = example(250)
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    x, y, groups = load_model_data(data)
    assert "PageValues" not in x.columns
    assert len(FEATURES) == 16
    assert groups[0] == groups[-1]
    x2, y2, groups2 = load_model_data(data, True)
    assert "PageValues" in x2.columns
    assert np.array_equal(groups, groups2)
    assert np.array_equal(y, y2)


def test_split_is_group_isolated():
    x, y, groups = load_model_data(example())
    train, test = split_sessions(y, groups)
    assert len(train) + len(test) == len(x)
    assert not set(groups[train]).intersection(groups[test])
    assert np.unique(y[train]).size == np.unique(y[test]).size == 2


def test_metrics_known_outcomes_and_bootstrap_repeatability():
    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.8, 0.6, 0.2, 0.7, 0.4])
    result = evaluate(y, p)
    assert result["true_positive"] == 3
    assert result["false_positive"] == 0
    assert result["average_precision"] == pytest.approx(1.0)
    assert bootstrap_ap(y, p, 30) == bootstrap_ap(y, p, 30)
    with pytest.raises(ValueError):
        evaluate(y, np.array([-0.1, 0.8, 0.6, 0.2, 0.7, 0.4]))


def test_modeling_pipeline_artifacts(tmp_path: Path):
    source = tmp_path / "example.csv"
    example().to_csv(source, index=False)
    output = tmp_path / "modeling"
    result = run(source, output)
    assert result["source_sessions"] == 420
    assert result["training_sessions"] + result["test_sessions"] == 420
    assert result["group_overlap"] == 0
    assert 0 <= result["selected_cv_average_precision"] <= 1
    assert len(list((output / "tables").glob("*.csv"))) == 5
    assert len(list((output / "charts").glob("*.png"))) == 6
    comparison = pd.read_csv(output / "tables/model_comparison.csv")
    assert len(comparison) == 3
    assert comparison["test_sessions"].nunique() == 1
    assert (comparison["average_precision"].between(0, 1)).all()
    assert 0.05 <= result["train_cv_selected_threshold"] <= 0.50
    test_policy = result["selected_threshold_heldout_metrics"]
    assert test_policy["test_sessions"] == result["test_sessions"]
    assert (test_policy["true_positive"] + test_policy["false_positive"] +
            test_policy["true_negative"] + test_policy["false_negative"]) == result["test_sessions"]
    sensitivity = pd.read_csv(output / "tables/threshold_sensitivity.csv")
    assert len(sensitivity) == 4
    assert sensitivity.iloc[0]["policy"] == "Chosen using training-only out-of-fold F1"
    assert (output / "charts/confusion_matrix_cv_threshold.png").exists()


def test_threshold_selected_from_oof_training_scores():
    y = np.array([0, 0, 1, 1, 1, 0])
    scores = np.array([0.04, 0.07, 0.12, 0.20, 0.29, 0.43])
    cutoff, grid = select_training_threshold(y, scores)
    assert 0.05 <= cutoff <= 0.5
    assert len(grid) == 19
    assert cutoff == pytest.approx(0.1)
    assert grid["cv_f1"].between(0, 1).all()
