from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/online_shoppers_intention.csv"
OUTPUT = ROOT / "outputs/modeling"
SEED = 42

NUMERIC = [
    "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "SpecialDay",
]
CATEGORICAL = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
FEATURES = NUMERIC + CATEGORICAL


def load_model_data(df: pd.DataFrame, include_page_values: bool = False) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    needed = set(FEATURES + ["Revenue"] + (["PageValues"] if include_page_values else []))
    if missing := sorted(needed - set(df.columns)):
        raise ValueError(f"Required model columns missing: {missing}")
    if df.empty:
        raise ValueError("Empty dataset")
    outcome = df["Revenue"].astype("string").str.strip().str.lower()
    if not outcome.isin(["true", "false"]).all():
        raise ValueError("Invalid Revenue labels")
    y = outcome.eq("true").astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("Two outcome classes are required")
    numeric = NUMERIC + (["PageValues"] if include_page_values else [])
    x = df[numeric + CATEGORICAL].copy()
    for column in numeric:
        x[column] = pd.to_numeric(x[column], errors="raise")
        if not np.isfinite(x[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid numeric values: {column}")
    for column in CATEGORICAL:
        if x[column].isna().any():
            raise ValueError(f"Missing categorical value: {column}")
        x[column] = x[column].astype("string").str.strip().astype(str)
        if x[column].eq("").any():
            raise ValueError(f"Blank categorical value: {column}")
    grouping_features = x.drop(columns=["PageValues"], errors="ignore")
    groups = pd.util.hash_pandas_object(grouping_features, index=False).to_numpy(dtype="uint64")
    return x, y, groups


def preprocessing(include_page_values: bool = False) -> ColumnTransformer:
    columns = NUMERIC + (["PageValues"] if include_page_values else [])
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], remainder="drop")


def model_pipeline(estimator, include_page_values: bool = False) -> Pipeline:
    return Pipeline([("features", preprocessing(include_page_values)), ("classifier", estimator)])


def split_sessions(y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    x_index = np.arange(len(y))
    train_idx, test_idx = next(folds.split(x_index, y, groups))
    if set(groups[train_idx]).intersection(groups[test_idx]):
        raise AssertionError("Identical feature groups leaked across train/test")
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
        raise ValueError("Train/test split requires both purchase classes")
    return train_idx, test_idx


def evaluate(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    scores = np.asarray(scores, dtype=float)
    if scores.shape != y.shape or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid probability scores")
    if len(np.unique(y)) != 2:
        raise ValueError("Both classes required for evaluation")
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "test_sessions": int(len(y)), "test_purchases": int(y.sum()),
        "test_purchase_prevalence": float(np.mean(y)),
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y, scores)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def bootstrap_ap(y: np.ndarray, scores: np.ndarray, repeats: int = 400) -> tuple[float, float]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(SEED)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    values = np.empty(repeats)
    for i in range(repeats):
        indices = np.concatenate((rng.choice(positive, len(positive), replace=True),
                                  rng.choice(negative, len(negative), replace=True)))
        values[i] = average_precision_score(y[indices], scores[indices])
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))


def select_training_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Choose a threshold using only training-fold out-of-fold probabilities."""
    candidates = np.linspace(0.05, 0.50, 19)
    rows = []
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "cv_precision": float(precision_score(y, predicted, zero_division=0)),
            "cv_recall": float(recall_score(y, predicted, zero_division=0)),
            "cv_f1": float(f1_score(y, predicted, zero_division=0)),
        })
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["cv_f1", "threshold"], ascending=[False, False]).iloc[0]
    return float(best["threshold"]), grid


def save_curves(y: np.ndarray, scored: dict[str, np.ndarray], chosen: str, charts: Path, cv_threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, scores in scored.items():
        if name == "Majority-prior baseline":
            continue
        precision, recall, _ = precision_recall_curve(y, scores)
        ax.plot(recall, precision, label=f"{name}: AP {average_precision_score(y, scores):.3f}")
    ax.axhline(y.mean(), color="#888888", linestyle="--", alpha=0.7, label="Test prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Held-out precision–recall curves", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(charts / "precision_recall.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, scores in scored.items():
        if name == "Majority-prior baseline":
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(y, scores)
        ax.plot(false_positive_rate, true_positive_rate, label=f"{name}: AUC {roc_auc_score(y, scores):.3f}")
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", label="No discrimination")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate",
           title="Held-out ROC curves", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(charts / "roc_curves.png", dpi=160)
    plt.close(fig)

    for threshold, filename, description in (
        (0.50, "confusion_matrix.png", "fixed 0.50 threshold"),
        (cv_threshold, "confusion_matrix_cv_threshold.png", "threshold chosen from training CV"),
    ):
        predicted = (scored[chosen] >= threshold).astype(int)
        matrix = confusion_matrix(y, predicted, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6.4, 5))
        im = ax.imshow(matrix, cmap="Blues")
        for i in (0, 1):
            for j in (0, 1):
                ax.text(j, i, f"{matrix[i, j]:,}", va="center", ha="center", fontsize=16)
        ax.set_xticks([0, 1], ["No purchase", "Purchase"])
        ax.set_yticks([0, 1], ["No purchase", "Purchase"])
        ax.set(xlabel="Predicted", ylabel="Recorded outcome",
               title=f"{chosen} | {description} ({threshold:.3f})")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(charts / filename, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name in ("Logistic regression", "Random forest"):
        observed, predicted_bins = calibration_curve(y, scored[name], n_bins=8, strategy="quantile")
        ax.plot(predicted_bins, observed, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "--", color="#888888", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed purchase frequency",
           title="Held-out calibration (quantile bins)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(charts / "calibration.png", dpi=160)
    plt.close(fig)


def run(source: Path = SOURCE, output: Path = OUTPUT) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"Dataset missing: {source}")
    data = pd.read_csv(source)
    X, y, groups = load_model_data(data)
    train_idx, test_idx = split_sessions(y, groups)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_groups = groups[train_idx]
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    choices = {
        "Logistic regression": (
            model_pipeline(LogisticRegression(max_iter=1500, random_state=SEED)),
            {"classifier__C": [0.2, 1.0], "classifier__class_weight": [None, "balanced"]},
        ),
        "Random forest": (
            model_pipeline(RandomForestClassifier(n_estimators=120, n_jobs=1, random_state=SEED)),
            {"classifier__max_depth": [10, None], "classifier__min_samples_leaf": [5, 20]},
        ),
    }
    fitted = {}
    tuning = []
    for label, (pipeline, parameters) in choices.items():
        search = GridSearchCV(pipeline, parameters, scoring="average_precision", cv=cv,
                              refit=True, n_jobs=1, error_score="raise")
        search.fit(X_train, y_train, groups=train_groups)
        fitted[label] = search.best_estimator_
        tuning.append({"model": label, "cv_average_precision": float(search.best_score_),
                       "parameters": json.dumps(search.best_params_, sort_keys=True)})
    chosen = max(tuning, key=lambda row: row["cv_average_precision"])["model"]
    threshold_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + 1)
    oof_probabilities = cross_val_predict(
        fitted[chosen], X_train, y_train, groups=train_groups, cv=threshold_cv,
        method="predict_proba", n_jobs=1,
    )[:, 1]
    chosen_threshold, threshold_grid = select_training_threshold(y_train, oof_probabilities)
    training_threshold_row = threshold_grid.loc[
        np.isclose(threshold_grid["threshold"], chosen_threshold)
    ].iloc[0]

    baseline = DummyClassifier(strategy="prior")
    baseline.fit(np.zeros((len(train_idx), 1)), y_train)
    predictions = {"Majority-prior baseline": baseline.predict_proba(np.zeros((len(test_idx), 1)))[:, 1]}
    predictions.update({label: estimator.predict_proba(X_test)[:, 1] for label, estimator in fitted.items()})
    metrics = []
    for label, values in predictions.items():
        metrics.append({"model": label, **evaluate(y_test, values)})

    tables = output / "tables"
    charts = output / "charts"
    tables.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(tables / "model_comparison.csv", index=False)
    pd.DataFrame(tuning).to_csv(tables / "cross_validation_tuning.csv", index=False)
    save_curves(y_test, predictions, chosen, charts, chosen_threshold)

    intervals = bootstrap_ap(y_test, predictions[chosen])
    importance = permutation_importance(fitted[chosen], X_test, y_test,
                                        n_repeats=3, random_state=SEED,
                                        scoring="average_precision", n_jobs=1)
    ranking = pd.DataFrame({"feature": FEATURES,
                            "ap_decrease_mean": importance.importances_mean,
                            "ap_decrease_std": importance.importances_std})
    ranking = ranking.sort_values("ap_decrease_mean", ascending=False)
    ranking.to_csv(tables / "permutation_importance.csv", index=False)
    top = ranking.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["ap_decrease_mean"], xerr=top["ap_decrease_std"], capsize=2)
    ax.axvline(0, linestyle="--", color="#666666", linewidth=1)
    ax.set(xlabel="Mean held-out average-precision decrease on permutation",
           title=f"{chosen}: permutation importance (3 repeats)")
    fig.tight_layout()
    fig.savefig(charts / "permutation_importance.png", dpi=160)
    plt.close(fig)

    threshold_rows = [
        {"policy": "Chosen using training-only out-of-fold F1", **evaluate(y_test, predictions[chosen], chosen_threshold)},
    ] + [
        {"policy": f"Exploratory fixed threshold {threshold:.2f}",
         **evaluate(y_test, predictions[chosen], threshold)}
        for threshold in (0.25, 0.5, 0.75)
    ]
    pd.DataFrame(threshold_rows).to_csv(tables / "threshold_sensitivity.csv", index=False)
    chosen_threshold_test = evaluate(y_test, predictions[chosen], chosen_threshold)

    # This separate comparison is intentionally excluded from selecting the main model.
    fixed = model_pipeline(LogisticRegression(C=1.0, max_iter=1500, random_state=SEED))
    without_page = clone(fixed).fit(X_train, y_train)
    with_page_X, _, with_page_groups = load_model_data(data, include_page_values=True)
    if not np.array_equal(with_page_groups, groups):
        raise AssertionError("PageValues comparison must use identical split groups")
    with_page = model_pipeline(LogisticRegression(C=1.0, max_iter=1500, random_state=SEED), True)
    with_page.fit(with_page_X.iloc[train_idx], y_train)
    sensitivity = pd.DataFrame([
        {"feature_set": "Without PageValues", **evaluate(y_test, without_page.predict_proba(X_test)[:, 1])},
        {"feature_set": "With PageValues (transaction-linked; potential leakage)",
         **evaluate(y_test, with_page.predict_proba(with_page_X.iloc[test_idx])[:, 1])},
    ])
    sensitivity.to_csv(tables / "pagevalues_sensitivity.csv", index=False)

    if int(data.duplicated(subset=FEATURES).sum()) > 0:
        duplicate_note = f"{int(data.duplicated(subset=FEATURES).sum()):,} rows repeat the chosen input-feature vector; identical profiles stay in one partition."
    else:
        duplicate_note = "No identical selected feature profiles were observed."
    result = {
        "source_sessions": int(len(data)), "source_purchases": int(y.sum()),
        "training_sessions": int(len(train_idx)), "test_sessions": int(len(test_idx)),
        "training_purchases": int(y_train.sum()), "test_purchases": int(y_test.sum()),
        "group_overlap": 0, "chosen_model": chosen,
        "selected_cv_average_precision": float(max(row["cv_average_precision"] for row in tuning)),
        "train_cv_selected_threshold": chosen_threshold,
        "train_cv_threshold_f1": float(training_threshold_row["cv_f1"]),
        "train_cv_threshold_precision": float(training_threshold_row["cv_precision"]),
        "train_cv_threshold_recall": float(training_threshold_row["cv_recall"]),
        "selected_threshold_heldout_metrics": chosen_threshold_test,
        "chosen_test_ap_interval_95": list(intervals), "seed": SEED,
        "n_selected_features": len(FEATURES), "excluded_feature": "PageValues",
        "duplicate_note": duplicate_note, "metrics": metrics,
        "tuning": tuning,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    winner = next(row for row in metrics if row["model"] == chosen)
    report = [
        "# Purchase-completion classification — held-out evaluation", "",
        f"- Source: **{len(data):,}** sessions; **{int(y.sum()):,}** purchases.",
        f"- Training: **{len(train_idx):,}** sessions; held-out evaluation: **{len(test_idx):,}** sessions.",
        "- Source rows remain intact. Five-fold stratified group splitting isolates identical feature profiles; a three-fold stratified group CV selects hyperparameters on training data only.",
        f"- {duplicate_note}",
        "- One-hot encoded categorical codes and scaled numeric measures are fitted only on training folds.",
        "- Models: majority-prior baseline, logistic regression, and random forest. The selection metric is **cross-validated average precision** (PR-AUC summary).",
        f"- Selected by training CV: **{chosen}**, mean CV average precision **{result['selected_cv_average_precision']:.3f}**.",
        f"- Selected model held-out average precision: **{winner['average_precision']:.3f}**; ROC-AUC: **{winner['roc_auc']:.3f}**.",
        f"- Fixed-test, stratified-session bootstrap 95% interval for held-out average precision: **{intervals[0]:.3f}–{intervals[1]:.3f}** (400 replicates; treats evaluation records as independent).",
        f"- Held-out purchase prevalence: **{winner['test_purchase_prevalence']:.3f}**; compare average precision to that baseline, not to accuracy alone.",
        f"- At prespecified probability threshold 0.50: precision **{winner['precision']:.3f}**, recall **{winner['recall']:.3f}**, F1 **{winner['f1']:.3f}**.",
        f"- Threshold **{chosen_threshold:.3f}** was selected using 3-fold out-of-fold **training-only** F1, from 19 fixed candidates (0.05 to 0.50), independently of test performance.",
        f"- At that training-selected threshold, held-out precision **{chosen_threshold_test['precision']:.3f}**, recall **{chosen_threshold_test['recall']:.3f}**, F1 **{chosen_threshold_test['f1']:.3f}**; TP **{chosen_threshold_test['true_positive']}**, FP **{chosen_threshold_test['false_positive']}**, FN **{chosen_threshold_test['false_negative']}**, TN **{chosen_threshold_test['true_negative']}**.",
        "- The optimized threshold changes the precision–recall trade-off; it does not change average precision, ROC-AUC or calibration. Training OOF F1 can be optimistic because the model family/hyperparameters were previously chosen using the same training cohort.",
        "", "## Cautions", "",
        "- These are retrospective complete-session classifications, NOT validated live purchase forecasts.",
        "- `PageValues` is excluded from all principal models because it is transaction-linked; the separate sensitivity table illustrates how including it changes performance, not a deployable result.",
        "- No person IDs, chronological timestamps, external validation site, sales amounts, or randomized interventions are available.",
        "- A group split reduces exact-feature overlap but does not prove records are independent or protect against unobserved shared customers or time drift.",
        "- Hyperparameters, model family and decision threshold are selected using training folds; the held-out test is never used to optimize them. Only the training-selected threshold is a pre-test decision policy; other thresholds are exploratory.",
        "- Permutation importance estimates associations with held-out average precision, not causal importance; correlated predictors can obscure attribution.",
        "- Random partitioning does not support claims about future-site performance. Uncertainty intervals are exploratory under independent-session assumptions.",
        "", "Source: Sakar & Kastro (2018), UCI Online Shoppers Purchasing Intention Dataset (CC BY 4.0).", "",
    ]
    (output / "model_report.md").write_text("\n".join(report), encoding="utf-8")
    return result


def main() -> None:
    result = run()
    print(f"PASS - Modeling: {result['training_sessions']:,} train, {result['test_sessions']:,} test; selected {result['chosen_model']}")
    print(f"Cross-validated average precision: {result['selected_cv_average_precision']:.3f}")


if __name__ == "__main__":
    main()
