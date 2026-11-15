
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.conversion_deep_dive import prepare_sessions  # noqa: E402

SOURCE = ROOT / "data" / "raw" / "online_shoppers_intention.csv"
OUTPUT = ROOT / "outputs" / "statistical_review"

CATEGORICAL = ["VisitorType", "weekend_label", "page_depth", "bounce_band"]
NUMERIC = ["ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates"]


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Adjust a prespecified family of p-values to control false-discovery rate."""
    if not p_values:
        return []
    p = np.asarray(p_values, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("P-values must be finite and within [0, 1]")
    order = np.argsort(p)
    ranks = np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate((p[order] * len(p) / ranks)[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result.tolist()


def categorical_tests(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variable in CATEGORICAL:
        tab = pd.crosstab(df[variable], df["purchased"], dropna=False)
        tab = tab.reindex(columns=[0, 1], fill_value=0)
        tab = tab.loc[tab.sum(axis=1) > 0]
        if tab.shape[0] < 2 or (tab.sum(axis=0) == 0).any():
            raise ValueError(f"Not enough categories or outcomes for {variable}")
        chi2, p_value, dof, expected = chi2_contingency(tab, correction=False)
        n = int(tab.to_numpy().sum())
        if n != len(df):
            raise AssertionError(f"Reconciliation failed for {variable}")
        small_cells = int((expected < 5).sum())
        okay = bool(expected.min() >= 1 and small_cells / expected.size <= 0.2)
        records.append({
            "variable": variable,
            "sessions": n,
            "categories": len(tab),
            "chi_square": float(chi2),
            "degrees_of_freedom": int(dof),
            "p_value": float(p_value),
            "cramers_v": float(np.sqrt(chi2 / (n * min(tab.shape[0] - 1, tab.shape[1] - 1)))),
            "min_expected_cell": float(expected.min()),
            "expected_cells_below_5": small_cells,
            "chi_square_expected_count_check": "PASS" if okay else "CAUTION",
        })
    return pd.DataFrame.from_records(records)


def numeric_tests(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variable in NUMERIC:
        buyers = df.loc[df["purchased"] == 1, variable].to_numpy(dtype=float)
        non_buyers = df.loc[df["purchased"] == 0, variable].to_numpy(dtype=float)
        if not (np.isfinite(buyers).all() and np.isfinite(non_buyers).all()):
            raise ValueError(f"Non-finite observations in {variable}")
        if len(buyers) == 0 or len(non_buyers) == 0:
            raise ValueError("Both purchase outcomes are required")
        u, p_value = mannwhitneyu(buyers, non_buyers, alternative="two-sided", method="asymptotic")
        records.append({
            "variable": variable,
            "purchasing_sessions": len(buyers),
            "nonpurchasing_sessions": len(non_buyers),
            "median_purchasing": float(np.median(buyers)),
            "median_nonpurchasing": float(np.median(non_buyers)),
            "mann_whitney_u": float(u),
            "p_value": float(p_value),
            "rank_biserial_purchasing_vs_nonpurchasing": float(2 * u / (len(buyers) * len(non_buyers)) - 1),
        })
    return pd.DataFrame.from_records(records)


def visitor_gap_bootstrap(df: pd.DataFrame, repetitions: int = 2000, seed: int = 2026) -> dict:
    """Session-level within-group bootstrap of New vs Returning conversion gap."""
    if repetitions < 100:
        raise ValueError("At least 100 bootstrap repetitions required")
    records = {}
    for visitor in ("New_Visitor", "Returning_Visitor"):
        outcomes = df.loc[df["VisitorType"] == visitor, "purchased"]
        if len(outcomes) == 0:
            raise ValueError(f"Missing visitor category: {visitor}")
        records[visitor] = {"n": int(len(outcomes)), "purchases": int(outcomes.sum())}
    new = records["New_Visitor"]
    returning = records["Returning_Visitor"]
    new_rate = new["purchases"] / new["n"]
    returning_rate = returning["purchases"] / returning["n"]
    rng = np.random.default_rng(seed)
    # Binomial resampling is equivalent to bootstrapping the observed 0/1 outcomes.
    sampled_new = rng.binomial(new["n"], new_rate, size=repetitions) / new["n"]
    sampled_returning = rng.binomial(returning["n"], returning_rate, size=repetitions) / returning["n"]
    lower, upper = np.percentile(100 * (sampled_new - sampled_returning), [2.5, 97.5])
    return {
        "new_sessions": new["n"], "new_purchases": new["purchases"],
        "returning_sessions": returning["n"], "returning_purchases": returning["purchases"],
        "new_conversion_pct": 100 * new_rate,
        "returning_conversion_pct": 100 * returning_rate,
        "difference_percentage_points": 100 * (new_rate - returning_rate),
        "bootstrap_95pct_lower_pp": float(lower),
        "bootstrap_95pct_upper_pp": float(upper),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
    }


def analyze(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = prepare_sessions(data)
    if not df["purchased"].nunique() == 2:
        raise ValueError("Both purchasing and non-purchasing sessions are required")
    categorical = categorical_tests(df)
    numeric = numeric_tests(df)
    all_adjusted = benjamini_hochberg(
        categorical["p_value"].tolist() + numeric["p_value"].tolist()
    )
    categorical["p_value_bh_8tests"] = all_adjusted[:len(categorical)]
    numeric["p_value_bh_8tests"] = all_adjusted[len(categorical):]
    return categorical, numeric, visitor_gap_bootstrap(df)


def run(source: Path = SOURCE, output: Path = OUTPUT) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    data = pd.read_csv(source)
    categorical, numeric, gap = analyze(data)
    output.mkdir(parents=True, exist_ok=True)
    categorical.to_csv(output / "categorical_tests.csv", index=False, float_format="%.10g")
    numeric.to_csv(output / "numeric_tests.csv", index=False, float_format="%.10g")
    pd.DataFrame([gap]).to_csv(output / "visitor_conversion_gap.csv", index=False, float_format="%.10g")
    count_check = int(categorical["sessions"].min())
    if count_check != len(data):
        raise AssertionError("Categorical tests did not reconcile to all sessions")
    strongest = categorical.sort_values("cramers_v", ascending=False).iloc[0]
    notes = [
        "# Statistical review — browsing sessions", "",
        f"- Observations: **{len(data):,}** sessions; **{int(data['Revenue'].sum()):,}** purchases.",
        "- Eight preselected, exploratory comparisons: four categorical chi-square tests and four numeric Mann–Whitney U tests.",
        "- P-values are adjusted together using Benjamini–Hochberg false-discovery-rate control.",
        f"- Strongest tested categorical association by Cramér's V: **{strongest['variable']}** (V = **{strongest['cramers_v']:.3f}**).",
        "", "## Visitor-type conversion gap", "",
        f"- New visitors: **{gap['new_purchases']:,} / {gap['new_sessions']:,} ({gap['new_conversion_pct']:.2f}%)**.",
        f"- Returning visitors: **{gap['returning_purchases']:,} / {gap['returning_sessions']:,} ({gap['returning_conversion_pct']:.2f}%)**.",
        f"- Observed gap (new minus returning): **{gap['difference_percentage_points']:+.2f} percentage points**.",
        f"- Exploratory 95% bootstrap interval: **{gap['bootstrap_95pct_lower_pp']:.2f} to {gap['bootstrap_95pct_upper_pp']:.2f} percentage points**, using {gap['bootstrap_repetitions']:,} fixed-seed within-group session resamples.",
        "", "## Interpret with care", "",
        "- Chi-square expected-cell checks are in `categorical_tests.csv`; comparisons marked CAUTION are not strong grounds for asymptotic inference.",
        "- P-values alone do not describe practical significance; review Cramér's V, rank-biserial effect sizes, group counts, and distributions.",
        "- UCI describes one session per user, but the released CSV has no user identifiers to independently verify this design.",
        "- The bootstrap assumes independent session observations within visitor groups; that assumption is not independently testable from the CSV.",
        "- Page-depth and bounce-rate groups reflect observed session behavior, not treatments or causal purchase drivers.",
        "- The dataset is observational and does not establish causation, customer-level behavior, or complete annual seasonality.",
        "- All original sessions, including 125 later exact-match rows, are retained; the duplicate sensitivity result belongs to the separate data audit.",
        "- `PageValues` is excluded from the tested features because it is transaction-linked.",
        "", "Source: UCI Online Shoppers Purchasing Intention Dataset (Sakar & Kastro, 2018; CC BY 4.0).", "",
    ]
    (output / "statistical_summary.md").write_text("\n".join(notes), encoding="utf-8")
    return {"sessions": len(data), "categorical_tests": len(categorical), "numeric_tests": len(numeric), "outputs": 4}


def main() -> None:
    result = run()
    print(f"PASS - {result['sessions']:,} sessions; {result['categorical_tests']} categorical and {result['numeric_tests']} numeric tests")
    print("Generated 3 CSV tables and 1 Markdown summary")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
