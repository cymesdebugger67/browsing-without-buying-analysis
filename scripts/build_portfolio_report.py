from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def build_report(outputs: Path = OUTPUTS) -> Path:
    audit = json.loads((outputs / "audit" / "data_audit.json").read_text(encoding="utf-8"))
    gap = pd.read_csv(outputs / "statistical_review" / "visitor_conversion_gap.csv").iloc[0]
    categorical = pd.read_csv(outputs / "statistical_review" / "categorical_tests.csv")
    numeric = pd.read_csv(outputs / "statistical_review" / "numeric_tests.csv")
    engagement = pd.read_csv(outputs / "deep_dive" / "tables" / "engagement_by_outcome.csv")
    depth = pd.read_csv(outputs / "deep_dive" / "tables" / "conversion_by_visitor_depth.csv")

    sessions, purchases = int(audit["sessions"]), int(audit["purchase_sessions"])
    if sessions <= 0 or not 0 <= purchases <= sessions:
        raise ValueError("Invalid audit session totals")
    if (int(gap["new_sessions"]) + int(gap["returning_sessions"]) > sessions or
            int(gap["new_purchases"]) + int(gap["returning_purchases"]) > purchases):
        raise ValueError("Visitor conversion-gap groups exceed audited totals")
    if int(depth["sessions"].sum()) != sessions or int(depth["purchases"].sum()) != purchases:
        raise ValueError("Depth table does not reconcile to the audit totals")
    if not (categorical["sessions"] == sessions).all():
        raise ValueError("Categorical test counts differ from audited session total")
    if len(categorical) != 4 or len(numeric) != 4:
        raise ValueError("Expected four categorical and four numeric comparisons")
    if abs(float(gap["difference_percentage_points"]) -
           (float(gap["new_conversion_pct"]) - float(gap["returning_conversion_pct"]))) > 0.02:
        raise ValueError("Visitor conversion-gap table is inconsistent")

    medians = engagement.loc[engagement["metric"] == "ProductRelated"].set_index("outcome")
    buyers_median = float(medians.loc["Purchasing", "median"])
    nonbuyers_median = float(medians.loc["Non-purchasing", "median"])
    top_category = categorical.sort_values("cramers_v", ascending=False).iloc[0]
    top_numeric = numeric.loc[numeric["rank_biserial_purchasing_vs_nonpurchasing"].abs().idxmax()]
    depth_subset = depth.loc[(depth["page_depth"] == "21+ pages") &
                             (depth["VisitorType"].isin(["New_Visitor", "Returning_Visitor"]))]
    depth_lines = []
    for _, row in depth_subset.sort_values("VisitorType").iterrows():
        depth_lines.append(
            f"- {row['VisitorType'].replace('_', ' ')} visiting 21+ product pages: "
            f"**{float(row['conversion_rate_pct']):.2f}%** "
            f"({int(row['purchases']):,} / {int(row['sessions']):,} sessions)."
        )

    doc = [
        "# Browsing Without Buying — Analytical Case Study", "",
        "**Project owner:** Cyril James Lunzaga", "",
        "## Objective", "",
        "Understand how recorded online shopping sessions differ by purchase outcome, "
        "visitor category, browsing activity, and traffic characteristics. These findings "
        "are observational, not conversion drivers or customer-level predictions.", "",
        "## Dataset and preparation", "",
        f"- **{sessions:,} sessions**, {purchases:,} ending in a purchase "
        f"(**{100 * purchases / sessions:.2f}%** conversion).",
        f"- **{int(audit['missing_cells']):,}** missing source cells; "
        f"**{int(audit['exact_duplicate_rows']):,}** later exact-match rows retained.",
        f"- Removing the later exact-match rows changes the overall conversion rate by "
        f"**{float(audit['duplicate_sensitivity_pp']):+.3f} percentage points**.",
        "- UCI describes one session per user, but the CSV has no user IDs to "
        "verify the design independently. Exact-match rows are not assumed to be "
        "duplicate events, and month labels do not establish a complete time series.", "",
        "## Main findings", "",
        "### Visitor category", "",
        f"- New visitors: **{float(gap['new_conversion_pct']):.2f}%** "
        f"({int(gap['new_purchases']):,} / {int(gap['new_sessions']):,}).",
        f"- Returning visitors: **{float(gap['returning_conversion_pct']):.2f}%** "
        f"({int(gap['returning_purchases']):,} / {int(gap['returning_sessions']):,}).",
        f"- Observed gap: **{float(gap['difference_percentage_points']):+.2f} percentage points** "
        "(new minus returning).",
        f"- Exploratory session-level 95% bootstrap interval for the gap: "
        f"**{float(gap['bootstrap_95pct_lower_pp']):.2f} to "
        f"{float(gap['bootstrap_95pct_upper_pp']):.2f} percentage points**.", "",
        "### Browsing engagement", "",
        f"- Median product pages: **{buyers_median:g}** for purchasing sessions "
        f"vs **{nonbuyers_median:g}** for non-purchasing sessions.",
        *depth_lines, "",
        "### Statistical context", "",
        f"- Largest tested categorical association: **{top_category['variable']}**, "
        f"Cramér's V = **{float(top_category['cramers_v']):.3f}**.",
        f"- Largest absolute tested numeric rank-biserial effect: "
        f"**{top_numeric['variable']}**, "
        f"r = **{float(top_numeric['rank_biserial_purchasing_vs_nonpurchasing']):+.3f}**.",
        "- Eight exploratory p-values were adjusted together with Benjamini–Hochberg; "
        "effect sizes and group counts matter more than significance alone.", "",
        "## Selected visual evidence", "",
        "![Conversion by visitor type](charts/conversion_by_visitor.png)", "",
        "![Conversion by page depth](charts/conversion_by_page_depth.png)", "",
        "![Conversion across bounce-rate bands](deep_dive/charts/conversion_by_bounce_band.png)", "",
        "## Evidence and limitations", "",
        "- [Data audit](audit/data_audit.md)",
        "- [Detailed conversion comparison](deep_dive/summary.md)",
        "- [Statistical review](statistical_review/statistical_summary.md)",
        "- `PageValues` is not treated as an independent driver because it is linked "
        "to transaction-related page values.",
        "- The CSV has no user IDs to verify UCI's one-session-per-user design; "
        "independence-based inferential statistics and bootstrap intervals warrant caution.",
        "- No cart-event history, customer-level identity, sales amounts, or experimental "
        "exposure is available; these results cannot establish abandonment, revenue impact, "
        "causal effects, or individualized recommendations.", "",
        "**Source:** UCI Online Shoppers Purchasing Intention Dataset "
        "(Sakar & Kastro, 2018; CC BY 4.0). "
        "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset", "",
    ]
    result = outputs / "portfolio_case_study.md"
    result.write_text("\n".join(doc), encoding="utf-8")
    return result


def main() -> None:
    result = build_report()
    print(f"PASS - Case-study report created: {result}")


if __name__ == "__main__":
    main()
