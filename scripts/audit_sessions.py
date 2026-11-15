from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "online_shoppers_intention.csv"
OUTPUT = ROOT / "outputs" / "audit"

EXPECTED_COLUMNS = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
    "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType",
    "Weekend", "Revenue",
]
NONNEGATIVE = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "PageValues",
]
UNIT_INTERVAL = ["BounceRates", "ExitRates", "SpecialDay"]


def audit(df: pd.DataFrame, sha256: str) -> tuple[dict, pd.DataFrame]:
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))
    if missing_columns or extra_columns:
        raise ValueError(
            f"Unexpected source schema. Missing={missing_columns}; Extra={extra_columns}"
        )

    outcome = df["Revenue"].astype("string").str.strip().str.lower()
    invalid_outcome = ~outcome.isin(["true", "false"])
    if invalid_outcome.any():
        raise ValueError(f"Invalid or missing Revenue labels: {int(invalid_outcome.sum())}")
    purchased = outcome.eq("true")

    missing_cells = int(df.isna().sum().sum())
    duplicates = df.duplicated(keep="first")
    unique = ~duplicates
    negative_counts = {
        column: int((pd.to_numeric(df[column], errors="coerce") < 0).sum())
        for column in NONNEGATIVE
    }
    outside_unit_counts = {
        column: int((~pd.to_numeric(df[column], errors="coerce").between(0, 1)).sum())
        for column in UNIT_INTERVAL
    }
    invalid_numeric = int(
        sum(pd.to_numeric(df[c], errors="coerce").isna().sum()
            for c in NONNEGATIVE + UNIT_INTERVAL)
    )

    n = len(df)
    if n == 0:
        raise ValueError("Source dataset has no sessions")

    result = {
        "source_file": SOURCE.name,
        "sha256": sha256,
        "sessions": n,
        "columns": int(df.shape[1]),
        "missing_cells": missing_cells,
        "purchase_sessions": int(purchased.sum()),
        "nonpurchase_sessions": int((~purchased).sum()),
        "conversion_rate_pct": round(100 * float(purchased.mean()), 4),
        "exact_duplicate_rows": int(duplicates.sum()),
        "duplicate_purchase_rows": int((duplicates & purchased).sum()),
        "duplicate_nonpurchase_rows": int((duplicates & ~purchased).sum()),
        "deduplicated_sessions": int(unique.sum()),
        "deduplicated_conversion_pct": round(100 * float(purchased[unique].mean()), 4),
        "pagevalues_zero_sessions": int(df["PageValues"].eq(0).sum()),
        "pagevalues_zero_pct": round(100 * float(df["PageValues"].eq(0).mean()), 2),
        "months_observed": sorted(df["Month"].astype(str).unique().tolist()),
        "visitor_type_counts": {str(k): int(v) for k, v in df["VisitorType"].value_counts(dropna=False).items()},
        "negative_value_counts": negative_counts,
        "outside_0_1_counts": outside_unit_counts,
        "unparseable_numeric_cells": invalid_numeric,
    }
    result["duplicate_sensitivity_pp"] = round(
        result["deduplicated_conversion_pct"] - result["conversion_rate_pct"], 4
    )

    profile = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "distinct_values": [int(df[c].nunique(dropna=False)) for c in df.columns],
    })
    return result, profile


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"Missing original CSV: {SOURCE}")
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest().upper()
    data = pd.read_csv(SOURCE)
    metrics, profile = audit(data, digest)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "data_audit.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    profile.to_csv(OUTPUT / "column_profile.csv", index=False)

    doc = [
        "# Data audit — Browsing Without Buying",
        "",
        "## Source and scope",
        "",
        "- Source: UCI Online Shoppers Purchasing Intention Dataset (Sakar & Kastro, 2018).",
        "- UCI dataset: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset",
        "- License: CC BY 4.0; raw CSV is not committed to Git.",
        f"- Local source SHA-256: `{digest}`.",
        "- Unit of analysis: a recorded shopping session; the label is session purchase completion.",
        "",
        "## Observed data quality",
        "",
        f"- Sessions: **{metrics['sessions']:,}** across **{metrics['columns']}** columns.",
        f"- Missing cells: **{metrics['missing_cells']:,}**.",
        f"- Purchasing sessions: **{metrics['purchase_sessions']:,}**; non-purchasing: **{metrics['nonpurchase_sessions']:,}**.",
        f"- Observed conversion: **{metrics['conversion_rate_pct']:.3f}%**.",
        f"- Exact duplicate feature rows: **{metrics['exact_duplicate_rows']:,}**, including **{metrics['duplicate_purchase_rows']:,}** purchasing duplicates.",
        f"- Removing subsequent exact duplicates changes conversion to **{metrics['deduplicated_conversion_pct']:.3f}%**, a shift of **{metrics['duplicate_sensitivity_pp']:+.3f} percentage points**.",
        f"- Zero PageValues: **{metrics['pagevalues_zero_sessions']:,}** sessions (**{metrics['pagevalues_zero_pct']:.2f}%**).",
        f"- Months represented: {', '.join(metrics['months_observed'])}.",
        f"- Negative values in tested nonnegative fields: **{sum(metrics['negative_value_counts'].values())}**.",
        f"- Values outside 0–1 in tested rate/proximity fields: **{sum(metrics['outside_0_1_counts'].values())}**.",
        f"- Unparseable numeric cells in tested fields: **{metrics['unparseable_numeric_cells']}**.",
        "",
        "## Decisions and limitations",
        "",
        "- Retain all source sessions for primary results: identical measured features do not prove duplicate events.",
        "- Treat deduplication as a sensitivity analysis, not an automatic cleaning operation.",
        "- Treat operating system, browser, region and traffic type integers as category identifiers.",
        "- Do not infer a year-over-year trend or a complete seasonal cycle from the available month labels.",
        "- PageValues is related to transaction-linked page value and must not be presented as an independent causal driver of purchases.",
        "- Associations in this observational dataset do not establish causation.",
        "",
        "See `column_profile.csv` for column-level types, missingness, and cardinality.",
    ]
    (OUTPUT / "data_audit.md").write_text("\n".join(doc) + "\n", encoding="utf-8")

    print(f"PASS - {metrics['sessions']:,} sessions, {metrics['columns']} columns")
    print(f"Missing cells: {metrics['missing_cells']}")
    print(f"Conversion: {metrics['conversion_rate_pct']:.3f}% ({metrics['purchase_sessions']:,} purchasing sessions)")
    print(f"Exact duplicate rows: {metrics['exact_duplicate_rows']} (purchasing: {metrics['duplicate_purchase_rows']}; non-purchasing: {metrics['duplicate_nonpurchase_rows']})")
    print(f"Deduplicated conversion: {metrics['deduplicated_conversion_pct']:.3f}% ({metrics['duplicate_sensitivity_pp']:+.3f} pp)")
    print(f"PageValues zeros: {metrics['pagevalues_zero_sessions']:,}")
    print(f"Range violations: {sum(metrics['negative_value_counts'].values()) + sum(metrics['outside_0_1_counts'].values())}")
    print(f"Audit outputs: {OUTPUT}")


if __name__ == "__main__":
    main()