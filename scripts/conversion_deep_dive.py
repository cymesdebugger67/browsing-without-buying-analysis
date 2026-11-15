from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "online_shoppers_intention.csv"
OUTPUT = ROOT / "outputs" / "deep_dive"

REQUIRED = {
    "Revenue", "VisitorType", "Weekend", "ProductRelated", "TrafficType",
    "ProductRelated_Duration", "BounceRates", "ExitRates",
}
DEPTH_LABELS = ["0 pages", "1–5 pages", "6–20 pages", "21+ pages"]
BOUNCE_LABELS = ["Zero", "(0, 0.02]", "(0.02, 0.05]", "Above 0.05"]
MIN_GROUP_SIZE = 100


def prepare_sessions(data: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED - set(data.columns))
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    if data.empty:
        raise ValueError("Dataset contains no sessions")

    df = data.copy()
    outcome = df["Revenue"].astype("string").str.strip().str.lower()
    if not outcome.isin(["true", "false"]).all():
        raise ValueError("Revenue must contain only true or false values")
    df["purchased"] = outcome.eq("true").astype(int)

    weekend = df["Weekend"].astype("string").str.strip().str.lower()
    if not weekend.isin(["true", "false"]).all():
        raise ValueError("Weekend must contain only true or false values")
    df["weekend_label"] = weekend.map({"true": "Weekend", "false": "Weekday"})

    visitor = df["VisitorType"].astype("string").str.strip()
    if visitor.isna().any() or visitor.eq("").any():
        raise ValueError("VisitorType has missing or blank values")
    df["VisitorType"] = visitor

    for col in ["ProductRelated", "TrafficType", "ProductRelated_Duration", "BounceRates", "ExitRates"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if df[col].isna().any():
            raise ValueError(f"{col} includes missing values")
    if (df[["ProductRelated", "ProductRelated_Duration", "TrafficType"]] < 0).any().any():
        raise ValueError("Negative counts, durations, or traffic codes detected")
    if not df["BounceRates"].between(0, 1).all() or not df["ExitRates"].between(0, 1).all():
        raise ValueError("BounceRates and ExitRates must lie between 0 and 1")
    if (df["ProductRelated"] % 1 != 0).any() or (df["TrafficType"] % 1 != 0).any():
        raise ValueError("ProductRelated and TrafficType must be integer-valued")

    df["page_depth"] = pd.cut(
        df["ProductRelated"], bins=[-1, 0, 5, 20, float("inf")],
        labels=DEPTH_LABELS,
    )
    df["bounce_band"] = pd.cut(
        df["BounceRates"], bins=[-0.00000001, 0, 0.02, 0.05, float("inf")],
        labels=BOUNCE_LABELS,
    )
    if df[["page_depth", "bounce_band"]].isna().any().any():
        raise ValueError("A session could not be classified into the defined bands")
    return df


def conversion_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = (
        df.groupby(columns, observed=True, dropna=False)["purchased"]
        .agg(sessions="size", purchases="sum")
        .reset_index()
    )
    result["nonpurchase_sessions"] = result["sessions"] - result["purchases"]
    result["conversion_rate_pct"] = (100 * result["purchases"] / result["sessions"]).round(2)
    result["interpret_with_caution"] = result["sessions"] < MIN_GROUP_SIZE
    if int(result["sessions"].sum()) != len(df) or int(result["purchases"].sum()) != int(df["purchased"].sum()):
        raise AssertionError(f"Reconciliation failed for {columns}")
    return result


def engagement_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, value in [("Non-purchasing", 0), ("Purchasing", 1)]:
        subset = df.loc[df["purchased"] == value]
        for variable in ["ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates"]:
            series = subset[variable]
            rows.append({
                "outcome": label,
                "metric": variable,
                "sessions": len(subset),
                "mean": round(float(series.mean()), 4),
                "p25": round(float(series.quantile(0.25)), 4),
                "median": round(float(series.median()), 4),
                "p75": round(float(series.quantile(0.75)), 4),
            })
    return pd.DataFrame(rows)


def save_visitor_depth_heatmap(data: pd.DataFrame, output: Path) -> None:
    visitors = data.groupby("VisitorType", observed=True)["sessions"].sum().sort_values(ascending=False).index.tolist()
    matrix = data.pivot(index="VisitorType", columns="page_depth", values="conversion_rate_pct").reindex(index=visitors, columns=DEPTH_LABELS)
    counts = data.pivot(index="VisitorType", columns="page_depth", values="sessions").reindex(index=visitors, columns=DEPTH_LABELS)
    arr = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    masked = np.ma.masked_invalid(arr)
    color_map = plt.colormaps["Blues"].with_extremes(bad="#eeeeee")
    im = ax.imshow(masked, vmin=0, vmax=max(30, float(np.nanmax(arr))), cmap=color_map, aspect="auto")
    ax.set_xticks(range(len(DEPTH_LABELS)), DEPTH_LABELS)
    ax.set_yticks(range(len(visitors)), visitors)
    ax.set(title="Purchase conversion by visitor type and product-page depth", xlabel="Product pages visited")
    for i in range(len(visitors)):
        for j in range(len(DEPTH_LABELS)):
            if pd.notna(matrix.iloc[i, j]):
                n = int(counts.iloc[i, j])
                note = "*" if n < MIN_GROUP_SIZE else ""
                ax.text(j, i, f"{matrix.iloc[i, j]:.1f}%\nn={n:,}{note}",
                        ha="center", va="center", fontsize=8,
                        color="white" if arr[i, j] > float(np.nanmax(arr)) * 0.52 else "black")
    fig.colorbar(im, ax=ax, label="Sessions ending in purchase (%)")
    fig.text(0.5, 0.015, "* Fewer than 100 sessions; interpret cautiously. Observational associations only.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_traffic_chart(data: pd.DataFrame, output: Path) -> None:
    # Restrict the comparison to traffic codes supported by at least 100 sessions.
    shown = data.loc[data["sessions"] >= MIN_GROUP_SIZE].sort_values("sessions", ascending=False).head(10)
    shown = shown.sort_values("conversion_rate_pct")
    if shown.empty:
        fig, ax = plt.subplots(figsize=(8.8, 4.0))
        ax.text(0.5, 0.5, "No traffic types with 100 or more sessions", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    pos = np.arange(len(shown))
    ax.barh(pos, shown["conversion_rate_pct"].to_numpy())
    ax.set_yticks(pos, [f"Code {int(x)}" for x in shown["TrafficType"]])
    ax.set(xlabel="Sessions ending in purchase (%)", title="Conversion across high-volume traffic codes")
    ax.set_xlim(0, max(10, shown["conversion_rate_pct"].max() * 1.24))
    for i, (_, row) in enumerate(shown.iterrows()):
        ax.text(row["conversion_rate_pct"] + 0.35, i, f"n={int(row['sessions']):,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_bounce_chart(data: pd.DataFrame, output: Path) -> None:
    ordered = data.set_index("bounce_band").reindex(BOUNCE_LABELS).dropna(subset=["sessions"])
    fig, ax = plt.subplots(figsize=(8.3, 4.4))
    x = np.arange(len(ordered))
    ax.bar(x, ordered["conversion_rate_pct"])
    ax.set_xticks(x, ordered.index)
    ax.set(ylabel="Sessions ending in purchase (%)", xlabel="Session bounce-rate band",
           title="Conversion across bounce-rate bands")
    for i, (_, row) in enumerate(ordered.iterrows()):
        ax.text(i, row["conversion_rate_pct"] + 0.4, f"n={int(row['sessions']):,}",
                ha="center", fontsize=8)
    ax.set_ylim(0, max(10, ordered["conversion_rate_pct"].max() * 1.18))
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(source: Path = SOURCE, output: Path = OUTPUT) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    df = prepare_sessions(pd.read_csv(source))
    tables = {
        "conversion_by_visitor_depth": conversion_table(df, ["VisitorType", "page_depth"]),
        "conversion_by_visitor_weekend": conversion_table(df, ["VisitorType", "weekend_label"]),
        "conversion_by_traffic_type": conversion_table(df, ["TrafficType"]),
        "conversion_by_bounce_band": conversion_table(df, ["bounce_band"]),
        "engagement_by_outcome": engagement_table(df),
    }
    folder = output / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    charts = output / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", index=False)
    save_visitor_depth_heatmap(tables["conversion_by_visitor_depth"], charts / "visitor_depth_heatmap.png")
    save_traffic_chart(tables["conversion_by_traffic_type"], charts / "high_volume_traffic_codes.png")
    save_bounce_chart(tables["conversion_by_bounce_band"], charts / "conversion_by_bounce_band.png")
    median = tables["engagement_by_outcome"].query("metric == 'ProductRelated'").set_index("outcome")
    rate = 100 * float(df["purchased"].mean())
    report = [
        "# Session conversion — deeper descriptive analysis", "",
        f"- Sessions: **{len(df):,}**; purchasing: **{int(df['purchased'].sum()):,}**.",
        f"- Observed purchase conversion: **{rate:.2f}%**.",
        "- Original source rows retained; no automatic deduplication or outlier exclusion.",
        f"- Median product pages, purchasing sessions: **{median.at['Purchasing', 'median']:g}**.",
        f"- Median product pages, non-purchasing sessions: **{median.at['Non-purchasing', 'median']:g}**.",
        "", "## Evidence", "",
        "- `tables/conversion_by_visitor_depth.csv` compares visitor categories and page-depth bands.",
        "- `tables/conversion_by_visitor_weekend.csv` compares visitor categories across weekdays/weekends.",
        "- `tables/conversion_by_traffic_type.csv` retains original traffic-type codes and sample sizes.",
        "- `tables/conversion_by_bounce_band.csv` compares rates across prespecified bounce-rate bands.",
        "- `tables/engagement_by_outcome.csv` reports means, medians, and quartiles without deleting extreme values.",
        "- The three accompanying charts display selected descriptive comparisons.",
        "", "## Interpretation limits", "",
        "- Conversion rates are **session-level** descriptive statistics, not causal effects.",
        "- Groups with fewer than 100 sessions are flagged for caution, not silently removed from the tables.",
        "- TrafficType is a categorical identifier; numeric codes are not ranked quantities.",
        "- Bounce-rate bands are descriptive; their boundaries are analysis choices, not validated risk thresholds.",
        "- `PageValues` was intentionally excluded from these comparisons because of its transaction-linked definition.",
        "- Avoid interpreting observed engagement differences as evidence that increasing page views causes purchases.",
        "", "Source: UCI Online Shoppers Purchasing Intention Dataset (Sakar & Kastro, 2018; CC BY 4.0).",
    ]
    (output / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"sessions": len(df), "purchases": int(df["purchased"].sum()), "tables": len(tables), "charts": 3}


def main() -> None:
    result = run()
    print(f"PASS - {result['sessions']:,} sessions; {result['purchases']:,} purchases")
    print(f"Generated {result['tables']} tables, {result['charts']} charts, and one Markdown summary")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
