from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.conversion_deep_dive import prepare_sessions

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/online_shoppers_intention.csv"
OUTPUT = ROOT / "outputs/advanced_segmentation"
Z_95 = 1.959963984540054


def wilson_interval(success: int, total: int) -> tuple[float, float]:
    if total <= 0 or success < 0 or success > total:
        raise ValueError("Invalid binomial count")
    p = success / total
    denominator = 1 + Z_95**2 / total
    midpoint = (p + Z_95**2 / (2 * total)) / denominator
    spread = Z_95 * np.sqrt(p * (1 - p) / total + Z_95**2 / (4 * total**2)) / denominator
    return 100 * (midpoint - spread), 100 * (midpoint + spread)


def summarize(df: pd.DataFrame, group: str) -> pd.DataFrame:
    result = (df.groupby(group, observed=True, dropna=False)["purchased"]
              .agg(sessions="size", purchases="sum").reset_index())
    result["nonpurchase_sessions"] = result["sessions"] - result["purchases"]
    result["conversion_rate_pct"] = 100 * result["purchases"] / result["sessions"]
    bounds = [wilson_interval(int(row.purchases), int(row.sessions)) for row in result.itertuples()]
    result["wilson_95_lower_pct"] = [x[0] for x in bounds]
    result["wilson_95_upper_pct"] = [x[1] for x in bounds]
    result["sample_share_pct"] = 100 * result["sessions"] / len(df)
    result["small_sample_flag"] = result["sessions"] < 100
    if (result["sessions"].sum() != len(df) or
            result["purchases"].sum() != df["purchased"].sum()):
        raise AssertionError(f"Group reconciliation failed: {group}")
    return result


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    df = prepare_sessions(data)
    for col in ("Region", "Browser", "OperatingSystems", "SpecialDay"):
        if col not in df:
            raise ValueError(f"Missing column: {col}")
        df[col] = pd.to_numeric(df[col], errors="raise")
        if df[col].isna().any() or not np.isfinite(df[col].to_numpy()).all():
            raise ValueError(f"Invalid values in {col}")
    if not df["SpecialDay"].between(0, 1).all():
        raise ValueError("SpecialDay outside [0, 1]")
    df["special_day_band"] = np.where(df["SpecialDay"] == 0, "Zero proximity", "Nonzero proximity")
    df["Region"] = df["Region"].astype(int).astype(str)
    df["Browser"] = df["Browser"].astype(int).astype(str)
    df["OperatingSystems"] = df["OperatingSystems"].astype(int).astype(str)
    return df


def save_donut(labels: list[str], values: list[int], title: str, output: Path) -> None:
    if not values or sum(values) <= 0:
        raise ValueError("Cannot plot empty composition")
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    wedges, _ = ax.pie(values, startangle=90, counterclock=False, labels=None,
                       wedgeprops={"width": 0.38, "edgecolor": "white"})
    ax.text(0, 0.10, f"{sum(values):,}", fontsize=21, weight="bold", ha="center")
    ax.text(0, -0.13, "sessions", fontsize=11, ha="center")
    legend = [f"{label}: {value:,} ({value / sum(values):.1%})" for label, value in zip(labels, values)]
    ax.legend(wedges, legend, loc="center left", bbox_to_anchor=(0.88, 0.5), frameon=False)
    ax.set_title(title, fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_interval_chart(table: pd.DataFrame, column: str, title: str, output: Path) -> None:
    displayed = table.loc[table["sessions"] >= 100].sort_values("sessions", ascending=False).head(12)
    displayed = displayed.sort_values("conversion_rate_pct")
    fig, ax = plt.subplots(figsize=(9.0, max(4.0, len(displayed) * 0.45 + 1.3)))
    if not displayed.empty:
        y = np.arange(len(displayed))
        rates = displayed["conversion_rate_pct"].to_numpy()
        error = np.stack((rates - displayed["wilson_95_lower_pct"].to_numpy(),
                          displayed["wilson_95_upper_pct"].to_numpy() - rates))
        ax.errorbar(rates, y, xerr=error, fmt="o", capsize=3, color="#275c84", ecolor="#7792a8")
        ax.set_yticks(y, [f"{column} {v}  (n={n:,})" for v, n in zip(displayed[column], displayed["sessions"])])
        ax.set_xlim(left=0)
        ax.set_xlabel("Purchase conversion (%) and Wilson 95% interval")
    else:
        ax.text(0.5, 0.5, "No groups with >=100 sessions", transform=ax.transAxes, ha="center")
        ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_page_distribution(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    upper = float(np.percentile(df["ProductRelated"], 99))
    bins = np.linspace(0, max(10, upper), 32)
    for outcome, title in [(0, "Non-purchasing"), (1, "Purchasing")]:
        x = df.loc[df["purchased"] == outcome, "ProductRelated"].clip(upper=upper)
        ax.hist(x, bins=bins, density=True, alpha=0.48, label=title)
    ax.set(xlabel="Product pages viewed (capped at overall 99th percentile for display)",
           ylabel="Relative density", title="Product-page engagement by session outcome")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(source: Path = SOURCE, output: Path = OUTPUT) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"Dataset missing: {source}")
    data = prepare(pd.read_csv(source))
    tables_dir = output / "tables"
    charts_dir = output / "charts"
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    table_fields = {
        "region": "Region", "browser": "Browser", "operating_system": "OperatingSystems",
        "visitor_type": "VisitorType", "product_depth": "page_depth", "special_day": "special_day_band",
        "weekend": "weekend_label", "traffic_type": "TrafficType",
    }
    tables = {}
    for name, col in table_fields.items():
        tables[name] = summarize(data, col)
        tables[name].to_csv(tables_dir / f"conversion_by_{name}.csv", index=False)
    save_donut(["Purchasing", "Non-purchasing"],
               [int(data["purchased"].sum()), int((1 - data["purchased"]).sum())],
               "Purchase completion share", charts_dir / "purchase_share_donut.png")
    visitor = tables["visitor_type"].sort_values("sessions", ascending=False)
    save_donut(visitor["VisitorType"].astype(str).tolist(), visitor["sessions"].astype(int).tolist(),
               "Recorded visitor-category composition", charts_dir / "visitor_mix_donut.png")
    depth_order = ["0 pages", "1–5 pages", "6–20 pages", "21+ pages"]
    depth = tables["product_depth"].set_index("page_depth").reindex(depth_order).dropna(subset=["sessions"])
    save_donut(depth.index.astype(str).tolist(), depth["sessions"].astype(int).tolist(),
               "Session composition by product-page depth", charts_dir / "page_depth_donut.png")
    save_interval_chart(tables["region"], "Region", "Conversion across recorded regions", charts_dir / "region_intervals.png")
    save_interval_chart(tables["browser"], "Browser", "Conversion across recorded browser codes", charts_dir / "browser_intervals.png")
    save_page_distribution(data, charts_dir / "product_pages_by_outcome.png")
    purchase_total = int(data["purchased"].sum())
    overall_low, overall_high = wilson_interval(purchase_total, len(data))
    report = [
        "# Advanced segmentation — recorded shopping sessions", "",
        f"- Sessions: **{len(data):,}**; purchasing: **{purchase_total:,}**; observed conversion: **{purchase_total / len(data):.2%}**.",
        f"- Approximate 95% Wilson interval for overall session conversion: **{overall_low:.2f}–{overall_high:.2f}%**.",
        "- Eight categorical breakdowns reconcile to the same full sample; no records are deleted.",
        "- Groups below 100 sessions remain in tables and are flagged; interval charts display groups with at least 100 sessions.",
        "- Wilson intervals are descriptive binomial intervals under an independence assumption; they do not account for unobserved dependencies.",
        "- Browser, region, operating-system, and traffic identifiers are categorical codes, not numeric rankings.",
        "- Donut charts show composition, not comparative conversion; conversion charts include denominators and uncertainty intervals.",
        "- Product-page visualization caps values only for display; all source values remain in calculations.",
        "- Segment differences are observational; no experiment, revenue, or causal impact is inferred.",
        "", "Source: Sakar & Kastro (2018), UCI Online Shoppers Purchasing Intention Dataset (CC BY 4.0).", "",
    ]
    (output / "summary.md").write_text("\n".join(report), encoding="utf-8")
    return {"sessions": len(data), "purchases": purchase_total, "tables": len(tables), "charts": 6}


def main() -> None:
    summary = run()
    print(f"PASS - Advanced segmentation: {summary['sessions']:,} sessions, {summary['tables']} tables, {summary['charts']} charts")


if __name__ == "__main__":
    main()
