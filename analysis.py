# %% Imports and paths
"""Descriptive online shopping conversion analysis. Run: python analysis.py"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/raw/online_shoppers_intention.csv"
OUT = ROOT / "outputs"


def summarize(df, group):
    out = (df.groupby(group, observed=True, dropna=False)
           .agg(sessions=("purchased", "size"), conversion_rate=("purchased", "mean"))
           .reset_index())
    out["conversion_rate_pct"] = (out.pop("conversion_rate") * 100).round(2)
    return out


def bar(table, group, name, title):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(table[group].astype(str), table["conversion_rate_pct"])
    ax.set(ylabel="Sessions with purchase (%)", title=title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUT / "charts" / name, dpi=150)
    plt.close(fig)


def main():
    if not SOURCE.exists():
        raise SystemExit(f"Dataset missing: {SOURCE}\nGet the exact CSV from the README.md link.")
    df = pd.read_csv(SOURCE)
    required = {"Revenue", "VisitorType", "Month", "ProductRelated"}
    if required - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    raw_rows = len(df)
    # Revenue is Boolean in the source; explicit normalization also supports CSV text.
    flag = df["Revenue"].astype("string").str.strip().str.lower()
    df = df.loc[flag.isin(["true", "false", "1", "0"])].copy()
    df["purchased"] = flag.loc[df.index].isin(["true", "1"]).astype(int)
    df["ProductRelated"] = pd.to_numeric(df["ProductRelated"], errors="coerce")
    df["page_depth"] = pd.cut(df["ProductRelated"], [-1, 0, 5, 20, float("inf")],
                              labels=["0 pages", "1–5 pages", "6–20 pages", "21+ pages"])
    df["VisitorType"] = df["VisitorType"].astype("string").str.strip()
    df["Month"] = df["Month"].astype("string").str.strip()
    if df.empty:
        raise ValueError("No sessions with valid Revenue values")
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "charts").mkdir(parents=True, exist_ok=True)
    visitors = summarize(df, "VisitorType").sort_values("sessions", ascending=False)
    month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    months = summarize(df, "Month")
    months["Month"] = pd.Categorical(months["Month"], categories=month_order, ordered=True)
    months = months.sort_values("Month")
    depth = summarize(df.dropna(subset=["page_depth"]), "page_depth")
    for name, data in [("conversion_by_visitor_type", visitors), ("conversion_by_month", months),
                       ("conversion_by_page_depth", depth)]:
        data.to_csv(OUT / "tables" / f"{name}.csv", index=False)
    bar(visitors, "VisitorType", "conversion_by_visitor.png", "Conversion by visitor type")
    bar(months, "Month", "conversion_by_month.png", "Conversion by month")
    bar(depth, "page_depth", "conversion_by_page_depth.png", "Conversion by pages viewed")
    report = f"""# Findings — Browsing Without Buying

- Source rows: {raw_rows:,}; sessions analyzed: {len(df):,}.
- Overall observed session conversion rate: **{df['purchased'].mean() * 100:.1f}%**.
- Visitors, month, and page-depth comparisons are saved to `tables/` and `charts/`.

## Interpretation and next step
Differences between groups describe correlation, not causal conversion drivers. Each row is a session, **not necessarily a distinct person**. Do not use `PageValues` to argue that browsing behavior causes revenue.
"""
    (OUT / "findings.md").write_text(report, encoding="utf-8")
    print(f"Done. {len(df):,} sessions analyzed. Results: {OUT}")


if __name__ == "__main__":
    main()
