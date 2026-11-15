from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OWNER = "Cyril James Lunzaga"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset"


def load_evidence(output: Path) -> dict:
    required = ["audit/data_audit.json", "statistical_review/visitor_conversion_gap.csv",
                "statistical_review/categorical_tests.csv", "statistical_review/numeric_tests.csv",
                "advanced_segmentation/tables/conversion_by_region.csv",
                "modeling/metrics.json", "modeling/tables/model_comparison.csv",
                "modeling/tables/pagevalues_sensitivity.csv",
                "advanced_segmentation/tables/conversion_by_product_depth.csv",
                "advanced_segmentation/tables/conversion_by_weekend.csv",
                "advanced_segmentation/tables/conversion_by_special_day.csv"]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError("Run the full analysis first. Missing: " + ", ".join(missing))
    audit = json.loads((output / required[0]).read_text(encoding="utf-8"))
    gap = pd.read_csv(output / required[1]).iloc[0]
    categorical = pd.read_csv(output / required[2])
    numeric = pd.read_csv(output / required[3])
    regions = pd.read_csv(output / required[4])
    model = json.loads((output / required[5]).read_text(encoding="utf-8"))
    comparison = pd.read_csv(output / required[6])
    sensitivity = pd.read_csv(output / required[7])
    depth = pd.read_csv(output / required[8])
    weekend = pd.read_csv(output / required[9])
    special = pd.read_csv(output / required[10])
    if (int(audit["sessions"]) != int(model["source_sessions"]) or
            int(audit["purchase_sessions"]) != int(model["source_purchases"]) or
            int(regions["sessions"].sum()) != int(audit["sessions"]) or
            int(regions["purchases"].sum()) != int(audit["purchase_sessions"])):
        raise ValueError("Source audit, modeling and regional counts do not reconcile")
    if int(model["training_sessions"]) + int(model["test_sessions"]) != int(audit["sessions"]):
        raise ValueError("Train/test counts do not reconcile")
    for label, frame in (("depth", depth), ("weekend", weekend), ("special-day", special)):
        if int(frame["sessions"].sum()) != int(audit["sessions"]) or int(frame["purchases"].sum()) != int(audit["purchase_sessions"]):
            raise ValueError(f"Source and {label} counts do not reconcile")
    if len(comparison) != 3 or len(sensitivity) != 2:
        raise ValueError("Model results incomplete")
    if model["chosen_model"] not in ("Logistic regression", "Random forest"):
        raise ValueError("Invalid selected model")
    return dict(audit=audit, gap=gap, categorical=categorical, numeric=numeric,
                regions=regions, model=model, comparison=comparison, sensitivity=sensitivity,
                depth=depth, weekend=weekend, special=special)


def build(output: Path = OUT, readme: Path | None = None) -> tuple[Path, Path]:
    e = load_evidence(output)
    a = e["audit"]
    gap = e["gap"]
    model = e["model"]
    comparison = e["comparison"]
    winner = comparison.set_index("model").loc[model["chosen_model"]]
    baseline = comparison.set_index("model").loc["Majority-prior baseline"]
    biggest_cat = e["categorical"].sort_values("cramers_v", ascending=False).iloc[0]
    effects = e["numeric"].assign(abs_effect=e["numeric"]["rank_biserial_purchasing_vs_nonpurchasing"].abs())
    biggest_numeric = effects.sort_values("abs_effect", ascending=False).iloc[0]
    def segment_row(frame: pd.DataFrame, field: str, value: str) -> pd.Series:
        selected = frame.loc[frame[field].eq(value)]
        if len(selected) != 1:
            raise ValueError(f"Expected one segment for {field}={value!r}")
        return selected.iloc[0]

    depth_low = segment_row(e["depth"], "page_depth", "1–5 pages")
    depth_high = segment_row(e["depth"], "page_depth", "21+ pages")
    weekday = segment_row(e["weekend"], "weekend_label", "Weekday")
    weekend = segment_row(e["weekend"], "weekend_label", "Weekend")
    special_zero = segment_row(e["special"], "special_day_band", "Zero proximity")
    special_nonzero = segment_row(e["special"], "special_day_band", "Nonzero proximity")
    segment_findings = (
        f"**Browsing depth.** Sessions with 21+ product pages had observed purchase completion of "
        f"**{depth_high['conversion_rate_pct']:.2f}%** ({int(depth_high['purchases']):,}/{int(depth_high['sessions']):,}), "
        f"versus **{depth_low['conversion_rate_pct']:.2f}%** ({int(depth_low['purchases']):,}/{int(depth_low['sessions']):,}) "
        "for 1–5 pages. This is a full-session association, not proof that increasing page views creates purchases. "
        "See the [depth table](outputs/advanced_segmentation/tables/conversion_by_product_depth.csv).\n\n"
        f"**Weekend label.** Weekend sessions converted at **{weekend['conversion_rate_pct']:.2f}%** "
        f"({int(weekend['purchases']):,}/{int(weekend['sessions']):,}) versus "
        f"**{weekday['conversion_rate_pct']:.2f}%** ({int(weekday['purchases']):,}/{int(weekday['sessions']):,}) "
        "on weekdays. This comparison is unadjusted for traffic mix and month; it is not a causal weekend effect. "
        "See the [weekend table](outputs/advanced_segmentation/tables/conversion_by_weekend.csv).\n\n"
        f"**SpecialDay indicator.** Sessions with nonzero SpecialDay proximity converted at "
        f"**{special_nonzero['conversion_rate_pct']:.2f}%** "
        f"({int(special_nonzero['purchases']):,}/{int(special_nonzero['sessions']):,}) "
        f"versus **{special_zero['conversion_rate_pct']:.2f}%** "
        f"({int(special_zero['purchases']):,}/{int(special_zero['sessions']):,}) for zero proximity. "
        "SpecialDay is a source-defined proximity feature, not a verified calendar event or promotion indicator. "
        "See the [SpecialDay table](outputs/advanced_segmentation/tables/conversion_by_special_day.csv)."
    )
    segment_findings_case = segment_findings.replace(
        "(outputs/advanced_segmentation/", "(advanced_segmentation/"
    )
    test_n = int(model["test_sessions"])
    cv_ap = float(model["selected_cv_average_precision"])
    ap = float(winner["average_precision"])
    ap_base = float(baseline["average_precision"])
    chosen_threshold = float(model["train_cv_selected_threshold"])
    chosen_policy = model["selected_threshold_heldout_metrics"]
    metrics_rows = [
        "| Model | Test average precision | ROC-AUC | Precision @0.50 | Recall @0.50 | Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.itertuples(index=False):
        metrics_rows.append(f"| {row.model} | {row.average_precision:.3f} | {row.roc_auc:.3f} | {row.precision:.3f} | {row.recall:.3f} | {row.brier_score:.3f} |")
    metrics_markdown = "\n".join(metrics_rows)
    finding_text = (
        f"The model selected using **training-only cross-validation** was **{model['chosen_model']}** "
        f"(mean CV average precision **{cv_ap:.3f}**). On the **{test_n:,}-session held-out group split**, "
        f"average precision was **{ap:.3f}**, compared with **{ap_base:.3f}** for the prior-only baseline; "
        f"ROC-AUC was **{float(winner['roc_auc']):.3f}**. "
        "This is retrospective session classification, not a deployed or time-forward validated forecasting system."
    )
    source = f"[UCI Online Shoppers Purchasing Intention Dataset]({SOURCE_URL})"
    readme_text = f"""# Browsing Without Buying

**E-commerce conversion analytics · statistical evidence · session-level classification**

**Project author:** {OWNER}

## Executive summary

Why do some recorded shopping sessions end in a purchase while others do not? This project uses reproducible descriptive analysis, uncertainty estimates, statistical tests, and machine-learning comparisons to examine the recorded sessions. It distinguishes **observed relationships** from **predictive classification performance** and from interventions that the data cannot support.

| Key metric | Observed value |
| --- | ---: |
| Recorded sessions | **{int(a['sessions']):,}** |
| Purchasing sessions | **{int(a['purchase_sessions']):,}** |
| Non-purchasing sessions | **{int(a['nonpurchase_sessions']):,}** |
| Observed session conversion | **{float(a['conversion_rate_pct']):.2f}%** |
| Source attributes | **{int(a['columns'])}** (17 inputs + Revenue) |
| Missing cells | **{int(a['missing_cells'])}** |
| Exact-match later rows, retained | **{int(a['exact_duplicate_rows'])}** |

## Project objectives

1. Audit the original dataset and quantify data-quality and duplicate-record sensitivity.
2. Compare conversion across visitor type, browsing depth, traffic sources, browser and regional categories, and other recorded session attributes.
3. Evaluate the strength and uncertainty of exploratory associations using effect sizes, intervals, and multiple-testing control.
4. Compare a prior-only baseline, logistic regression, and random forest using a group-isolated holdout and training-only hyperparameter selection.
5. Turn the evidence into practical hypotheses for **future testing**, without inventing customer journeys, cart abandonment, revenue or causal effects.

## Dataset and analytical scope

- **Source:** {source}, Sakar & Kastro (2018), DOI [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q); license **CC BY 4.0**.
- **Unit:** recorded shopping session; `Revenue` is a binary indicator that the recorded session ended in a purchase.
- **Input:** 18 CSV columns. Categorical integer codes are treated as labels, not ordinal magnitudes.
- **Integrity:** original records are retained. Excluding later exact-match rows changes overall conversion by **{float(a['duplicate_sensitivity_pp']):+.3f} percentage points**, but matching features do not establish duplicate events.
- **Leakage boundary:** `PageValues` is excluded from principal models and explanatory results because its definition is transaction-linked. A separate sensitivity experiment documents the difference.
- **Data ownership:** the raw CSV stays under `data/raw/` locally and is never committed to Git.

## Findings from exploratory analysis

**Visitor category.** Recorded New_Visitor sessions converted at **{float(gap['new_conversion_pct']):.2f}%** ({int(gap['new_purchases']):,}/{int(gap['new_sessions']):,}), versus **{float(gap['returning_conversion_pct']):.2f}%** ({int(gap['returning_purchases']):,}/{int(gap['returning_sessions']):,}) for Returning_Visitor sessions, a **{float(gap['difference_percentage_points']):+.2f} percentage-point** descriptive difference. A 2,000-replicate session bootstrap produced an exploratory 95% interval of **{float(gap['bootstrap_95pct_lower_pp']):+.2f} to {float(gap['bootstrap_95pct_upper_pp']):+.2f} percentage points**.

**Statistical associations.** The largest Cramér's V in the prespecified categorical test family was **{biggest_cat['variable']}** (**{float(biggest_cat['cramers_v']):.3f}**); the largest absolute numeric rank-biserial effect was **{biggest_numeric['variable']}** (**{float(biggest_numeric['rank_biserial_purchasing_vs_nonpurchasing']):+.3f}**). The original eight tests use Benjamini–Hochberg adjustment. These comparisons do not prove causes.

**Additional segmentation.** The upgrade adds region, browser, operating-system and special-day breakdowns, Wilson intervals for categorical conversion estimates, and composition plots. Groups with fewer than 100 records remain in CSV evidence but are flagged as sparse; interval charts emphasize adequately sized groups. Codes lack published business labels and should not be translated into invented audience profiles.

### Additional quantified patterns

{segment_findings}

### Session composition and conversion

![Purchase and non-purchase session share](outputs/advanced_segmentation/charts/purchase_share_donut.png)

![Visitor-category composition](outputs/advanced_segmentation/charts/visitor_mix_donut.png)

![Product-page depth composition](outputs/advanced_segmentation/charts/page_depth_donut.png)

![Purchase rate by visitor category](outputs/charts/conversion_by_visitor.png)

![Purchase rate by browsing depth](outputs/charts/conversion_by_page_depth.png)

![Purchase conversion across regions with Wilson intervals](outputs/advanced_segmentation/charts/region_intervals.png)

![Browsing-depth distribution by outcome](outputs/advanced_segmentation/charts/product_pages_by_outcome.png)

For the full collection, see [visitor–depth heatmap](outputs/deep_dive/charts/visitor_depth_heatmap.png), [bounce-rate bands](outputs/deep_dive/charts/conversion_by_bounce_band.png), [traffic-type chart](outputs/deep_dive/charts/high_volume_traffic_codes.png), [month comparison](outputs/charts/conversion_by_month.png), and [browser intervals](outputs/advanced_segmentation/charts/browser_intervals.png).

## Predictive algorithms and tuning

{finding_text}

{metrics_markdown}

At the fixed **0.50** decision threshold, the CV-selected model's held-out recall is **{float(winner['recall']):.1%}**. This is a crucial operational limitation: good ranking performance need not translate into strong recall at a default cutoff. A separate decision threshold of **{chosen_threshold:.3f}**, selected by training-only out-of-fold F1 (not test performance), yields held-out precision **{float(chosen_policy['precision']):.3f}**, recall **{float(chosen_policy['recall']):.3f}**, and F1 **{float(chosen_policy['f1']):.3f}**. These policies are compared in the [threshold evaluation table](outputs/modeling/tables/threshold_sensitivity.csv). Threshold tuning changes the precision–recall trade-off, not the underlying ranking quality or deployment readiness. **Average precision**, supported by ROC-AUC, confusion matrices and Brier score, is emphasized because only about {float(a['conversion_rate_pct']):.1f}% of sessions are purchases.

**Model design:** nominal category one-hot encoding and numeric standardization in a scikit-learn pipeline; exact matching input profiles assigned to the same partition; approximately one fifth held out via five-fold stratified group splitting; three-fold group-aware cross-validation on the training portion; a small, documented hyperparameter grid for logistic regression and random forest. The model family is selected using **training CV scores**, not held-out metrics. The selected model's held-out average precision has a 400-replicate, class-stratified bootstrap interval of **{model['chosen_test_ap_interval_95'][0]:.3f}–{model['chosen_test_ap_interval_95'][1]:.3f}**, conditional on session-level independence.

![Precision–recall comparison on the holdout](outputs/modeling/charts/precision_recall.png)

![ROC comparison](outputs/modeling/charts/roc_curves.png)

![Selected classifier confusion matrix, fixed 0.50](outputs/modeling/charts/confusion_matrix.png)

![Selected classifier confusion matrix, training-selected threshold](outputs/modeling/charts/confusion_matrix_cv_threshold.png)

![Held-out calibration](outputs/modeling/charts/calibration.png)

![Permutation-based feature importance](outputs/modeling/charts/permutation_importance.png)

Read the [model evaluation](outputs/modeling/model_report.md), [model comparison table](outputs/modeling/tables/model_comparison.csv), [training CV tuning table](outputs/modeling/tables/cross_validation_tuning.csv), [permutation importance](outputs/modeling/tables/permutation_importance.csv), and [transaction-linked PageValues sensitivity](outputs/modeling/tables/pagevalues_sensitivity.csv).

## Execution and technical processing

The pipeline reads the original source CSV, validates its shape and labels, retains all source records, generates descriptive summaries, computes statistical evidence, creates advanced segmentation figures, trains and evaluates classifiers, then builds the README and consolidated case-study report **using the actual numerical outputs**.

```text
source CSV (local, ignored by Git)
    → audit + original descriptive analysis + statistical evidence
    → advanced categorical segmentation + uncertainty plots
    → group-isolated train/test modeling + train-only CV tuning
    → analytical reports + README + reproducibility verification
```

**Run on Windows PowerShell** from the repository root:

```powershell
py -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m scripts.run_upgrade --verify
```

The last command reruns the existing and upgraded pipeline, validates and executes five notebooks, runs the automated test suite, and checks that reports and figures were produced. The notebook executor does not save execution outputs back into the tracked notebooks. For Linux/macOS, use the matching virtual-environment Python interpreter. For manual execution, inspect [the reproducibility and modeling guide](docs/advanced_methodology.md).

**Notebooks:** [01 Source quality](notebooks/01_data_audit.ipynb) · [02 Browsing patterns](notebooks/02_browsing_patterns.ipynb) · [03 Statistical evidence](notebooks/03_statistical_evidence.ipynb) · [04 Segment uncertainty](notebooks/04_advanced_segmentation.ipynb) · [05 Modeling assessment](notebooks/05_predictive_modeling.ipynb).

## Implications, limitations and conclusion

- Segments with reproducible descriptive differences can motivate **A/B-test hypotheses**, usability reviews, or more precise event-data collection. No intervention has been tested here.
- Model evaluation is **retrospective full-session discrimination**. Product-page totals and durations reflect behavior observed across the session, so this is **not** a proven pre-session or real-time purchase predictor.
- User identities, raw event sequences, cart events, exact timestamps, transaction amounts, and an external validation site are absent; customer retention, cart abandonment, revenue loss and causal ROI cannot be estimated.
- The source's one-session-per-user design is reported by UCI but cannot be independently confirmed from the supplied CSV. Random group splitting prevents identical measured input profiles crossing the holdout, not unobserved customer or temporal dependence.
- Month labels are incomplete as a calendar series, and small categorical groups can show unstable estimates. No customer-targeting policy should be built from these observational segments.

**Conclusion:** The dataset supports a rigorous description of purchase-completion patterns and a reproducible internal classification benchmark. Its limitations prevent confident intervention or real-world deployment claims. Stronger next-stage evidence would require timestamped events, identified consented cohorts, a defined decision point, forward-time validation, and randomized experiments.

## Evidence, methods and credits

- [Full advanced case study](outputs/advanced_case_study.md)
- [Original case study](outputs/portfolio_case_study.md)
- [Data audit](outputs/audit/data_audit.md) · [Original statistical review](outputs/statistical_review/statistical_summary.md) · [Segmentation summary](outputs/advanced_segmentation/summary.md)
- [Original conversion methodology](docs/conversion_methodology.md) · [Statistical methods](docs/statistical_methods.md) · [Modeling and reproducibility](docs/advanced_methodology.md)
- **Author:** {OWNER}
- **Dataset:** Sakar, C. & Kastro, Y. (2018), {source}, CC BY 4.0. Code and analytical interpretation are independent of the dataset authors.
"""
    case_study = f"""# Browsing Without Buying — Extended analytical case study

**Author:** {OWNER}

## Business problem and scope

Determine which recorded shopping-session characteristics differ between purchases and non-purchases, quantify uncertainty, and establish whether full-session measurements can discriminate purchasing outcomes better than a no-skill prevalence baseline. This is not a study of cart abandonment, revenue loss, identified customers, or causal interventions.

## Audited dataset

- Original UCI source rows: **{int(a['sessions']):,}**; purchasing: **{int(a['purchase_sessions']):,}**; conversion: **{float(a['conversion_rate_pct']):.2f}%**.
- Missing source cells: **{int(a['missing_cells'])}**; exact-match later rows retained: **{int(a['exact_duplicate_rows'])}**.
- Raw data hash: `{a['sha256']}`.
- Original-record deduplication sensitivity: **{float(a['duplicate_sensitivity_pp']):+.3f} percentage points**.

## Exploratory and statistical evidence

- New versus returning session conversion: **{float(gap['new_conversion_pct']):.2f}%** versus **{float(gap['returning_conversion_pct']):.2f}%**, difference **{float(gap['difference_percentage_points']):+.2f} pp**, exploratory bootstrap interval **[{float(gap['bootstrap_95pct_lower_pp']):+.2f}, {float(gap['bootstrap_95pct_upper_pp']):+.2f}] pp**.
- Largest categorical effect in the prespecified family: **{biggest_cat['variable']}**, Cramér's V **{float(biggest_cat['cramers_v']):.3f}**.
- Largest absolute numeric rank-biserial effect: **{biggest_numeric['variable']}**, signed value **{float(biggest_numeric['rank_biserial_purchasing_vs_nonpurchasing']):+.3f}**.
- Original eight tests use shared BH adjustment. Further region, browser, operating-system and session-composition analyses are explicitly exploratory and descriptive.

## Additional descriptive comparisons

{segment_findings_case}

## Modeling evidence

{finding_text}

{metrics_markdown}

- Test AP interval, exploratory conditional-on-sessions bootstrap: **[{model['chosen_test_ap_interval_95'][0]:.3f}, {model['chosen_test_ap_interval_95'][1]:.3f}]**.
- At the fixed 0.50 cutoff, test precision **{float(winner['precision']):.3f}** and recall **{float(winner['recall']):.3f}**. The training-OOF-F1-selected cutoff **{chosen_threshold:.3f}** gives held-out precision **{float(chosen_policy['precision']):.3f}**, recall **{float(chosen_policy['recall']):.3f}**, F1 **{float(chosen_policy['f1']):.3f}**. These results are retrospective threshold diagnostics, not a production decision rule.
- Group-isolated holdout has **{int(model['training_sessions']):,}** training and **{test_n:,}** testing records, with no identical feature-group overlap.
- `PageValues` is excluded from principal modeling; [leakage sensitivity](modeling/tables/pagevalues_sensitivity.csv) is kept separate.

## Interpretation

Identified correlations cannot establish that changing visitor behavior causes conversion. Full-session classifiers cannot be marketed as real-time prediction without validating a specific point within the session, prospective features and forward-time generalization. No actual intervention, checkout event, user identifier or monetary measure is present.

## Evidence links

- [Statistical report](statistical_review/statistical_summary.md)
- [Advanced segmentation](advanced_segmentation/summary.md)
- [Model diagnostics](modeling/model_report.md)
- [Selected model PR chart](modeling/charts/precision_recall.png)
- [Model importance](modeling/charts/permutation_importance.png)
- [Advanced modeling methods](../docs/advanced_methodology.md)

Source: Sakar & Kastro (2018), {source}, CC BY 4.0.
"""
    readme_path = readme or ROOT / "README.md"
    readme_path.write_text(readme_text, encoding="utf-8")
    case_path = output / "advanced_case_study.md"
    case_path.write_text(case_study, encoding="utf-8")
    return readme_path, case_path


def main() -> None:
    readme, report = build()
    print(f"PASS - README generated from evidence: {readme}")
    print(f"PASS - Extended case study generated: {report}")


if __name__ == "__main__":
    main()
