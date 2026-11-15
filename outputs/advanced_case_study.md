# Browsing Without Buying — Extended analytical case study

**Author:** Cyril James Lunzaga

## Business problem and scope

Determine which recorded shopping-session characteristics differ between purchases and non-purchases, quantify uncertainty, and establish whether full-session measurements can discriminate purchasing outcomes better than a no-skill prevalence baseline. This is not a study of cart abandonment, revenue loss, identified customers, or causal interventions.

## Audited dataset

- Original UCI source rows: **12,330**; purchasing: **1,908**; conversion: **15.47%**.
- Missing source cells: **0**; exact-match later rows retained: **125**.
- Raw data hash: `B3055EE355F59134D851D32641183CB4A8B45DEF7124D2F50442A042F358E0D9`.
- Original-record deduplication sensitivity: **+0.158 percentage points**.

## Exploratory and statistical evidence

- New versus returning session conversion: **24.91%** versus **13.93%**, difference **+10.98 pp**, exploratory bootstrap interval **[+8.81, +13.16] pp**.
- Largest categorical effect in the prespecified family: **page_depth**, Cramér's V **0.186**.
- Largest absolute numeric rank-biserial effect: **ExitRates**, signed value **-0.406**.
- Original eight tests use shared BH adjustment. Further region, browser, operating-system and session-composition analyses are explicitly exploratory and descriptive.

## Additional descriptive comparisons

**Browsing depth.** Sessions with 21+ product pages had observed purchase completion of **21.96%** (1,229/5,597), versus **4.12%** (96/2,331) for 1–5 pages. This is a full-session association, not proof that increasing page views creates purchases. See the [depth table](advanced_segmentation/tables/conversion_by_product_depth.csv).

**Weekend label.** Weekend sessions converted at **17.40%** (499/2,868) versus **14.89%** (1,409/9,462) on weekdays. This comparison is unadjusted for traffic mix and month; it is not a causal weekend effect. See the [weekend table](advanced_segmentation/tables/conversion_by_weekend.csv).

**SpecialDay indicator.** Sessions with nonzero SpecialDay proximity converted at **6.16%** (77/1,251) versus **16.53%** (1,831/11,079) for zero proximity. SpecialDay is a source-defined proximity feature, not a verified calendar event or promotion indicator. See the [SpecialDay table](advanced_segmentation/tables/conversion_by_special_day.csv).

## Modeling evidence

The model selected using **training-only cross-validation** was **Random forest** (mean CV average precision **0.389**). On the **2,466-session held-out group split**, average precision was **0.361**, compared with **0.155** for the prior-only baseline; ROC-AUC was **0.771**. This is retrospective session classification, not a deployed or time-forward validated forecasting system.

| Model | Test average precision | ROC-AUC | Precision @0.50 | Recall @0.50 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority-prior baseline | 0.155 | 0.500 | 0.000 | 0.000 | 0.131 |
| Logistic regression | 0.323 | 0.741 | 0.636 | 0.037 | 0.119 |
| Random forest | 0.361 | 0.771 | 0.594 | 0.050 | 0.115 |

- Test AP interval, exploratory conditional-on-sessions bootstrap: **[0.320, 0.401]**.
- At the fixed 0.50 cutoff, test precision **0.594** and recall **0.050**. The training-OOF-F1-selected cutoff **0.225** gives held-out precision **0.332**, recall **0.592**, F1 **0.425**. These results are retrospective threshold diagnostics, not a production decision rule.
- Group-isolated holdout has **9,864** training and **2,466** testing records, with no identical feature-group overlap.
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

Source: Sakar & Kastro (2018), [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), CC BY 4.0.
