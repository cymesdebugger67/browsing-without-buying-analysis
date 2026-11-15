# Statistical methods — Browsing Without Buying

This phase compares **recorded shopping sessions**. UCI describes a one-session-per-user sampling design, but the released data has no identifiers to independently verify this claim. The source is the UCI Online Shoppers Purchasing Intention Dataset (Sakar & Kastro, 2018; CC BY 4.0). All rows remain in the primary sample; the 125 later exact matches are documented separately in the data audit.

## Planned comparisons

- **Categorical:** VisitorType, weekday/weekend, prespecified product-page depth bands, and prespecified bounce-rate bands versus session purchase completion. Pearson chi-square tests (no continuity correction) are summarized with **Cramér's V**, group counts, and a diagnostic of low expected cell counts. Small-cell comparisons must be treated cautiously.
- **Numeric:** ProductRelated, ProductRelated_Duration, BounceRates, and ExitRates across purchasing/non-purchasing sessions. Two-sided **Mann–Whitney U** comparisons are summarized with medians and signed **rank-biserial correlations**. Many tied or zero values may complicate interpretation; these are distribution comparisons, not necessarily median tests.
- **Multiplicity:** Apply **Benjamini–Hochberg** correction to the entire prespecified family of eight exploratory p-values. Both raw and adjusted values appear in CSV outputs.
- **Visitor gap:** Compare New_Visitor and Returning_Visitor purchase rates and calculate an exploratory 95% **within-group session bootstrap** percentile interval (2,000 replicates, fixed seed). The interval assumes sessions are independent within groups; the dataset provides no cross-session visitor IDs to verify that assumption.

TrafficType has sparse individual codes and remains descriptive rather than entering this planned test family. `PageValues` is excluded from primary comparisons because its construction is transaction-linked. These are **associations**, not causes. No model estimates interventions or predicts purchases.

## Reproduce

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\scripts\statistical_review.py
.\.venv\Scripts\python.exe -m pytest -q --basetemp=".venv/pytest-tmp"
```

Outputs appear in `outputs/statistical_review/`:

- `categorical_tests.csv`
- `numeric_tests.csv`
- `visitor_conversion_gap.csv`
- `statistical_summary.md`

Results and conclusions should be reviewed against the actual CSV outputs before quoting them in the public portfolio.
