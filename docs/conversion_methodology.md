# Conversion analysis — methodology checkpoint

## Unit of analysis

Each record represents a recorded shopping session. `Revenue=True` denotes a session ending in a purchase. The primary denominator is **all source sessions**, including rows with identical recorded values; the separate data audit reports deduplication sensitivity.

## Primary descriptive comparisons

- `VisitorType` × product-page depth (`0`, `1–5`, `6–20`, `21+` pages)
- `VisitorType` × weekday/weekend indicator
- Traffic source codes (`TrafficType`) treated as categorical identifiers
- Bounce-rate bands: exactly zero, (0, 0.02], (0.02, 0.05], and above 0.05
- Product-page count, duration, bounce rate, and exit rate summarized by purchase outcome using median, quartiles, and mean

For each categorical cell, **conversion rate = 100 × purchasing sessions / sessions in the cell**. All category tables reconcile their sample sizes and purchase counts to the dataset totals. Group sizes under 100 are flagged as exploratory. This threshold is a practical display rule, not a statistical significance criterion.

## Interpretation safeguards

The month field has only 10 observed categories and no complete multi-year time series. Session conversion does not measure cart abandonment or distinct-customer purchasing frequency. Associations cannot establish causes or expected results from changing site behavior.

The `PageValues` variable is excluded from the primary explanatory comparisons because its source definition is transaction-linked. Traffic type integers are not treated as continuous quantities. Descriptive thresholds and band boundaries are analytical choices rather than externally validated decision rules.

## Dataset attribution

Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. [Dataset page](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset). DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). License: CC BY 4.0. Raw source files are not included in Git.

## Execution

The root `analysis.py` and `scripts/audit_sessions.py` remain independent. Run `scripts/conversion_deep_dive.py` to write five analytical tables, three charts, and a generated Markdown summary to `outputs/deep_dive/`. Generated results are untracked until deliberately curated for portfolio presentation.
