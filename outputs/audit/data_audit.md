# Data audit — Browsing Without Buying

## Source and scope

- Source: UCI Online Shoppers Purchasing Intention Dataset (Sakar & Kastro, 2018).
- UCI dataset: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- License: CC BY 4.0; raw CSV is not committed to Git.
- Local source SHA-256: `B3055EE355F59134D851D32641183CB4A8B45DEF7124D2F50442A042F358E0D9`.
- Unit of analysis: a recorded shopping session; the label is session purchase completion.

## Observed data quality

- Sessions: **12,330** across **18** columns.
- Missing cells: **0**.
- Purchasing sessions: **1,908**; non-purchasing: **10,422**.
- Observed conversion: **15.475%**.
- Exact duplicate feature rows: **125**, including **0** purchasing duplicates.
- Removing subsequent exact duplicates changes conversion to **15.633%**, a shift of **+0.158 percentage points**.
- Zero PageValues: **9,600** sessions (**77.86%**).
- Months represented: Aug, Dec, Feb, Jul, June, Mar, May, Nov, Oct, Sep.
- Negative values in tested nonnegative fields: **0**.
- Values outside 0–1 in tested rate/proximity fields: **0**.
- Unparseable numeric cells in tested fields: **0**.

## Decisions and limitations

- Retain all source sessions for primary results: identical measured features do not prove duplicate events.
- Treat deduplication as a sensitivity analysis, not an automatic cleaning operation.
- Treat operating system, browser, region and traffic type integers as category identifiers.
- Do not infer a year-over-year trend or a complete seasonal cycle from the available month labels.
- PageValues is related to transaction-linked page value and must not be presented as an independent causal driver of purchases.
- Associations in this observational dataset do not establish causation.

See `column_profile.csv` for column-level types, missingness, and cardinality.
