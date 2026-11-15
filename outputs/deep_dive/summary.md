# Session conversion — deeper descriptive analysis

- Sessions: **12,330**; purchasing: **1,908**.
- Observed purchase conversion: **15.47%**.
- Original source rows retained; no automatic deduplication or outlier exclusion.
- Median product pages, purchasing sessions: **29**.
- Median product pages, non-purchasing sessions: **16**.

## Evidence

- `tables/conversion_by_visitor_depth.csv` compares visitor categories and page-depth bands.
- `tables/conversion_by_visitor_weekend.csv` compares visitor categories across weekdays/weekends.
- `tables/conversion_by_traffic_type.csv` retains original traffic-type codes and sample sizes.
- `tables/conversion_by_bounce_band.csv` compares rates across prespecified bounce-rate bands.
- `tables/engagement_by_outcome.csv` reports means, medians, and quartiles without deleting extreme values.
- The three accompanying charts display selected descriptive comparisons.

## Interpretation limits

- Conversion rates are **session-level** descriptive statistics, not causal effects.
- Groups with fewer than 100 sessions are flagged for caution, not silently removed from the tables.
- TrafficType is a categorical identifier; numeric codes are not ranked quantities.
- Bounce-rate bands are descriptive; their boundaries are analysis choices, not validated risk thresholds.
- `PageValues` was intentionally excluded from these comparisons because of its transaction-linked definition.
- Avoid interpreting observed engagement differences as evidence that increasing page views causes purchases.

Source: UCI Online Shoppers Purchasing Intention Dataset (Sakar & Kastro, 2018; CC BY 4.0).
