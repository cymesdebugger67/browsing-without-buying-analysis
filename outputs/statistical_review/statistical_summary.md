# Statistical review — browsing sessions

- Observations: **12,330** sessions; **1,908** purchases.
- Eight preselected, exploratory comparisons: four categorical chi-square tests and four numeric Mann–Whitney U tests.
- P-values are adjusted together using Benjamini–Hochberg false-discovery-rate control.
- Strongest tested categorical association by Cramér's V: **page_depth** (V = **0.186**).

## Visitor-type conversion gap

- New visitors: **422 / 1,694 (24.91%)**.
- Returning visitors: **1,470 / 10,551 (13.93%)**.
- Observed gap (new minus returning): **+10.98 percentage points**.
- Exploratory 95% bootstrap interval: **8.81 to 13.16 percentage points**, using 2,000 fixed-seed within-group session resamples.

## Interpret with care

- Chi-square expected-cell checks are in `categorical_tests.csv`; comparisons marked CAUTION are not strong grounds for asymptotic inference.
- P-values alone do not describe practical significance; review Cramér's V, rank-biserial effect sizes, group counts, and distributions.
- UCI describes one session per user, but the released CSV has no user identifiers to independently verify this design.
- The bootstrap assumes independent session observations within visitor groups; that assumption is not independently testable from the CSV.
- Page-depth and bounce-rate groups reflect observed session behavior, not treatments or causal purchase drivers.
- The dataset is observational and does not establish causation, customer-level behavior, or complete annual seasonality.
- All original sessions, including 125 later exact-match rows, are retained; the duplicate sensitivity result belongs to the separate data audit.
- `PageValues` is excluded from the tested features because it is transaction-linked.

Source: UCI Online Shoppers Purchasing Intention Dataset (Sakar & Kastro, 2018; CC BY 4.0).
