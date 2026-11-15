# Advanced analytical methods and reproducibility

## Research scope

The study measures purchase completion at the **recorded session** level. The original UCI data contains 12,330 rows, 17 predictors and a binary `Revenue` outcome. The data lacks event sequences, exact timestamps, transaction amounts, and customer identifiers. No reported result establishes causality, cart abandonment, or commercial impact.

## Reproducible pipeline

From the repository root, using Python 3.15 and a virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.run_upgrade --verify
```

This runs the original source audit, descriptive analysis, deep dive, eight-test statistical review, expanded segmentation, model comparison, and report builders. Verification then executes five notebooks (in memory, without overwriting saved notebooks), runs the tests, and checks dependencies.

The original CSV is placed at `data/raw/online_shoppers_intention.csv` and ignored by Git. Do not stage raw CSVs, `.venv`, or temporary release-review files. Generated `outputs/` are ignored by default; select reviewed reports, tables and PNGs deliberately at release time.

## Descriptive inference

- The primary denominator retains all original source rows, including 125 later exact matches; identical measured fields do not establish duplicate recording.
- Region, browser, operating-system and traffic codes are categorical labels with no inferred business meaning.
- The advanced categorical summaries include counts, purchases, conversion percentages, and approximate two-sided 95% Wilson binomial intervals.
- All breakdowns must reconcile their counts to the source totals. Groups under 100 sessions are flagged as sparse; charts that compare rates display groups above that exploratory threshold.
- The original eight prespecified comparisons use chi-square/Cramér's V or Mann–Whitney/signed rank-biserial correlations, with Benjamini–Hochberg correction across the family. New segment displays are descriptive, not a second uncontrolled set of claims of significance.
- Standard intervals and independent-session bootstrap assume independent records; absence of user IDs prevents independent verification.

## Classification benchmark

### Feature definition

Principal models use administrative, informational and product-page counts/durations, bounce and exit rates, SpecialDay, month, operating-system code, browser code, region code, traffic-type code, visitor category and weekend label. Category identifiers are one-hot encoded; numeric features are scaled. All transformations are inside training-only scikit-learn pipelines. `PageValues` is excluded because its reported transaction-linked definition makes it problematic for prospective use. Its influence is documented only in a separately labeled sensitivity analysis using identical split indices.

### Partition, tuning and selection

- The train/test split is the first fold of a shuffled **five-fold StratifiedGroupKFold** (seed 42), typically about 80/20; realized sample sizes are reported.
- Group identifiers are deterministic hashes of all **principal input features**; exact matching feature vectors must stay together across the holdout. No shared user identity can be inferred or guaranteed.
- **Three-fold stratified group CV** is applied on training records only. The model selection metric is mean cross-validated **average precision**.
- Algorithms: prior-only DummyClassifier (no-skill prevalence baseline), LogisticRegression (C = 0.2 or 1.0; class_weight = None or balanced), RandomForestClassifier (120 trees; maximum depth = 10 or unrestricted; minimum leaf size = 5 or 20).
- The model family is chosen on the training CV result, **not test performance**. The test set remains held out for a one-time descriptive comparison. The fixed 0.50 threshold is the reference binary classification policy. Separately, the CV-selected model receives three-fold group-aware out-of-fold probabilities on training records only (a different random seed), and we select the threshold maximizing training out-of-fold F1 over 19 predetermined cutoffs from 0.05 to 0.50, breaking ties toward the higher threshold. The held-out test is used only to evaluate both decision policies. Other fixed test thresholds (0.25 and 0.75) are exploratory. This training-OOF threshold tuning is not nested within the hyperparameter/model-family search and its apparent training performance may be optimistic.
- Report average precision, ROC-AUC, precision, recall, F1, balanced accuracy, Brier score, confusion counts and calibration plots at a fixed 0.50 decision cutoff and an independently test-held-out training-selected cutoff. A stratified class-wise bootstrap (400 replicates, seed 42) gives an exploratory interval for held-out average precision; this is conditional on the test set and does not capture sampling/site drift.
- Permutation importance is calculated on the holdout for the CV-selected pipeline using three shuffles per original input. Correlated predictors can mask each other; these estimates are neither causal nor stable under new data without further validation.

### Interpretation and validation boundaries

All principal predictors are **full-session measurements**. This is a retrospective classification benchmark, **not a proven early-session, live intent, or forward-time forecasting model**. Random group splitting avoids exact input profiles crossing test/training but cannot prevent unknown user overlap or measure temporal drift. The dataset cannot justify expected sales, impact of interventions, or individualized targeting. Any operational use would require timestamped event capture, target population definition, feature availability at the decision time, forward-in-time and external validation, governance, and controlled experiments.

## Main files

- `scripts/advanced_segmentation.py` — reconciled categorical tables, Wilson intervals and six figures.
- `scripts/model_purchase_intent.py` — strict input/target processing, grouping, train-only tuning, metrics, uncertainty, model plots, feature and `PageValues` sensitivity.
- `scripts/build_upgrade_report.py` — reads actual saved evidence, checks count reconciliation, generates `README.md` and `outputs/advanced_case_study.md`.
- `scripts/run_upgrade.py` — one-command reproducibility and optional integrated QA.
- `scripts/verify_upgrade_release.py` — final read-only real-dataset SHA, figure, table, link and Git-exclusion audit.
- `notebooks/04_advanced_segmentation.ipynb` — descriptive uncertainty walkthrough.
- `notebooks/05_predictive_modeling.ipynb` — model design, split checks and metric review.

**Dataset attribution:** Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*, UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5F88Q. License: CC BY 4.0. Source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset.
