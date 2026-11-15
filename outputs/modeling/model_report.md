# Purchase-completion classification — held-out evaluation

- Source: **12,330** sessions; **1,908** purchases.
- Training: **9,864** sessions; held-out evaluation: **2,466** sessions.
- Source rows remain intact. Five-fold stratified group splitting isolates identical feature profiles; a three-fold stratified group CV selects hyperparameters on training data only.
- 125 rows repeat the chosen input-feature vector; identical profiles stay in one partition.
- One-hot encoded categorical codes and scaled numeric measures are fitted only on training folds.
- Models: majority-prior baseline, logistic regression, and random forest. The selection metric is **cross-validated average precision** (PR-AUC summary).
- Selected by training CV: **Random forest**, mean CV average precision **0.389**.
- Selected model held-out average precision: **0.361**; ROC-AUC: **0.771**.
- Fixed-test, stratified-session bootstrap 95% interval for held-out average precision: **0.320–0.401** (400 replicates; treats evaluation records as independent).
- Held-out purchase prevalence: **0.155**; compare average precision to that baseline, not to accuracy alone.
- At prespecified probability threshold 0.50: precision **0.594**, recall **0.050**, F1 **0.092**.
- Threshold **0.225** was selected using 3-fold out-of-fold **training-only** F1, from 19 fixed candidates (0.05 to 0.50), independently of test performance.
- At that training-selected threshold, held-out precision **0.332**, recall **0.592**, F1 **0.425**; TP **226**, FP **455**, FN **156**, TN **1629**.
- The optimized threshold changes the precision–recall trade-off; it does not change average precision, ROC-AUC or calibration. Training OOF F1 can be optimistic because the model family/hyperparameters were previously chosen using the same training cohort.

## Cautions

- These are retrospective complete-session classifications, NOT validated live purchase forecasts.
- `PageValues` is excluded from all principal models because it is transaction-linked; the separate sensitivity table illustrates how including it changes performance, not a deployable result.
- No person IDs, chronological timestamps, external validation site, sales amounts, or randomized interventions are available.
- A group split reduces exact-feature overlap but does not prove records are independent or protect against unobserved shared customers or time drift.
- Hyperparameters, model family and decision threshold are selected using training folds; the held-out test is never used to optimize them. Only the training-selected threshold is a pre-test decision policy; other thresholds are exploratory.
- Permutation importance estimates associations with held-out average precision, not causal importance; correlated predictors can obscure attribution.
- Random partitioning does not support claims about future-site performance. Uncertainty intervals are exploratory under independent-session assumptions.

Source: Sakar & Kastro (2018), UCI Online Shoppers Purchasing Intention Dataset (CC BY 4.0).
