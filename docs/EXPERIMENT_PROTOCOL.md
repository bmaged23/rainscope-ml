# Experiment protocol (specified before viewing final test results)

## Scope

Binary classification of the existing dataset's `RainTomorrow` label from the current historical daily weather record. Same-station temporal generalization is assessed. No new weather observations are collected, and no operational forecast lead time is promised.

## Partitions

- Training observations: before 2014-12-30.
- Validation observations: 2015-01-01 through 2015-12-29.
- Final test observations: 2016-01-01 onward.
- December 30 and 31 before each boundary are excluded as a two-day embargo, allowing next-day target windows to remain separated.
- Missing/invalid target labels are excluded before splitting. Exact duplicate rows are removed. Conflicting station/date duplicates abort the run for inspection.

These are globally ordered dates, not independent random station splits. Weather correlation and climate/station coverage shifts can still affect interpretation.

## Preprocessing

A schema explicitly lists predictor columns, excluding target and raw date. Training-fitted median imputation with missing indicators handles numeric fields. Standard scaling and one-hot encoding are fitted only on training. Unseen categorical labels map to zero one-hot vectors; TabICLv2's ordinal adapter uses -1. Derived columns are cyclic month and within-day pressure/humidity differences. No target encoding or arbitrary outlier clipping is used.

TabICLv2's preprocessing is fitted only on its sampled training context. The context cap is a disclosed resource limit, not data from validation or test.

## Candidate comparison

Fixed single configurations: prior-climatology baseline, logistic regression, random forest, histogram gradient boosting, TabM and TabICLv2. Configuration and package versions are recorded. This is not an equal-compute hyperparameter competition. TabM is a recent learned tabular neural network; TabICLv2 is a pretrained in-context foundation model. Merely using newer software does not make a classical model a newly invented architecture.

Use validation average precision to select the model. Thresholds are selected by validation F1, with a deterministic tie rule. TabM early stopping uses validation log loss. Persist the selection before testing. No final-test result should be used to revise features, thresholds or hyperparameters.

If a model fails, record its exception. Fix technical errors with training/validation-only smoke tests before running the complete final evaluation. Never fill a leaderboard with fabricated or differently sourced scores.

## Diagnostics

Rain-class precision/recall/F1, accuracy and balanced accuracy, ROC-AUC, average precision, Brier and log loss, confusion matrix, reliability curve and station slices. Average precision is not labelled trapezoidal PR-AUC. Report descriptive month-cluster bootstrap intervals for selected-model AP/F1. Report per-phase wall time, process RSS/CPU and PyTorch peak allocated VRAM, including their measurement limits.

Validation-only permutation importance describes sensitivity; correlated features make causal interpretations inappropriate. Public tabular foundation-model pretraining contamination cannot be excluded for this well-known dataset. These results are not directly comparable to the upstream random-split accuracy.

## Follow-up work, separate from this final holdout

A station-group holdout, rolling-origin backtests, multiple seeds, feature ablation and larger tuning budgets are worthwhile extensions. Predefine new evaluation periods or outer folds before using them for further model development.

## Baseline interpretation

The prior-climatology estimator emits the training rain frequency for every record. Since the same validation-F1 threshold rule is applied to all candidates, its selected threshold may classify every record as rain. Its average precision equals the positive-class prevalence; a low thresholded accuracy is not the majority-class accuracy.
