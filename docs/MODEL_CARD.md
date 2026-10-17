# Model card: RainScope ML selected classifier

## Purpose

Research classification of a historical daily Australian station record into `RainTomorrow = Yes/No`, with a probability and validation-selected decision threshold. The exact selected estimator and configuration are recorded in `reports/selection.json`; only recorded successful candidates are evaluated.

## Intended use

Portfolio research, reproducible comparative ML experiments, and exploration of tabular model trade-offs. The dashboard exposes saved metrics and permits local batch predictions after training.

## Out of scope

Emergency alerts, aviation, agricultural safety decisions, real-time forecasting, other countries, unseen station networks, or claims about current weather. Feature availability is defined by the daily dataset, not a validated forecast issuance time.

## Training and evaluation

Versioned dataset fingerprint, chronological partitions and embargoes are in `reports/data_audit.json`. The full protocol is in `docs/EXPERIMENT_PROTOCOL.md`. Validation selects both the estimator and rain-class F1 threshold. Final test periods must not be used for further tuning. Same stations occur in multiple partitions. TabICLv2 uses a smaller, training-only context due to the workstation memory budget, unlike full-data candidates.

## Outputs and probability interpretation

`rain_probability` is the model's predicted positive-class probability. `rain_prediction` applies the recorded threshold, which may differ from 0.5. Calibration curves and Brier loss are diagnostics; no post-hoc probability calibration was fitted in this experiment, and no claim of universally calibrated probabilities is made.

## Limitations

A bounded single-seed search cannot establish a universally best algorithm. Historical weather distribution, missingness mechanisms and station coverage may shift. High missingness in Sunshine/Evaporation makes dependence on measurement availability plausible. Spatial correlation is not removed by temporal splitting. Month-cluster confidence intervals are descriptive, not a guarantee of future accuracy. Foundation-model training-data overlap with this public dataset cannot be excluded.

## Artefacts

Fitted models, including any retained context rows and downloaded checkpoints, remain local and are excluded from Git. The published project contains code and aggregate evaluation evidence, not third-party model weights or raw observations. Load only locally generated model artifacts that you trust.

## Authorship

New pipeline/application/evaluation implementation developed for Bola Maged Ramses Nessim with AI coding assistance. Dataset, original repository and model architecture authors retain their attribution. See `docs/ATTRIBUTION.md`.
