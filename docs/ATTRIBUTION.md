# Attribution and project lineage

RainScope ML was developed for **Bola Maged Ramses Nessim** as a new engineering and evaluation implementation. It originates from reviewing his fork of [tjkyner/australian-rain-prediction](https://github.com/tjkyner/australian-rain-prediction). The fork's default branch was identical to the upstream branch at review time; the upstream notebook and its results must not be represented as Bola's original work.

The new contribution is the packaged pipeline, explicit schema validation, chronological/embargo splits, validation-only selection, recent-model adapters, measured experiments, resource monitoring, error reports, tests, dashboard and reproduction documentation. TabM and TabICL are models developed by their cited authors, not new architectures invented in this project. The implementation was developed with AI coding assistance and should be understood and reviewed before presentation in an interview.

The upstream GPL-3.0 licence text is retained in `LICENSE`; this project's implementation is distributed under GPL-3.0. Dependencies and pretrained checkpoints retain their own licences. Raw observations and model checkpoints are not redistributed in this repository. `docs/upstream.json` records the exact source snapshot, download URL and SHA-256 used for the experiment.

The upstream README traces weather observations to the Australian Bureau of Meteorology via the Kaggle/Rattle Rain in Australia dataset. That lineage is not a blanket grant to redistribute all raw data. Download from the recorded source and check applicable source terms for your own use.

## Model references

- [TabM (ICLR 2025), official implementation](https://github.com/yandex-research/tabm): parameter-efficient ensembling of MLP-like models; official package used here. Train member losses independently and average probabilities at inference.
- [TabICLv2 (ICML 2026), official implementation](https://github.com/soda-inria/tabicl): pretrained tabular in-context learning; checkpoint `tabicl-classifier-v2-20260212.ckpt`. Checkpoint downloaded locally, not committed.
- [scikit-learn](https://scikit-learn.org/stable/): baseline, logistic regression, random forest and histogram gradient boosting; evaluation and preprocessing.

Prior Labs' current TabPFN code and checkpoint licences are separate. It was researched as an alternative but is not described as evaluated unless it appears in an actual report. Recent models are included because they make an informative comparison, not because newness guarantees better performance.
