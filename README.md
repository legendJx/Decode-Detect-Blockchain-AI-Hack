# Decode-Detect-Blockchain-AI-Hack

Team **API Maxxers**' entry for the Decode & Detect hackathon (https://hackathon.ukcbt.org):
an anti-money-laundering detector that scores every payment on an XRPL-style ledger from
0 (normal) to 1 (suspicious).

The data is a synthetic ledger of about 40,000 payments per split, mostly normal activity.
Laundering scenarios are planted in it at three difficulty levels: easy, medium and hard
(including "relay" cohorts hidden among 99 decoys each). The competition score is
100 × average precision on the hidden test labels.

## Results

Public leaderboard, by submission:

| Version | Main change | Score | Top-1% precision | Top-1% recall |
|---|---|---|---|---|
| v5 | Flow and topology features (pass-through chains, fan-in/out, issuer hopping) | 63.5 | 91% | 29% |
| v6 | Loss × seed ensemble (binary + `rank_xendcg`, 3 seeds) | 66.4 | 93% | 30% |
| v8 | Scenario-level aggregates, dead features dropped, tuned settings | **66.7** | 90% | 29% |
| v9 | "Blob" fix, shape-relative medium features | 66.0 | 88% | 28% |
| v10 | Account-fingerprint features removed, robust core blend (final, this repo) | | | |

## How it works

The final score is a 50/50 rank average of two model families. Each family is 6 LightGBM
models: binary log-loss and `rank_xendcg`, 3 seeds each.

- **Full model:** 430 features. Payment, flow, scenario-level and shape-relative features,
  with oversized "blob" components blanked.
- **Core model:** 126 short-window features (5-minute bursts, pass-through, chains, base
  rules), the features that look most alike in train and test.

Feature engineering (`aml_features.py`, which never reads labels):

- **Base rules R01–R14:** found from the labelled training data, e.g. fast relay hubs,
  round trips and meaningless hops.
- **Flow linking:** payments are linked when money passes straight through an account or
  leaves a common source together. Connected components are taken over several time windows
  (5 minutes, 30 minutes, "fresh pair" 1 hour / 4 hours), plus amount-matched flows. About
  95% of planted payments go between accounts that have never traded before, vs 28% of
  background payments.
- **Scenario aggregates:** each payment inherits the size, duration, value preservation and
  topology mix of its component.
- **Per-payment signals:** direct vs path payments, slippage, issuer hopping, fan-in/out
  velocity, account age and bursts, and how long a hub holds money before forwarding it.

Validation: `StratifiedGroupKFold(5)` grouped by scenario, so a whole laundering scenario is
always in one fold. This is checked in the notebook: 0 of 1,497 scenarios are split across
folds.

## What we learned

- **Cross-validation can lie in quiet ways.** Two kinds of leakage pushed CV up while the
  leaderboard stayed flat or fell:
  - Giant ~4,600-payment components, where features summarised half the ledger at once.
  - Per-account weekly totals that act as account IDs. Adversarial validation (a model
    guessing train vs test, no labels) separated the splits perfectly, AUC 1.0, using them.

  v10 removes both and checks every feature for how differently it behaves in train and test.
- **Easy cases are close to the ceiling.** Easy AP is about 0.78; even an oracle given the
  true scenario duration only reaches about 0.83.
- **Medium and relay cases are designed to look normal at the single-payment level.** No
  payment-level feature beats AUC 0.56 for them. Shape-relative scenario features reach
  0.77, but medium cases are only 0.6% of payments, so they move the overall score little.
- **Experiments that didn't work** are recorded in section 5b of the notebook:
  - bigger trees;
  - up-weighting medium and hard cases;
  - a medium specialist model;
  - two ways of rebuilding relay cohorts;
  - second-stage smoothing.

## Layout

The competition data is not included (`.gitignore` keeps it out). Download it from the
hackathon site and place it next to the code:

```
aml_lightgbm_notebook_v10.ipynb
aml_features.py
train/        # training download: transactions, transfers, blocks, market_rates, labels.csv
test-data/    # the unzipped test download
```

The notebook is saved with its outputs, so the charts and results can be read on GitHub
without running it. Its sections:

1. Read the data.
2. Tidy the data.
3. Look at the data.
4. Feature matrix, plus adversarial validation.
5. Train and test the model: grouped CV, hyperparameters, loss functions.
6. Diagnostics: ranked list, feature and rule importance.
7. Final model and the submission CSV.
8. Plain-language summary.

Every chart has a short "how to read this chart" note under it.

## Run it

```sh
brew install libomp   # macOS only: OpenMP runtime required by LightGBM
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute aml_lightgbm_notebook_v10.ipynb --output aml_lightgbm_notebook_v10.executed.ipynb --ExecutePreprocessor.timeout=3600
```

This writes `predictions_aml_lightgbm_v10.csv`: 40,101 rows of `TRANSACTION_HASH,score`,
sorted by score descending, one row per transaction, scores in [0, 1].

The run is deterministic:

- fixed seeds (model seeds 0–2, fold `random_state=42`);
- LightGBM's `deterministic=True, force_col_wise=True`;
- a fixed thread count (`num_threads=8`).

It takes about 6–7 minutes on an Apple M4 Pro laptop, CPU only, with peak memory under 2 GB.
Requires Python 3.14.3; dependencies are pinned in `requirements.txt`.
