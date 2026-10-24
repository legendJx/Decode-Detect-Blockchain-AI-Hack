# Decode-Detect-Blockchain-AI-Hack

Team **API Maxxers**' entry for the Decode & Detect hackathon (https://hackathon.ukcbt.org).

LightGBM detector on hand-made payment-flow features. Built on the organisers' starter
notebook, extended with two rule groups:

- **Fast pass-through**: for each payment, the share the receiver forwards within 1 and 5
  minutes, and the share the sender had itself received in the 1 and 5 minutes before. Targets
  fast, connected bursts where money passes straight through.
- **Scenario volume**: total USD moved by the payment's sender and receiver within +/- 1 hour
  (about one scenario's length), raw and relative to what those accounts usually move in 2 hours.

Model: small, regularised LightGBM (7 leaves, 400 trees, learning rate 0.03, at least 200
payments per leaf, 80% row/column sampling, L2 = 5). Held-back scores (100 × average precision):

| Check | Score |
|---|---|
| Scenario-grouped 5-fold (whole scenarios held out) | **44.4** |
| Random 80/20 split (optimistic: scenarios leak across the split) | 47.8 |

The last cell plots the money flow around one suspicious and one non-suspicious training
transaction, for investigation only; it does not affect the predictions.

## Layout

Place the data next to the code (the data is not in this repo; `.gitignore` keeps it out):

```
detector.ipynb
payments.py
train/        # from the training download (transactions, transfers, blocks, market_rates, labels.csv)
test-data/    # the unzipped test download
```

## Exact command

```sh
brew install libomp   # macOS only: OpenMP runtime required by LightGBM
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute detector.ipynb --output detector.executed.ipynb
```

This writes `predictions_lightgbm.csv` (40,101 rows, `TRANSACTION_HASH,score`) to the current
directory. The run is deterministic: fixed seeds (`random_state=0`) and LightGBM's
`deterministic=True, force_col_wise=True` with a fixed thread count (`n_jobs=4`).

## Hardware and time

Apple M4 Pro laptop, CPU only, about a minute end to end (the grouped 5-fold check trains the
model five extra times). No GPU and little memory needed.

## Language

Python 3.14.3. Dependencies pinned in `requirements.txt`.
