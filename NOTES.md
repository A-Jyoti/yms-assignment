# NOTES — will a Nifty-50 stock leave its ±2% band today?

## Summary

A 6-feature logistic regression that estimates, before 09:15, each stock's probability of trading more than 2% away from the previous close. It flags a stock when that probability is above **0.70**. The inputs are:

- the stock's option-implied daily volatility;
- the stock's recent realised high–low range;
- the whole market's realised range over the last few days;
- the F&O market's 3-month baseline volatility (taken out of the z-score columns);
- two flags from the NSE results calendar: results due today, and results out since the last close.

`predict.py` uses only the Python standard library. The learned parameters are in `model.json`.

All tuning used **only** `data/` (training info files and `train_labels.csv`). No file in `evaluator/data/` was used to build, select or tune anything (see the last section).

## Files

| File | Purpose |
|---|---|
| `predict.py` | Grader entry point. Loads the info files dated ≤ D and the rows of `user_historical_info.csv` dated ≤ D, rebuilds the price history, computes the features and writes `<D>.predictions.csv` |
| `features.py` | Shared feature code (standard library only): NSE 2026 trading calendar, history reconstruction, EWMAs, market baseline |
| `model.json` | Standardisation, coefficients, threshold and fallbacks (written by `train.py`) |
| `train.py` | Rebuilds the training table exactly as `predict.py` would see it, fits the model and, with `--cv`, prints the validation and stress tables quoted below |
| `prepare_user_historical_info.py` | Builds `user_historical_info.csv` from `nse_results_calendar.txt` |
| `nse_results_calendar.txt` | Board-meeting dates for financial results, Mar–Oct 2026, for all 50 stocks (137 meetings) |
| `user_historical_info.csv` | `date,name,results_today,results_after_prev_close` for all 138 dates × 50 names |

Reproduce with:

```
python prepare_user_historical_info.py
python train.py --cv                      # needs numpy + scikit-learn (requirements.txt); predict.py needs neither
python evaluator/run_evaluation.py .
```

## Rebuilding history without look-ahead

Each info file for day D holds `pclose/phi/plo`: the close, high and low of the **previous trading day**. I added the NSE 2026 holiday list to `features.py`, so each of these daily bars can be placed on its actual date. With that, `predict.py` rebuilds every stock's daily high–low range history from the info files it is given. That history covers training days and earlier test days, and never a later day.

For training, the same history is rebuilt from the training info files. It is topped up with the high/low implied by `train_labels.csv` for a training day T, but only when the next trading day after T is itself a dataset date. That is exactly the case in which, at grading time, the info file of that next date would show T's bar. As a result, the training features are built from the same kind of information as the grading features. Test days' bars, which are blanked in `data/`, are never reconstructed.

There are 29 masked training dates: the day before each is a test day, so their VIX and previous-day columns are blank. These dates are left out of the fit. That leaves **68 dates, 3,382 rows** with complete open-time features. Masked dates still feed the history of later dates through their labels.

## Features

All of these are known before 09:15 on D.

| Feature | Definition | Why |
|---|---|---|
| `liv` | log(vix / 100 / √252): the option-implied 1-σ daily move from the previous session | Forward-looking volatility for the stock |
| `lrv10` | log EWMA (half-life 10 trading days, 40-day window) of the stock's daily (high − low) / close | Realised volatility; corrects for stocks whose options are priced richer or cheaper than they actually move |
| `lm_rv3` | log of the cross-sectional mean of each stock's half-life-3 EWMA range | Market-wide turbulence over the last few days (volatility clustering) |
| `lM` | log of M, where `vix_zscore_smooth = (vix − M)/S`. M is recovered each day by regressing `vix` on the z-score across the 50 stocks (an exact fit, since M and S are common to all stocks) | The F&O universe's 3-month baseline. With current IV held fixed, a high baseline means implied vol is sticky and above what is being realised, so the coefficient is negative |
| `results_today` | Results board meeting scheduled on D | Results released during market hours |
| `results_after_prev_close` | Results board meeting on the previous trading day or on a weekend/holiday before D | Results released after the last close, so the reaction falls on D |

The results calendar comes from NSE's public event calendar (`https://www.nseindia.com/api/event-calendar?index=equities&symbol=<SYMBOL>&from_date=01-03-2026&to_date=31-10-2026`). Only meetings whose purpose includes *Financial Results* are kept. Under SEBI LODR Reg. 29, results meetings are announced to the exchange in advance, so both flags are public before the open.

One meeting was moved to an earlier date: POWERGRID's meeting was originally announced for 16 May and moved to 15 May by an intimation published on 15 May itself. The calendar keeps 16 May, the date that was public before the open on the 15th.

In the training data, `results_after_prev_close` rows moved **82.9%** of the time, against 41.2% overall. On calm days the gap is similar: 80.9% against a same-day average of 28.5%. These moves are stock-specific, so they keep precision up on quiet market days.

## Model

The model is a logistic regression (scikit-learn, L2 penalty, C = 1) on the standardised features. Coefficients per 1 SD:

| liv | lrv10 | lm_rv3 | lM | results_today | results_after_prev_close |
|---|---|---|---|---|---|
| +0.49 | +0.33 | +0.51 | −0.34 | +0.05 | +0.21 (≈ +1.8 log-odds for a reporting stock) |

Cross-validated AUC as features were added, with folds made of whole dates. Random CV is 8 folds; time-block CV is 6 contiguous blocks.

| Features | Random CV | Time-block CV |
|---|---|---|
| implied vol only | 0.701 | 0.684 |
| + stock realised range | 0.716 | 0.694 |
| + market realised range | 0.722 | 0.706 |
| + 3-month baseline M | 0.729 | 0.728 |
| + results flags (market IV dropped, no longer useful) | **0.736** | **0.733** |

## Choosing the threshold on training dates only

The precision bar is the binding constraint. For a calibrated model, the score-optimal cut-off is about 0.75 × score², roughly 0.4. Out of fold, that cut-off gives pooled precision just under 0.60, which fails. The real question is therefore how much margin to keep. The README warns that the test days are calmer (34% of rows move, against 41% in training). Calmness matters because most errors come from **day-level shocks**: on a day that turns out quieter than its regime suggests, every flag on that day suffers at once. Bootstrapping days shows precision falls by about 1.5 points for every point the day-level base rate comes in below the model's expectation.

So the threshold was chosen on out-of-fold probabilities from the 68 training dates, under two lenses. For each candidate threshold I measured:

1. **pooled out-of-fold** precision, recall and score;
2. a **calm-test stress**: thousands of bootstrapped 41-day samples of training days, keeping only the samples whose realised base rate is 32–37%, like the README's description of the test set. For those samples I report mean precision, 10th-percentile precision, mean score, P(precision > 0.60) and E[score × pass].

Reproduced by `python train.py --cv`. Values are ranges over 3 random fold seeds plus the time-block split:

| threshold | pooled precision | pooled recall | pooled score | calm precision | calm P(pass) | calm E[score × pass] |
|---|---|---|---|---|---|---|
| 0.60 | 0.674–0.691 | 0.37–0.39 | 0.75–0.76 | 0.614–0.626 | 0.60–0.70 | 0.43–0.50 |
| 0.65 | 0.711–0.731 | 0.31 | 0.73–0.74 | 0.652–0.668 | 0.87–0.92 | 0.60–0.63 |
| 0.675 | 0.723–0.748 | 0.27–0.28 | 0.72 | 0.662–0.691 | 0.90–0.96 | 0.60–0.64 |
| **0.70** | **0.743–0.768** | **0.23–0.25** | **0.70–0.71** | **0.688–0.718** | **0.97–1.00** | **0.63–0.65** |
| 0.725 | 0.749–0.780 | 0.20 | 0.67 | 0.702–0.730 | 0.98–1.00 | 0.61–0.63 |

At 0.70, the calm-stress samples score 0.65–0.66 with precision around 0.70; across all 68 dates the pooled figures are 0.70–0.71 and 0.74–0.77 (table above). Under the calm stress, 0.70 gives the best or near-best expected score with a pass probability of about 97–100%, so I fixed it there. Lower thresholds score better on average but fail too often when the evaluation days are quiet.

I also ran a strict expanding-window walk-forward: train only on earlier dates, starting from 27 Apr. It is harsher, because the first fits see only the very volatile March–April regime and are then applied to the calmer May–October period. At 0.70 it gives precision 0.635 and recall 0.10. The test days, though, are interleaved with the training days over the same period, and the final model is fit on all of them. That setting is what the random and block splits reproduce.

## What I tried that did not help

- **Gradient boosting** (HistGradientBoosting): random-CV AUC about 0.73, but time-block AUC fell to 0.67–0.68, because it learns patterns specific to a period. Logistic regression is stable across blocks.
- **US overnight move**: S&P 500 closes from FRED, using the US session between the last Indian close and the open, either alone or interacted with the IT stocks. No CV gain. India's big days in Mar–Apr 2026 were not mainly US-driven.
- Day-over-day change in the stock's IV; IV relative to its 1-month or 3-month average; traded value relative to its 20-day median; the previous day's close-to-close return; day of week; days since the previous session; recent 2%-hit rate; EWMAs of max-move instead of range; the spread term S; market-level IV once M is included. None improved CV log-loss or AUC.
- Capping flags per day, to spread out the day-level risk: at equal calm-scenario precision it gave lower recall than a single probability cut-off.
- Interactions (stock IV × market range, IV × M): no gain.

## Robustness in `predict.py`

- If a stock's `vix` is blank, the latest earlier `vix` for that stock is used. Missing history falls back to training-mean offsets stored in `model.json`.
- If M cannot be recovered, the last available day's M is used.
- Any unexpected error writes `False` for every stock, never a missing file.
- One date takes about 0.07 s.

## About the test dates

`evaluator/data/` (test labels, `info_all.csv`, `labels_all.csv`, test dates) was not read for training, feature design, model selection or the threshold. The only test-related information used was the README's general statement that the test days are calmer, which motivated the calm-stress criterion above. The figures in that criterion come from training days only.

After the model and threshold were frozen and committed, the grading command (`--data evaluator/data`) was run once, purely as an end-to-end check that the submission runs in grading mode. It ran on all 41 dates with no failures: precision 0.736, recall 0.201, score 0.665, PASS. This is in line with the calm-stress estimate above. Nothing in the model, features or threshold was changed afterwards; only this sentence was added.
