#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = 0.70
C_REG = 1.0


def load_labels(path):
    out = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            out[(r['date'], r['name'])] = (r['moved'].strip() == 'True', float(r['up_move_pct']), float(r['down_move_pct']))
    return out


def build_table(info_dir, labels_path, user_path, all_dates_path):
    info = F.load_info_dir(info_dir)
    labels = load_labels(labels_path)
    user = F.read_user_info(user_path)
    all_dates = set(open(all_dates_path).read().split())
    hist = F.History()
    rows = []
    for d in sorted(info):
        hist.add_info(d, info[d])
        per = F.raw_features(d, info, hist, user)
        for name, f in per.items():
            moved = labels[(d, name)][0]
            rows.append(dict(date=d, name=name, y=int(moved), **f))
        if F.next_trading_day(d) in all_dates:
            for name in info[d]:
                _, up, dn = labels[(d, name)]
                hist.add_label(d, name, up, dn)
    return rows


def complete(rows):
    return [r for r in rows if all(r[k] is not None for k in F.FEATURES)]


def matrix(rows):
    return np.array([[r[k] for k in F.FEATURES] for r in rows], float), np.array([r['y'] for r in rows], int)


def fit(X, y):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    m = LogisticRegression(C=C_REG, max_iter=5000).fit((X - mu) / sd, y)
    return mu, sd, m.coef_[0], m.intercept_[0]


def predict(model, X):
    mu, sd, w, b = model
    z = ((X - mu) / sd) @ w + b
    return 1.0 / (1.0 + np.exp(-z))


def score(p, r):
    return 0.0 if p <= 0 or r <= 0 else math.sqrt(1.0 / (0.75 / p + 0.25 / r))


def pr(y, flags):
    tp = int((y & flags).sum())
    k = int(flags.sum())
    pos = int(y.sum())
    p = tp / k if k else 0.0
    r = tp / pos if pos else 0.0
    return p, r, score(p, r), k


def oof(rows, folds):
    X, y = matrix(rows)
    dates = np.array([r['date'] for r in rows])
    out = np.full(len(rows), np.nan)
    for fd in folds:
        te = np.isin(dates, list(fd))
        model = fit(X[~te], y[~te])
        out[te] = predict(model, X[te])
    return out


def random_folds(dates, k, seed):
    d = np.array(sorted(dates))
    np.random.RandomState(seed).shuffle(d)
    return [set(d[i::k]) for i in range(k)]


def block_folds(dates, k):
    d = sorted(dates)
    n = len(d)
    return [set(d[int(i * n / k):int((i + 1) * n / k)]) for i in range(k)]


def calm_stress(rows, p, taus, n_boot=12000, n_days=41, lo=0.32, hi=0.37, seed=0):
    dates = sorted({r['date'] for r in rows})
    idx = {}
    for i, r in enumerate(rows):
        idx.setdefault(r['date'], []).append(i)
    y = np.array([r['y'] for r in rows], bool)
    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(n_boot):
        s = rng.choice(dates, n_days, replace=True)
        ii = np.concatenate([idx[x] for x in s])
        if lo <= y[ii].mean() <= hi:
            samples.append(ii)
    res = {}
    for t in taus:
        vals = np.array([pr(y[ii], p[ii] > t)[:3] for ii in samples])
        ok = vals[:, 0] > 0.6
        res[t] = (len(samples), vals[:, 0].mean(), np.percentile(vals[:, 0], 10), vals[:, 1].mean(),
                  vals[:, 2].mean(), ok.mean(), (vals[:, 2] * ok).mean())
    return res


def report(rows):
    dates = sorted({r['date'] for r in rows})
    y = np.array([r['y'] for r in rows], bool)
    taus = [0.55, 0.6, 0.65, 0.675, 0.7, 0.725, 0.75]
    splits = [('random 8-fold seed %d' % s, random_folds(dates, 8, s)) for s in (1, 2, 3)]
    splits.append(('6 time blocks', block_folds(dates, 6)))
    for name, folds in splits:
        p = oof(rows, folds)
        print('\n%s: out-of-fold rows %d, base rate %.3f' % (name, len(rows), y.mean()))
        stress = calm_stress(rows, p, taus)
        print('  tau   flagged  prec   recall score | calm 41-day samples (base rate 32-37%%, n=%d): prec  p10   recall score P(pass) E[score*pass]'
              % stress[taus[0]][0])
        for t in taus:
            pp, rr, ss, k = pr(y, p > t)
            n, cp, c10, cr, cs, cpass, ce = stress[t]
            print('  %.3f %6d  %.3f  %.3f  %.3f | %55s %.3f %.3f %.3f  %.3f  %.3f    %.3f' % (t, k, pp, rr, ss, '', cp, c10, cr, cs, cpass, ce))
    p_wf = np.full(len(rows), np.nan)
    X, yy = matrix(rows)
    dd = np.array([r['date'] for r in rows])
    for i in range(20, len(dates)):
        tr = np.isin(dd, dates[:i])
        te = dd == dates[i]
        p_wf[te] = predict(fit(X[tr], yy[tr]), X[te])
    ok = ~np.isnan(p_wf)
    print('\nexpanding-window walk-forward (train on all earlier dates, first test date %s):' % dates[20])
    for t in taus:
        pp, rr, ss, k = pr(y[ok], p_wf[ok] > t)
        print('  %.3f flagged %4d prec %.3f recall %.3f score %.3f' % (t, k, pp, rr, ss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--info', default=os.path.join(HERE, 'data', 'historical_info'))
    ap.add_argument('--labels', default=os.path.join(HERE, 'data', 'train_labels.csv'))
    ap.add_argument('--user', default=os.path.join(HERE, 'user_historical_info.csv'))
    ap.add_argument('--all-dates', default=os.path.join(HERE, 'data', 'all_dates.txt'))
    ap.add_argument('--out', default=os.path.join(HERE, 'model.json'))
    ap.add_argument('--cv', action='store_true')
    a = ap.parse_args()
    rows_all = build_table(a.info, a.labels, a.user, a.all_dates)
    rows = complete(rows_all)
    print('training rows %d of %d (%d dates with complete open-time features)'
          % (len(rows), len(rows_all), len({r['date'] for r in rows})))
    if a.cv:
        report(rows)
    X, y = matrix(rows)
    mu, sd, w, b = fit(X, y)
    by_date = {}
    for r in rows:
        by_date.setdefault(r['date'], r)
    model = {
        'features': F.FEATURES,
        'mean': [float(v) for v in mu],
        'scale': [float(v) for v in sd],
        'coef': [float(v) for v in w],
        'intercept': float(b),
        'threshold': THRESHOLD,
        'fallback': {
            'lrv10_minus_liv': float(np.mean([r['lrv10'] - r['liv'] for r in rows])),
            'lm_rv3_minus_m_liv': float(np.mean([r['lm_rv3'] - r['m_liv'] for r in by_date.values()])),
            'lM': float(np.mean([r['lM'] for r in by_date.values()])),
        },
        'trained_on': {'rows': len(rows), 'dates': len(by_date), 'first': min(by_date), 'last': max(by_date)},
    }
    with open(a.out, 'w') as f:
        json.dump(model, f, indent=1)
    p_in = predict((mu, sd, w, b), X)
    pp, rr, ss, k = pr(y.astype(bool), p_in > THRESHOLD)
    print('\nfinal model coefficients (standardised):')
    for k_, c in zip(F.FEATURES, w):
        print('  %-26s %+.3f' % (k_, c))
    print('  %-26s %+.3f' % ('intercept', b))
    print('in-sample at threshold %.2f: flagged %d, precision %.3f, recall %.3f, score %.3f (optimistic)' % (THRESHOLD, k, pp, rr, ss))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
