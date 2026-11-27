#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model():
    with open(os.path.join(HERE, 'model.json')) as f:
        return json.load(f)


def probabilities(date, info_dir, user_path, model):
    info = F.load_info_dir(info_dir, upto=date)
    user = F.read_user_info(user_path, upto=date)
    hist = F.History()
    for d in sorted(info):
        hist.add_info(d, info[d])
    last_vix, last_lM = F.latest_vix_and_baseline(info, date)
    per = F.raw_features(date, info, hist, user, last_vix=last_vix, last_lM=last_lM)
    fb = model['fallback']
    out = {}
    for name, f in per.items():
        liv = f['liv'] if f['liv'] is not None else f['m_liv']
        if liv is None:
            out[name] = None
            continue
        x = {
            'liv': liv,
            'lrv10': f['lrv10'] if f['lrv10'] is not None else liv + fb['lrv10_minus_liv'],
            'lm_rv3': f['lm_rv3'] if f['lm_rv3'] is not None else (f['m_liv'] if f['m_liv'] is not None else liv) + fb['lm_rv3_minus_m_liv'],
            'lM': f['lM'] if f['lM'] is not None else fb['lM'],
            'results_today': f['results_today'],
            'results_after_prev_close': f['results_after_prev_close'],
        }
        z = model['intercept']
        for k, mu, sd, w in zip(model['features'], model['mean'], model['scale'], model['coef']):
            z += w * (x[k] - mu) / sd
        out[name] = 1.0 / (1.0 + math.exp(-z))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True)
    ap.add_argument('--historical-info', required=True)
    ap.add_argument('--user-historical-info', default=None)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    names = list(F.read_info_file(os.path.join(a.historical_info, a.date + '_n50.info.csv')))
    model = load_model()
    try:
        probs = probabilities(a.date, a.historical_info, a.user_historical_info, model)
    except Exception:
        probs = {}
    thr = model['threshold']
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, a.date + '.predictions.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'prediction'])
        for n in names:
            p = probs.get(n)
            w.writerow([n, bool(p is not None and p > thr)])


if __name__ == '__main__':
    main()
