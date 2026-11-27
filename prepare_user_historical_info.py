#!/usr/bin/env python3
import argparse
import csv
import os

from features import prev_trading_day

HERE = os.path.dirname(os.path.abspath(__file__))


def load_calendar(path):
    cal = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] == 'symbol':
                continue
            cal[parts[0]] = sorted(p.replace('-', '') for p in parts[1:])
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dates', default=os.path.join(HERE, 'data', 'all_dates.txt'))
    ap.add_argument('--names', default=os.path.join(HERE, 'data', 'n50_names.txt'))
    ap.add_argument('--calendar', default=os.path.join(HERE, 'nse_results_calendar.txt'))
    ap.add_argument('--out', default=os.path.join(HERE, 'user_historical_info.csv'))
    a = ap.parse_args()
    dates = open(a.dates).read().split()
    names = open(a.names).read().split()
    cal = load_calendar(a.calendar)
    missing = [n for n in names if n not in cal]
    if missing:
        raise SystemExit('no results calendar for: %s' % ' '.join(missing))
    n_today = n_prev = 0
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'name', 'results_today', 'results_after_prev_close'])
        for d in dates:
            t1 = prev_trading_day(d)
            for n in names:
                meetings = cal[n]
                today = int(d in meetings)
                after_prev = int(any(t1 <= m < d for m in meetings))
                n_today += today
                n_prev += after_prev
                w.writerow([d, n, today, after_prev])
    print('wrote %s: %d dates x %d names, results_today=%d, results_after_prev_close=%d'
          % (a.out, len(dates), len(names), n_today, n_prev))


if __name__ == '__main__':
    main()
