import csv
import datetime as dt
import math
import os

NSE_HOLIDAYS = frozenset([
    '20251225',
    '20260126', '20260303', '20260326', '20260331', '20260403', '20260414', '20260501',
    '20260528', '20260626', '20260914', '20261002', '20261020', '20261110', '20261124', '20261225',
])

NUM_COLS = ['pclose', 'phi', 'plo', 'pvolume', 'mddv_20', 'vix', 'vix_1m', 'vix_3m', 'vix_6m', 'vix_1y',
            'vix_zscore_smooth', 'vix_zscore_raw']

FEATURES = ['liv', 'lrv10', 'lm_rv3', 'lM', 'results_today', 'results_after_prev_close']

SQRT252 = math.sqrt(252.0)
LOOKBACK = 40
HL_STOCK = 10.0
HL_MARKET = 3.0
MIN_STOCKS = 10


def to_date(s):
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def to_str(d):
    return d.strftime('%Y%m%d')


def is_trading_day(s):
    return to_date(s).weekday() < 5 and s not in NSE_HOLIDAYS


def prev_trading_day(s):
    d = to_date(s) - dt.timedelta(days=1)
    while not is_trading_day(to_str(d)):
        d -= dt.timedelta(days=1)
    return to_str(d)


def next_trading_day(s):
    d = to_date(s) + dt.timedelta(days=1)
    while not is_trading_day(to_str(d)):
        d += dt.timedelta(days=1)
    return to_str(d)


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read_info_file(path):
    rows = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            x = {c: num(r.get(c)) for c in NUM_COLS}
            x['vix_asof'] = (r.get('vix_asof') or '').strip()
            rows[r['name'].strip()] = x
    return rows


def load_info_dir(path, upto=None):
    out = {}
    for fn in sorted(os.listdir(path)):
        if not fn.endswith('_n50.info.csv'):
            continue
        d = fn[:8]
        if upto is not None and d > upto:
            continue
        out[d] = read_info_file(os.path.join(path, fn))
    return out


def read_user_info(path, upto=None):
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            d = (r.get('date') or '').strip()
            if not d or (upto is not None and d > upto):
                continue
            out[(d, (r.get('name') or '').strip())] = r
    return out


class History:
    def __init__(self):
        self.hl = {}
        self.rel = {}

    def add_info(self, date, rows):
        t = prev_trading_day(date)
        for name, r in rows.items():
            hi, lo, pc = r['phi'], r['plo'], r['pclose']
            if pc and hi and lo and hi >= lo:
                self.hl.setdefault(name, {})[t] = (hi - lo) / pc

    def add_label(self, day, name, up_pct, down_pct):
        self.rel.setdefault(name, {})[day] = (up_pct + down_pct) / 100.0

    def day_range(self, name, day):
        v = self.hl.get(name, {}).get(day)
        if v is None:
            v = self.rel.get(name, {}).get(day)
        return v


def ewma(pairs, halflife):
    lam = 0.5 ** (1.0 / halflife)
    sw = sx = 0.0
    for lag, x in pairs:
        if x is None:
            continue
        w = lam ** lag
        sw += w
        sx += w * x
    return sx / sw if sw > 0 else None


def safe_log(x):
    return math.log(x) if x is not None and x > 0 else None


def market_baseline(rows):
    pts = [(r['vix_zscore_smooth'], r['vix']) for r in rows.values()
           if r['vix'] is not None and r['vix_zscore_smooth'] is not None]
    if len(pts) < 2:
        return None
    n = float(len(pts))
    mz = sum(z for z, _ in pts) / n
    mv = sum(v for _, v in pts) / n
    szz = sum((z - mz) ** 2 for z, _ in pts)
    if szz <= 0:
        return None
    s = sum((z - mz) * (v - mv) for z, v in pts) / szz
    m = mv - s * mz
    return safe_log(m / 100.0 / SQRT252)


def flag(user_row, key):
    if not user_row:
        return 0.0
    v = num(user_row.get(key))
    return 1.0 if v else 0.0


def raw_features(date, info, hist, user=None, last_vix=None, last_lM=None):
    rows = info[date]
    days = []
    t = date
    for _ in range(LOOKBACK):
        t = prev_trading_day(t)
        days.append(t)
    lM = market_baseline(rows)
    if lM is None:
        lM = last_lM
    per = {}
    for name, r in rows.items():
        vix = r['vix']
        if vix is None and last_vix:
            vix = last_vix.get(name)
        ranges = [(i, hist.day_range(name, d)) for i, d in enumerate(days)]
        u = (user or {}).get((date, name))
        per[name] = {
            'liv': safe_log(vix / 100.0 / SQRT252) if vix else None,
            'rv10': ewma(ranges, HL_STOCK),
            'rv3': ewma(ranges, HL_MARKET),
            'lM': lM,
            'results_today': flag(u, 'results_today'),
            'results_after_prev_close': flag(u, 'results_after_prev_close'),
        }
    rv3 = [f['rv3'] for f in per.values() if f['rv3'] is not None]
    m_rv3 = sum(rv3) / len(rv3) if len(rv3) >= MIN_STOCKS else None
    livs = [f['liv'] for f in per.values() if f['liv'] is not None]
    m_liv = sum(livs) / len(livs) if livs else None
    for f in per.values():
        f['lrv10'] = safe_log(f['rv10'])
        f['lm_rv3'] = safe_log(m_rv3)
        f['m_liv'] = m_liv
    return per


def latest_vix_and_baseline(info, date):
    last_vix, last_lM = {}, None
    for d in sorted(x for x in info if x < date):
        rows = info[d]
        for name, r in rows.items():
            if r['vix'] is not None:
                last_vix[name] = r['vix']
        b = market_baseline(rows)
        if b is not None:
            last_lM = b
    return last_vix, last_lM
