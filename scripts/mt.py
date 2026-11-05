"""Shared library for the midterm demo branch (self-contained; repo files are only imported, never edited)."""
import sys, os, time, json
import numpy as np, pandas as pd
from scipy.signal import find_peaks
from sklearn.metrics import f1_score, accuracy_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
MT = os.path.join(ROOT, "midterm")
CSV = os.path.join(MT, "data_v32.csv")
OUT = os.path.join(MT, "out")
STEP_DAYS = 5
T = 73
NAMES = ["Fallow", "Vegetative", "Reproductive", "Senescence"]


def load():
    df = pd.read_csv(CSV)
    X = df[[f"NDVI_{i:02d}" for i in range(1, T + 1)]].values.astype(np.float32)
    Y = df[[f"stage_{i:02d}" for i in range(1, T + 1)]].values.astype(np.int64)
    df["cell"] = (df.lon // 0.02).astype(int).astype(str) + "_" + (df.lat // 0.02).astype(int).astype(str)
    doy = ((df.start_doy.values[:, None] - 1 + STEP_DAYS * np.arange(T)[None, :]) % 365) + 1
    return df, X, Y, doy


def features(X, doy):
    """(N,T,3): [ndvi, sin(doy), cos(doy)]"""
    ang = 2 * np.pi * doy / 365.0
    return np.stack([X, np.sin(ang), np.cos(ang)], axis=-1).astype(np.float32)


# ---------- threshold rule (re-implementation of notebook cell 24, window level) ----------
THR, MIN_AMP, SPAN = 0.2, 0.15, 35


def detect(s, thr=THR):
    good = []
    peaks, _ = find_peaks(s, prominence=0.25, distance=12)
    for p in peaks:
        lo, hi = max(0, p - SPAN), min(len(s) - 1, p + SPAN)
        left, right = s[lo:p + 1], s[p:hi + 1]
        rmin = right.min()
        cut = False
        for j in range(1, len(right) - 1):
            if right[j] <= right[j - 1] and right[j] <= right[j + 1] and right[j] < s[p] - 0.5 * (s[p] - rmin) \
                    and right[j:j + 11].max() - right[j] > 0.1:
                right = right[:j + 1]; cut = True; break
        bl, br = left.min(), right.min()
        ok = (s[p] - bl >= MIN_AMP) and (s[p] - br >= MIN_AMP)
        if (lo == 0 and left.argmin() == 0) or (hi == len(s) - 1 and not cut and right.argmin() == len(right) - 1):
            ok = False
        if ok:
            li = np.where(left < bl + thr * (s[p] - bl))[0]
            ri = np.where(right < br + thr * (s[p] - br))[0]
            ok = len(li) > 0 and len(ri) > 0
        if ok:
            sos, eos = lo + li[-1] + 1, p + ri[0]
            ok = sos < p < eos
        if ok:
            good.append((int(sos), int(p), int(eos)))
    return good


def rule_labels(s, thr=THR):
    y = np.zeros(len(s), dtype=np.int64)
    t = np.arange(len(s))
    for a, p, b in detect(s, thr):
        d = int(np.floor(0.12 * (b - a)))
        y[(t >= a) & (t < p - d)] = 1
        y[(t >= p - d) & (t <= p + d)] = 2
        y[(t > p + d) & (t <= b)] = 3
    return y


# ---------- metrics ----------
def seg_metrics(Y, P):
    y, p = Y.ravel(), P.ravel()
    f = f1_score(y, p, average=None, labels=[0, 1, 2, 3], zero_division=0)
    return dict(acc=accuracy_score(y, p), macro_f1=float(f.mean()), f1=[float(v) for v in f])


def events(y, ndvi):
    sos = [t for t in range(1, len(y)) if y[t - 1] == 0 and y[t] > 0]
    eos = [t for t in range(1, len(y)) if y[t - 1] == 3 and y[t] == 0]
    pos, t = [], 0
    while t < len(y):
        if y[t] in (1, 2):
            u = t
            while u + 1 < len(y) and y[u + 1] in (1, 2):
                u += 1
            pos.append(t + int(np.argmax(ndvi[t:u + 1])))
            t = u + 1
        else:
            t += 1
    return dict(sos=sos, pos=pos, eos=eos)


def date_metrics(Y, P, X, tol_steps=2, max_match=12):
    """Match every true event to the nearest predicted event of the same type, only if within max_match steps
    (60 days = half the spacing of two rice crops); otherwise it counts as a miss.
    Returns MAE in days over matched events, hit-rate within tol_steps*5 days, miss-rate."""
    out = {}
    for k in ("sos", "pos", "eos"):
        errs, n_true, n_hit = [], 0, 0
        for y, p, x in zip(Y, P, X):
            te, pe = events(y, x)[k], events(p, x)[k]
            for a in te:
                n_true += 1
                e = min((abs(a - b) for b in pe), default=None)
                if e is not None and e <= max_match:
                    errs.append(e * STEP_DAYS)
                    n_hit += e <= tol_steps
        out[k] = dict(mae_days=float(np.mean(errs)) if errs else None,
                      hit10=n_hit / max(n_true, 1), miss=1 - len(errs) / max(n_true, 1), n=n_true)
    return out


# ---------- cyclic Viterbi: Fallow->Veg->Repro->Senescence->Fallow, stay or advance one stage ----------
def viterbi(logp):
    """logp (T,4) -> best path under transitions c->c or c->(c+1)%4."""
    Tn = logp.shape[0]
    NEG = -1e9
    trans = np.full((4, 4), NEG)
    for c in range(4):
        trans[c, c] = 0.0
        trans[c, (c + 1) % 4] = 0.0
    dp = np.zeros((Tn, 4)); bp = np.zeros((Tn, 4), dtype=np.int64)
    dp[0] = logp[0]
    for t in range(1, Tn):
        cand = dp[t - 1][:, None] + trans
        bp[t] = cand.argmax(0)
        dp[t] = cand.max(0) + logp[t]
    path = np.zeros(Tn, dtype=np.int64)
    path[-1] = dp[-1].argmax()
    for t in range(Tn - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def viterbi_batch(prob):
    return np.stack([viterbi(np.log(np.clip(p, 1e-8, 1))) for p in prob])


def retrograde_rate(P):
    """Share of windows whose predicted stage sequence violates the cyclic order (any step not stay/advance)."""
    bad = 0
    for p in P:
        d = (p[1:] - p[:-1]) % 4
        bad += bool(np.any(d > 1))
    return bad / len(P)


def interp_mask(x, keep):
    """Linear re-interpolation of a series after dropping steps (keep = boolean array)."""
    idx = np.arange(len(x))
    return np.interp(idx, idx[keep], x[keep]).astype(np.float32)
