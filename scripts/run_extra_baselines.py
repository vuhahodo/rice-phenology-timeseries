"""Two more classical baselines on the same split/metrics:
  B3  double-logistic fit (TIMESAT/Gobin style) around each NDVI peak, then the 20%-amplitude rule on the fitted curve
  B4  NDVI-slope method: SOS = steepest rise before the peak, EOS = steepest fall after it (derivative family; idea
      from NDVI-difference methods such as Tian 2023, not a re-implementation of that paper)
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import mt, numpy as np, json, warnings
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
warnings.filterwarnings("ignore")

df, X, Y, doy = mt.load()
te = np.where(df.split.values == "test")[0]


def dlog(t, a, b, t1, k1, t2, k2):
    return a + b * (1 / (1 + np.exp(-k1 * (t - t1))) - 1 / (1 + np.exp(-k2 * (t - t2))))


def dlog_events(s, thr=0.2):
    """TIMESAT-style: one double-logistic per peak (fitted between the troughs that separate neighbouring peaks);
    SOS/EOS = where the fitted curve crosses thr x amplitude on the up/down limb."""
    ev = []
    peaks = find_peaks(s, prominence=0.25, distance=12)[0]
    for j, p in enumerate(peaks):
        lo = max(0, p - mt.SPAN) if j == 0 else max(int(peaks[j - 1] + np.argmin(s[peaks[j - 1]:p + 1])), p - mt.SPAN)
        hi = min(len(s) - 1, p + mt.SPAN) if j == len(peaks) - 1 else min(int(p + np.argmin(s[p:peaks[j + 1] + 1])), p + mt.SPAN)
        if hi - lo < 8: continue
        t = np.arange(lo, hi + 1).astype(float); y = s[lo:hi + 1]
        try:
            prm, _ = curve_fit(dlog, t, y, p0=[y.min(), y.max() - y.min(), (lo + p) / 2, 0.4, (p + hi) / 2, 0.4],
                               bounds=([-0.3, 0.1, lo - 5, 0.02, p - 5, 0.02], [0.8, 1.2, p + 5, 3, hi + 10, 3]), maxfev=6000)
        except Exception:
            continue
        f = dlog(t, *prm); pk = int(np.argmax(f)); left, right = f[:pk + 1], f[pk:]
        if f[pk] - left.min() < mt.MIN_AMP or f[pk] - right.min() < mt.MIN_AMP: continue
        li = np.where(left < left.min() + thr * (f[pk] - left.min()))[0]; ri = np.where(right < right.min() + thr * (f[pk] - right.min()))[0]
        if len(li) == 0 or len(ri) == 0: continue
        sos, eos = lo + int(li[-1]) + 1, lo + pk + int(ri[0])
        if sos < lo + pk < eos: ev.append((sos, lo + pk, eos))
    return ev


def stages(L, events):
    y = np.zeros(L, dtype=np.int64); t = np.arange(L)
    for a, p, b in events:
        d = int(np.floor(0.12 * (b - a)))
        y[(t >= a) & (t < p - d)] = 1; y[(t >= p - d) & (t <= p + d)] = 2; y[(t > p + d) & (t <= b)] = 3
    return y


def slope_labels(s):
    ev = []; ds = np.diff(s)
    for p in find_peaks(s, prominence=0.25, distance=12)[0]:
        lo, hi = max(0, p - mt.SPAN), min(len(s) - 1, p + mt.SPAN)
        if p - lo < 2 or hi - p < 2: continue
        a = lo + int(np.argmax(ds[lo:p])) + 1; b = p + int(np.argmin(ds[p:hi])) + 1
        if a < p < b and s[p] - s[lo:p + 1].min() >= mt.MIN_AMP and s[p] - s[p:hi + 1].min() >= mt.MIN_AMP:
            ev.append((a, int(p), b))
    return stages(len(s), ev)


res = {}
for name, fn in (("B3_double_logistic", lambda s: stages(len(s), dlog_events(s))), ("B4_ndvi_slope", slope_labels)):
    P = np.stack([fn(s) for s in X[te]])
    m = mt.seg_metrics(Y[te], P); m["dates"] = mt.date_metrics(Y[te], P, X[te]); m["retrograde"] = mt.retrograde_rate(P)
    res[name] = m
    d = m["dates"]
    print("%-20s macroF1 %.3f acc %.3f | MAE SOS/POS/EOS %s | miss SOS/EOS %.2f/%.2f" % (name, m["macro_f1"], m["acc"],
          "/".join("%.1f" % d[k]["mae_days"] for k in ("sos", "pos", "eos")), d["sos"]["miss"], d["eos"]["miss"]), flush=True)
json.dump(res, open(os.path.join(mt.OUT, "extra_baselines.json"), "w"), indent=1, default=float)
