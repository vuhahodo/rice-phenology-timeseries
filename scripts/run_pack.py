"""Robustness pack for the final write-up (1D-CNN, 3 seeds, three input sets):
  scenario "standard": the project split (train/val/test by 2 km cell)
  scenario "year":     train = train-split windows of 2023-2024, val = val-split windows of 2023-2024,
                       test  = test-split windows of 2025  (spatial AND temporal hold-out, so no field is shared)
Outputs: macro-F1, hit rates within +-5/10/15 days, signed error, per-season MAE (Dong Xuan / He Thu),
and window-level bootstrap CIs (seed-ensemble) with the paired difference NDVI vs SAR+NDWI."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import mt, numpy as np, json, time, torch, torch.nn as nn
from src.models.cnn1d import NDVICNN1D

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
df, X, Y, doy = mt.load()
z = np.load(os.path.join(mt.MT, "sar", "rice_phenology_multi_v3.npz"), allow_pickle=False)
XM = np.nan_to_num(z["X"].astype(np.float32)); ch = [str(c) for c in z["channels"]]
assert np.abs(XM[:, :, ch.index("NDVI")] - X).max() < 1e-4 and (z["y"] == Y).all()
ndvi, ndwi, vv, vh = (XM[:, :, ch.index(c)] for c in ("NDVI", "NDWI", "VV", "VH"))
ang = 2 * np.pi * doy / 365.0
CFG = {"NDVI": [ndvi], "SAR+NDWI": [vv, vh, vh - vv, ndwi], "NDVI+NDWI+SAR": [ndvi, ndwi, vv, vh, vh - vv]}
sp, yr = df.split.values, df.year.values
SCEN = {
    "standard": (np.where(sp == "train")[0], np.where(sp == "val")[0], np.where(sp == "test")[0]),
    "year": (np.where((sp == "train") & (yr <= 2024))[0], np.where((sp == "val") & (yr <= 2024))[0], np.where((sp == "test") & (yr == 2025))[0]),
}
print({k: [len(i) for i in v] for k, v in SCEN.items()}, flush=True)
SEEDS = [0, 1, 2]


def build(chs, tr):
    A = np.stack(chs, -1).astype(np.float32); mu, sd = A[tr].mean((0, 1)), A[tr].std((0, 1)) + 1e-6
    return np.concatenate([(A - mu) / sd, np.sin(ang)[..., None], np.cos(ang)[..., None]], -1).astype(np.float32)


def predict(model, f):
    model.eval()
    with torch.no_grad(): return torch.softmax(model(torch.from_numpy(f)), -1).numpy()


def train(F, tr, va, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cw = np.bincount(Y[tr].ravel(), minlength=4).astype(float); cw = torch.tensor((cw.sum() / cw) ** 0.5, dtype=torch.float32); cw = cw / cw.mean()
    model = NDVICNN1D(F.shape[-1]); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2); lossf = nn.CrossEntropyLoss(weight=cw)
    xt, yt = torch.from_numpy(F[tr]), torch.from_numpy(Y[tr]); best, state, bad = -1, None, 0
    for ep in range(120):
        model.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 32):
            b = perm[i:i + 32]; opt.zero_grad(); lossf(model(xt[b]).reshape(-1, 4), yt[b].reshape(-1)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        f1 = mt.seg_metrics(Y[va], predict(model, F[va]).argmax(-1))["macro_f1"]
        if f1 > best: best, bad, state = f1, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 15: break
    model.load_state_dict(state); return model


def season_of(kind, d):
    if kind == "sos": return "DX" if (d >= 305 or d <= 90) else ("HT" if 91 <= d <= 243 else "khac")
    return "DX" if 32 <= d <= 151 else ("HT" if 152 <= d <= 304 else "khac")


def event_table(Yt, P, Xn, start_doy):
    """one row per true event: (kind, season, signed error in days or None if missed)."""
    rows = []
    for y, p, x, sd in zip(Yt, P, Xn, start_doy):
        te_, pe_ = mt.events(y, x), mt.events(p, x)
        for k in ("sos", "eos"):
            for a in te_[k]:
                d = ((sd - 1 + mt.STEP_DAYS * a) % 365) + 1
                c = [b - a for b in pe_[k] if abs(b - a) <= 12]
                rows.append((k, season_of(k, d), (min(c, key=abs) * mt.STEP_DAYS) if c else None))
    return rows


def summarize_events(rows):
    out = {}
    for k in ("sos", "eos"):
        r = [e for kk, s, e in rows if kk == k]; m = [e for e in r if e is not None]
        out[k] = dict(n=len(r), mae=float(np.mean(np.abs(m))) if m else None, signed_median=float(np.median(m)) if m else None,
                      hit5=float(np.mean([abs(e) <= 5 for e in m]) * len(m) / max(len(r), 1)) if r else None,
                      hit10=float(np.mean([abs(e) <= 10 for e in m]) * len(m) / max(len(r), 1)) if r else None,
                      hit15=float(np.mean([abs(e) <= 15 for e in m]) * len(m) / max(len(r), 1)) if r else None, miss=1 - len(m) / max(len(r), 1))
        for s in ("DX", "HT"):
            mm = [e for kk, ss, e in rows if kk == k and ss == s and e is not None]; nn_ = len([1 for kk, ss, e in rows if kk == k and ss == s])
            out[k]["mae_" + s] = float(np.mean(np.abs(mm))) if mm else None; out[k]["n_" + s] = nn_
    return out


res, ens = {}, {}
for sc, (tr, va, te) in SCEN.items():
    for name, chs in CFG.items():
        F = build(chs, tr); probs = []; runs = []
        for seed in SEEDS:
            t0 = time.time(); model = train(F, tr, va, seed); prob = predict(model, F[te]); probs.append(prob)
            Pv = mt.viterbi_batch(prob)
            runs.append(dict(seed=seed, macro_f1=mt.seg_metrics(Y[te], Pv)["macro_f1"], events=summarize_events(event_table(Y[te], Pv, X[te], df.start_doy.values[te]))))
            print("%-9s %-14s seed %d  F1 %.4f  (%.0fs)" % (sc, name, seed, runs[-1]["macro_f1"], time.time() - t0), flush=True)
        res.setdefault(sc, {})[name] = runs; ens[(sc, name)] = np.mean(probs, 0)
        json.dump(res, open(os.path.join(mt.OUT, "pack_results.json"), "w"), indent=1, default=float)

# ---- window-level bootstrap on the seed-ensemble ----
rng = np.random.default_rng(0); boot = {}
for sc, (tr, va, te) in SCEN.items():
    P = {n: mt.viterbi_batch(ens[(sc, n)]) for n in CFG}
    def f1w(idx, n): return mt.seg_metrics(Y[te][idx], P[n][idx])["macro_f1"]
    B = {n: [] for n in CFG}; D = []
    for _ in range(300):
        idx = rng.integers(0, len(te), len(te))
        for n in CFG: B[n].append(f1w(idx, n))
        D.append(B["NDVI"][-1] - B["SAR+NDWI"][-1])
    boot[sc] = {n: dict(mean=float(np.mean(B[n])), lo=float(np.percentile(B[n], 2.5)), hi=float(np.percentile(B[n], 97.5))) for n in CFG}
    boot[sc]["diff_NDVI_minus_SARNDWI"] = dict(mean=float(np.mean(D)), lo=float(np.percentile(D, 2.5)), hi=float(np.percentile(D, 97.5)))
    boot[sc]["n_test"] = int(len(te))
json.dump(boot, open(os.path.join(mt.OUT, "pack_bootstrap.json"), "w"), indent=1)
print(json.dumps(boot, indent=1)); print("DONE")
