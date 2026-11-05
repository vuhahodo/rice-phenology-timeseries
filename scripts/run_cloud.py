"""Cloud robustness: does SAR keep working when optical (NDVI/NDWI) is lost, and does masking-augmented training help?
Input: midterm/sar/rice_phenology_multi_v3.npz (channels NDVI, NDWI, VV, VH, obs_ndvi; optical channels are already gap-filled,
obs_ndvi==1 marks steps that were really observed).
Test-time cloud simulation: drop contiguous 30-day blocks (6 steps) of REAL optical observations until a target fraction of the
73 steps is lost, then re-interpolate NDVI and NDWI linearly. SAR is never masked (radar sees through cloud).
Labels stay as they are (derived from the clean NDVI), so the score is "how well the clean-NDVI label is still reproduced".
Usage: python -I midterm/scripts/run_cloud.py [--smoke]"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import mt, numpy as np, json, time, torch, torch.nn as nn
from src.models.cnn1d import NDVICNN1D

smoke = "--smoke" in sys.argv
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
df, X, Y, doy = mt.load()
sp = df.split.values
tr, va, te = [np.where(sp == s)[0] for s in ("train", "val", "test")]
z = np.load(os.path.join(mt.MT, "sar", "rice_phenology_multi_v3.npz"), allow_pickle=False)
XM = np.nan_to_num(z["X"].astype(np.float32)); ch = [str(c) for c in z["channels"]]
assert float(np.abs(XM[:, :, 0] - X).max()) < 1e-4 and bool((z["y"] == Y).all())
ndvi, ndwi, vv, vh, obs = (XM[:, :, ch.index(c)] for c in ("NDVI", "NDWI", "VV", "VH", "obs_ndvi"))
obs = obs > 0.5
T = mt.T; ang = 2 * np.pi * doy / 365.0
LEVELS = [0.0, 0.25, 0.5, 0.75]; BLOCK = 6


def drop_mask(i_rows, frac, rng):
    """Boolean (n,T): True = extra step dropped. Blocks are placed on real observations; at least 2 observed steps always remain."""
    fr = np.broadcast_to(np.asarray(frac, dtype=float), (len(i_rows),))
    dropped = np.zeros((len(i_rows), T), bool)
    for r in range(len(i_rows)):
        if fr[r] <= 0: continue
        for s0 in rng.integers(0, T - BLOCK + 1, size=max(1, int(round(fr[r] * T / BLOCK)))): dropped[r, s0:s0 + BLOCK] = True
        if (obs[i_rows[r]] & ~dropped[r]).sum() < 2: dropped[r] = False
    return dropped


def reinterp(A, rows, dropped):
    """Rows with dropped steps are re-interpolated linearly from the remaining real observations; other rows stay as stored."""
    out = A.copy()
    if dropped is None: return out
    for r in np.where(dropped.any(1))[0]: out[r] = mt.interp_mask(A[r], obs[rows[r]] & ~dropped[r])
    return out


def feats(chs, rows, stats=None):
    A = np.stack(chs, -1).astype(np.float32)
    if stats is None: stats = (A[tr].mean((0, 1)), A[tr].std((0, 1)) + 1e-6)
    A = (A - stats[0]) / stats[1]
    return np.concatenate([A, np.sin(ang[rows])[..., None], np.cos(ang[rows])[..., None]], -1).astype(np.float32), stats


def raw_inputs(rows, dropped, which):
    n_, w_ = reinterp(ndvi[rows], rows, dropped), reinterp(ndwi[rows], rows, dropped)
    d = {"NDVI": [n_], "SAR": [vv[rows], vh[rows], vh[rows] - vv[rows]], "SAR+NDWI": [vv[rows], vh[rows], vh[rows] - vv[rows], w_],
         "NDVI+NDWI+SAR": [n_, w_, vv[rows], vh[rows], vh[rows] - vv[rows]]}
    return d[which]


CONFIGS = [("NDVI", False), ("NDVI", True), ("SAR", False), ("SAR+NDWI", False), ("SAR+NDWI", True), ("NDVI+NDWI+SAR", False), ("NDVI+NDWI+SAR", True)]
if smoke: CONFIGS = CONFIGS[:2]
cw = np.bincount(Y[tr].ravel(), minlength=4).astype(float); cw = torch.tensor((cw.sum() / cw) ** 0.5, dtype=torch.float32); cw = cw / cw.mean()


def predict(model, f):
    model.eval()
    with torch.no_grad(): return torch.softmax(model(torch.from_numpy(f)), -1).numpy()


# fixed test masks per level so every model sees identical clouds
rng_te = np.random.default_rng(2026)
TEST_KEEP = {p: drop_mask(te, p, rng_te) for p in LEVELS}
SEEDS = [0] if smoke else [0, 1, 2]; EPOCHS = 3 if smoke else 120
res = {}
for which, aug in CONFIGS:
    name = f"{which}{' +mask-aug' if aug else ''}"; res[name] = []
    allrows = np.arange(len(df)); stats = feats(raw_inputs(allrows, None, which), allrows)[1]
    for seed in SEEDS:
        torch.manual_seed(seed); rng = np.random.default_rng(100 + seed); t0 = time.time()
        model = NDVICNN1D(feats(raw_inputs(tr[:2], None, which), tr[:2], stats)[0].shape[-1])
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2); lossf = nn.CrossEntropyLoss(weight=cw)
        Fva = feats(raw_inputs(va, None, which), va, stats)[0]; yt = torch.from_numpy(Y[tr]); best, state, bad = -1, None, 0
        Ftr_clean = feats(raw_inputs(tr, None, which), tr, stats)[0]
        for ep in range(EPOCHS):
            if aug and which != "SAR":
                dr = drop_mask(tr, rng.uniform(0.0, 0.7, size=len(tr)), rng)
                Ftr = feats(raw_inputs(tr, dr, which), tr, stats)[0]
            else: Ftr = Ftr_clean
            xt = torch.from_numpy(Ftr); model.train(); perm = torch.randperm(len(tr))
            for i in range(0, len(tr), 32):
                b = perm[i:i + 32]; opt.zero_grad()
                lossf(model(xt[b]).reshape(-1, 4), yt[b].reshape(-1)).backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            f1 = mt.seg_metrics(Y[va], predict(model, Fva).argmax(-1))["macro_f1"]
            if f1 > best: best, bad, state = f1, 0, {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 15: break
        model.load_state_dict(state); out = dict(seed=seed, val=best, epochs=ep + 1, by_level={})
        for p in LEVELS:
            prob = predict(model, feats(raw_inputs(te, TEST_KEEP[p], which), te, stats)[0]); Pv = mt.viterbi_batch(prob)
            m = mt.seg_metrics(Y[te], Pv); d = mt.date_metrics(Y[te], Pv, X[te])
            out["by_level"][str(p)] = dict(macro_f1=m["macro_f1"], sos_mae=d["sos"]["mae_days"], eos_mae=d["eos"]["mae_days"])
        # natural cloudiness of the test windows (fraction of steps really observed is 1 - miss_ratio), no extra masking
        prob = predict(model, feats(raw_inputs(te, TEST_KEEP[0.0], which), te, stats)[0]); Pv = mt.viterbi_batch(prob)
        mr = df.miss_ratio.values[te]; qs = np.quantile(mr, [1 / 3, 2 / 3]); grp = np.digitize(mr, qs)
        out["by_natural_cloud"] = {g: mt.seg_metrics(Y[te][grp == i], Pv[grp == i])["macro_f1"] for i, g in enumerate(["low", "mid", "high"])}
        out["natural_cloud_ranges"] = [float(mr.min()), float(qs[0]), float(qs[1]), float(mr.max())]
        res[name].append(out)
        print("%-24s seed %d val %.3f | " % (name, seed, best) + " ".join("%.0f%%: %.3f" % (p * 100, out["by_level"][str(p)]["macro_f1"]) for p in LEVELS) + "  %.0fs" % (time.time() - t0), flush=True)
        json.dump(dict(levels=LEVELS, block_steps=BLOCK, results=res), open(os.path.join(mt.OUT, "cloud_results%s.json" % ("_smoke" if smoke else "")), "w"), indent=1, default=float)
print("DONE")
