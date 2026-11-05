"""No-NDVI branch: train the 1D-CNN on SAR (VV, VH, VH-VV) / NDWI inputs instead of NDVI.
Input file: rice_phenology_multi_v3.npz written by notebook cell 22 (X: N x 73 x 5 with channels NDVI, NDWI, VV, VH, obs_ndvi).
Usage: python -I midterm/scripts/run_sar.py [path/to/npz] [--smoke]
Labels are still derived from NDVI, so a good SAR-only score says "the information is present in SAR",
not "SAR reproduces ground-truth phenology"."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import mt, numpy as np, json, time, torch, torch.nn as nn
from src.models.cnn1d import NDVICNN1D

smoke = "--smoke" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NPZ = args[0] if args else os.path.join(mt.MT, "sar", "rice_phenology_multi_v3.npz")
TAG = "smoke" if smoke else "real"
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

df, X, Y, doy = mt.load()
sp = df.split.values
tr, va, te = [np.where(sp == s)[0] for s in ("train", "val", "test")]
z = np.load(NPZ, allow_pickle=False)   # downloaded file: never unpickle; split.npy (object array) is therefore not read
XM = z["X"].astype(np.float32); ch = [str(c) for c in z["channels"]]
assert XM.shape[:2] == (len(df), mt.T), "shape mismatch %s" % (XM.shape,)
ndvi_diff = float(np.abs(XM[:, :, ch.index("NDVI")] - X).max())
same_y = bool((z["y"] == Y).all()); same_split = True   # split comes from the CSV; rows are verified through NDVI and labels
print("channels", ch, "| max |NDVI_npz - NDVI_csv| = %.2e | labels identical: %s" % (ndvi_diff, same_y), flush=True)
assert ndvi_diff < 1e-4 and same_y, "npz rows are not aligned with the CSV"
for c in ("NDWI", "VV", "VH"):
    k = ch.index(c); print("  %-5s mean %.3f std %.3f  NaN %d" % (c, np.nanmean(XM[:, :, k]), np.nanstd(XM[:, :, k]), np.isnan(XM[:, :, k]).sum()))
XM = np.nan_to_num(XM)
ndvi, ndwi, vv, vh = (XM[:, :, ch.index(c)] for c in ("NDVI", "NDWI", "VV", "VH"))
ang = 2 * np.pi * doy / 365.0
CONFIGS = {
    "NDVI+DOY (control)":     [ndvi],
    "SAR (VV,VH,VH-VV)+DOY":  [vv, vh, vh - vv],
    "SAR+NDWI+DOY":           [vv, vh, vh - vv, ndwi],
    "NDVI+NDWI+SAR+DOY":      [ndvi, ndwi, vv, vh, vh - vv],
}
def build(chs):
    A = np.stack(chs, axis=-1).astype(np.float32)
    mu, sd = A[tr].mean((0, 1)), A[tr].std((0, 1)) + 1e-6          # z-score with train statistics only
    A = (A - mu) / sd
    return np.concatenate([A, np.sin(ang)[..., None], np.cos(ang)[..., None]], -1).astype(np.float32)

cw = np.bincount(Y[tr].ravel(), minlength=4).astype(float); cw = torch.tensor((cw.sum() / cw) ** 0.5, dtype=torch.float32); cw = cw / cw.mean()
def predict(model, f):
    model.eval()
    with torch.no_grad(): return torch.softmax(model(torch.from_numpy(f)), -1).numpy()

SEEDS = [0] if smoke else [0, 1, 2, 3, 4]; EPOCHS = 3 if smoke else 120
res = {}
for name, chs in CONFIGS.items():
    F = build(chs); res[name] = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = NDVICNN1D(F.shape[-1]); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2); lossf = nn.CrossEntropyLoss(weight=cw)
        xt, yt = torch.from_numpy(F[tr]), torch.from_numpy(Y[tr]); best, state, bad, t0 = -1, None, 0, time.time()
        for ep in range(EPOCHS):
            model.train(); perm = torch.randperm(len(tr))
            for i in range(0, len(tr), 32):
                b = perm[i:i + 32]; opt.zero_grad()
                lossf(model(xt[b]).reshape(-1, 4), yt[b].reshape(-1)).backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            f1 = mt.seg_metrics(Y[va], predict(model, F[va]).argmax(-1))["macro_f1"]
            if f1 > best: best, bad, state = f1, 0, {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 15: break
        model.load_state_dict(state); prob = predict(model, F[te]); P = prob.argmax(-1); Pv = mt.viterbi_batch(prob)
        r = dict(seed=seed, val=best, epochs=ep + 1, in_ch=F.shape[-1], raw=mt.seg_metrics(Y[te], P), viterbi=mt.seg_metrics(Y[te], Pv),
                 dates=mt.date_metrics(Y[te], Pv, X[te]), retro_raw=mt.retrograde_rate(P))
        res[name].append(r)
        print("%-24s seed %d  val %.4f  test raw %.4f viterbi %.4f  MAE SOS/EOS %.1f/%.1f  ep %d  %.0fs" % (
            name, seed, best, r["raw"]["macro_f1"], r["viterbi"]["macro_f1"], r["dates"]["sos"]["mae_days"], r["dates"]["eos"]["mae_days"], ep + 1, time.time() - t0), flush=True)
        json.dump(dict(npz=NPZ, channels=ch, results=res), open(os.path.join(mt.OUT, "sar_results_%s.json" % TAG), "w"), indent=1, default=float)
print("DONE")
