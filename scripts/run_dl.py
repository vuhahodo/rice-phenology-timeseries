import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import mt, numpy as np, json, time, torch, torch.nn as nn
from src.models.cnn1d import NDVICNN1D
from src.models.bilstm import NDVIBiLSTM
from src.models.transformer import NDVITransformer

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
df, X, Y, doy = mt.load()
sp = df.split.values
tr, va, te = [np.where(sp == s)[0] for s in ("train", "val", "test")]
F = mt.features(X, doy)
RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = [0, 1, 2, 3, 4]
MODELS = {
    "CNN1D":            (lambda: NDVICNN1D(3), 3),
    "BiLSTM":           (lambda: NDVIBiLSTM(3), 3),
    "Transformer_NDVI": (lambda: NDVITransformer(1), 1),
    "Transformer_DOY":  (lambda: NDVITransformer(3), 3),
}
# fixed random masks shared by every model (robustness test)
mrng = np.random.default_rng(123)
masks = {r: np.stack([(mrng.random(mt.T) >= r) | (np.arange(mt.T) == 0) | (np.arange(mt.T) == mt.T - 1)
                      for _ in range(len(te))]) for r in RATES}

def masked_inputs(r):
    Xm = X[te].copy()
    if r > 0:
        for i in range(len(te)):
            Xm[i] = mt.interp_mask(Xm[i], masks[r][i])
    return mt.features(Xm, doy[te])

cw = np.bincount(Y[tr].ravel(), minlength=4).astype(float)
cw = torch.tensor((cw.sum() / cw) ** 0.5, dtype=torch.float32); cw = cw / cw.mean()

def predict(model, feats, c):
    model.eval()
    with torch.no_grad():
        out = []
        for i in range(0, len(feats), 256):
            out.append(torch.softmax(model(torch.from_numpy(feats[i:i + 256][:, :, :c])), -1).numpy())
    return np.concatenate(out)

results, store = {}, {}
for name, (ctor, c) in MODELS.items():
    results[name] = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = ctor(); npar = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
        lossf = nn.CrossEntropyLoss(weight=cw)
        xt = torch.from_numpy(F[tr][:, :, :c]); yt = torch.from_numpy(Y[tr])
        best, best_state, bad, t0 = -1, None, 0, time.time()
        for ep in range(120):
            model.train(); perm = torch.randperm(len(tr))
            for i in range(0, len(tr), 32):
                b = perm[i:i + 32]
                opt.zero_grad()
                loss = lossf(model(xt[b]).reshape(-1, 4), yt[b].reshape(-1))
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            f1 = mt.seg_metrics(Y[va], predict(model, F[va], c).argmax(-1))["macro_f1"]
            if f1 > best:
                best, bad = f1, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 15: break
        train_s = time.time() - t0
        model.load_state_dict(best_state)
        prob = predict(model, F[te], c); P = prob.argmax(-1); Pv = mt.viterbi_batch(prob)
        r = dict(seed=seed, val_macro_f1=best, epochs=ep + 1, train_s=train_s, params=npar,
                 raw=dict(**mt.seg_metrics(Y[te], P), dates=mt.date_metrics(Y[te], P, X[te]), retrograde=mt.retrograde_rate(P)),
                 viterbi=dict(**mt.seg_metrics(Y[te], Pv), dates=mt.date_metrics(Y[te], Pv, X[te]), retrograde=mt.retrograde_rate(Pv)),
                 robust={})
        for rate in RATES:
            pm = predict(model, masked_inputs(rate), c)
            r["robust"][str(rate)] = mt.seg_metrics(Y[te], mt.viterbi_batch(pm))["macro_f1"]
        results[name].append(r)
        if seed == 0: store[name] = prob
        print("%-17s seed %d  val %.4f  test raw %.4f viterbi %.4f  retro %.3f->%.3f  ep %3d  %.0fs  params %d" % (
            name, seed, best, r["raw"]["macro_f1"], r["viterbi"]["macro_f1"], r["raw"]["retrograde"], r["viterbi"]["retrograde"], ep + 1, train_s, npar), flush=True)
        json.dump(results, open(os.path.join(mt.OUT, "dl_results.json"), "w"), indent=1, default=float)
np.savez(os.path.join(mt.OUT, "dl_test_prob_seed0.npz"), **store)
print("DONE")
