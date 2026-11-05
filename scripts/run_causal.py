"""Online (causal) stage recognition: a unidirectional LSTM sees only steps 0..t when labelling step t.
Compared with the bidirectional LSTM from run_dl.py (same cell type, same inputs, same training recipe).
Also reports detection delay: signed error (predicted - true) of each transition, in days (positive = late)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import mt, numpy as np, json, time, torch, torch.nn as nn
from src.models.bilstm import NDVIBiLSTM

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
df, X, Y, doy = mt.load()
sp = df.split.values
tr, va, te = [np.where(sp == s)[0] for s in ("train", "val", "test")]
F = mt.features(X, doy)


class CausalLSTM(nn.Module):
    def __init__(self, c, hidden=64):
        super().__init__()
        self.rnn = nn.LSTM(c, hidden, batch_first=True); self.out = nn.Linear(hidden, 4)
    def forward(self, x):
        return self.out(self.rnn(x)[0])


def signed_delay(Y_, P_, X_, tol=12):
    out = {"sos": [], "eos": []}
    for y, p, x in zip(Y_, P_, X_):
        te_, pe_ = mt.events(y, x), mt.events(p, x)
        for k in out:
            for a in te_[k]:
                c = [b - a for b in pe_[k] if abs(b - a) <= tol]
                if c: out[k].append(min(c, key=abs) * mt.STEP_DAYS)
    return {k: dict(median=float(np.median(v)), mean=float(np.mean(v)), n=len(v)) for k, v in out.items()}


cw = np.bincount(Y[tr].ravel(), minlength=4).astype(float); cw = torch.tensor((cw.sum() / cw) ** 0.5, dtype=torch.float32); cw = cw / cw.mean()
def predict(model, feats):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(feats)), -1).numpy()

res = {}
for name, ctor in (("LSTM_causal", lambda: CausalLSTM(3)), ("BiLSTM", lambda: NDVIBiLSTM(3))):
    res[name] = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed); np.random.seed(seed)
        model = ctor(); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2); lossf = nn.CrossEntropyLoss(weight=cw)
        xt, yt = torch.from_numpy(F[tr]), torch.from_numpy(Y[tr]); best, state, bad = -1, None, 0
        for ep in range(120):
            model.train(); perm = torch.randperm(len(tr))
            for i in range(0, len(tr), 32):
                b = perm[i:i + 32]; opt.zero_grad()
                lossf(model(xt[b]).reshape(-1, 4), yt[b].reshape(-1)).backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            f1 = mt.seg_metrics(Y[va], predict(model, F[va]).argmax(-1))["macro_f1"]
            if f1 > best: best, bad, state = f1, 0, {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 15: break
        model.load_state_dict(state); P = predict(model, F[te]).argmax(-1)
        r = dict(seed=seed, val=best, macro_f1=mt.seg_metrics(Y[te], P)["macro_f1"], f1=mt.seg_metrics(Y[te], P)["f1"],
                 dates=mt.date_metrics(Y[te], P, X[te]), delay=signed_delay(Y[te], P, X[te]), retro=mt.retrograde_rate(P))
        res[name].append(r)
        print("%-12s seed %d  test macroF1 %.4f  delay median SOS %.0f EOS %.0f d  retro %.2f" % (name, seed, r["macro_f1"], r["delay"]["sos"]["median"], r["delay"]["eos"]["median"], r["retro"]), flush=True)
        json.dump(res, open(os.path.join(mt.OUT, "causal_results.json"), "w"), indent=1, default=float)
print("DONE")
