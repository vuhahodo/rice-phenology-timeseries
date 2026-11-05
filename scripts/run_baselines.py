import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import mt, numpy as np, pandas as pd, json, time, itertools
from sklearn.ensemble import RandomForestClassifier

df, X, Y, doy = mt.load()
sp = df.split.values
tr, va, te = [np.where(sp == s)[0] for s in ("train", "val", "test")]
res, store = {}, {}

def evaluate(name, P, idx, extra=None):
    m = mt.seg_metrics(Y[idx], P)
    m["dates"] = mt.date_metrics(Y[idx], P, X[idx])
    m["retrograde"] = mt.retrograde_rate(P)
    if extra: m.update(extra)
    return m

# ---- B0 calendar: per (province, day-of-year) majority class learned on train ----
prov = (df.province.values == "Da Nang").astype(int)
cnt = np.zeros((2, 366, 4))
for i in tr:
    for t in range(mt.T):
        cnt[prov[i], doy[i, t], Y[i, t]] += 1
glob = cnt.sum(0)
def cal_pred(idx):
    P = np.zeros((len(idx), mt.T), dtype=np.int64)
    for a, i in enumerate(idx):
        for t in range(mt.T):
            c = cnt[prov[i], doy[i, t]]
            P[a, t] = (c if c.sum() > 0 else glob[doy[i, t]]).argmax()
    return P
res["B0_calendar"] = evaluate("B0", cal_pred(te), te); store["B0_calendar"] = cal_pred(te)

# ---- B1 threshold rule (label generator re-implemented at window level) ----
Pr = np.stack([mt.rule_labels(s) for s in X[te]])
nd = np.array([len(mt.detect(s)) for s in X[te]])
nt = np.array([int(sum((y[t-1] == 0 and y[t] > 0) for t in range(1, len(y))) + (y[0] > 0)) for y in Y[te]])
fired = Pr > 0
agree_when_fired = float((Pr[fired] == Y[te][fired]).mean())
full = nd == nt
res["B1_threshold_rule"] = evaluate("B1", Pr, te, dict(
    agree_when_rule_fires=agree_when_fired,
    windows_detected_all_seasons=float(full.mean()),
    exact_match_on_those=float(np.mean([(a == b).all() for a, b in zip(Pr[full], Y[te][full])]))))
store["B1_threshold_rule"] = Pr

# ---- B2 Random Forest multi-output (BreizhCrops-style), config picked on validation ----
def feat(idx):
    return np.concatenate([X[idx], df.start_doy.values[idx, None] / 365.0], axis=1)
grid = [dict(max_depth=d, min_samples_leaf=l) for d, l in itertools.product([15, 25, None], [1, 3])]
best = None
for g in grid:
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=-1, random_state=0, **g).fit(feat(tr), Y[tr])
    Pv = rf.predict(feat(va))
    f1 = mt.seg_metrics(Y[va], Pv)["macro_f1"]
    print("RF", g, "val macroF1 %.4f (%.0fs)" % (f1, time.time() - t0), flush=True)
    if best is None or f1 > best[0]:
        best = (f1, g, rf)
f1v, gbest, rf = best
print("selected", gbest, flush=True)
t0 = time.time(); proba = rf.predict_proba(feat(te)); inf_s = time.time() - t0
prob = np.zeros((len(te), mt.T, 4))
for t in range(mt.T):
    prob[:, t, rf.estimators_[0].classes_ if False else rf.classes_[t].astype(int)] = proba[t]
Prf = prob.argmax(-1)
res["B2_random_forest"] = evaluate("B2", Prf, te, dict(config=gbest, val_macro_f1=f1v, seeds=1))
store["B2_random_forest"] = Prf
Pvit = mt.viterbi_batch(prob)
res["B2_random_forest+viterbi"] = evaluate("B2v", Pvit, te)
np.savez(os.path.join(mt.OUT, "rf_test_prob.npz"), prob=prob)
np.savez(os.path.join(mt.OUT, "baseline_test_preds.npz"), **{k: v for k, v in store.items()})
json.dump(res, open(os.path.join(mt.OUT, "baselines.json"), "w"), indent=1, default=float)
for k, v in res.items():
    d = v["dates"]
    print("%-28s macroF1 %.3f  acc %.3f | MAE d  SOS %s POS %s EOS %s | retro %.2f" % (
        k, v["macro_f1"], v["acc"], *(("%.1f" % d[e]["mae_days"]) for e in ("sos", "pos", "eos")), v["retrograde"]))
print({k: res["B1_threshold_rule"][k] for k in ("agree_when_rule_fires", "windows_detected_all_seasons", "exact_match_on_those")})
