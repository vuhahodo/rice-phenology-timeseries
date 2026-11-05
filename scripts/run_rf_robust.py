import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import mt, numpy as np, json
from sklearn.ensemble import RandomForestClassifier
df, X, Y, doy = mt.load()
sp = df.split.values
tr, va, te = [np.where(sp == s)[0] for s in ("train", "val", "test")]
RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
mrng = np.random.default_rng(123)   # identical generation order to run_dl.py
masks = {r: np.stack([(mrng.random(mt.T) >= r) | (np.arange(mt.T) == 0) | (np.arange(mt.T) == mt.T - 1) for _ in range(len(te))]) for r in RATES}
feat = lambda Xn, idx: np.concatenate([Xn, df.start_doy.values[idx, None] / 365.0], axis=1)
rf = RandomForestClassifier(n_estimators=500, max_depth=25, max_features="sqrt", n_jobs=-1, random_state=0).fit(feat(X[tr], tr), Y[tr])
out = {}
for r in RATES:
    Xm = X[te].copy()
    if r > 0:
        for i in range(len(te)): Xm[i] = mt.interp_mask(Xm[i], masks[r][i])
    proba = rf.predict_proba(feat(Xm, te))
    prob = np.zeros((len(te), mt.T, 4))
    for t in range(mt.T): prob[:, t, rf.classes_[t].astype(int)] = proba[t]
    out[str(r)] = mt.seg_metrics(Y[te], mt.viterbi_batch(prob))["macro_f1"]
    print(r, round(out[str(r)], 4), flush=True)
json.dump(out, open(os.path.join(mt.OUT, "rf_robust.json"), "w"), indent=1)
