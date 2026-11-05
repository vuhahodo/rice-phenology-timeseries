"""SICKLE (India) Sentinel-1 -> harvest / sowing day, evaluated against farmer-reported dates (train+val plots only; test is withheld).
Phase 1 reads S1 scenes straight from the zip (never extracts it) and caches the per-scene median VV/VH.
Phase 2 predicts HARVESTING_DAY / SOWING_DAY with plot-grouped 5-fold CV and compares with the season-mean baseline.
Data stays local; only aggregate results are written. Usage: python -I run_sickle_s1.py [extract|model|all]"""
import sys, os, io, re, json, time, zipfile, datetime as dt; sys.path.insert(0, os.path.dirname(__file__))
import mt, numpy as np, pandas as pd

ROOT = mt.ROOT
ZIP = os.path.join(ROOT, "data", "external", "sickle", "raw", "sickle_dataset.zip")
TAB = os.path.join(ROOT, "data", "external", "sickle", "tabular", "sickle_dataset_tabular.csv")
CACHE = os.path.join(mt.OUT, "sickle_s1_cache.json")
MONTH = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
mode = sys.argv[1] if len(sys.argv) > 1 else "all"

tab = pd.read_csv(TAB)
pad = tab[(tab.CROP.str.lower() == "paddy") & tab.SPLIT.isin(["train", "val"]) & tab.HARVESTING_DAY.notna()].copy()
print("paddy train+val rows with harvest day:", len(pad), "| plots:", pad.PLOT_ID.nunique(), "| seasons:", pad.STANDARD_SEASON.value_counts().to_dict(), flush=True)


def season_start(r):
    m = MONTH[str(r.STANDARD_SEASON).split("-")[0][:3].lower()]
    return dt.date(int(r.YEAR), m, 1)

pad["start"] = pad.apply(season_start, axis=1)
# consistency of the "days since season start" convention (sowing date - start == SOWING_DAY)
ok = pad.dropna(subset=["SOWING_DATE", "SOWING_DAY"])
imp = [(pd.to_datetime(r.SOWING_DATE).date() - r.start).days - r.SOWING_DAY for r in ok.itertuples()]
print("season-start convention: sowing-date minus start minus SOWING_DAY  -> zero in %.1f%% of %d rows" % (100 * np.mean(np.array(imp) == 0), len(imp)), flush=True)

if mode in ("extract", "all"):
    t0 = time.time(); z = zipfile.ZipFile(ZIP); print("zip opened: %d entries (%.0fs)" % (len(z.namelist()), time.time() - t0), flush=True)
    want = set(map(str, pad.UNIQUE_ID)); idx = {}
    for nm in z.namelist():
        if "/S1/npy/" in nm and nm.endswith(".npz"):
            p = nm.split("/"); uid = p[-2]
            if uid in want: idx.setdefault(uid, []).append(nm)
    print("samples with S1 scenes: %d / %d | scenes: %d" % (len(idx), len(want), sum(map(len, idx.values()))), flush=True)
    cache = {}
    for k, (uid, names) in enumerate(idx.items()):
        rows = []
        for nm in names:
            m = re.search(r"_(\d{8})T", nm.split("/")[-1])
            if not m: continue
            a = np.load(io.BytesIO(z.read(nm)), allow_pickle=False)
            rows.append((m.group(1), float(np.median(a["VV"])), float(np.median(a["VH"])), float(a["angle"].ravel()[0])))
        cache[uid] = sorted(rows)
        if k % 100 == 0: print("  read %d/%d samples (%.0fs)" % (k, len(idx), time.time() - t0), flush=True)
    json.dump(cache, open(CACHE, "w")); print("cached ->", CACHE, flush=True)

if mode in ("model", "all"):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    cache = json.load(open(CACHE)); STEP, NG = 6, 46; grid = np.arange(NG) * STEP
    feats, keep = [], []
    for r in pad.itertuples():
        sc = cache.get(str(r.UNIQUE_ID))
        if not sc: continue
        d = np.array([(dt.datetime.strptime(s[0], "%Y%m%d").date() - r.start).days for s in sc], float)
        sel = (d >= -12) & (d <= grid[-1] + 12)
        if sel.sum() < 8: continue
        o = np.argsort(d[sel]); dd = d[sel][o]
        vv, vh = np.array([s[1] for s in sc])[sel][o], np.array([s[2] for s in sc])[sel][o]
        ivv, ivh = np.interp(grid, dd, vv), np.interp(grid, dd, vh)
        # simple, interpretable timing features: flooded/bare minimum of VH and the later maximum
        jmin = int(np.argmin(ivh[:NG // 2])); jmax = jmin + int(np.argmax(ivh[jmin:]))
        feats.append(np.concatenate([ivv, ivh, ivh - ivv, [grid[jmin], grid[jmax], ivh[jmax] - ivh[jmin], len(dd)]])); keep.append(r.Index)
    P = pad.loc[keep].copy(); F = np.array(feats); print("usable rows:", len(P), "| plots:", P.PLOT_ID.nunique(), flush=True)
    out = {"n_rows": int(len(P)), "n_plots": int(P.PLOT_ID.nunique())}
    # Many rows carry season-boundary values (sowing = day 1, harvest = last day of the standard season; apr-aug has ONE value for all 48 rows).
    # Report both: every row, and "informative" rows that exclude those boundary values.
    hmax = P.groupby("STANDARD_SEASON").HARVESTING_DAY.transform("max"); hnun = P.groupby("STANDARD_SEASON").HARVESTING_DAY.transform("nunique")
    INFO = {"HARVESTING_DAY": ((P.HARVESTING_DAY != hmax) & (hnun > 1)).values, "SOWING_DAY": ((P.SOWING_DAY != 1) & (hnun > 1)).values}
    for subset in ("all_rows", "informative_rows"):
        out[subset] = {}
        for tgt in ("HARVESTING_DAY", "SOWING_DAY"):
            m = P[tgt].notna().values
            if subset == "informative_rows": m = m & INFO[tgt]
            Pm, Fm = P[m], F[m]; y = Pm[tgt].values.astype(float); g = Pm.PLOT_ID.values; seas = Pm.STANDARD_SEASON.values
            pred_b, pred_g = np.zeros(len(y)), np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=5).split(Fm, y, g):
                mean_by = pd.Series(y[tr]).groupby(seas[tr]).mean()
                base = np.array([mean_by.get(s_, y[tr].mean()) for s_ in seas])
                pred_b[te] = base[te]
                gb = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0).fit(Fm[tr], y[tr] - base[tr])
                pred_g[te] = base[te] + gb.predict(Fm[te])
            res = {}
            for nm, pr in (("season_mean_baseline", pred_b), ("S1_gradient_boosting", pred_g)):
                e = np.abs(pr - y); res[nm] = dict(mae=float(e.mean()), median_ae=float(np.median(e)), hit7=float((e <= 7).mean()), hit10=float((e <= 10).mean()), hit15=float((e <= 15).mean()))
            rng = np.random.default_rng(0); ug = np.unique(g); dif = []
            for _ in range(500):
                pick = rng.choice(ug, len(ug)); ii = np.concatenate([np.where(g == q)[0] for q in pick])
                dif.append(np.abs(pred_b[ii] - y[ii]).mean() - np.abs(pred_g[ii] - y[ii]).mean())
            res["mae_gain_baseline_minus_S1"] = dict(mean=float(np.mean(dif)), lo=float(np.percentile(dif, 2.5)), hi=float(np.percentile(dif, 97.5)))
            res["n"] = int(len(y)); res["n_plots"] = int(len(ug)); res["target_std_days"] = float(y.std())
            res["per_season"] = {s_: dict(n=int((seas == s_).sum()), baseline_mae=float(np.abs(pred_b - y)[seas == s_].mean()), s1_mae=float(np.abs(pred_g - y)[seas == s_].mean())) for s_ in np.unique(seas) if (seas == s_).sum() >= 15}
            out[subset][tgt] = res
            print(subset, tgt, "n=%d plots=%d | baseline MAE %.2f -> S1 MAE %.2f | gain %.2f [%.2f, %.2f] | hit<=7d %.2f -> %.2f" % (
                len(y), len(ug), res["season_mean_baseline"]["mae"], res["S1_gradient_boosting"]["mae"], res["mae_gain_baseline_minus_S1"]["mean"],
                res["mae_gain_baseline_minus_S1"]["lo"], res["mae_gain_baseline_minus_S1"]["hi"], res["season_mean_baseline"]["hit7"], res["S1_gradient_boosting"]["hit7"]), flush=True)
    json.dump(out, open(os.path.join(mt.OUT, "sickle_s1_results.json"), "w"), indent=1)
print("DONE")
