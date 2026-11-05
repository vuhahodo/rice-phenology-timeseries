import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import mt, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

O, FD = mt.OUT, os.path.join(mt.MT, "figs")
base = json.load(open(f"{O}/baselines.json")); dl = json.load(open(f"{O}/dl_results.json")); rfr = json.load(open(f"{O}/rf_robust.json"))
df, X, Y, doy = mt.load(); te = np.where(df.split.values == "test")[0]
COL = ["#B8AD93", "#5FA86B", "#D9A83A", "#C9653F"]; INK, MUTED, GREEN, AMBER = "#1B2A21", "#5F6B63", "#17322A", "#C98A2B"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
ms = lambda a: (float(np.mean(a)), float(np.std(a)))
S = {}
def dl_row(name, key):
    runs = dl[name]; g = lambda f: [f(r) for r in runs]
    return dict(n_seeds=len(runs),
        macro_f1=ms(g(lambda r: r[key]["macro_f1"])), acc=ms(g(lambda r: r[key]["acc"])),
        f1=[ms([r[key]["f1"][c] for r in runs]) for c in range(4)],
        mae={k: ms(g(lambda r: r[key]["dates"][k]["mae_days"])) for k in ("sos", "pos", "eos")},
        miss={k: ms(g(lambda r: r[key]["dates"][k]["miss"])) for k in ("sos", "pos", "eos")},
        hit10={k: ms(g(lambda r: r[key]["dates"][k]["hit10"])) for k in ("sos", "pos", "eos")},
        retro=ms(g(lambda r: r[key]["retrograde"])),
        params=runs[0]["params"], train_s=ms(g(lambda r: r["train_s"])), epochs=ms(g(lambda r: r["epochs"])))
for n in dl: S[n] = dict(raw=dl_row(n, "raw"), viterbi=dl_row(n, "viterbi"),
                         robust={k: ms([r["robust"][k] for r in dl[n]]) for k in dl[n][0]["robust"]})
S["_base"] = base; S["_rf_robust"] = rfr
json.dump(S, open(f"{O}/summary.json", "w"), indent=1, default=float)

# ---- Fig: macro-F1 bars ----
rows = [("Lịch mùa vụ", base["B0_calendar"]["macro_f1"], None, None, "#9AA49D"),
        ("Luật ngưỡng 20%", base["B1_threshold_rule"]["macro_f1"], None, None, "#9AA49D"),
        ("Random Forest", base["B2_random_forest"]["macro_f1"], None, None, "#6C8576")]
order = ["CNN1D", "BiLSTM", "Transformer_NDVI", "Transformer_DOY"]
lab = {"CNN1D": "1D-CNN", "BiLSTM": "BiLSTM", "Transformer_NDVI": "Transformer\n(chỉ NDVI)", "Transformer_DOY": "Transformer\n(+DOY)"}
for n in order:
    if n in S: rows.append((lab[n], S[n]["viterbi"]["macro_f1"][0], S[n]["viterbi"]["macro_f1"][1], S[n]["raw"]["macro_f1"][0], GREEN if n == "CNN1D" else "#4C8A62"))
fig, ax = plt.subplots(figsize=(13, 4.3))
for i, (nm, v, sd, raw, c) in enumerate(rows):
    ax.bar(i, v, color=c, width=.62, yerr=sd if sd else None, error_kw=dict(ecolor=INK, capsize=4, lw=1.4))
    ax.text(i, v + (sd or 0) + .012, ("%.3f" % v).replace(".", ","), ha="center", fontsize=11, fontweight="bold")
    if raw is not None: ax.plot([i - .31, i + .31], [raw, raw], color=AMBER, lw=2.4)
ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=10); ax.set_ylim(0.5, 1.0); ax.set_ylabel("Macro-F1 (test)")
ax.plot([], [], color=AMBER, lw=2.4, label="trước giải mã Viterbi"); ax.legend(frameon=False, loc="upper left", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FD}/fig_results.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ---- Fig: robustness ----
fig, ax = plt.subplots(figsize=(8.6, 4.6)); xs = [0, 10, 20, 30, 40, 50]
ax.plot(xs, [rfr[str(r)] for r in [0.0, .1, .2, .3, .4, .5]], "-o", color="#9AA49D", lw=2.4, label="Random Forest")
sty = {"CNN1D": (GREEN, "1D-CNN"), "BiLSTM": ("#D9A83A", "BiLSTM"), "Transformer_DOY": ("#4C8A62", "Transformer (+DOY)")}
for n, (c, l) in sty.items():
    if n in S:
        m = np.array([S[n]["robust"][k][0] for k in ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]]); s = np.array([S[n]["robust"][k][1] for k in ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]])
        ax.plot(xs, m, "-o", color=c, lw=2.4, label=l); ax.fill_between(xs, m - s, m + s, color=c, alpha=.15)
ax.set_xlabel("% bước NDVI bị che thêm khi kiểm thử (nội suy tuyến tính lại)"); ax.set_ylabel("Macro-F1 (test)"); ax.legend(frameon=False, fontsize=10, loc="lower left", bbox_to_anchor=(0.0, 0.2))
fig.tight_layout(); fig.savefig(f"{FD}/fig_robust.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ---- Fig: qualitative masks (windows chosen by fixed rule: best-model+Viterbi accuracy percentiles 50/25/5) ----
if os.path.exists(f"{O}/dl_test_prob_seed0.npz"):
    pr = np.load(f"{O}/dl_test_prob_seed0.npz")["CNN1D"]; Pc = mt.viterbi_batch(pr)
    Prf = mt.viterbi_batch(np.load(f"{O}/rf_test_prob.npz")["prob"])
    acc = (Pc == Y[te]).mean(1); order_ = np.argsort(acc)
    pick = [order_[int(len(te) * q)] for q in (0.5, 0.25, 0.05)]
    fig = plt.figure(figsize=(14.5, 6.6)); gs = fig.add_gridspec(4, 3, height_ratios=[1.5, .16, .16, .16], hspace=.18, wspace=.14)
    cm = ListedColormap(COL)
    for k, w in enumerate(pick):
        i = te[w]; a = fig.add_subplot(gs[0, k]); a.plot(np.arange(mt.T) * 5, X[i], color=INK, lw=2); a.set_ylim(-.05, 1); a.set_xlim(0, 365); a.set_xticks([])
        a.set_title("%s %d · %d vụ · độ chính xác bước %.0f%%" % (df.province[i].replace("Da Nang", "Đà Nẵng").replace("Quang Nam", "Quảng Nam"), df.year[i], df.n_seasons[i], acc[w] * 100), loc="left", fontsize=10.5)
        if k == 0: a.set_ylabel("NDVI")
        for r, (y, nm) in enumerate(((Y[i], "Nhãn"), (Prf[w], "RF"), (Pc[w], "CNN+Viterbi"))):
            b = fig.add_subplot(gs[r + 1, k]); b.imshow(y[None], aspect="auto", cmap=cm, vmin=0, vmax=3, extent=(0, 365, 0, 1)); b.set_yticks([]); b.set_xticks([] if r < 2 else [0, 100, 200, 300])
            for s_ in b.spines.values(): s_.set_visible(False)
            if k == 0: b.set_ylabel(nm, rotation=0, ha="right", va="center", fontsize=10)
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in range(4)]
    fig.legend(h, ["Ngoài vụ", "Sinh trưởng", "Sinh sản", "Suy giảm"], loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(.5, -.03))
    fig.savefig(f"{FD}/fig_qualitative.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    # confusion matrix for the best model+Viterbi
    from sklearn.metrics import confusion_matrix
    cmx = confusion_matrix(Y[te].ravel(), Pc.ravel(), normalize="true"); json.dump(cmx.tolist(), open(f"{O}/cm_cnn.json", "w"))
print("ok")
for n in order:
    if n in S: print(n, "viterbi F1 %.3f±%.3f  raw %.3f  retro raw %.3f  MAE sos/pos/eos %.1f/%.1f/%.1f  seeds %d" % (*S[n]["viterbi"]["macro_f1"], S[n]["raw"]["macro_f1"][0], S[n]["raw"]["retro"][0], *(S[n]["viterbi"]["mae"][k][0] for k in ("sos", "pos", "eos")), S[n]["viterbi"]["n_seeds"]))
