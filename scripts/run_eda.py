import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import mt, numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

df, X, Y, doy = mt.load()
COL = ["#B8AD93", "#5FA86B", "#D9A83A", "#C9653F"]   # fallow / vegetative / reproductive / senescence
INK, MUTED = "#1B2A21", "#5F6B63"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
sp = df.split.values
eda = {}
eda["n"] = len(df); eda["fields"] = int(df.field_id.nunique()); eda["cols"] = int(df.shape[1]) - 1  # minus helper column "cell" added by mt.load()
eda["split"] = df.split.value_counts().to_dict()
eda["province_by_split"] = pd.crosstab(df.split, df.province).to_dict()
eda["year"] = df.year.value_counts().sort_index().to_dict()
eda["n_seasons"] = df.n_seasons.value_counts().sort_index().to_dict()
eda["miss_mean"] = float(df.miss_ratio.mean()); eda["miss_p90"] = float(df.miss_ratio.quantile(.9)); eda["miss_max"] = float(df.miss_ratio.max())
eda["class_share"] = (np.bincount(Y.ravel(), minlength=4) / Y.size).tolist()
eda["cells"] = int(df.cell.nunique()); eda["cells_multi_split"] = int((df.groupby("cell").split.nunique() > 1).sum())
eda["windows_in_multi_split_cells"] = int(df.cell.isin(df.groupby("cell").split.nunique()[lambda s: s > 1].index).sum())
eda["fields_multi_split"] = int((df.groupby("field_id").split.nunique() > 1).sum())
eda["fields_multi_window"] = int((df.groupby("field_id").size() > 1).sum())

# ---------- label sensitivity: re-run the SAME rule with threshold 10/20/30% ----------
res = {}
for thr in (0.1, 0.2, 0.3):
    res[thr] = [mt.detect(s, thr) for s in X]
d_sos, d_eos = [], []
for i in range(len(X)):
    a, b, c = res[0.1][i], res[0.2][i], res[0.3][i]
    if len(a) == len(b) == len(c) and len(b) > 0:
        for k in range(len(b)):
            d_sos.append((a[k][0], b[k][0], c[k][0])); d_eos.append((a[k][2], b[k][2], c[k][2]))
d_sos, d_eos = np.array(d_sos) * mt.STEP_DAYS, np.array(d_eos) * mt.STEP_DAYS
def spread(a):  # max-min across the 3 thresholds, per season
    s = a.max(1) - a.min(1); return dict(median=float(np.median(s)), p90=float(np.percentile(s, 90)), mean=float(s.mean()), n=len(s))
eda["label_sensitivity_days"] = dict(sos=spread(d_sos), eos=spread(d_eos),
    sos_20_vs_10=float(np.abs(d_sos[:, 1] - d_sos[:, 0]).mean()), sos_30_vs_20=float(np.abs(d_sos[:, 2] - d_sos[:, 1]).mean()),
    eos_20_vs_10=float(np.abs(d_eos[:, 1] - d_eos[:, 0]).mean()), eos_30_vs_20=float(np.abs(d_eos[:, 2] - d_eos[:, 1]).mean()))
# season length
eda["season_len_days"] = dict(median=float(np.median((d_eos[:, 1] - d_sos[:, 1]))), p10=float(np.percentile(d_eos[:, 1] - d_sos[:, 1], 10)), p90=float(np.percentile(d_eos[:, 1] - d_sos[:, 1], 90)))
json.dump(eda, open(os.path.join(mt.OUT, "eda.json"), "w"), indent=1, default=float)
print(json.dumps(eda, indent=1, default=float))

fig_dir = os.path.join(mt.MT, "figs")
# ---------- Fig 1: dataset overview ----------
fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1, 1, 1]))
ct = pd.crosstab(df.split, df.province).loc[["train", "val", "test"]]
ax[0].bar(ct.index, ct["Quang Nam"], color="#4C8A62", label="Quảng Nam"); ax[0].bar(ct.index, ct["Da Nang"], bottom=ct["Quang Nam"], color="#9CC3A8", label="Đà Nẵng")
for i, s in enumerate(ct.index): ax[0].text(i, ct.loc[s].sum() + 20, int(ct.loc[s].sum()), ha="center", fontsize=10)
ax[0].set_title("Cửa sổ theo tập và tỉnh", loc="left", fontsize=12); ax[0].legend(frameon=False, fontsize=9, loc="upper right")
sh = [np.bincount(Y[sp == s].ravel(), minlength=4) / (sp == s).sum() / mt.T * 100 for s in ("train", "val", "test")]
bot = np.zeros(3)
for c in range(4):
    v = np.array([h[c] for h in sh]); ax[1].bar(["train", "val", "test"], v, bottom=bot, color=COL[c], label=mt.NAMES[c]); bot += v
ax[1].set_title("Tỷ lệ bốn lớp (% bước)", loc="left", fontsize=12); ax[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5)) if False else None
ax[2].hist(df.miss_ratio * 100, bins=24, color="#7C8F84"); ax[2].axvline(df.miss_ratio.mean() * 100, color=COL[3], lw=2)
ax[2].text(df.miss_ratio.mean() * 100 + 1.5, ax[2].get_ylim()[1] * .88, "TB %.1f%%" % (df.miss_ratio.mean() * 100), color=COL[3], fontsize=10)
ax[2].set_title("Tỷ lệ bước thiếu ảnh gốc (%)", loc="left", fontsize=12)
ns = df.n_seasons.value_counts().sort_index(); ax[3].bar([str(k) for k in ns.index], ns.values, color="#4C8A62")
for i, v in enumerate(ns.values): ax[3].text(i, v + 20, str(v), ha="center", fontsize=10)
ax[3].set_title("Số vụ lúa trong một cửa sổ", loc="left", fontsize=12)
h = [plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in range(4)]
fig.legend(h, ["Ngoài vụ", "Sinh trưởng", "Sinh sản", "Suy giảm"], loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_data_overview.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------- Fig 2: example windows with stage strips ----------
def strip(ax, y):
    ax.imshow(y[None, :], aspect="auto", cmap=ListedColormap(COL), vmin=0, vmax=3, extent=(0, mt.T * 5, 0, 1)); ax.set_yticks([]); ax.set_xticks([])
    for sp_ in ax.spines.values(): sp_.set_visible(False)
ex = [int(np.where((df.n_seasons == 2) & (df.miss_ratio < .15))[0][0]), int(np.where((df.n_seasons == 1))[0][3]), int(np.argsort(-df.miss_ratio.values)[5])]
fig, axes = plt.subplots(3, 2, figsize=(13.5, 6.4), gridspec_kw=dict(height_ratios=[1, 0.16, 0.0001]) if False else None)
fig = plt.figure(figsize=(14, 6.2)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.14], hspace=0.05, wspace=0.18)
for k, i in enumerate(ex):
    a = fig.add_subplot(gs[0, k]); a.plot(np.arange(mt.T) * 5, X[i], color=INK, lw=2)
    a.set_ylim(-0.05, 1.0); a.set_xlim(0, 365); a.set_xticks([])
    a.set_title("%s %d · %d vụ · thiếu %.0f%%" % (df.province[i].replace("Da Nang", "Đà Nẵng").replace("Quang Nam", "Quảng Nam"), df.year[i], df.n_seasons[i], df.miss_ratio[i] * 100), loc="left", fontsize=11)
    if k == 0: a.set_ylabel("NDVI")
    b = fig.add_subplot(gs[1, k]); strip(b, Y[i]); b.set_xlabel("ngày trong cửa sổ", fontsize=9, color=MUTED); b.set_xticks([0, 100, 200, 300]); b.tick_params(labelsize=9)
fig.legend(h, ["Ngoài vụ", "Sinh trưởng", "Sinh sản", "Suy giảm"], loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.05))
fig.savefig(os.path.join(fig_dir, "fig_examples.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3: label sensitivity (stacked, larger type) ----------
fig, ax = plt.subplots(2, 1, figsize=(7.4, 5.8), sharex=True)
for a, d, nm in ((ax[0], d_sos, "SOS"), (ax[1], d_eos, "EOS")):
    s_ = d.max(1) - d.min(1)
    a.hist(s_, bins=np.arange(-2.5, 62.5, 5), color="#4C8A62"); a.axvline(np.median(s_), color=COL[3], lw=2.4)
    a.set_title("%s: trung vị %.0f ngày, P90 %.0f ngày" % (nm, np.median(s_), np.percentile(s_, 90)), loc="left", fontsize=14)
    a.set_ylabel("số mùa vụ", fontsize=12); a.tick_params(labelsize=12)
ax[1].set_xlabel("độ lệch của mốc khi đổi ngưỡng 10 / 20 / 30% (ngày)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_label_sensitivity.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
print("figs done")
