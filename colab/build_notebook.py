"""Builds rice_vn_full_pipeline.ipynb (runs on Kaggle and on Colab). Run: python build_notebook.py"""
import json, os, nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r'''
# Giám sát lúa Việt Nam từ Sentinel-1 và Sentinel-2: pipeline đầy đủ

Notebook tự đủ, chạy trên **Kaggle** hoặc **Colab**. Dựa trên notebook thu thập dữ liệu của nhóm (Earth Engine, Đà Nẵng và Quảng Nam, 2023–2025) và kế hoạch đã chốt.

| Phần | Việc | Cần Earth Engine |
|---|---|---|
| 1 | Đăng nhập Earth Engine | có |
| 2 | Trích ~1.050 điểm **không phải lúa** (5 kênh, 3 năm) | có |
| 3 | Ghép cửa sổ 73 bước cho điểm mới, nạp dữ liệu lúa của nhóm | không |
| 4 | **Bài A**: phân loại lúa so với không phải lúa (RF và 1D-CNN, chia theo ô lưới 2 km) | không |
| 5 | **Bài B**: mặt nạ giai đoạn lúa (1D-CNN + Viterbi), so NDVI với SAR + NDWI | không |

**Đăng nhập Earth Engine** (một trong ba cách, thử theo thứ tự):
1. Kaggle Secret tên `EE_SERVICE_ACCOUNT_JSON` (nội dung file khóa JSON của service account); chạy ngầm được.
2. Một dataset Kaggle riêng tư chứa file khóa JSON của service account, gắn vào notebook.
3. Chạy tay: notebook in ra một liên kết, bạn đăng nhập Google và dán mã xác nhận.

Nếu không có thông tin đăng nhập, notebook bỏ qua phần 2 và bài A, vẫn chạy bài B.

**Lưu ý trung thực:** nhãn "lúa" dẫn xuất từ luật NDVI/NDWI của nhóm; nhãn "không phải lúa" lấy từ WorldCover. Đây đều là nhãn yếu; kết quả bài A cho biết mô hình tái tạo được các nhãn đó đến đâu, chưa phải độ chính xác ngoài đồng.
''')

code(r'''
# ===== 0. Cấu hình =====
import os, sys, json, time, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

SMOKE = os.environ.get("SMOKE", "0") == "1"       # 1 = chạy thử nhanh bằng dữ liệu giả, không cần Earth Engine
ON_KAGGLE = os.path.exists("/kaggle/working")
WORK = "/kaggle/working" if ON_KAGGLE else os.environ.get("WORK", os.path.abspath("work"))
os.makedirs(WORK, exist_ok=True)
PROJECT_ID = "verdant-cargo-437911-h6"            # Project ID Earth Engine (kiểm tra lại trên thanh địa chỉ Google Cloud)
YEARS = [2023, 2024, 2025]
EPOCH_SCALE = 0.05 if SMOKE else 1.0
NR_DIR = os.path.join(WORK, "nonrice"); os.makedirs(NR_DIR, exist_ok=True)
RES_DIR = os.path.join(WORK, "results"); os.makedirs(RES_DIR, exist_ok=True)

def find_file(name):
    c = []
    if os.environ.get("RICE_DIR"): c.append(os.path.join(os.environ["RICE_DIR"], name))
    c += glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    c += [os.path.join(WORK, name), name]
    return next((p for p in c if os.path.exists(p)), None)

RICE_FILES = {"rice_phenology_multi_v3.npz": "1HFooXWc-6HzBjK5bQkMJUywJPY4ni4Xr",   # id file trong thư mục Drive của nhóm
              "phenology_dataset_v3.csv": "1WwgH-RQWxZK-6Ox3Z2T-VAZAyTNaHvu0"}
paths = {}
for nm, gid in RICE_FILES.items():
    p = find_file(nm)
    if p is None:                                  # Colab: tải từ thư mục Drive chia sẻ công khai của nhóm
        os.system("pip -q install gdown"); import gdown
        p = gdown.download(id=gid, output=os.path.join(WORK, nm), quiet=True)
    paths[nm] = p
print("SMOKE =", SMOKE, "| nền tảng:", "Kaggle" if ON_KAGGLE else "khác", "\n", paths)
''')

code(r'''
# ===== Hàm dùng chung: chỉ số, Viterbi, mô hình =====
import torch, torch.nn as nn
from scipy.signal import savgol_filter
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier

T, STEP = 73, 5
NAMES = ["Fallow", "Vegetative", "Reproductive", "Senescence"]
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

class ConvBlock(nn.Module):
    def __init__(self, i, o, k=5, d=1, p=0.1):
        super().__init__()
        self.conv = nn.Conv1d(i, o, k, padding=(k - 1) * d // 2, dilation=d); self.bn = nn.BatchNorm1d(o)
        self.act = nn.ReLU(); self.drop = nn.Dropout(p); self.res = nn.Conv1d(i, o, 1) if i != o else nn.Identity()
    def forward(self, x): return self.drop(self.act(self.bn(self.conv(x)))) + self.res(x)

class SeqCNN(nn.Module):                            # nhãn cho từng bước (bài B)
    def __init__(self, c_in, n_cls=4, ch=(32, 64, 64), dil=(1, 2, 4)):
        super().__init__(); L, prev = [], c_in
        for c, d in zip(ch, dil): L.append(ConvBlock(prev, c, 5, d)); prev = c
        self.blocks = nn.ModuleList(L); self.head = nn.Conv1d(prev, n_cls, 1)
    def forward(self, x):
        h = x.transpose(1, 2)
        for b in self.blocks: h = b(h)
        return self.head(h).transpose(1, 2)

class ClsCNN(nn.Module):                            # nhãn cho cả cửa sổ (bài A)
    def __init__(self, c_in, n_cls=2, ch=(32, 64, 64), dil=(1, 2, 4)):
        super().__init__(); L, prev = [], c_in
        for c, d in zip(ch, dil): L.append(ConvBlock(prev, c, 5, d)); prev = c
        self.blocks = nn.ModuleList(L); self.fc = nn.Linear(prev, n_cls)
    def forward(self, x):
        h = x.transpose(1, 2)
        for b in self.blocks: h = b(h)
        return self.fc(h.mean(-1))

def seg_metrics(Y, P):
    f = f1_score(Y.ravel(), P.ravel(), average=None, labels=[0, 1, 2, 3], zero_division=0)
    return dict(acc=float(accuracy_score(Y.ravel(), P.ravel())), macro_f1=float(f.mean()), f1=[float(v) for v in f])

def events(y, ndvi):
    sos = [t for t in range(1, len(y)) if y[t - 1] == 0 and y[t] > 0]
    eos = [t for t in range(1, len(y)) if y[t - 1] == 3 and y[t] == 0]
    pos, t = [], 0
    while t < len(y):
        if y[t] in (1, 2):
            u = t
            while u + 1 < len(y) and y[u + 1] in (1, 2): u += 1
            pos.append(t + int(np.argmax(ndvi[t:u + 1]))); t = u + 1
        else: t += 1
    return dict(sos=sos, pos=pos, eos=eos)

def date_metrics(Y, P, X, max_match=12):
    out = {}
    for k in ("sos", "pos", "eos"):
        errs, n = [], 0
        for y, p, x in zip(Y, P, X):
            te, pe = events(y, x)[k], events(p, x)[k]
            for a in te:
                n += 1; e = min((abs(a - b) for b in pe), default=None)
                if e is not None and e <= max_match: errs.append(e * STEP)
        out[k] = dict(mae_days=float(np.mean(errs)) if errs else None, miss=1 - len(errs) / max(n, 1), n=n)
    return out

def viterbi(logp):                                  # chỉ cho phép giữ nguyên hoặc tiến một giai đoạn (vòng 0-1-2-3-0)
    Tn = logp.shape[0]; NEG = -1e9; tr = np.full((4, 4), NEG)
    for c in range(4): tr[c, c] = 0.0; tr[c, (c + 1) % 4] = 0.0
    dp = np.zeros((Tn, 4)); bp = np.zeros((Tn, 4), dtype=np.int64); dp[0] = logp[0]
    for t in range(1, Tn):
        cand = dp[t - 1][:, None] + tr; bp[t] = cand.argmax(0); dp[t] = cand.max(0) + logp[t]
    path = np.zeros(Tn, dtype=np.int64); path[-1] = dp[-1].argmax()
    for t in range(Tn - 1, 0, -1): path[t - 1] = bp[t, path[t]]
    return path
def viterbi_batch(prob): return np.stack([viterbi(np.log(np.clip(p, 1e-8, 1))) for p in prob])

def fit_predict(model, Xtr, ytr, Xte, epochs, seed, cw=None, Xva=None, yva=None, seq=False, patience=15):
    torch.manual_seed(seed); np.random.seed(seed); model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss(weight=None if cw is None else torch.tensor(cw, dtype=torch.float32, device=DEV))
    xt, yt = torch.from_numpy(Xtr).to(DEV), torch.from_numpy(ytr).to(DEV)
    def predict(A):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(A), 256): out.append(torch.softmax(model(torch.from_numpy(A[i:i + 256]).to(DEV)), -1).cpu().numpy())
        return np.concatenate(out)
    best, state, bad = -1, None, 0
    for ep in range(max(2, int(epochs))):
        model.train(); perm = torch.randperm(len(xt), device=DEV)
        for i in range(0, len(xt), 32):
            b = perm[i:i + 32]; opt.zero_grad()
            lossf(model(xt[b]).reshape(-1, model_out(model)), yt[b].reshape(-1)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if Xva is not None:
            f = f1_score(yva.ravel(), predict(Xva).argmax(-1).ravel(), average="macro", zero_division=0)
            if f > best: best, bad, state = f, 0, {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
    if state is not None: model.load_state_dict(state)
    return predict(Xte)
def model_out(m): return m.head.out_channels if hasattr(m, "head") else m.fc.out_features
print("thiết bị:", DEV)
''')

code(r'''
# ===== 1. Đăng nhập Earth Engine =====
EE_OK = False
EE_INTERACTIVE = (not ON_KAGGLE) or os.environ.get("EE_INTERACTIVE", "0") == "1"   # Kaggle chạy ngầm: không hỏi tương tác; muốn đăng nhập tay thì đặt True
HAVE_RAW = all(find_file(f) for f in [f"raw_nonrice_{y}.csv" for y in YEARS] + [f"raw_extra_nonrice_{y}.csv" for y in YEARS])

def ee_login():
    import ee
    key_json = None
    try:
        from kaggle_secrets import UserSecretsClient
        key_json = UserSecretsClient().get_secret("EE_SERVICE_ACCOUNT_JSON")
    except Exception: pass
    if key_json is None:
        for f in glob.glob("/kaggle/input/**/*.json", recursive=True):
            try:
                d = json.load(open(f))
                if d.get("type") == "service_account" and "client_email" in d: key_json = json.dumps(d); break
            except Exception: pass
    if key_json:
        d = json.loads(key_json); kf = os.path.join(WORK, "_ee_key.json"); open(kf, "w").write(key_json)
        try: ee.Initialize(ee.ServiceAccountCredentials(d["client_email"], kf), project=d.get("project_id", PROJECT_ID))
        finally: os.remove(kf)                       # không để lại khóa trong thư mục kết quả
        print("Đăng nhập bằng service account:", d["client_email"])
    elif EE_INTERACTIVE:
        ee.Authenticate(); ee.Initialize(project=PROJECT_ID)   # tương tác: mở liên kết, đăng nhập, dán mã
        print("Đăng nhập tương tác thành công")
    else:
        raise RuntimeError("Chưa có khóa service account. Đặt EE_INTERACTIVE = True (chạy tay) hoặc gắn dataset chứa khóa JSON.")
    ee.Number(1).getInfo()                           # thử một lệnh thật để chắc quyền đã có

if SMOKE or HAVE_RAW:
    print("Bỏ qua đăng nhập Earth Engine (đã có file thô hoặc đang chạy thử).")
else:
    try: ee_login(); EE_OK = True
    except Exception as e:
        print("KHÔNG đăng nhập được Earth Engine:", type(e).__name__, str(e)[:300])
        print("-> bỏ qua phần trích dữ liệu và bài A; bài B vẫn chạy.")
''')

code(r'''
# ===== 2. Trích điểm KHÔNG phải lúa từ Earth Engine (hàm chép từ notebook của nhóm) =====
def run_extraction():
    import ee
    global points, n, roi, region
    gaul = ee.FeatureCollection("FAO/GAUL/2015/level1"); vn = gaul.filter(ee.Filter.eq("ADM0_NAME", "Viet Nam"))
    region = vn.filter(ee.Filter.Or(ee.Filter.stringContains("ADM1_NAME", "Nang"), ee.Filter.eq("ADM1_NAME", "Quang Nam")))
    print(region.aggregate_array("ADM1_NAME").getInfo()); roi = region.geometry()
    worldcover = ee.Image("ESA/WorldCover/v200/2021").select("Map"); cropland = worldcover.eq(40)
    def prep25(img):
        scl = img.select("SCL"); clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return (img.normalizedDifference(["B8", "B4"]).rename("NDVI").addBands(img.normalizedDifference(["B3", "B8"]).rename("NDWI"))
                .updateMask(clear).copyProperties(img, ["system:time_start"]))
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate("2025-01-01", "2026-01-01")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)).map(prep25))
    p = s2.select("NDVI").reduce(ee.Reducer.percentile([5, 95])); amp = p.select("NDVI_p95").subtract(p.select("NDVI_p5"))
    w90 = s2.select("NDWI").reduce(ee.Reducer.percentile([90])).rename("ndwi_p90")
    rice_candidate = cropland.And(amp.gt(0.35)).And(p.select("NDVI_p95").gt(0.6)).And(w90.gt(-0.2)).selfMask().clip(roi)   # luật của nhóm
    lc = worldcover.where(rice_candidate.unmask(0).eq(1), 99).rename("lc")
    CLASSES = [40, 10, 30, 20, 50, 80]                       # cây trồng khác, cây thân gỗ, cỏ, bụi, đô thị, nước
    PER = {"Nang": [60, 60, 30, 20, 20, 20], "Quang Nam": [240, 240, 120, 80, 80, 80]}
    def sample(name, counts):
        g = region.filter(ee.Filter.stringContains("ADM1_NAME", name)).geometry()
        return lc.stratifiedSample(numPoints=0, classBand="lc", region=g, scale=30, classValues=CLASSES, classPoints=counts,
                                   seed=7, geometries=True, tileScale=8).map(lambda f: f.set("province", name))
    pts = ee.FeatureCollection([sample(k, v) for k, v in PER.items()]).flatten()
    n = pts.size().getInfo(); lst = pts.toList(n)
    points = ee.FeatureCollection(ee.List.sequence(0, n - 1).map(lambda i: ee.Feature(lst.get(i)).set("field_id", ee.Number(i).int().add(100000))))
    print("Số điểm không phải lúa:", n, points.aggregate_histogram("lc").getInfo())

    def get_s2(year):
        def prep(img):
            scl = img.select("SCL"); clear = scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
            return img.normalizedDifference(["B8", "B4"]).rename("NDVI").updateMask(clear).copyProperties(img, ["system:time_start"])
        return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate(f"{year-1}-12-15", f"{year+1}-01-15").map(prep)
    def sample_stack(stack, batch, geometries):
        rows, plist = [], points.toList(n)
        for b in range(0, n, batch):
            chunk = ee.FeatureCollection(plist.slice(b, min(b + batch, n))); out = None
            for attempt in range(3):
                try: out = stack.sampleRegions(collection=chunk, scale=10, geometries=geometries).getInfo(); break
                except Exception as e: print("Retry", attempt, str(e)[:120]); time.sleep(5)
            if out is None: continue
            for f in out["features"]:
                d = dict(f["properties"])
                if geometries: d["lon"], d["lat"] = f["geometry"]["coordinates"]
                rows.append(d)
            print(f"   {min(b+batch, n)}/{n}")
        return pd.DataFrame(rows)
    def extract_year(year):
        col = get_s2(year); dummy = ee.Image.constant(0).rename("NDVI").updateMask(ee.Image.constant(0)); start = ee.Date(f"{year}-01-01"); bands = []
        for i in range(73):
            c = start.advance(i * 5, "day")
            bands.append(col.filterDate(c.advance(-5, "day"), c.advance(10, "day")).merge(ee.ImageCollection([dummy])).median().unmask(-9999).rename(f"t{i:02d}"))
        return sample_stack(ee.Image.cat(bands), 100, True)
    def s2_ndwi_col(year):
        def prep(img):
            scl = img.select("SCL"); clear = scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
            return img.normalizedDifference(["B3", "B8"]).rename("NDWI").updateMask(clear).copyProperties(img, ["system:time_start"])
        return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate(f"{year-1}-12-15", f"{year+1}-01-15").map(prep)
    def s1_col(year):
        col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(roi).filterDate(f"{year-1}-12-15", f"{year+1}-01-15")
               .filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).select(["VV", "VH"]))
        return col.map(lambda im: im.focalMedian(30, "circle", "meters").copyProperties(im, ["system:time_start"]))
    def make_stack(col, names, prefixes, year):
        start = ee.Date(f"{year}-01-01"); dummy = ee.Image.constant([0] * len(names)).rename(names).updateMask(ee.Image.constant(0)); out = []
        for i in range(73):
            c = start.advance(i * 5, "day")
            out.append(col.filterDate(c.advance(-5, "day"), c.advance(10, "day")).merge(ee.ImageCollection([dummy])).median()
                       .select(names).unmask(-9999).rename([f"{pr}{i:02d}" for pr in prefixes]))
        return ee.Image.cat(out)
    def extract_extra(year):
        return sample_stack(ee.Image.cat([make_stack(s2_ndwi_col(year), ["NDWI"], ["w"], year), make_stack(s1_col(year), ["VV", "VH"], ["v", "h"], year)]), 50, False)
    for y in YEARS:                                          # có thể chạy lại: bỏ qua file đã có
        a, b = os.path.join(NR_DIR, f"raw_nonrice_{y}.csv"), os.path.join(NR_DIR, f"raw_extra_nonrice_{y}.csv")
        if not os.path.exists(a): print("Năm", y, "NDVI"); extract_year(y).to_csv(a, index=False)
        if not os.path.exists(b): print("Năm", y, "NDWI, VV, VH"); extract_extra(y).to_csv(b, index=False)

if EE_OK and not HAVE_RAW:
    run_extraction()
    HAVE_RAW = True
elif HAVE_RAW and not SMOKE:
    for f in [f"raw_nonrice_{y}.csv" for y in YEARS] + [f"raw_extra_nonrice_{y}.csv" for y in YEARS]:
        src = find_file(f); dst = os.path.join(NR_DIR, f)
        if src and os.path.abspath(src) != os.path.abspath(dst): import shutil; shutil.copy(src, dst)
print("Đã có file thô điểm không phải lúa:", HAVE_RAW or SMOKE)
''')

code(r'''
# ===== 3a. Nạp dữ liệu lúa của nhóm và kiểm tra khớp dòng =====
z = np.load(paths["rice_phenology_multi_v3.npz"], allow_pickle=False)           # không bật pickle với file tải về
XR = np.nan_to_num(z["X"].astype(np.float32)); CH = [str(c) for c in z["channels"]]
dfr = pd.read_csv(paths["phenology_dataset_v3.csv"])
YR = dfr[[f"stage_{i:02d}" for i in range(1, T + 1)]].values.astype(np.int64)
ND = dfr[[f"NDVI_{i:02d}" for i in range(1, T + 1)]].values.astype(np.float32)
assert XR.shape[:2] == (len(dfr), T) and (z["y"] == YR).all() and np.abs(XR[:, :, CH.index("NDVI")] - ND).max() < 1e-4, "npz và CSV không khớp dòng"
print("Lúa:", XR.shape, "| kênh:", CH, "| tập:", dfr.split.value_counts().to_dict())
dfr["cell"] = (dfr.lon // 0.02).astype(int).astype(str) + "_" + (dfr.lat // 0.02).astype(int).astype(str)

# vị trí bắt đầu cửa sổ trong chuỗi 3 năm (219 bước) của từng cửa sổ lúa; dùng để cắt cửa sổ điểm mới cùng phân bố thời gian
def start_index(year, start_doy):
    m = (int(start_doy) - 1) // 5
    for k in range(3):
        st = m + 73 * k
        if st <= 146 and 2023 + (st + 36) // 73 == int(year): return st
    return m
ST_RICE = np.array([start_index(y, d) for y, d in zip(dfr.year, dfr.start_doy)])
''')

code(r'''
# ===== 3b. Ghép cửa sổ cho điểm không phải lúa (cùng cách xử lý như notebook của nhóm) =====
def make_fake_raw(npts=160, seed=0):                   # CHỈ để chạy thử code (SMOKE=1), không phải dữ liệu thật
    rng = np.random.default_rng(seed); ids = 100000 + np.arange(npts); lc = rng.choice([40, 10, 30, 20, 50, 80], npts)
    lon, lat = rng.uniform(107.8, 108.6, npts), rng.uniform(15.0, 16.1, npts)
    for y in YEARS:
        tt = np.arange(73)
        def curve(): return np.clip(0.3 + 0.2 * np.sin(tt / 12 + rng.uniform(0, 6)) + rng.normal(0, 0.03, 73), -0.1, 0.95)
        pd.DataFrame([dict(field_id=int(i), lc=int(c), province="Quang Nam", lon=a, lat=b, **{f"t{j:02d}": (-9999 if rng.random() < 0.25 else v) for j, v in enumerate(curve())})
                      for i, c, a, b in zip(ids, lc, lon, lat)]).to_csv(os.path.join(NR_DIR, f"raw_nonrice_{y}.csv"), index=False)
        pd.DataFrame([dict(field_id=int(i), lc=int(c), province="Quang Nam",
                           **{f"w{j:02d}": (-9999 if rng.random() < 0.3 else float(rng.uniform(-0.6, 0.1))) for j in range(73)},
                           **{f"v{j:02d}": float(rng.normal(-11, 2)) for j in range(73)}, **{f"h{j:02d}": float(rng.normal(-18, 2)) for j in range(73)})
                      for i, c in zip(ids, lc)]).to_csv(os.path.join(NR_DIR, f"raw_extra_nonrice_{y}.csv"), index=False)
if SMOKE: make_fake_raw()

def fill_smooth(M):
    a = M.interpolate(axis=1, limit_direction="both").values.astype(float)
    return savgol_filter(a, window_length=9, polyorder=2, axis=1)

XN, META_N = None, None
if HAVE_RAW or SMOKE:
    fr = lambda y: pd.read_csv(os.path.join(NR_DIR, f"raw_nonrice_{y}.csv")).drop_duplicates("field_id").set_index("field_id")
    fe = lambda y: pd.read_csv(os.path.join(NR_DIR, f"raw_extra_nonrice_{y}.csv")).drop_duplicates("field_id").set_index("field_id")
    base = fr(YEARS[0])[["lon", "lat", "province", "lc"]]
    def stack(loader, prefix):
        parts = [loader(y).reindex(base.index)[[f"{prefix}{i:02d}" for i in range(73)]].replace(-9999, np.nan) for y in YEARS]
        M = pd.concat(parts, axis=1); M.columns = range(M.shape[1]); return M
    Mt, Mw, Mv, Mh = stack(fr, "t"), stack(fe, "w"), stack(fe, "v"), stack(fe, "h")
    keep = ((Mt.isna().mean(1) < 0.6) & (Mv.isna().mean(1) < 0.6)).values
    Mt, Mw, Mv, Mh, base = Mt[keep], Mw[keep], Mv[keep], Mh[keep], base[keep]
    obs = (~Mt.isna()).values.astype(float)
    ndvi = np.clip(fill_smooth(Mt), -0.1, 1); ndwi = np.clip(fill_smooth(Mw.fillna(Mw.mean().mean())), -1, 1)
    vv, vh = fill_smooth(Mv), fill_smooth(Mh)
    C_all = np.stack([ndvi, ndwi, vv, vh, obs], -1).astype(np.float32)             # [điểm, 219, 5], cùng thứ tự kênh với dữ liệu lúa
    rng = np.random.default_rng(0); wins, meta = [], []
    for k, (pid, row) in enumerate(base.iterrows()):
        got = 0
        for _ in range(8):
            if got == 2: break
            st = int(rng.choice(ST_RICE))
            if 1 - C_all[k, st:st + T, 4].mean() > 0.5: continue                    # quá nhiều mây, như luật của nhóm
            wins.append(C_all[k, st:st + T]); meta.append(dict(field_id=pid, lc=int(row.lc), lon=row.lon, lat=row.lat, st=st)); got += 1
    XN, META_N = np.array(wins), pd.DataFrame(meta)
    META_N["cell"] = (META_N.lon // 0.02).astype(int).astype(str) + "_" + (META_N.lat // 0.02).astype(int).astype(str)
    print("Điểm không phải lúa:", len(base), "| cửa sổ:", XN.shape, "| theo lớp WorldCover:", META_N.lc.value_counts().to_dict())
else:
    print("Chưa có dữ liệu điểm không phải lúa: bài A sẽ bị bỏ qua.")
''')

code(r'''
# ===== 4. Bài A: phân loại lúa so với không phải lúa (chia theo ô lưới 2 km, 5 phần) =====
def summary_feats(A):
    d = np.diff(A, axis=1); p5, p95 = np.percentile(A, 5, axis=1), np.percentile(A, 95, axis=1)
    return np.concatenate([A.mean(1), A.std(1), A.min(1), A.max(1), p5, p95, p95 - p5, A.argmax(1) / T, A.argmin(1) / T, d.max(1), d.min(1), np.abs(d).mean(1)], 1)

RES_A = {}
if XN is not None:
    X_all = np.concatenate([XR, XN]); y_all = np.r_[np.ones(len(XR)), np.zeros(len(XN))].astype(np.int64)
    lc_all = np.r_[np.full(len(XR), 99), META_N.lc.values]; groups = np.r_[dfr.cell.values, META_N.cell.values]
    SETS = {"Quang hoc (NDVI+NDWI)": [0, 1], "SAR (VV,VH)": [2, 3], "Tat ca (NDVI,NDWI,VV,VH)": [0, 1, 2, 3]}
    n_splits = min(5, len(np.unique(groups))); folds = list(GroupKFold(n_splits=n_splits).split(X_all, y_all, groups))
    print("Rice:", int(y_all.sum()), "| không phải lúa:", int((1 - y_all).sum()), "| nhóm ô lưới:", len(np.unique(groups)))
    for sname, idx in SETS.items():
        for mname in ("RandomForest", "CNN"):
            fm, pooled = [], np.zeros(len(y_all), dtype=np.int64)
            for tr, te in folds:
                if mname == "RandomForest":
                    rf = RandomForestClassifier(300, n_jobs=-1, random_state=0, class_weight="balanced").fit(summary_feats(X_all[tr][:, :, idx]), y_all[tr])
                    pred = rf.predict(summary_feats(X_all[te][:, :, idx]))
                else:
                    A = X_all[:, :, idx]; mu, sd = A[tr].mean((0, 1)), A[tr].std((0, 1)) + 1e-6; A = ((A - mu) / sd).astype(np.float32)
                    cw = (len(tr) / (2 * np.bincount(y_all[tr], minlength=2))).tolist()
                    pred = fit_predict(ClsCNN(len(idx)), A[tr], y_all[tr], A[te], 40 * EPOCH_SCALE, 0, cw=cw, seq=False).argmax(-1)
                pooled[te] = pred
                p, r, f, _ = precision_recall_fscore_support(y_all[te], pred, labels=[1], zero_division=0)
                fm.append(dict(oa=accuracy_score(y_all[te], pred), macro_f1=f1_score(y_all[te], pred, average="macro"), f1_rice=float(f[0]), prec_rice=float(p[0]), rec_rice=float(r[0])))
            agg = {k: (float(np.mean([m[k] for m in fm])), float(np.std([m[k] for m in fm]))) for k in fm[0]}
            non = {int(c): float((pooled[(lc_all == c)] == 0).mean()) for c in np.unique(lc_all[lc_all != 99])}   # tỷ lệ nhận đúng "không phải lúa" theo lớp phủ
            RES_A[f"{sname} | {mname}"] = dict(folds=agg, correct_nonrice_by_worldcover_class=non)
            print("%-26s %-12s OA %.3f±%.3f  macro-F1 %.3f±%.3f  F1(lúa) %.3f  | đúng theo lớp: %s" % (sname, mname, *agg["oa"], *agg["macro_f1"], agg["f1_rice"][0], non), flush=True)
    json.dump(RES_A, open(os.path.join(RES_DIR, "taskA_results.json"), "w"), indent=1)
else:
    print("Bỏ qua bài A (chưa có dữ liệu điểm không phải lúa).")
''')

code(r'''
# ===== 5. Bài B: mặt nạ giai đoạn lúa (1D-CNN + Viterbi), so NDVI với SAR + NDWI =====
doy = ((dfr.start_doy.values[:, None] - 1 + STEP * np.arange(T)[None, :]) % 365) + 1
ang = 2 * np.pi * doy / 365.0
ndvi_r, ndwi_r, vv_r, vh_r = (XR[:, :, CH.index(c)] for c in ("NDVI", "NDWI", "VV", "VH"))
CFG_B = {"NDVI": [ndvi_r], "SAR+NDWI": [vv_r, vh_r, vh_r - vv_r, ndwi_r], "NDVI+NDWI+SAR": [ndvi_r, ndwi_r, vv_r, vh_r, vh_r - vv_r]}
sp = dfr.split.values; tr_i, va_i, te_i = [np.where(sp == s)[0] for s in ("train", "val", "test")]
cwB = np.bincount(YR[tr_i].ravel(), minlength=4).astype(float); cwB = ((cwB.sum() / cwB) ** 0.5); cwB = (cwB / cwB.mean()).tolist()
RES_B = {}
for name, chs in CFG_B.items():
    A = np.stack(chs, -1).astype(np.float32); mu, sd = A[tr_i].mean((0, 1)), A[tr_i].std((0, 1)) + 1e-6
    F = np.concatenate([(A - mu) / sd, np.sin(ang)[..., None], np.cos(ang)[..., None]], -1).astype(np.float32); RES_B[name] = []
    for seed in ([0] if SMOKE else [0, 1, 2]):
        t0 = time.time()
        prob = fit_predict(SeqCNN(F.shape[-1]), F[tr_i], YR[tr_i], F[te_i], 120 * EPOCH_SCALE, seed, cw=cwB, Xva=F[va_i], yva=YR[va_i], seq=True)
        Pv = viterbi_batch(prob); P = prob.argmax(-1)
        r = dict(seed=seed, raw=seg_metrics(YR[te_i], P), viterbi=seg_metrics(YR[te_i], Pv), dates=date_metrics(YR[te_i], Pv, ND[te_i]))
        RES_B[name].append(r)
        print("%-15s seed %d  macro-F1 thô %.3f -> Viterbi %.3f | MAE SOS/EOS %.1f/%.1f ngày  (%.0fs)" % (name, seed, r["raw"]["macro_f1"], r["viterbi"]["macro_f1"],
              r["dates"]["sos"]["mae_days"], r["dates"]["eos"]["mae_days"], time.time() - t0), flush=True)
json.dump(RES_B, open(os.path.join(RES_DIR, "taskB_results.json"), "w"), indent=1)
print("\nTóm tắt bài B (macro-F1 sau Viterbi, trung bình các seed):")
for k, v in RES_B.items(): print("  %-15s %.3f" % (k, np.mean([x["viterbi"]["macro_f1"] for x in v])))
''')

code(r'''
# ===== 6. Lưu kết quả =====
summary = dict(smoke=SMOKE, ee_logged_in=EE_OK, have_nonrice_raw=bool(HAVE_RAW or SMOKE), taskA=RES_A or None, taskB={k: float(np.mean([x["viterbi"]["macro_f1"] for x in v])) for k, v in RES_B.items()})
json.dump(summary, open(os.path.join(RES_DIR, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1)[:1500]); print("\nKết quả nằm trong:", RES_DIR, "| file thô điểm không phải lúa:", NR_DIR)
''')

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
out = os.path.join(HERE, "rice_vn_full_pipeline.ipynb")
nbf.write(nb, out); print("written", out, "| cells:", len(cells))
