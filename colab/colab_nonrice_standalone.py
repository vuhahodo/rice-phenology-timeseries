# ============================================================================
# BẢN TỰ ĐỦ: dán vào file Colab mới ("Untitled"), mỗi khối "CELL n" là một cell, chạy lần lượt từ trên xuống.
# Mục đích: lấy ~1.050 điểm KHÔNG phải lúa ở Đà Nẵng và Quảng Nam, 2023-2025, 5 kênh (NDVI, NDWI, VV, VH).
# Kết quả lưu vào Google Drive: MyDrive/PTDL/nonrice/ (6 file csv). Chạy lại cell 4 sẽ bỏ qua file đã có.
# CHƯA KIỂM THỬ: máy của Claude không có Earth Engine. Nếu báo lỗi, gửi lại nội dung lỗi.
# Các hàm trích được chép nguyên từ notebook Rice_Phenology_DaNang_QuangNam (cell 14-17 và 20).
# ============================================================================


# ============================ CELL 1: đăng nhập ============================
import os, time
import ee
import pandas as pd
import numpy as np
from google.colab import drive

ee.Authenticate()                       # mở liên kết, đăng nhập đúng tài khoản Google của bạn
PROJECT_ID = "DAN_PROJECT_ID_CUA_BAN"   # <-- thay bằng Project ID đã đăng ký Earth Engine (dạng ten-project-123456)
ee.Initialize(project=PROJECT_ID)

drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/PTDL/nonrice"
os.makedirs(OUT, exist_ok=True)
YEARS = [2023, 2024, 2025]
print("Earth Engine OK. Lưu vào:", OUT)


# ====================== CELL 2: vùng, mặt nạ, lấy điểm ======================
gaul = ee.FeatureCollection("FAO/GAUL/2015/level1")
vn = gaul.filter(ee.Filter.eq("ADM0_NAME", "Viet Nam"))
region = vn.filter(ee.Filter.Or(
    ee.Filter.stringContains("ADM1_NAME", "Nang"),
    ee.Filter.eq("ADM1_NAME", "Quang Nam")))
print(region.aggregate_array("ADM1_NAME").getInfo())     # cần ra 2 tỉnh
roi = region.geometry()

worldcover = ee.Image("ESA/WorldCover/v200/2021").select("Map")
cropland = worldcover.eq(40)

def prep25(img):
    scl = img.select("SCL")
    clear = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return ndvi.addBands(ndwi).updateMask(clear).copyProperties(img, ["system:time_start"])

s2_25 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi)
         .filterDate("2025-01-01", "2026-01-01").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)).map(prep25))
ndvi_p = s2_25.select("NDVI").reduce(ee.Reducer.percentile([5, 95]))
ndvi_amp = ndvi_p.select("NDVI_p95").subtract(ndvi_p.select("NDVI_p5"))
ndwi_p90 = s2_25.select("NDWI").reduce(ee.Reducer.percentile([90])).rename("ndwi_p90")

# luật "ứng viên lúa" của nhóm (cell 15 của notebook)
rice_candidate = (cropland.And(ndvi_amp.gt(0.35)).And(ndvi_p.select("NDVI_p95").gt(0.6))
                  .And(ndwi_p90.gt(-0.2))).selfMask().clip(roi)

# lớp phủ: pixel lúa -> 99; giữ nguyên các lớp WorldCover còn lại
rice_px = rice_candidate.unmask(0).eq(1)
lc = worldcover.where(rice_px, 99).rename("lc")

# 40 = cây trồng khác, 10 = cây thân gỗ/rừng, 30 = cỏ, 20 = bụi, 50 = đô thị, 80 = nước
CLASSES = [40, 10, 30, 20, 50, 80]
PER_PROVINCE = {"Nang": [60, 60, 30, 20, 20, 20],
                "Quang Nam": [240, 240, 120, 80, 80, 80]}

def sample_nonrice(name, counts):
    geom = region.filter(ee.Filter.stringContains("ADM1_NAME", name)).geometry()
    pts = lc.stratifiedSample(numPoints=0, classBand="lc", region=geom, scale=30,
                              classValues=CLASSES, classPoints=counts,
                              seed=7, geometries=True, tileScale=8)
    return pts.map(lambda f: f.set("province", name))

pts_nr = ee.FeatureCollection([sample_nonrice(k, v) for k, v in PER_PROVINCE.items()]).flatten()
n = pts_nr.size().getInfo()
lst = pts_nr.toList(n)
points = ee.FeatureCollection(ee.List.sequence(0, n - 1).map(
    lambda i: ee.Feature(lst.get(i)).set("field_id", ee.Number(i).int().add(100000))))
print("Số điểm không phải lúa:", n)
print(points.aggregate_histogram("lc").getInfo())


# ======================= CELL 3: các hàm trích chuỗi =======================
def get_s2(year):
    def prep(img):
        scl = img.select("SCL")
        clear = (scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return ndvi.updateMask(clear).copyProperties(img, ["system:time_start"])
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi)
            .filterDate(f"{year-1}-12-15", f"{year+1}-01-15").map(prep))

def extract_year(year, batch=100):
    col = get_s2(year)
    dummy = ee.Image.constant(0).rename("NDVI").updateMask(ee.Image.constant(0))
    start = ee.Date(f"{year}-01-01")
    bands = []
    for i in range(73):
        c = start.advance(i * 5, "day")
        comp = (col.filterDate(c.advance(-5, "day"), c.advance(10, "day"))
                .merge(ee.ImageCollection([dummy])).median().unmask(-9999).rename(f"t{i:02d}"))
        bands.append(comp)
    stack = ee.Image.cat(bands)
    rows, plist = [], points.toList(n)
    for b in range(0, n, batch):
        chunk = ee.FeatureCollection(plist.slice(b, min(b + batch, n)))
        out = None
        for attempt in range(3):
            try:
                out = stack.sampleRegions(collection=chunk, scale=10, geometries=True).getInfo()
                break
            except Exception as e:
                print("Retry", attempt, e); time.sleep(5)
        if out is None:
            continue
        for f in out["features"]:
            d = dict(f["properties"]); d["lon"], d["lat"] = f["geometry"]["coordinates"]; rows.append(d)
        print(f"  {year} NDVI: {min(b+batch, n)}/{n}")
    return pd.DataFrame(rows)

def s2_ndwi_col(year):
    def prep(img):
        scl = img.select("SCL")
        clear = (scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
        ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
        return ndwi.updateMask(clear).copyProperties(img, ["system:time_start"])
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi)
            .filterDate(f"{year-1}-12-15", f"{year+1}-01-15").map(prep))

def s1_col(year):
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(roi)
           .filterDate(f"{year-1}-12-15", f"{year+1}-01-15")
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .select(["VV", "VH"]))
    return col.map(lambda im: im.focalMedian(30, "circle", "meters").copyProperties(im, ["system:time_start"]))

def make_stack(col, band_names, prefixes, year):
    start = ee.Date(f"{year}-01-01")
    dummy = (ee.Image.constant([0] * len(band_names)).rename(band_names).updateMask(ee.Image.constant(0)))
    out = []
    for i in range(73):
        c = start.advance(i * 5, "day")
        comp = (col.filterDate(c.advance(-5, "day"), c.advance(10, "day"))
                .merge(ee.ImageCollection([dummy])).median()
                .select(band_names).unmask(-9999)
                .rename([f"{p}{i:02d}" for p in prefixes]))
        out.append(comp)
    return ee.Image.cat(out)

def extract_extra(year, batch=50):
    stack = ee.Image.cat([make_stack(s2_ndwi_col(year), ["NDWI"], ["w"], year),
                          make_stack(s1_col(year), ["VV", "VH"], ["v", "h"], year)])
    rows, plist = [], points.toList(n)
    for b in range(0, n, batch):
        chunk = ee.FeatureCollection(plist.slice(b, min(b + batch, n)))
        res = None
        for attempt in range(3):
            try:
                res = stack.sampleRegions(collection=chunk, scale=10, geometries=False).getInfo()
                break
            except Exception as e:
                print("Retry", attempt, e); time.sleep(5)
        if res is None:
            continue
        for f in res["features"]:
            rows.append(dict(f["properties"]))
        print(f"  {year} NDWI/SAR: {min(b+batch, n)}/{n}")
    return pd.DataFrame(rows)


# ================== CELL 4: chạy (có thể chạy lại để tiếp tục) ==================
for y in YEARS:
    f_ndvi = f"{OUT}/raw_nonrice_{y}.csv"
    f_extra = f"{OUT}/raw_extra_nonrice_{y}.csv"
    if not os.path.exists(f_ndvi):
        print("Năm", y, "- NDVI"); extract_year(y).to_csv(f_ndvi, index=False)
    if not os.path.exists(f_extra):
        print("Năm", y, "- NDWI, VV, VH"); extract_extra(y).to_csv(f_extra, index=False)
print("Xong. Thư mục MyDrive/PTDL/nonrice/ có 6 file csv. Chia sẻ thư mục theo link công khai rồi gửi lại.")
