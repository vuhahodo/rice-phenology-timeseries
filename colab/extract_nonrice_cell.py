# ============================================================================
# CELL MỚI cho notebook Rice_Phenology_DaNang_QuangNam (chạy trong Colab, SAU cell 20)
# Mục đích: lấy thêm ~1.050 điểm KHÔNG phải lúa (cây trồng khác, cây thân gỗ, cỏ, bụi, đô thị, nước)
# để làm bài phân loại "lúa so với cây khác". Tái dùng đúng hàm extract_year / extract_extra của notebook.
#
# CHƯA KIỂM THỬ: máy của Claude không có Earth Engine, nên cell này chưa được chạy thử.
# Nếu báo lỗi, gửi lại nội dung lỗi để sửa.
#
# Cần chạy trước, theo thứ tự: cell 2, 3, 4 (đăng nhập Earth Engine), 12, 13, 14, 15, 16 (gắn Drive, tạo `points`), 17.
# Cell 20 định nghĩa hàm extract_extra rồi tự chạy trích dữ liệu rất lâu: chạy cell 20, thấy in "Năm 2023" thì bấm Stop
# (các hàm đã được định nghĩa xong trước đó). KHÔNG chạy cell 18 (nó cũng tự trích lại dữ liệu lúa).
# ============================================================================
import time
YEARS = [2023, 2024, 2025]   # cell 18 định nghĩa biến này, nhưng ta không chạy cell 18

# 1) Lớp phủ: pixel lúa của luật cũ -> 99, giữ nguyên các lớp WorldCover còn lại
rice_px = rice_candidate.unmask(0).eq(1)
lc = worldcover.where(rice_px, 99).rename("lc")

# 2) Số điểm mỗi lớp (có thể chỉnh): 40 = cây trồng khác, 10 = cây thân gỗ/rừng, 30 = cỏ, 20 = bụi, 50 = đô thị, 80 = nước
CLASSES = [40, 10, 30, 20, 50, 80]
PER_PROVINCE = {"Nang": [60, 60, 30, 20, 20, 20],            # Đà Nẵng
                "Quang Nam": [240, 240, 120, 80, 80, 80]}    # Quảng Nam (tổng khoảng 1.050 điểm cả hai tỉnh)

def sample_nonrice(name, counts):
    geom = region.filter(ee.Filter.stringContains("ADM1_NAME", name)).geometry()
    pts = lc.stratifiedSample(numPoints=0, classBand="lc", region=geom, scale=30,
                              classValues=CLASSES, classPoints=counts,
                              seed=7, geometries=True, tileScale=8)
    return pts.map(lambda f: f.set("province", name))

pts_nr = ee.FeatureCollection([sample_nonrice(k, v) for k, v in PER_PROVINCE.items()]).flatten()
n_nr = pts_nr.size().getInfo()
lst = pts_nr.toList(n_nr)
points_nr = ee.FeatureCollection(ee.List.sequence(0, n_nr - 1).map(
    lambda i: ee.Feature(lst.get(i)).set("field_id", ee.Number(i).int().add(100000))))   # id >= 100000 để không trùng id lúa
print("Số điểm không phải lúa:", n_nr)
print(points_nr.aggregate_histogram("lc").getInfo())     # đếm theo lớp WorldCover

# 3) Trích chuỗi: hai hàm của notebook đọc biến toàn cục `points` và `n`, nên tạm thời thay rồi khôi phục
points_rice, n_rice = points, n
points, n = points_nr, n_nr
try:
    for y in YEARS:                                       # [2023, 2024, 2025]
        print("Năm", y, "- NDVI")
        extract_year(y).to_csv(f"{OUT}/raw_nonrice_{y}.csv", index=False)
        print("Năm", y, "- NDWI, VV, VH")
        extract_extra(y).to_csv(f"{OUT}/raw_extra_nonrice_{y}.csv", index=False)
finally:
    points, n = points_rice, n_rice                       # khôi phục để các cell sau không bị ảnh hưởng
print("Xong. Tải 6 file về máy: raw_nonrice_2023/2024/2025.csv và raw_extra_nonrice_2023/2024/2025.csv")
