import re
p = "build_pptx.js"; s = open(p, encoding="utf-8").read()

# helper + url table
s = s.replace('const hl = { fill: { color: "E3F0E0" }, bold: true };',
 'const hl = { fill: { color: "E3F0E0" }, bold: true };\n'
 'const lk = (t, u, o = {}) => ({ text: t, options: { hyperlink: { url: u, tooltip: u }, ...o } });\n'
 'const URL = { breizh: "https://arxiv.org/abs/1905.11893", breizhgit: "https://github.com/dl4sits/breizhcrops", gobin: "https://www.mdpi.com/2072-4292/15/8/2090", tian: "https://www.mdpi.com/1424-8220/23/13/5876", review: "https://doi.org/10.3390/s26123834", sickle: "https://arxiv.org/abs/2312.00069", gee: "https://doi.org/10.1016/j.rse.2017.06.031", s2: "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED", s1: "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD", wc: "https://esa-worldcover.org/en", wcee: "https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200" };', 1)

# clickable paper names on the baseline paper cards
a = 'tx(s, h, { x: x + 0.2, y: 1.92, w: 3.53, h: 0.45, fontSize: 18, bold: true, color: C.text2 });'
assert a in s
s = s.replace(a, 'tx(s, [lk(h, [URL.breizh, URL.gobin, URL.tian][i], { bold: true })], { x: x + 0.2, y: 1.92, w: 3.53, h: 0.45, fontSize: 18, bold: true, color: C.text2 });')

# data caption with source links
old = re.search(r'  tx\(s, "Chọn điểm từ lớp cropland WorldCover 2021.*?\n', s).group(0)
new = ('  tx(s, [{ text: "Nguồn: " }, lk("Sentinel-2 SR", URL.s2), { text: ", " }, lk("Sentinel-1 GRD", URL.s1), { text: " và " }, lk("ESA WorldCover 2021", URL.wc), { text: " qua " }, lk("Google Earth Engine", URL.gee), '
       '{ text: ". Chọn điểm từ lớp cropland, lọc ứng viên lúa bằng biên độ NDVI và NDWI, lấy mẫu phân tầng 300 điểm Đà Nẵng và 700 điểm Quảng Nam." }], { x: 0.9, y: 6.2, w: 11.5, h: 0.6, fontSize: 14, align: "center" });\n')
s = s.replace(old, new, 1)

# SICKLE header link
b = 'tx(s, h, { x: x + 1.2, y: 2.0, w: 4.6, h: 0.45, fontSize: 20, bold: true, color: C.text2 });'
assert b in s
s = s.replace(b, 'tx(s, i === 1 ? [lk(h, URL.sickle, { bold: true })] : h, { x: x + 1.2, y: 2.0, w: 4.6, h: 0.45, fontSize: 20, bold: true, color: C.text2 });')

# BreizhCrops evidence slide replaces the conditional block
i0 = s.index("  // ================= 8b Baseline papers"); i1 = s.index("  // ================= 9 Baseline results")
brz = '''  // ================= 8b BreizhCrops: evidence from the source paper (cropped from the real PDF)
  s = slideBase("Baseline", "Baseline: bằng chứng từ bài gốc", "BreizhCrops (Rußwurm và cs., 2020): Random Forest là baseline mạnh, mây xử lý bằng che và nội suy");
  s.addImage({ path: "papers/breizh_table.png", x: 0.6, y: 1.75, w: 7.3, h: 7.3 * 900 / 1591, shadow: shadow(), altText: "Bảng 2 của BreizhCrops: Random Forest so với các mạng sâu", hyperlink: { url: URL.breizh, tooltip: URL.breizh } });
  tx(s, [{ text: "Bảng 2 trong bài gốc. ", options: { bold: true } }, { text: "Random Forest (cột RF) đạt độ chính xác tổng 0,76–0,78, sát các mạng sâu (cao nhất 0,80–0,81): lý do nhóm lấy RF làm baseline B2." }], { x: 0.6, y: 5.95, w: 7.3, h: 0.8, fontSize: 15 });
  s.addImage({ path: "papers/breizh_clouds.png", x: 8.25, y: 1.75, w: 4.48, h: 4.48 * 255 / 767, shadow: shadow(), altText: "Đoạn trong BreizhCrops về nhiễu do mây" });
  tx(s, [{ text: "Mục 5 của bài gốc. ", options: { bold: true } }, { text: "Mây là nhiễu vốn có của ảnh quang học; cách thông dụng là phân loại, che và nội suy giá trị bị mây: đúng các bước tiền xử lý của nhóm." }], { x: 8.25, y: 3.4, w: 4.48, h: 1.35, fontSize: 15 });
  card(s, 8.25, 4.9, 4.48, 1.8, "sage", "breizh links");
  tx(s, "Mở nguồn (bấm để mở)", { x: 8.45, y: 4.98, w: 4.1, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  tx(s, [lk("Bài báo: arxiv.org/abs/1905.11893", URL.breizh, { breakLine: true }), lk("Mã nguồn: github.com/dl4sits/breizhcrops", URL.breizhgit)], { x: 8.45, y: 5.42, w: 4.1, h: 1.2, fontSize: 15, paraSpaceAfter: 6 });

'''
s = s[:i0] + brz + s[i1:]

# reference slides before the end slide
refs = '''  // ================= 18b/18c References (clickable)
  const refCard = (s, x, y, n, runs) => {
    card(s, x, y, 5.95, 2.4, "white", "ref " + n);
    numCircle(s, x + 0.18, y + 0.2, 0.55, "[" + n + "]", C.accent1, 13);
    tx(s, runs, { x: x + 0.9, y: y + 0.15, w: 4.9, h: 2.15, fontSize: 14.5, paraSpaceAfter: 3 });
  };
  const used = (t) => ({ text: t, options: { bold: true, color: C.text2 } });
  s = slideBase("Kết luận", "Tài liệu tham khảo (1/2): bài báo", "Bấm vào đường dẫn để mở nguồn");
  refCard(s, 0.6, 1.8, 1, [{ text: "Rußwurm, M., Pelletier, C., Zollner, M., Lefèvre, S., Körner, M. (2020). " }, { text: "BreizhCrops: A Time Series Dataset for Crop Type Mapping", options: { italic: true } }, { text: ". ISPRS Archives XLIII-B2-2020.", options: { breakLine: true } }, lk("arxiv.org/abs/1905.11893", URL.breizh, { breakLine: true }), lk("github.com/dl4sits/breizhcrops", URL.breizhgit, { breakLine: true }), used("Dùng ở: baseline B2 Random Forest, chia theo không gian, xử lý mây.")]);
  refCard(s, 6.78, 1.8, 2, [{ text: "Gobin và cs. (2023). Remote Sensing 15(8), 2090.", options: { breakLine: true } }, lk("mdpi.com/2072-4292/15/8/2090", URL.gobin, { breakLine: true }), used("Dùng ở: baseline B3, khớp double-logistic.")]);
  refCard(s, 0.6, 4.3, 3, [{ text: "Tian và cs. (2023). Sensors 23(13), 5876.", options: { breakLine: true } }, lk("mdpi.com/1424-8220/23/13/5876", URL.tian, { breakLine: true }), used("Dùng ở: baseline B4, ý tưởng độ dốc và chênh lệch NDVI.")]);
  refCard(s, 6.78, 4.3, 4, [{ text: "Tunio, M.H., Li, S., Ahmed, A., Lei, L., Liang, C. (2026). " }, { text: "Beyond the Visual Spectrum: From RGB-Based Learning to Hyperspectral Intelligence for Plant Disease Detection", options: { italic: true } }, { text: ". Sensors 26, 3834.", options: { breakLine: true } }, lk("doi.org/10.3390/s26123834", URL.review, { breakLine: true }), used("Dùng ở: tham khảo bối cảnh (học sâu 1D/3D-CNN trên dữ liệu ngoài dải nhìn thấy); không dùng cho số liệu nào.")]);

  s = slideBase("Kết luận", "Tài liệu tham khảo (2/2): dữ liệu và công cụ", "Bấm vào đường dẫn để mở nguồn");
  refCard(s, 0.6, 1.8, 5, [{ text: "Sani, D. và cs. (2023). " }, { text: "SICKLE: A Multi-Sensor Satellite Imagery Dataset Annotated with Multiple Key Cropping Parameters", options: { italic: true } }, { text: ". arXiv:2312.00069.", options: { breakLine: true } }, lk("arxiv.org/abs/2312.00069", URL.sickle, { breakLine: true }), used("Dùng ở: kiểm chứng ngoài bằng ngày gieo, thu hoạch do nông dân khai (Ấn Độ).")]);
  refCard(s, 6.78, 1.8, 6, [{ text: "Gorelick, N. và cs. (2017). " }, { text: "Google Earth Engine: Planetary-scale geospatial analysis for everyone", options: { italic: true } }, { text: ". Remote Sensing of Environment 202, 18–27.", options: { breakLine: true } }, lk("doi.org/10.1016/j.rse.2017.06.031", URL.gee, { breakLine: true }), used("Dùng ở: thu thập chuỗi NDVI, NDWI, VV, VH.")]);
  refCard(s, 0.6, 4.3, 7, [{ text: "Ảnh vệ tinh trên Earth Engine (Copernicus):", options: { breakLine: true } }, lk("Sentinel-2 SR Harmonized", URL.s2, { breakLine: true }), lk("Sentinel-1 GRD", URL.s1, { breakLine: true }), used("Dùng ở: NDVI, NDWI (Sentinel-2); VV, VH (Sentinel-1).")]);
  refCard(s, 6.78, 4.3, 8, [{ text: "ESA WorldCover 2021 (v200):", options: { breakLine: true } }, lk("esa-worldcover.org", URL.wc, { breakLine: true }), lk("Bản trên Earth Engine", URL.wcee, { breakLine: true }), used("Dùng ở: chọn điểm ruộng, cây trồng khác, rừng, cỏ, đô thị, nước.")]);

'''
e = "  // ================= 19 End"
assert e in s
s = s.replace(e, refs + e, 1)
open(p, "w", encoding="utf-8").write(s)
print("patched")
