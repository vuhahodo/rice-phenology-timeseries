// Rice phenology deck (PowerPoint). Numbers come from data.json (export_data.py); backgrounds from make_bg.py.
const SKILL = "C:/Users/Admin/.claude/skills/synced/80829326-5f07-4286-b071-bf9cb2d4e94c_844cb22d-3aff-4505-a5eb-23e4899ccf07/pptx";
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(SKILL + "/scripts/apply_theme.js");
const React = require("react"), RDS = require("react-dom/server"), sharp = require("sharp");
const fa = require("react-icons/fa");
const D = require("./data.json");

const THEME = {
  name: "Rice Phenology", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B2A22", lt1: "FFFFFF", dk2: "1F4D3A", lt2: "F4F9EE", accent1: "2F7A52", accent2: "D9A83A", accent3: "C9653F",
            accent4: "5FA86B", accent5: "5E8FA6", accent6: "8B7B5A", hlink: "2F7A52", folHlink: "5E8FA6" },
};
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "Theo dõi vòng đời cây lúa từ chuỗi vệ tinh NDVI và SAR";
pres.author = "Nhóm Time Series";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- number helpers (Vietnamese decimal comma)
const f3 = (v) => v.toFixed(3).replace(".", ","), f1 = (v) => v.toFixed(1).replace(".", ","), f2 = (v) => v.toFixed(2).replace(".", ",");
const pct = (v, d = 0) => (v * 100).toFixed(d).replace(".", ",") + "%";
const nf = (v) => String(v).replace(/\B(?=(\d{3})+(?!\d))/g, ".");

// ---------- icons
async function icon(name, color = "FFFFFF") {
  const svg = RDS.renderToStaticMarkup(React.createElement(fa[name], { color: "#" + color, size: "256" }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
const ICONS = {};
async function loadIcons() {
  for (const k of ["FaSeedling", "FaCloud", "FaTag", "FaSatelliteDish", "FaChartBar", "FaProjectDiagram", "FaCheck", "FaGlobeAsia", "FaBullseye", "FaFilter", "FaLayerGroup", "FaClipboardCheck", "FaCalendarAlt", "FaMapMarkedAlt", "FaSearch"]) {
    if (!fa[k]) throw new Error("missing icon " + k);
    ICONS[k] = await icon(k);
  }
}

// ---------- layouts (frames only; content is composed per slide)
const TITLE_FONT = { fontFace: "Cambria", bold: true };
pres.defineSlideMaster({
  title: "TITLE", background: { path: "bg_title.png" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 0.95, w: 8.6, h: 2.1, fontSize: 40, color: C.background1, margin: 0, valign: "top", align: "left", ...TITLE_FONT }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.1, w: 8.6, h: 0.6, fontSize: 20, color: C.background2, margin: 0, valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { path: "bg_content.png" },
  slideNumber: { x: 12.2, y: 7.0, w: 0.6, h: 0.3, fontSize: 12, color: C.text2, bold: true, align: "right" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 10.2, h: 0.75, fontSize: 34, color: C.text2, margin: 0, valign: "middle", align: "left", ...TITLE_FONT }, text: "" } },
    { text: { text: "Theo dõi vòng đời cây lúa từ chuỗi vệ tinh NDVI và SAR", options: { x: 0.6, y: 7.0, w: 7, h: 0.3, fontSize: 12, color: C.text2, margin: 0, isTextBox: true } } },
  ],
});

// ---------- drawing helpers
const shadow = () => ({ type: "outer", color: "1B2A22", opacity: 0.14, blur: 6, offset: 2, angle: 90 });
const tx = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, valign: "top", color: C.text1, fontSize: 16, ...o });
function card(s, x, y, w, h, kind = "white", name = "card") {
  const fills = { white: { color: C.background1 }, tint: { color: C.background2 }, dark: { color: C.text2 }, gold: { color: "FBF0DC" }, sage: { color: "E3F0E0" } };
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: fills[kind], line: { color: kind === "gold" ? "E2C48C" : "D3E2CC", width: 1 }, shadow: kind === "dark" ? undefined : shadow(), objectName: name });
}
function badge(s, x, y, d, iconName, fill = C.accent1) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: C.background1, width: 2 }, objectName: "badge " + iconName });
  const p = d * 0.27; s.addImage({ data: ICONS[iconName], x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, altText: iconName.replace("Fa", "") });
}
function numCircle(s, x, y, d, t, fill = C.accent1, size = 18) {
  s.addText(String(t), { x, y, w: d, h: d, shape: pres.shapes.OVAL, fill: { color: fill }, color: C.background1, bold: true, fontSize: size, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "num " + t });
}
function slideBase(section, title, sub) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (sub) tx(s, sub, { x: 0.6, y: 1.12, w: 11.4, h: 0.45, fontSize: 17, color: C.text2, italic: true });
  return s;
}
const hdr = (t, al = "left") => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: 14, align: al, valign: "middle" } });
const cell = (t, o = {}) => ({ text: String(t), options: { fontSize: 16, valign: "middle", border: [{ type: "none" }, { type: "none" }, { pt: 1, color: "D3E2CC" }, { type: "none" }], ...o } });
const hl = { fill: { color: "E3F0E0" }, bold: true };
const lk = (t, u, o = {}) => ({ text: t, options: { hyperlink: { url: u, tooltip: u }, ...o } });
const URL = { breizh: "https://arxiv.org/abs/1905.11893", breizhgit: "https://github.com/dl4sits/breizhcrops", gobin: "https://www.mdpi.com/2072-4292/15/8/2090", tian: "https://www.mdpi.com/1424-8220/23/13/5876", review: "https://doi.org/10.3390/s26123834", sickle: "https://arxiv.org/abs/2312.00069", gee: "https://doi.org/10.1016/j.rse.2017.06.031", s2: "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED", s1: "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD", wc: "https://esa-worldcover.org/en", wcee: "https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200" };

(async () => {
  await loadIcons();

  // ================= 1 Title
  pres.addSection({ title: "Mở đầu" });
  let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Mở đầu" });
  s.addText("Theo dõi vòng đời cây lúa từ chuỗi vệ tinh NDVI và SAR", { placeholder: "title" });
  s.addText("Đà Nẵng và Quảng Nam · Sentinel-2 và Sentinel-1 · 2023–2025", { placeholder: "body" });
  tx(s, "Văn Trọng Huy · 23AI018\nĐỗ Phúc Vũ Hà · 23AI012\nSử Thị Hà My · 23AI031", { x: 0.8, y: 3.95, w: 4.0, h: 1.0, fontSize: 16, color: C.background1, paraSpaceAfter: 2 });
  tx(s, "Nguyễn Hà Nhi · 23AI039\nTrần Thị Thùy Linh · 23AI029", { x: 5.0, y: 3.95, w: 4.2, h: 1.0, fontSize: 16, color: C.background1, paraSpaceAfter: 2 });
  tx(s, "PHÂN TÍCH CHUỖI THỜI GIAN · ĐỒ ÁN MÔN HỌC", { x: 0.8, y: 0.5, w: 8, h: 0.35, fontSize: 14, bold: true, color: C.accent2, charSpacing: 3 });

  // ================= 2 Outline
  s = slideBase("Mở đầu", "Nội dung", "Từ bối cảnh đến kế hoạch, theo thứ tự: dữ liệu, baseline, phương pháp");
  const items = ["Bối cảnh đề tài", "Mục tiêu đề tài", "Hướng làm", "Dữ liệu", "Baseline (paper nền)", "Phương pháp đề xuất", "Kết quả", "Hạn chế và kế hoạch"];
  items.forEach((t, i) => {
    const col = Math.floor(i / 4), row = i % 4, x = 0.9 + col * 6.1, y = 2.0 + row * 1.12;
    card(s, x, y, 5.6, 0.9, "white", "item " + (i + 1));
    numCircle(s, x + 0.17, y + 0.14, 0.62, String(i + 1).padStart(2, "0"), i < 3 ? C.accent1 : i < 5 ? C.accent2 : C.accent5, 16);
    tx(s, t, { x: x + 1.0, y, w: 4.4, h: 0.9, fontSize: 22, valign: "middle" });
  });

  // ================= 3 Context
  s = slideBase("Mở đầu", "Bối cảnh đề tài", "Vì sao cần theo dõi giai đoạn sinh trưởng của lúa bằng vệ tinh");
  tx(s, [
    { text: "Biết cây lúa đang ở giai đoạn nào theo từng thời điểm giúp theo dõi mùa vụ cho cả vùng, không cần đi từng thửa.", options: { bullet: true, breakLine: true } },
    { text: "Chuỗi ảnh vệ tinh theo thời gian là nguồn dữ liệu phù hợp: Sentinel-2 (quang học) và Sentinel-1 (radar).", options: { bullet: true } },
  ], { x: 0.7, y: 1.8, w: 11.8, h: 1.1, fontSize: 18, paraSpaceAfter: 6 });
  [["FaSeedling", "Lúa nhiều vụ", "Đà Nẵng và Quảng Nam có 1–2 vụ mỗi năm; mỗi vụ có 4 giai đoạn từ gieo đến thu hoạch."],
   ["FaCloud", "Ảnh quang học hay bị mây", `Trung bình ${pct(D.miss_mean)} số bước thiếu ảnh gốc; radar Sentinel-1 xuyên được mây.`],
   ["FaTag", "Thiếu nhãn ngoài đồng", "Nhóm chưa có ngày gieo, thu hoạch thật; nhãn phải suy ra từ chính ảnh vệ tinh."]].forEach(([ic, h, t], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 3.1, 3.93, 3.55, "white", "context " + (i + 1));
    badge(s, x + 1.4, 3.3, 1.1, ic, [C.accent1, C.accent5, C.accent3][i]);
    tx(s, h, { x: x + 0.25, y: 4.55, w: 3.43, h: 0.45, fontSize: 20, bold: true, color: C.text2, align: "center" });
    tx(s, t, { x: x + 0.3, y: 5.05, w: 3.33, h: 1.45, fontSize: 17 });
  });

  // ================= 4 Goals
  s = slideBase("Mở đầu", "Mục tiêu đề tài", "Bài toán gán nhãn chuỗi: mỗi bước 5 ngày thuộc một trong 4 giai đoạn");
  [["Gán 4 giai đoạn sinh trưởng cho từng bước 5 ngày của chuỗi"], ["Suy ra mốc SOS, POS, EOS (bắt đầu, đỉnh, kết thúc mùa) theo ngày"],
   ["Thử bỏ NDVI: dùng SAR và xử lý mây, vì nhãn đang sinh từ NDVI"], ["Nói rõ giới hạn của nhãn yếu, không tuyên bố hơn mức đo được"]].forEach(([t], i) => {
    const x = 0.6 + i * 3.06;
    card(s, x, 1.8, 2.9, 1.9, "tint", "goal " + (i + 1));
    numCircle(s, x + 0.2, 1.97, 0.5, i + 1, C.accent1, 16);
    tx(s, t, { x: x + 0.2, y: 2.6, w: 2.5, h: 1.05, fontSize: 16 });
  });
  card(s, 0.6, 3.95, 3.5, 2.7, "tint", "flow in");
  tx(s, "Đầu vào", { x: 0.85, y: 4.1, w: 3.0, h: 0.4, fontSize: 20, bold: true, color: C.text2 });
  tx(s, "1 cửa sổ 365 ngày\n73 giá trị, cách nhau 5 ngày", { x: 0.85, y: 4.65, w: 3.0, h: 1.8, fontSize: 19 });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 4.2, y: 5.05, w: 0.55, h: 0.5, fill: { color: C.accent2 }, line: { type: "none" } });
  card(s, 4.85, 3.95, 3.5, 2.7, "dark", "flow model");
  tx(s, "Mô hình chuỗi", { x: 5.1, y: 4.1, w: 3.0, h: 0.4, fontSize: 20, bold: true, color: C.background1 });
  tx(s, "Gán nhãn cho từng bước\n(sequence labelling)", { x: 5.1, y: 4.65, w: 3.0, h: 1.8, fontSize: 19, color: C.background1 });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 8.45, y: 5.05, w: 0.55, h: 0.5, fill: { color: C.accent2 }, line: { type: "none" } });
  card(s, 9.1, 3.95, 3.63, 2.7, "gold", "flow out");
  tx(s, "Đầu ra: 73 nhãn", { x: 9.35, y: 4.1, w: 3.2, h: 0.4, fontSize: 20, bold: true, color: C.text2 });
  [["Ngoài vụ", "8B7B5A"], ["Sinh trưởng", "2F7A52"], ["Sinh sản", "B8841F"], ["Suy giảm", "C9653F"]].forEach(([t, col], i) => {
    s.addText(t, { x: 9.35 + (i % 2) * 1.6, y: 4.7 + Math.floor(i / 2) * 0.62, w: 1.5, h: 0.5, shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, fill: { color: col }, color: C.background1, bold: true, fontSize: 15, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "chip " + t });
  });

  // ================= 5 Pipeline
  s = slideBase("Mở đầu", "Tổng quan luồng xử lý", "Năm bước từ ảnh vệ tinh đến kết quả đánh giá");
  [["FaGlobeAsia", "Thu thập", "Sentinel-2 và Sentinel-1 trên Google Earth Engine, 2023–2025, 1.000 điểm"],
   ["FaFilter", "Tiền xử lý", "Lọc mây bằng SCL, lưới 5 ngày, nội suy chỗ trống, làm mượt Savitzky–Golay"],
   ["FaTag", "Tạo nhãn", "Luật ngưỡng 20% biên độ NDVI: 4 giai đoạn và mốc SOS, POS, EOS"],
   ["FaProjectDiagram", "Mô hình", "1D-CNN gán nhãn từng bước, giải mã Viterbi giữ đúng thứ tự giai đoạn"],
   ["FaChartBar", "Đánh giá", "Macro-F1, sai số mốc theo ngày, kiểm thử theo năm, đối chiếu SICKLE"]].forEach(([ic, h, t], i) => {
    const x = 0.6 + i * 2.45;
    card(s, x, 2.15, 2.3, 3.7, "white", "step " + (i + 1));
    badge(s, x + 0.6, 1.8, 1.1, ic, [C.accent1, C.accent5, C.accent2, C.accent3, C.accent4][i]);
    numCircle(s, x + 0.12, 2.27, 0.45, i + 1, C.text2, 14);
    tx(s, h, { x: x + 0.15, y: 3.1, w: 2.0, h: 0.45, fontSize: 20, bold: true, color: C.text2, align: "center" });
    tx(s, t, { x: x + 0.18, y: 3.65, w: 1.94, h: 2.8, fontSize: 16.5 });
  });

  // ================= 6 Data
  pres.addSection({ title: "Dữ liệu" });
  s = slideBase("Dữ liệu", "Dữ liệu", "Tự thu thập từ Google Earth Engine, Đà Nẵng và Quảng Nam, 2023–2025");
  [[nf(D.n), "cửa sổ chuỗi thời gian"], [String(D.fields), "thửa ruộng"], ["73", "bước, mỗi bước 5 ngày"], ["5", "kênh: NDVI, NDWI, VV, VH, mặt nạ"], [`${D.split.train}/${D.split.val}/${D.split.test}`, "train / val / test, ô lưới 2 km"]].forEach(([b, l], i) => {
    const x = 0.6 + i * 2.45;
    card(s, x, 1.8, 2.3, 1.2, "tint", "stat " + (i + 1));
    tx(s, b, { x, y: 1.88, w: 2.3, h: 0.55, fontSize: i === 4 ? 24 : 30, bold: true, color: C.accent1, align: "center", valign: "middle" });
    tx(s, l, { x: x + 0.1, y: 2.45, w: 2.1, h: 0.5, fontSize: 14, align: "center" });
  });
  s.addImage({ path: "fig_data_overview.png", x: 1.15, y: 3.2, w: 11.0, h: 11.0 / 3.837, altText: "Thống kê bộ dữ liệu: tập chia, tỷ lệ lớp, bước thiếu ảnh, số vụ" });
  tx(s, [{ text: "Nguồn: " }, lk("Sentinel-2 SR", URL.s2), { text: ", " }, lk("Sentinel-1 GRD", URL.s1), { text: " và " }, lk("ESA WorldCover 2021", URL.wc), { text: " qua " }, lk("Google Earth Engine", URL.gee), { text: ". Chọn điểm từ lớp cropland, lọc ứng viên lúa bằng biên độ NDVI và NDWI, lấy mẫu phân tầng 300 điểm Đà Nẵng và 700 điểm Quảng Nam." }], { x: 0.9, y: 6.2, w: 11.5, h: 0.6, fontSize: 14, align: "center" });


  // ================= 6b Data: raw table (real Excel render of an excerpt of the CSV)
  s = slideBase("Dữ liệu", "Dữ liệu: bảng gốc", "Trích 14 dòng và 19 trong 155 cột của file phenology_dataset_v32csv.csv; dấu … là cột lược bỏ");
  s.addImage({ path: "data_screenshot.png", x: 0.6, y: 1.8, w: 12.13, h: 12.13 / 3.827, altText: "Bảng dữ liệu gốc: metadata, NDVI_01..73, stage_01..73" });
  [["tint", "9 cột metadata", "field_id, lon, lat, province, year, start_doy, n_seasons, miss_ratio, split."],
   ["gold", "73 cột NDVI", "NDVI_01 đến NDVI_73: chuỗi 365 ngày, mỗi bước 5 ngày (cột màu be)."],
   ["sage", "73 cột nhãn", "stage_01 đến stage_73: 0 ngoài vụ, 1 sinh trưởng, 2 sinh sản, 3 suy giảm (cột màu xanh)."]].forEach(([k, h, t], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 5.2, 3.93, 1.5, k, "col group " + (i + 1));
    tx(s, h, { x: x + 0.2, y: 5.28, w: 3.53, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
    tx(s, t, { x: x + 0.2, y: 5.72, w: 3.53, h: 0.95, fontSize: 15 });
  });

  // ================= 7 Labels
  s = slideBase("Dữ liệu", "Nhãn: vì sao cần thận trọng", "Nhãn do luật ngưỡng trên NDVI tạo ra, nên đo khả năng tái tạo luật đó");
  s.addImage({ path: "fig_examples.png", x: 0.6, y: 1.85, w: 6.9, h: 6.9 / 1.895, altText: "Ba chuỗi NDVI với dải nhãn giai đoạn bên dưới" });
  tx(s, "Dải màu dưới mỗi đường NDVI là nhãn giai đoạn của từng bước.", { x: 0.6, y: 5.6, w: 6.9, h: 0.4, fontSize: 14, align: "center" });
  [["dark", "Nhãn là hàm của NDVI", "Luật ngưỡng 20% biên độ phát hiện lại nhãn khớp 99,9%. Mô hình học NDVI vì thế là đang tái tạo luật gán nhãn."],
   ["gold", "Nhãn mờ khoảng 10 ngày", "Đổi ngưỡng 10% / 20% / 30% làm mốc SOS, EOS lệch trung vị 10 ngày; sai số dưới mức đó nằm trong độ mờ của nhãn."],
   ["sage", "Cách xử lý", "Báo cáo đúng tên là dự đoán nhãn vật hậu yếu, rồi kiểm tra ngoài bằng SAR và SICKLE."]].forEach(([k, h, t], i) => {
    const y = 1.8 + i * 1.68;
    card(s, 7.8, y, 4.93, 1.55, k, "label " + (i + 1));
    tx(s, h, { x: 8.0, y: y + 0.1, w: 4.55, h: 0.4, fontSize: 18, bold: true, color: k === "dark" ? C.background1 : C.text2 });
    tx(s, t, { x: 8.0, y: y + 0.52, w: 4.55, h: 1.0, fontSize: 14.5, color: k === "dark" ? C.background1 : C.text1 });
  });

  // ================= 8 Baseline papers
  pres.addSection({ title: "Baseline" });
  s = slideBase("Baseline", "Baseline: paper làm nền", "Nhóm lấy phương pháp từ các paper này rồi phát triển thêm");
  [["BreizhCrops (Rußwurm, 2020)", "Bộ chuẩn phân loại cây trồng từ chuỗi Sentinel-2: hơn 600.000 chuỗi, so Random Forest với CNN, RNN, Transformer, chia theo không gian.", "Nhóm lấy: Random Forest và cách chia không gian."],
   ["Gobin và cs. (2023)", "Vật hậu lúa mì, ngô, khoai tây từ Sentinel-2: khớp double-logistic lên fAPAR/fCover, điểm uốn ứng với giai đoạn.", "Nhóm lấy: khớp double-logistic."],
   ["Tian và cs. (2023)", "Trích xuất ruộng lúa bằng chênh lệch NDVI trước và sau thu hoạch, khử viền nước bằng NDWI, hồi quy ngày thu hoạch.", "Nhóm lấy: ý tưởng độ dốc, chênh lệch NDVI."]].forEach(([h, t, g], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.8, 3.93, 2.95, "tint", "paper " + (i + 1));
    tx(s, [lk(h, [URL.breizh, URL.gobin, URL.tian][i], { bold: true })], { x: x + 0.2, y: 1.92, w: 3.53, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
    tx(s, t, { x: x + 0.2, y: 2.4, w: 3.53, h: 1.6, fontSize: 14.5 });
    tx(s, g, { x: x + 0.2, y: 4.05, w: 3.53, h: 0.6, fontSize: 15, bold: true, color: C.accent1 });
  });
  card(s, 0.6, 4.95, 12.13, 0.95, "gold", "paper note");
  tx(s, "Các paper khác vùng, khác cây và (trừ BreizhCrops) dùng ground truth quan trắc. Nhãn của nhóm sinh từ NDVI nên không so F1 trực tiếp; nhóm cài lại phương pháp của họ trên cùng dữ liệu và cùng metric.", { x: 0.85, y: 5.0, w: 11.6, h: 0.85, fontSize: 15.5, valign: "middle" });
  [["B0", "Lịch mùa vụ"], ["B1", "Luật ngưỡng 20%"], ["B2", "Random Forest"], ["B3", "Double-logistic"], ["B4", "Độ dốc NDVI"]].forEach(([b, t], i) => {
    s.addText([{ text: b + "  ", options: { bold: true, color: C.accent3 } }, { text: t, options: { color: C.text1 } }], { x: 0.6 + i * 2.45, y: 6.1, w: 2.3, h: 0.55, shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, fill: { color: C.background1 }, line: { color: "D3E2CC", width: 1 }, fontSize: 15, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "base " + b });
  });


  // ================= 8b BreizhCrops: evidence from the source paper (cropped from the real PDF)
  s = slideBase("Baseline", "Baseline: bằng chứng từ bài gốc", "BreizhCrops (Rußwurm và cs., 2020): Random Forest là baseline mạnh, mây xử lý bằng che và nội suy");
  s.addImage({ path: "papers/breizh_table.png", x: 0.6, y: 1.75, w: 7.3, h: 7.3 * 900 / 1591, shadow: shadow(), altText: "Bảng 2 của BreizhCrops: Random Forest so với các mạng sâu", hyperlink: { url: URL.breizh, tooltip: URL.breizh } });
  tx(s, [{ text: "Bảng 2 trong bài gốc. ", options: { bold: true } }, { text: "Random Forest (cột RF) đạt độ chính xác tổng 0,76–0,78, sát các mạng sâu (cao nhất 0,80–0,81): lý do nhóm lấy RF làm baseline B2." }], { x: 0.6, y: 5.95, w: 7.3, h: 0.8, fontSize: 15 });
  s.addImage({ path: "papers/breizh_clouds.png", x: 8.25, y: 1.75, w: 4.48, h: 4.48 * 255 / 767, shadow: shadow(), altText: "Đoạn trong BreizhCrops về nhiễu do mây" });
  tx(s, [{ text: "Mục 5 của bài gốc. ", options: { bold: true } }, { text: "Mây là nhiễu vốn có của ảnh quang học; cách thông dụng là phân loại, che và nội suy giá trị bị mây: đúng các bước tiền xử lý của nhóm." }], { x: 8.25, y: 3.4, w: 4.48, h: 1.35, fontSize: 15 });
  card(s, 8.25, 4.9, 4.48, 1.8, "sage", "breizh links");
  tx(s, "Mở nguồn (bấm để mở)", { x: 8.45, y: 4.98, w: 4.1, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  tx(s, [lk("Bài báo: arxiv.org/abs/1905.11893", URL.breizh, { breakLine: true }), lk("Mã nguồn: github.com/dl4sits/breizhcrops", URL.breizhgit)], { x: 8.45, y: 5.42, w: 4.1, h: 1.2, fontSize: 15, paraSpaceAfter: 6 });

  // ================= 8c Tian et al. 2023: evidence from the source paper (cropped from the real PDF)
  s = slideBase("Baseline", "Baseline: bằng chứng từ bài gốc (B4)", "Tian và cs. (2023): NDVI lúa tách khỏi các lớp phủ khác và sụt mạnh sau thu hoạch — ý tưởng của B4 (slope/difference)");
  s.addImage({ path: "papers/tian_fig2_ndvi.png", x: 0.6, y: 1.75, w: 7.3, h: 7.3 * 1184 / 1800, shadow: shadow(), altText: "Hình 2 của Tian và cs.: đường cong NDVI theo thời gian của các loại lớp phủ", hyperlink: { url: URL.tian, tooltip: URL.tian } });
  tx(s, [{ text: "Hình 2 trong bài gốc. ", options: { bold: true } }, { text: "Đường NDVI ruộng lúa (vàng) tách khỏi rừng/đất khô/nước, và sụt mạnh ở lần đo sau cùng trước vụ kế tiếp — đúng dấu hiệu thu hoạch mà B4 dùng." }], { x: 0.6, y: 6.1, w: 7.3, h: 0.65, fontSize: 15 });
  s.addImage({ path: "papers/tian_fig3_diff.png", x: 8.25, y: 1.75, w: 4.48, h: 4.48 * 1040 / 2080, shadow: shadow(), altText: "Hình 3 của Tian và cs.: bản đồ NDVI trước và sau thu hoạch lúa" });
  tx(s, [{ text: "Hình 3 trong bài gốc. ", options: { bold: true } }, { text: "So khớp ảnh NDVI trước (a) và sau (b) vụ thu hoạch: vùng lúa đổi màu rõ rệt, là cơ sở cho phương pháp chênh lệch NDVI." }], { x: 8.25, y: 3.98, w: 4.48, h: 1.1, fontSize: 15 });
  card(s, 8.25, 5.25, 4.48, 1.45, "sage", "tian links");
  tx(s, "Mở nguồn (bấm để mở)", { x: 8.45, y: 5.33, w: 4.1, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  tx(s, [lk("mdpi.com/1424-8220/23/13/5876", URL.tian)], { x: 8.45, y: 5.75, w: 4.1, h: 0.8, fontSize: 15 });

  // ================= 9 Baseline results
  s = slideBase("Baseline", "Kết quả baseline", `${D.split.test} cửa sổ test; cấu hình chọn trên validation, báo cáo test sau khi khóa`);
  const bRow = (lbl, d, hlRow) => [cell(lbl, hlRow ? hl : {}), cell(f3(d.f1), { align: "right", ...(hlRow ? hl : {}) }), cell(`${f1(d.sos)} / ${f1(d.eos)}`, { align: "right", ...(hlRow ? hl : {}) }),
    cell(`${pct(d.miss_sos)} / ${pct(d.miss_eos)}`, { align: "right", ...(hlRow ? hl : {}) }), cell(pct(d.retro), { align: "right", ...(hlRow ? hl : {}) })];
  s.addTable([[hdr("Baseline"), hdr("Macro-F1", "right"), hdr("MAE SOS / EOS (ngày)", "right"), hdr("Bỏ sót SOS / EOS", "right"), hdr("Sai thứ tự", "right")],
    bRow("B0 · Lịch mùa vụ (không dùng NDVI)", D.B0), bRow("B1 · Luật ngưỡng 20%", D.B1), bRow("B2 · Random Forest", D.B2, true), bRow("B3 · Double-logistic", D.B3), bRow("B4 · Độ dốc NDVI", D.B4)],
    { x: 0.6, y: 1.8, w: 12.13, colW: [4.4, 1.7, 2.4, 2.1, 1.53], rowH: 0.5 });
  [["Sàn", `Lịch mùa vụ đã đạt ${f3(D.B0.f1)} mà không nhìn NDVI; khoảng cách thật của mô hình tính từ đây.`],
   ["Nhóm dùng NDVI", `Bốn baseline từ ${f3(D.B4.f1)} đến ${f3(D.B2.f1)}, khá sát nhau.`],
   ["Sai thứ tự", `Random Forest sai thứ tự giai đoạn ở ${pct(D.B2.retro)} cửa sổ vì đoán từng bước độc lập.`]].forEach(([h, t], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 4.95, 3.93, 1.75, "sage", "bres " + (i + 1));
    tx(s, h, { x: x + 0.2, y: 5.05, w: 3.53, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
    tx(s, t, { x: x + 0.2, y: 5.5, w: 3.53, h: 1.1, fontSize: 15.5 });
  });

  // ================= 10 From baseline to proposal
  s = slideBase("Baseline", "Từ baseline đến phương pháp đề xuất", "Mỗi hướng làm và điểm yếu của nó, rồi hướng nhóm chọn");
  const missLo = Math.min(D.B1.miss_sos, D.B4.miss_sos), missHi = Math.max(D.B1.miss_eos, D.B4.miss_eos);
  [["sage", "Baseline luật (B1, B4)", "Cách làm: tìm đỉnh, lấy điểm cắt hoặc điểm dốc nhất.", `Hạn chế: bỏ sót ${pct(missLo)}–${pct(Math.max(missHi, D.B4.miss_eos))} mốc ở hai mép cửa sổ; B4 còn sai thứ tự ${pct(D.B4.retro)}.`],
   ["sage", "Baseline học máy (B2)", "Cách làm: Random Forest dự đoán từng bước từ 73 giá trị NDVI.", `Hạn chế: các bước đoán độc lập, ${pct(D.B2.retro)} cửa sổ sai thứ tự; không dùng ngữ cảnh chuỗi.`],
   ["dark", "Đề xuất: 1D-CNN + Viterbi", "Cách làm: mạng tích chập giãn đọc cả chuỗi, Viterbi giữ thứ tự hợp lệ.", `Điểm mới: học cả trường hợp ở mép cửa sổ; 0% sai thứ tự; ít tham số (${nf(D.cnn.params)}); thay được NDVI bằng SAR.`]].forEach(([k, h, a, b], i) => {
    const x = 0.6 + i * 4.1, fg = k === "dark" ? C.background1 : C.text1;
    card(s, x, 1.8, 3.93, 3.3, k, "prop " + (i + 1));
    tx(s, h, { x: x + 0.2, y: 1.95, w: 3.53, h: 0.45, fontSize: 19, bold: true, color: k === "dark" ? C.background1 : C.text2 });
    tx(s, a, { x: x + 0.2, y: 2.5, w: 3.53, h: 1.1, fontSize: 15.5, color: fg });
    tx(s, b, { x: x + 0.2, y: 3.7, w: 3.53, h: 1.3, fontSize: 15.5, color: fg });
  });
  card(s, 0.6, 5.35, 12.13, 1.3, "gold", "prop note");
  tx(s, "Lý do chọn hướng này: nhãn là chuỗi giai đoạn có thứ tự, nên bài toán là gán nhãn chuỗi. Luật chỉ nhìn đỉnh, Random Forest không có ràng buộc thứ tự; mạng chuỗi kèm giải mã Viterbi xử lý được cả hai.", { x: 0.85, y: 5.45, w: 11.6, h: 1.1, fontSize: 17, valign: "middle" });

  // ================= 11 Method
  pres.addSection({ title: "Phương pháp và kết quả" });
  s = slideBase("Phương pháp và kết quả", "Phương pháp: 1D-CNN và Viterbi", "Học xác suất từng giai đoạn, rồi giải mã chuỗi hợp lệ");
  [["tint", "Đầu vào", "NDVI, hoặc VV, VH, NDWI + sin, cos của ngày; 73 bước"], ["dark", "1D-CNN", "3 khối tích chập giãn, kernel 5, kênh 32-64-64"], ["tint", "Phân loại", "Xác suất 4 giai đoạn cho mỗi bước"], ["gold", "Viterbi", "Chỉ giữ nguyên hoặc tiến một giai đoạn"]].forEach(([k, h, t], i) => {
    const x = 0.6 + i * 3.1, fg = k === "dark" ? C.background1 : C.text1;
    card(s, x, 1.8, 2.75, 1.5, k, "m " + h);
    tx(s, h, { x: x + 0.15, y: 1.88, w: 2.45, h: 0.4, fontSize: 18, bold: true, color: k === "dark" ? C.background1 : C.text2, align: "center" });
    tx(s, t, { x: x + 0.15, y: 2.3, w: 2.45, h: 0.95, fontSize: 14.5, color: fg, align: "center" });
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.78, y: 2.38, w: 0.3, h: 0.34, fill: { color: C.accent2 }, line: { type: "none" } });
  });
  card(s, 0.6, 3.55, 5.6, 3.1, "white", "state diagram");
  tx(s, "Thứ tự giai đoạn hợp lệ", { x: 0.85, y: 3.65, w: 5.1, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
  [["Ngoài\nvụ", "8B7B5A"], ["Sinh\ntrưởng", "2F7A52"], ["Sinh\nsản", "B8841F"], ["Suy\ngiảm", "C9653F"]].forEach(([t, col], i) => {
    const x = 0.8 + i * 1.3;
    s.addText(t, { x, y: 4.3, w: 1.0, h: 1.0, shape: pres.shapes.OVAL, fill: { color: col }, color: C.background1, bold: true, fontSize: 13, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "state " + i });
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 1.02, y: 4.65, w: 0.26, h: 0.3, fill: { color: C.text2 }, line: { type: "none" } });
  });
  s.addShape(pres.shapes.LINE, { x: 1.3, y: 5.65, w: 3.9, h: 0, line: { color: "8B7B5A", width: 2, dashType: "dash", endArrowType: "none" } });
  tx(s, "Sau Suy giảm quay lại Ngoài vụ (vòng một chiều, không đi lùi)", { x: 0.85, y: 5.8, w: 5.1, h: 0.7, fontSize: 14.5 });
  card(s, 6.5, 3.55, 6.23, 1.5, "gold", "why viterbi");
  tx(s, "Vì sao cần Viterbi", { x: 6.75, y: 3.63, w: 5.7, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  tx(s, `Dự đoán thô sai thứ tự giai đoạn ở ${pct(D.retro_min)}–${pct(D.retro_max)} cửa sổ; Viterbi đưa về 0%, không thêm tham số.`, { x: 6.75, y: 4.08, w: 5.7, h: 0.9, fontSize: 16 });
  card(s, 6.5, 5.2, 6.23, 1.45, "sage", "predict");
  tx(s, "Dự đoán gì", { x: 6.75, y: 5.28, w: 5.7, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  tx(s, "Mặt nạ 4 giai đoạn cho từng bước 5 ngày, mốc SOS, POS, EOS theo ngày, và nhận diện trực tuyến chỉ từ quá khứ.", { x: 6.75, y: 5.72, w: 5.7, h: 0.9, fontSize: 16 });

  // ================= 12 Main results (native chart)
  s = slideBase("Phương pháp và kết quả", "Kết quả chính", "Macro-F1 trên tập test; các mô hình sâu lấy trung bình 5 seed, sau giải mã Viterbi");
  const labels = ["Lịch mùa vụ", "Luật ngưỡng", "Random Forest", "1D-CNN", "BiLSTM", "Transformer (NDVI)", "Transformer (+DOY)"];
  const vals = [D.B0.f1, D.B1.f1, D.B2.f1, D.cnn.f1[0], D.lstm.f1[0], D.trn.f1[0], D.trd.f1[0]];
  s.addChart(pres.charts.BAR, [{ name: "Macro-F1", labels, values: vals }], {
    x: 0.6, y: 1.75, w: 12.13, h: 3.3, barDir: "col", chartColors: ["B8AD93", "B8AD93", "8FB39A", "1F4D3A", "5FA86B", "5FA86B", "5FA86B"], barGapWidthPct: 60,
    showValue: true, dataLabelFormatCode: "0.000", dataLabelPosition: "outEnd", dataLabelFontSize: 14, dataLabelColor: "1B2A22", dataLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 14, catAxisLabelColor: "1B2A22", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontSize: 12, valAxisLabelColor: "4A5F54", valAxisLabelFontFace: "+mn-lt",
    valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0", valGridLine: { color: "D3E2CC", size: 0.75 }, catGridLine: { style: "none" }, showLegend: false, showTitle: false,
  });
  [["1D-CNN + Viterbi", f3(D.cnn.f1[0]), `± ${f3(D.cnn.f1[1])} (5 seed)`, "dark"], ["Transformer, chỉ NDVI", f3(D.trn.f1[0]), "mô hình sâu thứ hai", "white"], ["Random Forest (B2)", f3(D.B2.f1), "baseline học máy", "white"], ["Lịch mùa vụ (B0)", f3(D.B0.f1), "sàn, không dùng NDVI", "white"]].forEach(([h, b, l, k], i) => {
    const x = 0.6 + i * 3.06, fg = k === "dark" ? C.background1 : C.text1;
    card(s, x, 5.2, 2.9, 1.5, k, "kpi " + (i + 1));
    tx(s, b, { x, y: 5.25, w: 2.9, h: 0.65, fontSize: 32, bold: true, color: k === "dark" ? C.accent2 : C.accent1, align: "center", valign: "middle" });
    tx(s, h, { x: x + 0.1, y: 5.9, w: 2.7, h: 0.35, fontSize: 15, bold: true, color: fg, align: "center" });
    tx(s, l, { x: x + 0.1, y: 6.25, w: 2.7, h: 0.35, fontSize: 13, color: fg, align: "center" });
  });

  // ================= 13 SAR input
  s = slideBase("Phương pháp và kết quả", "Bỏ NDVI: dùng đầu vào SAR", "1D-CNN, 5 seed; nhãn vẫn sinh từ NDVI");
  const sRow = (lbl, d, hlRow) => [cell(lbl, hlRow ? hl : {}), cell(f3(d.f1), { align: "right", ...(hlRow ? hl : {}) }), cell(`${f1(d.sos)} / ${f1(d.eos)}`, { align: "right", ...(hlRow ? hl : {}) })];
  s.addTable([[hdr("Đầu vào"), hdr("Macro-F1", "right"), hdr("MAE SOS / EOS (ngày)", "right")], sRow("NDVI (đối chứng)", D.sar.ndvi), sRow("Chỉ SAR: VV, VH, VH−VV", D.sar.sar, true), sRow("SAR + NDWI", D.sar.sarndwi), sRow("NDVI + NDWI + SAR", D.sar.all)],
    { x: 0.6, y: 1.8, w: 12.13, colW: [6.2, 2.7, 3.23], rowH: 0.5 });
  card(s, 0.6, 4.5, 5.95, 2.15, "sage", "sar read");
  tx(s, "Đọc kết quả", { x: 0.85, y: 4.6, w: 5.5, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  tx(s, `Chỉ SAR đạt ${f3(D.sar.sar.f1)}, thấp hơn NDVI ${f3(D.sar.ndvi.f1)} khoảng ${f1((D.sar.ndvi.f1 - D.sar.sar.f1) * 100)} điểm; thêm NDWI lên ${f3(D.sar.sarndwi.f1)}. Thông tin về giai đoạn có trong SAR nhưng chưa bằng NDVI.`, { x: 0.85, y: 5.05, w: 5.5, h: 1.55, fontSize: 17.5 });
  card(s, 6.78, 4.5, 5.95, 2.15, "gold", "sar limit");
  tx(s, "Chưa kết luận được", { x: 7.03, y: 4.6, w: 5.5, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  tx(s, "Nhãn sinh từ NDVI nên chấm điểm có lợi cho NDVI. Điểm SAR thấp chưa có nghĩa SAR kém trên vật hậu thật; chỉ ground truth ngoài đồng mới trả lời.", { x: 7.03, y: 5.05, w: 5.5, h: 1.55, fontSize: 17.5 });

  // ================= 14 SAR and cloud (native line chart)
  s = slideBase("Phương pháp và kết quả", "SAR và mây", "Che thêm khối 30 ngày quan sát quang học thật; SAR không bị che");
  const lv = D.cloud.levels.map((p) => Math.round(p * 100) + "%");
  s.addChart(pres.charts.LINE, [
    { name: "NDVI", labels: lv, values: D.cloud.ndvi }, { name: "NDVI, huấn luyện che mây", labels: lv, values: D.cloud.ndvi_aug },
    { name: "Chỉ SAR", labels: lv, values: D.cloud.sar }, { name: "SAR + NDWI", labels: lv, values: D.cloud.sarndwi }], {
    x: 0.6, y: 1.75, w: 7.4, h: 4.95, chartColors: ["1F4D3A", "5FA86B", "C9653F", "8B7B5A"], lineSize: 3, lineDataSymbolSize: 9, 
    showLegend: true, legendPos: "b", legendFontSize: 14, legendFontFace: "+mn-lt", legendColor: "1B2A22",
    catAxisLabelFontSize: 14, catAxisLabelColor: "1B2A22", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontSize: 13, valAxisLabelColor: "4A5F54", valAxisLabelFontFace: "+mn-lt",
    valAxisMinVal: 0.6, valAxisMaxVal: 0.92, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00", valGridLine: { color: "D3E2CC", size: 0.75 }, catGridLine: { style: "none" },
    ...(process.env.AT === "0" ? {} : { showCatAxisTitle: true, catAxisTitle: "% bước quang học bị mất thêm", catAxisTitleFontSize: 13, catAxisTitleColor: "4A5F54", catAxisTitleFontFace: "+mn-lt", showValAxisTitle: true, valAxisTitle: "Macro-F1 (test)", valAxisTitleFontSize: 13, valAxisTitleColor: "4A5F54", valAxisTitleFontFace: "+mn-lt" }),
  });
  const cross = 50 + 25 * (D.cloud.ndvi[2] - D.cloud.sar[0]) / (D.cloud.ndvi[2] - D.cloud.ndvi[3]);
  [["sage", "SAR giữ nguyên, NDVI giảm", `NDVI từ ${f3(D.cloud.ndvi[0])} xuống ${f3(D.cloud.ndvi[3])} khi mất 75% bước; chỉ SAR đứng yên ở ${f3(D.cloud.sar[0])} và chưa vượt NDVI cho đến khi mất khoảng ${Math.round(cross)}%.`, 1.75, 1.6],
   ["dark", "Cách xử lý: che mây khi huấn luyện", `NDVI + che mây ngẫu nhiên: ${f3(D.cloud.ndvi_aug[3])} ở 75%, hơn chỉ SAR; đổi lại kém ${f1((D.cloud.ndvi[0] - D.cloud.ndvi_aug[0]) * 100)} điểm khi trời quang.`, 3.55, 1.75],
   ["gold", "Giới hạn", "NDWI cũng là quang học nên bị mây như NDVI. Nhãn sinh từ NDVI lúc quang đãng nên phép thử thiên về NDVI.", 5.45, 1.3]].forEach(([k, h, t, y, hh]) => {
    card(s, 8.25, y, 4.48, hh, k, "cloud " + h);
    tx(s, h, { x: 8.45, y: y + 0.08, w: 4.1, h: 0.4, fontSize: 17, bold: true, color: k === "dark" ? C.background1 : C.text2 });
    tx(s, t, { x: 8.45, y: y + 0.5, w: 4.1, h: hh - 0.55, fontSize: 14.5, color: k === "dark" ? C.background1 : C.text1 });
  });

  // ================= 15 Task A
  s = slideBase("Phương pháp và kết quả", "Lúa so với cây khác", `Earth Engine, ${nf(1828)} cửa sổ lúa và ${nf(2058)} cửa sổ khác; chia 5 phần theo ô lưới 2 km`);
  const aRow = (lbl, rf, cn, w, hlRow) => [cell(lbl, hlRow ? hl : {}), cell(f3(rf), { align: "right", ...(hlRow ? hl : {}) }), cell(f3(cn), { align: "right", ...(hlRow ? hl : {}) }), cell(pct(w), { align: "right", ...(hlRow ? hl : {}) })];
  s.addTable([[hdr("Đầu vào"), hdr("Macro-F1 · Random Forest", "right"), hdr("Macro-F1 · 1D-CNN", "right"), hdr("Nhận đúng cây trồng khác (CNN)", "right")],
    aRow("NDVI + NDWI", D.taskA.opt_rf, D.taskA.opt_cnn, D.taskA.w_opt), aRow("Chỉ SAR (VV, VH)", D.taskA.sar_rf, D.taskA.sar_cnn, D.taskA.w_sar, true), aRow("Cả bốn kênh", D.taskA.all_rf, D.taskA.all_cnn, D.taskA.w_all)],
    { x: 0.6, y: 1.8, w: 12.13, colW: [3.6, 2.9, 2.5, 3.13], rowH: 0.52 });
  card(s, 0.6, 4.2, 5.95, 2.45, "gold", "taskA caution");
  tx(s, "Đừng dùng số quang học để so với SAR", { x: 0.85, y: 4.3, w: 5.5, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  tx(s, "Điểm lúa được chọn bằng luật trên NDVI và NDWI, nên tách chúng bằng NDVI gần như đã biết trước đáp án.", { x: 0.85, y: 4.8, w: 5.5, h: 1.7, fontSize: 18.5 });
  card(s, 6.78, 4.2, 5.95, 2.45, "sage", "taskA fair");
  tx(s, "Chỉ SAR là phép thử công bằng", { x: 7.03, y: 4.3, w: 5.5, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  tx(s, `Đạt ${f3(D.taskA.sar_cnn)}, tách rừng, đô thị, nước gần hoàn toàn nhưng chỉ nhận đúng ${pct(D.taskA.w_sar)} cây trồng khác: đây là phần khó thật của bài toán.`, { x: 7.03, y: 4.8, w: 5.5, h: 1.7, fontSize: 18.5 });

  // ================= 16 External checks
  s = slideBase("Phương pháp và kết quả", "Kiểm chứng ngoài", "Theo năm và với ngày thật của nông dân (SICKLE, Ấn Độ)");
  [["FaCalendarAlt", "Kiểm thử theo năm", "Huấn luyện 2023–2024, kiểm thử 2025 (70 cửa sổ).", [[f3(D.year.ndvi), "NDVI"], [f3(D.year.sarndwi), "SAR + NDWI"]], "Kết quả giữ vững qua năm; NDVI hơn SAR + NDWI một khoảng có thật trên nhãn dẫn xuất từ NDVI."],
   ["FaSatelliteDish", "SICKLE: ngày thu hoạch thật", "Sentinel-1 dự đoán ngày thu hoạch, 5 phần theo thửa.", [[f1(D.sickle.base), "MAE lịch vụ (ngày)"], [f1(D.sickle.s1), "MAE Sentinel-1 (ngày)"]], "Trên các dòng có thông tin thật, SAR chưa hơn lịch mùa vụ (cải thiện gần 0, CI chứa 0)."]].forEach(([ic, h, sub, nums, foot], i) => {
    const x = 0.6 + i * 6.18;
    card(s, x, 1.8, 5.95, 3.6, "white", "ext " + (i + 1));
    badge(s, x + 0.25, 1.95, 0.8, ic, i ? C.accent3 : C.accent1);
    tx(s, i === 1 ? [lk(h, URL.sickle, { bold: true })] : h, { x: x + 1.2, y: 2.0, w: 4.6, h: 0.45, fontSize: 20, bold: true, color: C.text2 });
    tx(s, sub, { x: x + 1.2, y: 2.5, w: 4.6, h: 0.4, fontSize: 15 });
    nums.forEach(([b, l], j) => {
      tx(s, b, { x: x + 0.4 + j * 2.7, y: 3.05, w: 2.5, h: 0.75, fontSize: 40, bold: true, color: C.accent1, valign: "middle" });
      tx(s, l, { x: x + 0.4 + j * 2.7, y: 3.8, w: 2.5, h: 0.35, fontSize: 14 });
    });
    tx(s, foot, { x: x + 0.4, y: 4.35, w: 5.2, h: 0.95, fontSize: 15.5 });
  });
  card(s, 0.6, 5.65, 12.13, 1.0, "gold", "ext note");
  tx(s, "Vì sao tách dòng có thông tin: khoảng 44% dòng của SICKLE rơi đúng ngày mặc định ở mép mùa vụ, nên so trên mọi dòng sẽ thổi phồng lợi ích của SAR.", { x: 0.85, y: 5.7, w: 11.6, h: 0.9, fontSize: 16.5, valign: "middle" });

  // ================= 17 Outputs
  pres.addSection({ title: "Kết luận" });
  s = slideBase("Kết luận", "Đầu ra và giải thích", "Những gì phương pháp cho ra, và tên đúng của kết quả");
  [["Mặt nạ 4 giai đoạn cho từng bước 5 ngày, luôn đúng thứ tự."], ["Mốc SOS, POS, EOS theo ngày, sai số vài ngày, nằm trong độ mờ của nhãn."], [`Nhận diện trực tuyến khi chỉ có quá khứ: LSTM nhân quả ${f3(D.causal.lstm)} so với BiLSTM ${f3(D.causal.bilstm)}.`]].forEach(([t], i) => {
    const y = 1.9 + i * 1.6;
    badge(s, 0.6, y + 0.17, 0.85, "FaCheck", C.accent1);
    card(s, 1.65, y, 6.4, 1.2, "white", "out " + (i + 1));
    tx(s, t, { x: 1.9, y, w: 5.9, h: 1.2, fontSize: 18, valign: "middle" });
  });
  card(s, 8.4, 1.9, 4.33, 4.4, "dark", "out name");
  tx(s, "Gọi đúng tên kết quả", { x: 8.7, y: 2.1, w: 3.7, h: 0.5, fontSize: 20, bold: true, color: C.accent2 });
  tx(s, "Dự đoán nhãn vật hậu yếu, suy ra từ NDVI, bằng chuỗi vệ tinh NDVI và SAR.", { x: 8.7, y: 2.75, w: 3.7, h: 1.8, fontSize: 22, bold: true, color: C.background1 });
  tx(s, "Chưa phải độ chính xác so với ngày gieo, thu hoạch ngoài đồng.", { x: 8.7, y: 4.75, w: 3.7, h: 1.2, fontSize: 16, color: C.background2 });

  // ================= 18 Plan
  s = slideBase("Kết luận", "Kế hoạch thực hiện tiếp", "Việc quan trọng nhất còn lại là ground truth ngoài đồng");
  const plan = [["Ground truth ngoài đồng", "Ngày sạ, thu hoạch thật của 100–150 thửa (hợp tác xã, Sở Nông nghiệp)", "FaMapMarkedAlt"],
    ["Nhãn mắt", "150 mẫu, 3 người chấm, tính kappa trên ảnh Google Earth", "FaSearch"],
    ["Mặt nạ quan sát và dự báo sớm", "Thêm kênh mặt nạ quan sát làm đầu vào; dự báo sớm EOS từ k bước đầu mùa", "FaLayerGroup"],
    ["Điểm lúa và cây khác xác nhận", "Xác nhận ngoài đồng; chia theo huyện để kiểm tra khái quát hóa", "FaBullseye"],
    ["Hoàn thiện", "Đánh giá lại, hoàn thiện báo cáo", "FaClipboardCheck"]];
  s.addShape(pres.shapes.LINE, { x: 1.15, y: 2.2, w: 0, h: 3.35, line: { color: "8FB39A", width: 2.5 } });
  plan.forEach(([h, t, ic], i) => {
    const y = 1.85 + i * 0.84;
    badge(s, 0.75, y, 0.8, ic, [C.accent1, C.accent5, C.accent2, C.accent3, C.text2][i]);
    tx(s, `Giai đoạn ${i + 1}`, { x: 1.8, y: y + 0.02, w: 2.0, h: 0.75, fontSize: 18, bold: true, color: C.accent1, valign: "middle" });
    tx(s, [{ text: h + ": ", options: { bold: true } }, { text: t }], { x: 3.8, y, w: 8.9, h: 0.8, fontSize: 17, valign: "middle" });
  });
  card(s, 0.6, 6.1, 12.13, 0.65, "gold", "plan note");
  tx(s, "Chưa có ground truth ngoài đồng thì các kết quả trên là dự đoán nhãn yếu.", { x: 0.85, y: 6.1, w: 11.6, h: 0.65, fontSize: 17, valign: "middle", bold: true });

  // ================= 18b/18c References (clickable)
  const refCard = (s, x, y, n, runs) => {
    card(s, x, y, 5.95, 2.4, "white", "ref " + n);
    numCircle(s, x + 0.18, y + 0.2, 0.55, "[" + n + "]", C.accent1, 13);
    tx(s, runs, { x: x + 0.9, y: y + 0.15, w: 4.9, h: 2.15, fontSize: 14.5, paraSpaceAfter: 3 });
  };
  const used = (t) => ({ text: t, options: { bold: true, color: C.text2 } });
  s = slideBase("Kết luận", "Tài liệu tham khảo (1/2): bài báo", "Bấm vào đường dẫn để mở nguồn");
  refCard(s, 0.6, 1.8, 1, [{ text: "Rußwurm, M., Pelletier, C., Zollner, M., Lefèvre, S., Körner, M. (2020). " }, { text: "BreizhCrops: A Time Series Dataset for Crop Type Mapping", options: { italic: true } }, { text: ". ISPRS Archives XLIII-B2-2020.", options: { breakLine: true } }, lk("arxiv.org/abs/1905.11893", URL.breizh, { breakLine: true }), lk("github.com/dl4sits/breizhcrops", URL.breizhgit, { breakLine: true }), used("Dùng ở: baseline B2 Random Forest, chia theo không gian, xử lý mây.")]);
  refCard(s, 6.78, 1.8, 2, [{ text: "Gobin và cs. (2023). " }, { text: "Crop Phenology Modelling Using Proximal and Satellite Sensor Data", options: { italic: true } }, { text: ". Remote Sensing 15(8), 2090.", options: { breakLine: true } }, lk("mdpi.com/2072-4292/15/8/2090", URL.gobin, { breakLine: true }), used("Dùng ở: baseline B3, khớp double-logistic.")]);
  refCard(s, 0.6, 4.3, 3, [{ text: "Tian, J., Tian, Y., Cao, Y., Wan, W., Liu, K. (2023). " }, { text: "Research on Rice Fields Extraction by NDVI Difference Method Based on Sentinel Data", options: { italic: true } }, { text: ". Sensors 23(13), 5876.", options: { breakLine: true } }, lk("mdpi.com/1424-8220/23/13/5876", URL.tian, { breakLine: true }), used("Dùng ở: baseline B4, ý tưởng độ dốc và chênh lệch NDVI.")]);
  refCard(s, 6.78, 4.3, 4, [{ text: "Tunio, M.H., Li, S., Ahmed, A., Lei, L., Liang, C. (2026). " }, { text: "Beyond the Visual Spectrum: From RGB-Based Learning to Hyperspectral Intelligence for Plant Disease Detection", options: { italic: true } }, { text: ". Sensors 26, 3834.", options: { breakLine: true } }, lk("doi.org/10.3390/s26123834", URL.review, { breakLine: true }), used("Dùng ở: tham khảo bối cảnh (học sâu 1D/3D-CNN trên dữ liệu ngoài dải nhìn thấy); không dùng cho số liệu nào.")]);

  s = slideBase("Kết luận", "Tài liệu tham khảo (2/2): dữ liệu và công cụ", "Bấm vào đường dẫn để mở nguồn");
  refCard(s, 0.6, 1.8, 5, [{ text: "Sani, D. và cs. (2023). " }, { text: "SICKLE: A Multi-Sensor Satellite Imagery Dataset Annotated with Multiple Key Cropping Parameters", options: { italic: true } }, { text: ". arXiv:2312.00069.", options: { breakLine: true } }, lk("arxiv.org/abs/2312.00069", URL.sickle, { breakLine: true }), used("Dùng ở: kiểm chứng ngoài bằng ngày gieo, thu hoạch do nông dân khai (Ấn Độ).")]);
  refCard(s, 6.78, 1.8, 6, [{ text: "Gorelick, N. và cs. (2017). " }, { text: "Google Earth Engine: Planetary-scale geospatial analysis for everyone", options: { italic: true } }, { text: ". Remote Sensing of Environment 202, 18–27.", options: { breakLine: true } }, lk("doi.org/10.1016/j.rse.2017.06.031", URL.gee, { breakLine: true }), used("Dùng ở: thu thập chuỗi NDVI, NDWI, VV, VH.")]);
  refCard(s, 0.6, 4.3, 7, [{ text: "Ảnh vệ tinh trên Earth Engine (Copernicus):", options: { breakLine: true } }, lk("Sentinel-2 SR Harmonized", URL.s2, { breakLine: true }), lk("Sentinel-1 GRD", URL.s1, { breakLine: true }), used("Dùng ở: NDVI, NDWI (Sentinel-2); VV, VH (Sentinel-1).")]);
  refCard(s, 6.78, 4.3, 8, [{ text: "ESA WorldCover 2021 (v200):", options: { breakLine: true } }, lk("esa-worldcover.org", URL.wc, { breakLine: true }), lk("Bản trên Earth Engine", URL.wcee, { breakLine: true }), used("Dùng ở: chọn điểm ruộng, cây trồng khác, rừng, cỏ, đô thị, nước.")]);

  // ================= 19 End
  s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Kết luận" });
  s.addText("Cảm ơn thầy và các bạn", { placeholder: "title" });
  tx(s, "Câu hỏi và góp ý", { x: 0.8, y: 2.15, w: 8.6, h: 0.6, fontSize: 22, color: C.background2 });

  await pres.writeFile({ fileName: process.env.OUT || "Slide_TimeSeries_Lua.pptx" });
  if (!process.env.NO_THEME) await applyTheme(process.env.OUT || "Slide_TimeSeries_Lua.pptx", THEME);
  console.log("written");
})().catch((e) => { console.error(e); process.exit(1); });
