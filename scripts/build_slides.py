import sys, os, json, base64, subprocess; sys.path.insert(0, os.path.dirname(__file__))
import mt, numpy as np

O, FD, SL = mt.OUT, os.path.join(mt.MT, "figs"), os.path.join(mt.MT, "slides")
E = json.load(open(f"{O}/eda.json")); B = json.load(open(f"{O}/baselines.json")); S = json.load(open(f"{O}/summary.json"))
B.update(json.load(open(f"{O}/extra_baselines.json")))
CZ = json.load(open(f"{O}/causal_results.json"))
SAR = json.load(open(f"{O}/sar_results_real.json"))
PK = json.load(open(f"{O}/pack_results.json")); PB = json.load(open(f"{O}/pack_bootstrap.json")); SK = json.load(open(f"{O}/sickle_s1_results.json"))
df, X, Y, doy = mt.load()
def img(name): return "data:image/png;base64," + base64.b64encode(open(f"{FD}/{name}", "rb").read()).decode()
def f3(v): return ("%.3f" % v).replace(".", ",")
def f1(v): return ("%.1f" % v).replace(".", ",")
def pm(t, d=3): return (("%." + str(d) + "f") % t[0] + " ± " + ("%." + str(d) + "f") % t[1]).replace(".", ",")
def pct(v, d=1): return (("%." + str(d) + "f") % (v * 100)).replace(".", ",") + "%"
def n(v): return f"{v:,}".replace(",", ".")
def ms(a): return float(np.mean(a)), float(np.std(a))

R = {k: S[k] for k in S if not k.startswith("_")}
trd, trn, cnn, lstm = R["Transformer_DOY"], R["Transformer_NDVI"], R["CNN1D"], R["BiLSTM"]
rf, cal, rule, dlg, slp = B["B2_random_forest"], B["B0_calendar"], B["B1_threshold_rule"], B["B3_double_logistic"], B["B4_ndvi_slope"]
ex0 = int(np.where((df.n_seasons == 2) & (df.miss_ratio < .15))[0][0])
pts = " ".join("%d,%d" % (i * 6.2, 150 - X[ex0][i] * 150) for i in range(mt.T))

CSS = """
@page{size:1440px 810px;margin:0}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{font-family:"Segoe UI","Be Vietnam Pro",Arial,sans-serif;color:#1B2A21;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.s{width:1440px;height:810px;position:relative;overflow:hidden;page-break-after:always;background:#fff;padding:54px 80px 0 80px}
.s.dark{background:#17322A;color:#fff}
.s.dark::before{content:"";position:absolute;left:0;top:0;bottom:0;width:20px;background:#C98A2B}
h1{font-size:46px;line-height:1.12;margin:0 0 6px;color:#17322A;font-weight:800;letter-spacing:-.4px}
.sub{font-size:22px;color:#5F6B63;margin:0 0 0}
.bar{width:150px;height:5px;background:#C98A2B;margin:14px 0 24px}
.pn{position:absolute;right:56px;bottom:22px;font-size:17px;color:#2F7A52;font-weight:700}
.sec{position:absolute;right:80px;top:24px;font-size:14.5px;letter-spacing:.12em;text-transform:uppercase;color:#B07A1E;font-weight:700}
.eyebrow{font-size:17px;letter-spacing:.12em;text-transform:uppercase;color:#C98A2B;font-weight:700}
p{margin:0}
.grid{display:grid;gap:22px}
.card{background:#F4F7F2;border:1.5px solid #D4DDD0;border-radius:14px;padding:20px 24px}
.card.w{background:#fff}
.card h3{margin:0 0 8px;font-size:23px;color:#17322A}
.card p,.card li{font-size:20.5px;line-height:1.42;color:#2C3A32}
.tag{font-size:15.5px;letter-spacing:.1em;text-transform:uppercase;font-weight:700;color:#2F7A52;margin-bottom:8px;display:block}
.tag.a{color:#B07A1E}
.big{font-size:46px;font-weight:800;color:#17322A;line-height:1}
.lbl{font-size:18px;color:#5F6B63;margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:20.5px}
th{font-size:15px;letter-spacing:.06em;text-transform:uppercase;color:#5F6B63;text-align:left;padding:9px 12px;border-bottom:2px solid #17322A;font-weight:700}
td{padding:11px 12px;border-bottom:1px solid #E3E8E0;vertical-align:top}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
table.cmp td{padding:6px 12px}
table.cl td{padding:4px 8px;font-size:16.5px}
table.cl th{padding:5px 8px;font-size:13px}
table.tight td{padding:8px 12px;font-size:19.5px}
tr.hl td{background:#E7F0E8;font-weight:700}
ul{margin:0;padding-left:26px}
li{margin-bottom:8px}
.flow{display:flex;align-items:stretch;gap:12px}
.box{flex:1;background:#F4F7F2;border:1.5px solid #D4DDD0;border-radius:12px;padding:15px 12px;text-align:center;font-size:19px;line-height:1.3}
.box b{display:block;font-size:21px;color:#17322A;margin-bottom:4px}
.box.dk{background:#17322A;color:#fff;border-color:#17322A}.box.dk b{color:#fff}
.box.am{background:#FBF0DC;border-color:#E2C48C}
.ar{align-self:center;font-size:28px;color:#C98A2B;font-weight:700}
.fig{display:block;margin:0 auto;max-width:100%}
.ok{color:#2F7A52;font-weight:800}.wn{color:#B07A1E;font-weight:800}.bd{color:#B5472A;font-weight:800}
.note{background:#FBF0DC;border-left:6px solid #C98A2B;border-radius:8px;padding:15px 22px;font-size:20.5px;line-height:1.42}
"""
slides = []
def add(body, cls="", sec=""):
    k = len(slides) + 1
    tag = f"<div class=sec>{sec}</div>" if sec else ""
    slides.append(f'<section class="s {cls}">{tag}{body}{"" if cls == "dark" else f"<div class=pn>{k}</div>"}</section>')
def head(t, st=""): return f"<h1>{t}</h1>" + (f"<p class=sub>{st}</p>" if st else "") + "<div class=bar></div>"
P1, P2, P3 = "Phần 1 · Dữ liệu", "Phần 2 · Baseline", "Phần 3 · Phương pháp và kết quả"

# 1 Title
add(f"""<div style="padding-left:20px"><div class=eyebrow style="color:#BFD1C5">Phân tích dữ liệu chuỗi thời gian · Đồ án môn học</div>
<div style="font-size:62px;font-weight:800;line-height:1.1;margin:22px 0 18px;max-width:760px">Theo dõi vòng đời cây lúa từ chuỗi vệ tinh NDVI và SAR</div>
<p style="font-size:25px;color:#D3E0D8;max-width:720px;line-height:1.4">Dữ liệu, baseline và phương pháp cho bài toán nhận diện giai đoạn sinh trưởng lúa</p>
<p style="font-size:20px;color:#E2B25E;font-weight:700;margin-top:26px">Đà Nẵng và Quảng Nam &nbsp;|&nbsp; Sentinel-2 và Sentinel-1 &nbsp;|&nbsp; 2023–2025</p>
<div style="margin-top:40px;font-size:21px;line-height:1.6;color:#fff"><b style="letter-spacing:.08em">THÀNH VIÊN</b><br>Văn Trọng Huy · 23AI018<br>Đỗ Phúc Vũ Hà · 23AI012<br>Sử Thị Hà My · 23AI031<br>Nguyễn Hà Nhi · 23AI039<br>Trần Thị Thùy Linh · 23AI029</div></div>
<svg style="position:absolute;right:90px;top:210px" width="520" height="190" viewBox="-6 -6 470 170"><polyline points="{pts}" fill="none" stroke="#E2B25E" stroke-width="4" stroke-linejoin="round"/></svg>
<p style="position:absolute;right:90px;top:430px;width:520px;text-align:center;font-size:20px;font-weight:700;color:#fff">{n(E['n'])} cửa sổ &nbsp;•&nbsp; 73 bước &nbsp;•&nbsp; 4 giai đoạn</p>""", "dark")

# 2 Outline
add(head("Nội dung", "Dữ liệu thu thập ra sao, paper làm nền cho baseline, phương pháp của nhóm và kết quả") + f"""
<div class=grid style="grid-template-columns:repeat(3,1fr)">
<div class=card><span class=tag>Phần 1</span><h3>Dữ liệu</h3><p>Thu thập từ Google Earth Engine (Sentinel-2 và Sentinel-1), tạo nhãn, thống kê {n(E['n'])} cửa sổ, kiểm tra chất lượng, độ mờ của nhãn.</p></div>
<div class=card><span class=tag>Phần 2</span><h3>Baseline</h3><p>Paper làm nền (BreizhCrops, Gobin 2023, Tian 2023) và năm baseline phát triển từ các paper đó, kèm kết quả.</p></div>
<div class=card><span class=tag>Phần 3</span><h3>Phương pháp và kết quả</h3><p>1D-CNN kết hợp giải mã Viterbi; đầu vào SAR và xử lý mây; lúa so với cây khác; nhận diện trực tuyến; kiểm thử theo năm và với ground truth SICKLE.</p></div></div>
<div class=note style="margin-top:34px"><b>Thông điệp chính:</b> phương pháp 1D-CNN kết hợp Viterbi đạt macro-F1 {f3(cnn['viterbi']['macro_f1'][0])}, vượt các baseline (0,64–0,77) và luôn đúng thứ tự giai đoạn. Nhãn sinh từ chính NDVI nên kết quả đo khả năng tái tạo luật gán nhãn, chưa phải độ chính xác so với ngày gieo hay thu hoạch ngoài đồng.</div>""")

# 3 Problem
add(head("Bài toán", "Gán giai đoạn mùa vụ cho từng bước thời gian trong chuỗi NDVI") + """
<div class=flow style="margin-bottom:30px"><div class=box><b>Đầu vào</b>Một cửa sổ 365 ngày<br>73 giá trị NDVI, cách nhau 5 ngày</div><div class=ar>→</div><div class="box dk"><b>Mô hình</b>nhãn cho từng bước<br>(sequence labelling)</div><div class=ar>→</div><div class=box><b>Đầu ra</b>73 nhãn giai đoạn<br>mỗi cửa sổ có 1 hoặc 2 vụ lúa</div></div>
<div class=grid style="grid-template-columns:repeat(4,1fr)"><div class=card style="border-top:7px solid #B8AD93"><h3>0 · Ngoài vụ</h3><p>Đất trống, ngập nước, giữa hai vụ</p></div><div class=card style="border-top:7px solid #5FA86B"><h3>1 · Sinh trưởng</h3><p>NDVI tăng nhanh sau gieo sạ</p></div><div class=card style="border-top:7px solid #D9A83A"><h3>2 · Sinh sản</h3><p>Quanh đỉnh NDVI, giai đoạn ngắn nhất</p></div><div class=card style="border-top:7px solid #C9653F"><h3>3 · Suy giảm</h3><p>NDVI giảm đến thu hoạch</p></div></div>
<p style="margin-top:28px;font-size:22px;line-height:1.45">Từ mặt nạ giai đoạn suy ra ba mốc: <b>SOS</b> (bắt đầu mùa), <b>POS</b> (đỉnh), <b>EOS</b> (kết thúc mùa). Thứ tự hợp lệ là một vòng: Ngoài vụ → Sinh trưởng → Sinh sản → Suy giảm → Ngoài vụ.</p>""", sec=P1)

# 4 Data collection
steps = ["Sentinel-2 SR<br>2023–2025", "Cloud mask SCL<br>bỏ mây, bóng, cirrus", "Lưới 5 ngày<br>median trong cửa sổ", "Nội suy<br>lấp khoảng trống", "Làm mượt<br>Savitzky–Golay (9, bậc 2)", "Phát hiện đỉnh<br>prominence 0,25", "Tạo nhãn<br>SOS, POS, EOS, 4 giai đoạn", "Lọc và chia tập<br>khoảng trống ≤ 10 bước; chia theo ô lưới"]
add(head("Dữ liệu được thu thập thế nào", "Tự xây dựng trên Google Earth Engine (Colab), 1.000 điểm ruộng lúa, 2023–2025") +
    '<div class=grid style="grid-template-columns:repeat(4,1fr);gap:12px">' + "".join(f'<div class=box><b>{i}</b>{t}</div>' for i, t in enumerate(steps, 1)) + f"""</div>
<div class=grid style="grid-template-columns:1fr 1fr;margin-top:24px"><div class=card><span class=tag>Chọn điểm</span><p>WorldCover 2021 lấy cropland; lọc ứng viên lúa bằng biên độ NDVI và NDWI; lấy ngẫu nhiên phân tầng 300 điểm Đà Nẵng và 700 điểm Quảng Nam (tọa độ 107,79–108,66°E, 14,98–16,15°N).</p></div>
<div class=card><span class=tag>Năm kênh thu được</span><p>Sentinel-2: <b>NDVI, NDWI</b>, mặt nạ quan sát (NDWI thiếu 35%). Sentinel-1 IW: <b>VV, VH</b>, lọc nhiễu hạt bằng median 30 m (thiếu 0,9%). Mảng đủ 5 kênh được lưu riêng; CSV huấn luyện chính chỉ giữ NDVI và nhãn.</p></div></div>""", sec=P1)

# 5 Dataset
add(head("Bộ dữ liệu cuối cùng", "Chia theo ô lưới 2 km để giảm rò rỉ không gian giữa các điểm gần nhau") + f"""
<div class=grid style="grid-template-columns:repeat(5,1fr);margin-bottom:20px">
<div><div class=big>{n(E['n'])}</div><div class=lbl>cửa sổ chuỗi thời gian</div></div><div><div class=big>{E['fields']}</div><div class=lbl>thửa (field_id)</div></div>
<div><div class=big>{E['cols']}</div><div class=lbl>cột: 9 metadata, 73 NDVI, 73 nhãn</div></div><div><div class=big style="font-size:31px;padding-top:9px;white-space:nowrap">{E['split']['train']} / {E['split']['val']} / {E['split']['test']}</div><div class=lbl>train / validation / test</div></div><div><div class=big>{pct(E['miss_mean'])}</div><div class=lbl>bước thiếu ảnh gốc, trung bình</div></div></div>
<img class=fig src="{img('fig_data_overview.png')}" style="width:1230px">
<p style="font-size:19px;color:#5F6B63;margin-top:10px">Năm 2023: {E['year']['2023']} · 2024: {E['year']['2024']} · 2025: {E['year']['2025']} · Đà Nẵng 401, Quảng Nam 1.427 · bốn lớp lệch vừa phải, Sinh sản chỉ {pct(E['class_share'][2])}</p>""", sec=P1)

# 6 Examples
add(head("Ví dụ cửa sổ và nhãn", "Dải màu bên dưới mỗi đường NDVI là nhãn giai đoạn theo từng bước") + f"""<img class=fig src="{img('fig_examples.png')}" style="width:1200px">
<p style="font-size:21px;margin-top:8px;text-align:center;color:#46544B">Hai vụ lúa tạo hai đỉnh NDVI. Cửa sổ thiếu 45–49% ảnh vẫn có đường cong liền do nội suy, nên mô hình phải học cả phần bị nội suy.</p>""", sec=P1)

# 7 Quality checks
L = E["label_sensitivity_days"]
add(head("Kiểm tra chất lượng dữ liệu", "Những gì đã kiểm chứng trên file CSV và những điểm cần nêu rõ") + f"""
<table><tr><th style="width:430px">Kiểm tra</th><th>Kết quả</th><th style="width:190px">Đánh giá</th></tr>
<tr><td>Ô trống (NaN) trong NDVI và nhãn</td><td>0 ô</td><td class=ok>Đạt</td></tr>
<tr><td>Một thửa nằm ở hai tập khác nhau</td><td>0 / {E['fields']} thửa ({E['fields_multi_window']} thửa có nhiều cửa sổ, đều cùng một tập)</td><td class=ok>Đạt</td></tr>
<tr><td>Ô lưới 2 km nằm ở nhiều tập</td><td>{E['cells_multi_split']} / {E['cells']} ô, tương ứng {E['windows_in_multi_split_cells']} cửa sổ ({pct(E['windows_in_multi_split_cells']/E['n'])})</td><td class=wn>Rò rỉ nhỏ</td></tr>
<tr><td>Phân bố lớp giữa train, val, test</td><td>Chênh lệch dưới 2 điểm phần trăm mỗi lớp</td><td class=ok>Đạt</td></tr>
<tr><td>Kiểm tra bằng mắt 50 mẫu (ảnh Sentinel-2 tại SOS, POS, EOS)</td><td>SOS 39/39 đạt, 11 chưa rõ · POS 45/45 · EOS 47/47 · đúng ruộng lúa 47/47</td><td class=wn>Mẫu nhỏ, 1 người chấm</td></tr>
<tr><td>Nhãn có độc lập với NDVI đầu vào không</td><td>Không: nhãn được sinh bằng luật ngưỡng trên chính chuỗi NDVI (xem trang sau)</td><td class=bd>Hạn chế chính</td></tr></table>""", sec=P1)

# 8 Label finding
add(head("Phát hiện: nhãn là hàm xác định của NDVI", "Cài lại luật gán nhãn ở mức cửa sổ và đo trên tập test") + f"""
<div class=grid style="grid-template-columns:1fr 1.25fr;gap:30px">
<div><div class=card style="margin-bottom:18px"><span class=tag>Luật ngưỡng 20% biên độ</span><p>Khi luật phát hiện được mùa vụ, nó khớp nhãn <b>{pct(rule['agree_when_rule_fires'])}</b> số bước. Sai số mốc SOS/POS/EOS xấp xỉ 0 ngày.</p><p style="margin-top:8px;font-size:18.5px;color:#5F6B63">Chỉ {pct(rule['windows_detected_all_seasons'],0)} cửa sổ được phát hiện đủ mùa vụ vì cửa sổ 73 bước thiếu ngữ cảnh 3 năm ở hai mép.</p></div>
<div class=card style="background:#FBF0DC;border-color:#E2C48C"><span class="tag a">Độ mờ của nhãn</span><p>Đổi ngưỡng 10% / 20% / 30% làm mốc lệch trung vị <b>{L['sos']['median']:.0f} ngày</b> (SOS) và <b>{L['eos']['median']:.0f} ngày</b> (EOS); P90 là {L['sos']['p90']:.0f} và {L['eos']['p90']:.0f} ngày.</p></div></div>
<div><img class=fig src="{img('fig_label_sensitivity.png')}" style="height:455px"><p style="font-size:21px;margin-top:10px;line-height:1.4"><b>Hệ quả:</b> sai số mốc dưới khoảng 10 ngày nằm trong độ mờ của chính nhãn, nên không dùng được để xếp hạng các mô hình tốt.</p></div></div>""", sec=P1)

# 9 Papers as baseline
add(head("Baseline: paper làm nền để phát triển", "Nhóm lấy phương pháp của các paper này làm baseline, rồi phát triển thêm") + """
<div class=grid style="grid-template-columns:1fr 1fr 1fr;gap:20px">
<div class=card style="border-top:7px solid #2F7A52"><span class=tag>Paper nền 1 · benchmark</span><h3>BreizhCrops</h3><p style="font-size:18.5px;color:#5F6B63;margin-bottom:8px">Rußwurm và cs., ISPRS Archives 2020</p><p>Bộ chuẩn phân loại cây trồng từ chuỗi Sentinel-2: hơn 600.000 chuỗi, 9 loại cây, 4 vùng chia train/val/test theo không gian. So Random Forest với CNN, RNN, Transformer.</p><p style="margin-top:10px"><b>Baseline lấy:</b> Random Forest và cách chia theo không gian.</p></div>
<div class=card style="border-top:7px solid #2F7A52"><span class=tag>Paper nền 2 · vật hậu</span><h3>Gobin và cs. 2023</h3><p style="font-size:18.5px;color:#5F6B63;margin-bottom:8px">Remote Sensing 15(8), 2090 (link thầy đưa)</p><p>Lúa mì đông, ngô, khoai tây ở Bỉ 2015–2017 (298 ruộng). Khớp double sigmoid lên fAPAR/fCover từ Sentinel-2; điểm uốn ứng với giai đoạn vươn thân và suy giảm; ground truth BBCH ngoài đồng.</p><p style="margin-top:10px"><b>Baseline lấy:</b> khớp double-logistic.</p></div>
<div class=card style="border-top:7px solid #C98A2B"><span class="tag a">Paper nền 3 · lúa</span><h3>Tian và cs. 2023</h3><p style="font-size:18.5px;color:#5F6B63;margin-bottom:8px">Sensors 23(13), 5876 (link thầy đưa)</p><p>Trích xuất ruộng lúa bằng chênh lệch NDVI trước và sau thu hoạch, khử viền nước bằng NDWI, hồi quy ngày thu hoạch theo độ cao và vĩ độ (32 điểm mẫu).</p><p style="margin-top:10px"><b>Baseline lấy:</b> ý tưởng độ dốc, chênh lệch NDVI.</p></div></div>
<div class=note style="margin-top:24px">Các bài này khác vùng, khác cây, và (trừ BreizhCrops) dùng ground truth quan trắc thực địa. Vì nhãn của nhóm dẫn xuất từ NDVI nên không so F1 trực tiếp với các bài này; nhóm cài lại phương pháp của họ trên cùng dữ liệu và cùng metric.</div>""", sec=P2)

# 10 Baseline choice
add(head("Năm baseline đã chốt", "Cùng split, cùng metric; chọn tham số trên validation, chỉ báo cáo test sau khi khóa") + """
<table class=tight><tr><th style="width:290px">Baseline</th><th>Cách làm</th><th style="width:330px">Nguồn</th></tr>
<tr><td><b>B0 · Lịch mùa vụ</b></td><td>Mỗi tỉnh và mỗi ngày trong năm lấy giai đoạn phổ biến nhất trong tập train</td><td>Ngưỡng sàn, không nhìn NDVI</td></tr>
<tr><td><b>B1 · Luật ngưỡng 20%</b></td><td>Tìm đỉnh, lấy điểm cắt 20% biên độ (chính luật tạo nhãn, cài ở mức cửa sổ)</td><td>Trần cơ học khi có NDVI</td></tr>
<tr class=hl><td><b>B2 · Random Forest</b></td><td>73 NDVI + ngày bắt đầu; 500 cây, max_features = sqrt; độ sâu 25 chọn trên validation</td><td>BreizhCrops (Rußwurm 2020)</td></tr>
<tr><td><b>B3 · Double-logistic</b></td><td>Khớp một double-logistic quanh mỗi đỉnh, lấy mốc 20% biên độ trên đường khớp</td><td>Gobin 2023</td></tr>
<tr><td><b>B4 · Độ dốc NDVI</b></td><td>SOS ở nơi NDVI tăng dốc nhất trước đỉnh, EOS ở nơi giảm dốc nhất sau đỉnh</td><td>Họ phương pháp đạo hàm, ý từ Tian 2023</td></tr></table>
<p style="font-size:18px;color:#76827A;margin-top:12px">B3 và B4 là bản cài đặt của nhóm cho cùng đầu ra 4 giai đoạn, không phải tái hiện nguyên bài gốc. B1 và B3 dựng mặt nạ từ các mốc nên luôn đúng thứ tự giai đoạn; B4 vẫn sai thứ tự ở khoảng 19% cửa sổ.</p>""", sec=P2)

# 11 Baseline results
def row(nm, d, cls=""):
    g = d["dates"]; return f"<tr class={cls}><td>{nm}</td><td class=n>{f3(d['macro_f1'])}</td><td class=n>{f3(d['acc'])}</td><td class=n>{f1(g['sos']['mae_days'])}</td><td class=n>{f1(g['pos']['mae_days'])}</td><td class=n>{f1(g['eos']['mae_days'])}</td><td class=n>{pct(g['sos']['miss'],0)} / {pct(g['eos']['miss'],0)}</td><td class=n>{pct(d['retrograde'],0)}</td></tr>"
add(head("Kết quả baseline trên tập test", f"{E['split']['test']} cửa sổ; chọn cấu hình trên validation, báo cáo test sau khi khóa") + f"""
<table class=cmp style="font-size:19.5px"><tr><th>Baseline</th><th class=n>Macro-F1</th><th class=n>Accuracy</th><th class=n>MAE SOS (ngày)</th><th class=n>MAE POS</th><th class=n>MAE EOS</th><th class=n>Bỏ sót SOS / EOS</th><th class=n>Sai thứ tự</th></tr>
{row('B0 · Lịch mùa vụ', cal)}{row('B1 · Luật ngưỡng 20%', rule)}{row('B2 · Random Forest', rf, 'hl')}{row('B2 · RF + giải mã Viterbi', B['B2_random_forest+viterbi'])}{row('B3 · Double-logistic', dlg)}{row('B4 · Độ dốc NDVI', slp)}</table>
<div class=grid style="grid-template-columns:1fr 1fr 1fr;margin-top:22px">
<div class=card><span class=tag>Sàn</span><p>Lịch mùa vụ đã đạt {f3(cal['macro_f1'])} mà không dùng NDVI, nên khoảng cách thật của mô hình phải tính từ đây.</p></div>
<div class=card><span class=tag>Các baseline dùng NDVI</span><p>Bốn baseline từ 0,746 đến 0,767: một nhóm sát nhau; B1 và B4 bỏ sót nhiều mùa vụ ở hai mép cửa sổ.</p></div>
<div class=card><span class=tag>Random Forest</span><p>{pct(rf['retrograde'],0)} cửa sổ sai thứ tự giai đoạn vì dự đoán từng bước độc lập; Viterbi đưa về 0%.</p></div></div>
<p style="font-size:17px;color:#76827A;margin-top:10px">MAE chỉ tính trên mốc thật có mốc dự đoán trong vòng 60 ngày; phần còn lại là bỏ sót. Các baseline chạy 1 lần (xác định); Random Forest 1 seed.</p>""", sec=P2)

# 12 Method
miss_lo = min(rule['dates']['sos']['miss'], rule['dates']['eos']['miss'], slp['dates']['sos']['miss'], slp['dates']['eos']['miss'])
miss_hi = max(rule['dates']['sos']['miss'], rule['dates']['eos']['miss'], slp['dates']['sos']['miss'], slp['dates']['eos']['miss'])
add(head("Phương pháp của nhóm", "Phát triển từ baseline: học mặt nạ giai đoạn bằng mạng chuỗi, giải mã Viterbi, thêm đầu vào SAR") + f"""
<div class=flow style="margin-bottom:24px"><div class=box><b>Đầu vào</b>NDVI, hoặc VV, VH + NDWI<br>cùng sin, cos của ngày<br>73 bước</div><div class=ar>→</div><div class="box dk"><b>1D-CNN</b>3 khối tích chập giãn<br>kernel 5, dilation 1-2-4<br>kênh 32-64-64</div><div class=ar>→</div><div class=box><b>Phân loại</b>conv 1×1 → xác suất<br>4 giai đoạn mỗi bước</div><div class=ar>→</div><div class="box am"><b>Viterbi</b>chỉ cho phép giữ nguyên<br>hoặc tiến một giai đoạn</div><div class=ar>→</div><div class=box><b>Đầu ra</b>mặt nạ, mốc SOS,<br>POS, EOS theo ngày</div></div>
<div class=grid style="grid-template-columns:1fr 1fr 1fr">
<div class=card><span class=tag>1 · Học từ dữ liệu</span><p>Các baseline luật bỏ sót {pct(miss_lo,0)}–{pct(miss_hi,0)} mốc ở hai mép cửa sổ; mạng chuỗi học cả các trường hợp đó. Chỉ {n(cnn['viterbi']['params'])} tham số.</p></div>
<div class=card><span class=tag>2 · Giải mã Viterbi</span><p>Dự đoán thô sai thứ tự giai đoạn ở {pct(min(m['raw']['retro'][0] for m in (cnn, lstm, trn, trd)),0)}–{pct(max(m['raw']['retro'][0] for m in (cnn, lstm, trn, trd)),0)} cửa sổ (tùy mô hình); Viterbi đưa về 0% mà không thêm tham số.</p></div>
<div class=card style="background:#FBF0DC;border-color:#E2C48C"><span class="tag a">3 · Đầu vào SAR</span><p>Thay NDVI bằng Sentinel-1 VV, VH và NDWI: nhãn sinh từ NDVI nên đây là cách thoát vòng lặp; SAR xuyên mây.</p></div></div>
<p style="margin-top:20px;font-size:21px;line-height:1.45"><b>Dự đoán gì:</b> mặt nạ 4 giai đoạn cho từng bước 5 ngày; mốc SOS, POS, EOS theo ngày; và nhận diện trực tuyến khi chỉ có dữ liệu đã quan sát.</p>""", sec=P3)

# 13 Setup
def prm(nm, d, desc): return f"<tr><td><b>{nm}</b></td><td>{desc}</td><td class=n>{n(d['viterbi']['params'])}</td></tr>"
add(head("Thiết lập thí nghiệm", "Cùng split, cùng đầu vào, cùng metric cho mọi mô hình") + f"""
<div class=grid style="grid-template-columns:1.25fr 1fr;gap:30px"><div>
<table><tr><th>Mô hình</th><th>Cấu hình</th><th class=n style="width:120px">Tham số</th></tr>
{prm('1D-CNN', cnn, '3 khối dilated conv, kênh 32-64-64')}{prm('BiLSTM', lstm, '1 lớp, ẩn 64, hai chiều')}{prm('Transformer (chỉ NDVI)', trn, '2 lớp, d=64, 4 đầu, mã hóa theo chỉ số bước')}{prm('Transformer (+DOY)', trd, 'như trên, thêm sin/cos của ngày')}</table></div>
<div class=card><span class=tag>Huấn luyện và đánh giá</span><ul style="font-size:20px;line-height:1.38"><li>Tối ưu AdamW, lr 2e-3, batch 32, tối đa 120 epoch</li><li>Cross-entropy có trọng số lớp; dừng sớm theo macro-F1 validation (kiên nhẫn 15)</li><li><b>5 seed</b> mỗi mô hình; báo cáo trung bình ± độ lệch chuẩn</li><li>Test chỉ dùng để báo cáo, không chọn mô hình</li><li>Metric: macro-F1, MAE mốc theo ngày, tỷ lệ chuỗi sai thứ tự</li></ul></div></div>
<p style="font-size:18px;color:#76827A;margin-top:14px">Ghi chú: 5 seed dùng chung một split, nên độ lệch chuẩn phản ánh dao động khởi tạo, chưa phản ánh dao động giữa các tập dữ liệu. Huấn luyện chạy trên CPU.</p>""", sec=P3)

# 14 Main results
def drow(nm, d, cls=""):
    v = d["viterbi"]; return f"<tr class={cls}><td>{nm}</td><td class=n>{pm(v['macro_f1'])}</td><td class=n>{pm(d['raw']['macro_f1'])}</td><td class=n>{pct(d['raw']['retro'][0],0)}</td><td class=n>{pm(v['f1'][2],3)}</td></tr>"
add(head("Kết quả chính trên tập test", "Macro-F1, trung bình ± độ lệch chuẩn trên 5 seed") + f"""
<img class=fig src="{img('fig_results.png')}" style="width:880px">
<table class=cmp style="margin-top:6px;font-size:18px"><tr><th>Mô hình</th><th class=n>Macro-F1 (Viterbi)</th><th class=n>Macro-F1 (thô)</th><th class=n>Sai thứ tự (thô)</th><th class=n>F1 lớp Sinh sản</th></tr>
{drow('1D-CNN', cnn, 'hl')}{drow('BiLSTM', lstm)}{drow('Transformer (chỉ NDVI)', trn)}{drow('Transformer (+DOY)', trd)}</table>
<div class=note style="margin-top:10px;font-size:18px;padding:9px 18px;line-height:1.35"><b>1D-CNN cao nhất</b> ({pm(cnn['viterbi']['macro_f1'])}) với ít tham số nhất ({n(cnn['viterbi']['params'])}), nên được chọn làm mô hình chính. Thêm mã hóa ngày vào Transformer chưa cho thấy lợi ích ({pm(trd['viterbi']['macro_f1'])} so với {pm(trn['viterbi']['macro_f1'])} chỉ NDVI). Transformer chỉ NDVI chạm trần 120 epoch ở 2/5 seed.</div>""", sec=P3)

# 15 Dates
def mrow(nm, d, cls=""):
    v = d["viterbi"]; return f"<tr class={cls}><td>{nm}</td><td class=n>{f1(v['mae']['sos'][0])}</td><td class=n>{f1(v['mae']['pos'][0])}</td><td class=n>{f1(v['mae']['eos'][0])}</td><td class=n>{pct(v['hit10']['sos'][0],0)} / {pct(v['hit10']['eos'][0],0)}</td><td class=n>{pct(v['miss']['sos'][0],0)} / {pct(v['miss']['eos'][0],0)}</td></tr>"
def brow(nm, d):
    g = d["dates"]; return f"<tr><td>{nm}</td><td class=n>{f1(g['sos']['mae_days'])}</td><td class=n>{f1(g['pos']['mae_days'])}</td><td class=n>{f1(g['eos']['mae_days'])}</td><td class=n>{pct(g['sos']['hit10'],0)} / {pct(g['eos']['hit10'],0)}</td><td class=n>{pct(g['sos']['miss'],0)} / {pct(g['eos']['miss'],0)}</td></tr>"
add(head("Sai số mốc SOS, POS, EOS theo ngày", "Mốc suy từ mặt nạ dự đoán; mỗi mốc thật ghép với mốc dự đoán gần nhất trong 60 ngày") + f"""
<table class=cmp style="font-size:19.5px"><tr><th>Mô hình</th><th class=n>MAE SOS</th><th class=n>MAE POS</th><th class=n>MAE EOS</th><th class=n>Trong ±10 ngày (SOS / EOS)</th><th class=n>Bỏ sót (SOS / EOS)</th></tr>
{brow('B0 · Lịch mùa vụ', cal)}{brow('B2 · Random Forest + Viterbi', B['B2_random_forest+viterbi'])}{brow('B3 · Double-logistic', dlg)}
{mrow('1D-CNN', cnn, 'hl')}{mrow('BiLSTM', lstm)}{mrow('Transformer (chỉ NDVI)', trn)}{mrow('Transformer (+DOY)', trd)}</table>
<div class=note style="margin-top:20px">Nhãn tự nó lệch khoảng <b>10 ngày</b> khi đổi ngưỡng 10–30%, còn MAE của các mô hình sâu chỉ 0–4 ngày: sai số nằm trong độ mờ của nhãn, nên không đủ cơ sở xếp hạng chúng theo mốc. MAE của POS gần 0 một phần vì POS được lấy là đỉnh NDVI trong đoạn dự đoán.</div>""", sec=P3)

# 16 Qualitative
add(head("Mặt nạ giai đoạn: nhãn so với dự đoán", "Ba cửa sổ test chọn theo quy tắc cố định: phân vị 50, 25 và 5 của độ chính xác 1D-CNN") + f"""<img class=fig src="{img('fig_qualitative.png')}" style="width:1260px">""", sec=P3)

# 17 Robustness
rb = lambda d, k: d["robust"][k][0]
add(head("Kháng chịu dữ liệu thiếu", "Che thêm 10–50% bước NDVI khi kiểm thử rồi nội suy tuyến tính lại; mô hình huấn luyện trên dữ liệu gốc") + f"""
<div class=grid style="grid-template-columns:1.25fr 1fr;gap:30px;align-items:center"><img class=fig src="{img('fig_robust.png')}" style="width:100%">
<div><div class=card style="margin-bottom:16px"><span class=tag>Con số (macro-F1, che 0% → 50%)</span><p>1D-CNN {f3(rb(cnn,'0.0'))} → {f3(rb(cnn,'0.5'))}<br>Transformer (+DOY) {f3(rb(trd,'0.0'))} → {f3(rb(trd,'0.5'))}<br>BiLSTM {f3(rb(lstm,'0.0'))} → {f3(rb(lstm,'0.5'))}<br>Random Forest {f3(S['_rf_robust']['0.0'])} → {f3(S['_rf_robust']['0.5'])}</p></div>
<div class=note style="font-size:19.5px">Mô hình sâu giảm khoảng 3,5 điểm khi che 50%, 1D-CNN dẫn đầu ở mọi mức. Random Forest gần như phẳng nhưng thấp hơn hẳn. Chuỗi đã được nội suy sẵn nên đây là thử nghiệm sơ bộ; trang sau so với SAR và thử cách xử lý.</div></div></div>""", sec=P3)

# 18 SAR
cfg = SAR["results"]; ctrl = cfg["NDVI+DOY (control)"]
srow_ = lambda nm, r, cls="": f"<tr class={cls}><td>{nm}</td><td class=n>{pm(ms([x['viterbi']['macro_f1'] for x in r]))}</td><td class=n>{f1(ms([x['dates']['sos']['mae_days'] for x in r])[0])} / {f1(ms([x['dates']['eos']['mae_days'] for x in r])[0])}</td></tr>"
sar_f = ms([x["viterbi"]["macro_f1"] for x in cfg["SAR (VV,VH,VH-VV)+DOY"]])[0]; ctl_f = ms([x["viterbi"]["macro_f1"] for x in ctrl])[0]
add(head("Đầu vào SAR: bỏ NDVI khỏi mô hình", "1D-CNN, 5 seed, z-score theo thống kê tập train; nhãn vẫn sinh từ NDVI") + f"""
<table><tr><th>Đầu vào (cộng sin, cos của ngày)</th><th class=n>Macro-F1 (Viterbi)</th><th class=n>MAE SOS / EOS (ngày)</th></tr>
{srow_('NDVI (đối chứng)', ctrl)}{srow_('SAR: VV, VH, VH−VV', cfg['SAR (VV,VH,VH-VV)+DOY'], 'hl')}{srow_('SAR + NDWI', cfg['SAR+NDWI+DOY'])}{srow_('NDVI + NDWI + SAR', cfg['NDVI+NDWI+SAR+DOY'])}</table>
<div class=grid style="grid-template-columns:1fr 1fr;margin-top:26px"><div class=card><span class=tag>Đọc kết quả</span><p>SAR không có NDVI đạt {f3(sar_f)}, thấp hơn đối chứng NDVI {f3(ctl_f)} khoảng {f1((ctl_f - sar_f) * 100)} điểm; thêm NDWI lên 0,850. Thông tin về giai đoạn có trong SAR, nhưng chưa bằng NDVI.</p></div>
<div class=card style="background:#FBF0DC;border-color:#E2C48C"><span class="tag a">Điều chưa được kết luận</span><p>Nhãn dẫn xuất từ NDVI nên chấm điểm sẵn có lợi cho NDVI và bất lợi cho SAR; điểm SAR thấp chưa có nghĩa SAR kém trên vật hậu thật. Chỉ ground truth ngoài đồng mới trả lời được. POS vẫn tính từ đỉnh NDVI khi đánh giá.</p></div></div>""", sec=P3)


# 18b SAR and cloud
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CL = json.load(open(f"{O}/cloud_results.json")); LV = CL["levels"]; CR = CL["results"]
cm = lambda name, p: float(np.mean([r["by_level"][str(p)]["macro_f1"] for r in CR[name]]))
def cl_row(lbl, name, cls=""):
    return f"<tr class={cls}><td>{lbl}</td>" + "".join(f"<td class=n>{f3(cm(name, p))}</td>" for p in LV) + "</tr>"
fig, ax = plt.subplots(figsize=(6.4, 3.15), dpi=170)
sty = [("NDVI", "NDVI", "#17322A", "-"), ("NDVI +mask-aug", "NDVI, huấn luyện che mây", "#2F7A52", "-"), ("SAR", "Chỉ SAR", "#C9653F", "--"), ("SAR+NDWI", "SAR + NDWI", "#B8AD93", "-")]
for nm, lb, col, ls in sty: ax.plot([p * 100 for p in LV], [cm(nm, p) for p in LV], ls, color=col, lw=2.4, marker="o", ms=5, label=lb)
ax.set_xlabel("% bước quang học bị mất thêm (khối 30 ngày)", fontsize=9); ax.set_ylabel("Macro-F1 (test)", fontsize=9); ax.set_xticks([p * 100 for p in LV]); ax.set_ylim(0.6, 0.92)
ax.grid(alpha=.25); ax.legend(fontsize=8, frameon=False, loc="lower left"); [ax.spines[k].set_visible(False) for k in ("top", "right")]; ax.tick_params(labelsize=8.5)
fig.tight_layout(); fig.savefig(f"{FD}/fig_cloud.png"); plt.close(fig)
sar_c = cm("SAR", 0.0); nd = [cm("NDVI", p) for p in LV]
cross = LV[2] * 100 + (LV[3] - LV[2]) * 100 * (nd[2] - sar_c) / (nd[2] - nd[3])
add(head("SAR và mây: khi mất dữ liệu quang học", "Che thêm khối 30 ngày của quan sát quang học thật rồi nội suy lại; SAR không bị che; 1D-CNN, 3 seed, 274 cửa sổ test") + f"""
<div class=grid style="grid-template-columns:1.05fr 1fr;gap:28px;align-items:start"><div><img class=fig src="{img('fig_cloud.png')}" style="width:100%">
<table class=cl style="margin-top:2px"><tr><th>Đầu vào</th><th class=n>0%</th><th class=n>25%</th><th class=n>50%</th><th class=n>75%</th></tr>
{cl_row('NDVI', 'NDVI')}{cl_row('NDVI + che mây', 'NDVI +mask-aug', 'hl')}{cl_row('Chỉ SAR', 'SAR')}{cl_row('SAR + NDWI', 'SAR+NDWI')}{cl_row('Cả bốn kênh', 'NDVI+NDWI+SAR')}{cl_row('Cả bốn kênh + che mây', 'NDVI+NDWI+SAR +mask-aug', 'hl')}</table></div>
<div><div class=card style="margin-bottom:14px"><span class=tag>Đọc kết quả</span><p>SAR không đổi ({f3(sar_c)}) vì radar xuyên mây, nhưng luôn thấp hơn NDVI trong dải thử. NDVI chỉ tụt xuống dưới mức SAR khi mất khoảng {cross:.0f}% số bước ({f3(nd[0])} → {f3(nd[3])} ở 75%).</p></div>
<div class=card style="margin-bottom:14px"><span class=tag>Cách xử lý mây hiệu quả nhất</span><p>Huấn luyện có che mây ngẫu nhiên: kém {f1((nd[0]-cm('NDVI +mask-aug',0.0))*100)} điểm khi trời quang, nhưng hơn {f1((cm('NDVI +mask-aug',0.75)-nd[3])*100)} điểm khi mất 75% và vượt cả chỉ SAR ({f3(cm('NDVI +mask-aug',0.75))}). Kênh NDWI cũng là quang học nên bị mây che như NDVI.</p></div>
<div class=note style="font-size:18.5px"><b>Giới hạn:</b> nhãn sinh từ NDVI lúc trời quang nên phép thử thiên về NDVI; chuỗi gốc đã nội suy khoảng 33% bước và bỏ cửa sổ thiếu quá 50%. Kết luận đúng là "SAR giữ nguyên, quang học giảm theo mây, huấn luyện che mây là cách xử lý tốt nhất", chưa phải "SAR chính xác hơn".</div></div></div>""", sec=P3)

# 19 Stricter validation
def pk(sc, c, kind, key): return float(np.mean([r["events"][kind][key] for r in PK[sc][c]]))
def pks(sc, c, kind, key):
    v = [r["events"][kind][key] for r in PK[sc][c] if r["events"][kind].get(key) is not None]; return float(np.mean(v)) if v else float("nan")
def prow(lbl, sc, c, cls=""):
    b = PB[sc][c]
    return (f"<tr class={cls}><td>{lbl}</td><td class=n>{f3(b['mean'])} [{f3(b['lo'])}; {f3(b['hi'])}]</td>"
            f"<td class=n>{f1(pk(sc,c,'sos','mae'))} / {f1(pk(sc,c,'eos','mae'))}</td>"
            f"<td class=n>{pct(pk(sc,c,'sos','hit5'),0)} / {pct(pk(sc,c,'sos','hit10'),0)} / {pct(pk(sc,c,'sos','hit15'),0)}</td>"
            f"<td class=n>{f1(pks(sc,c,'sos','mae_DX'))} / {f1(pks(sc,c,'sos','mae_HT'))} ; {f1(pks(sc,c,'eos','mae_DX'))} / {f1(pks(sc,c,'eos','mae_HT'))}</td></tr>")
dY = PB["year"]["diff_NDVI_minus_SARNDWI"]
add(head("Kiểm thử chặt hơn: theo năm, theo vụ, dung sai", "1D-CNN, 3 seed; khoảng tin cậy 95% bootstrap theo cửa sổ trên trung bình 3 seed") + f"""
<table class=cmp style="font-size:18.5px"><tr><th>Kịch bản · đầu vào</th><th class=n>Macro-F1 [CI 95%]</th><th class=n>MAE SOS / EOS (ngày)</th><th class=n>SOS trong ±5 / ±10 / ±15 ngày</th><th class=n>MAE Đông Xuân / Hè Thu (SOS ; EOS)</th></tr>
{prow('Chuẩn (274) · NDVI', 'standard', 'NDVI')}{prow('Chuẩn · SAR + NDWI', 'standard', 'SAR+NDWI')}{prow('Chuẩn · NDVI + NDWI + SAR', 'standard', 'NDVI+NDWI+SAR')}
{prow('Năm 2025 (70) · NDVI', 'year', 'NDVI', 'hl')}{prow('Năm 2025 · SAR + NDWI', 'year', 'SAR+NDWI')}{prow('Năm 2025 · NDVI + NDWI + SAR', 'year', 'NDVI+NDWI+SAR')}</table>
<div class=grid style="grid-template-columns:1fr 1fr 1fr;margin-top:18px">
<div class=card><span class=tag>Kiểm thử năm 2025</span><p>Huấn luyện 2023–2024, kiểm thử 2025 trên các ô lưới của tập test, không chung thửa hay ô với tập train. Kết quả giữ vững: NDVI {f3(PB['year']['NDVI']['mean'])}.</p></div>
<div class=card><span class=tag>NDVI so với SAR + NDWI</span><p>NDVI hơn {f3(dY['mean'])} điểm macro-F1, CI [{f3(dY['lo'])}; {f3(dY['hi'])}] không chứa 0. Chênh lệch này có thật trên nhãn dẫn xuất từ NDVI.</p></div>
<div class=card style="background:#FBF0DC;border-color:#E2C48C"><span class="tag a">Lưu ý</span><p>Chỉ 70 cửa sổ năm 2025 nên CI rộng. Vụ gán theo ngày lịch (Đông Xuân: SOS tháng 11–3, EOS tháng 2–5; Hè Thu: SOS tháng 4–8, EOS tháng 6–10), là phân loại gần đúng.</p></div></div>""", sec=P3)

# 20 Causal
cz, bl = CZ["LSTM_causal"], CZ["BiLSTM"]
g_ = lambda L_, f: ms([f(r) for r in L_])
def crow(nm, L_, cls=""):
    return (f"<tr class={cls}><td>{nm}</td><td class=n>{pm(g_(L_, lambda r: r['macro_f1']))}</td><td class=n>{pm(g_(L_, lambda r: r['f1'][1]), 3)}</td><td class=n>{pm(g_(L_, lambda r: r['f1'][2]), 3)}</td>"
            f"<td class=n>{f1(g_(L_, lambda r: r['dates']['sos']['mae_days'])[0])} / {f1(g_(L_, lambda r: r['dates']['eos']['mae_days'])[0])}</td>"
            f"<td class=n>{f1(g_(L_, lambda r: r['delay']['sos']['mean'])[0])} / {f1(g_(L_, lambda r: r['delay']['eos']['mean'])[0])}</td></tr>")
drop = ms([a["macro_f1"] for a in bl])[0] - ms([a["macro_f1"] for a in cz])[0]
add(head("Nhận diện trực tuyến: chỉ nhìn quá khứ", "LSTM một chiều gán nhãn bước t chỉ từ các bước 0..t; so với BiLSTM cùng cấu hình, 3 seed") + f"""
<table><tr><th>Mô hình</th><th class=n>Macro-F1</th><th class=n>F1 Sinh trưởng</th><th class=n>F1 Sinh sản</th><th class=n>MAE SOS / EOS (ngày)</th><th class=n>Độ trễ TB SOS / EOS (ngày)</th></tr>
{crow('LSTM nhân quả (chỉ quá khứ)', cz, 'hl')}{crow('BiLSTM (thấy cả tương lai)', bl)}</table>
<div class=grid style="grid-template-columns:1fr 1fr;margin-top:26px"><div class=card><span class=tag>Đọc kết quả</span><p>Bỏ phần tương lai làm macro-F1 giảm khoảng <b>{f1(drop*100)} điểm</b>; sai số EOS tăng từ {f1(g_(bl, lambda r: r['dates']['eos']['mae_days'])[0])} lên {f1(g_(cz, lambda r: r['dates']['eos']['mae_days'])[0])} ngày vì chưa thấy NDVI giảm, còn SOS gần như không đổi. Nhận diện giai đoạn hiện tại khi chỉ nhìn quá khứ vẫn khả thi.</p></div>
<div class=card style="background:#FBF0DC;border-color:#E2C48C"><span class="tag a">Chưa phải dự báo ngày thu hoạch</span><p>Đây mới là nhận diện giai đoạn hiện tại. Dự báo EOS từ k bước đầu mùa là việc tiếp theo. Độ trễ là sai số có dấu (dương = muộn).</p></div></div>""", sec=P3)

# 21 SICKLE
def srow(lbl, sub, tgt, cls=""):
    r = SK[sub][tgt]; b, g, d = r["season_mean_baseline"], r["S1_gradient_boosting"], r["mae_gain_baseline_minus_S1"]
    return (f"<tr class={cls}><td>{lbl}</td><td class=n>{r['n']} / {r['n_plots']}</td><td class=n>{f1(b['mae'])}</td><td class=n>{f1(g['mae'])}</td>"
            f"<td class=n>{f1(d['mean'])} [{f1(d['lo'])}; {f1(d['hi'])}]</td><td class=n>{pct(b['hit7'],0)} → {pct(g['hit7'],0)}</td></tr>")
nh = SK["all_rows"]["HARVESTING_DAY"]["n"]; ni = SK["informative_rows"]["HARVESTING_DAY"]["n"]; nsw = SK["informative_rows"]["SOWING_DAY"]["n"]
add(head("Kiểm chứng ngoài bằng SICKLE: ground truth thật", "Lúa Ấn Độ, Sentinel-1 VV/VH, ngày do nông dân khai; chỉ dùng train/val, chia 5 phần theo thửa") + f"""
<table class=cmp style="font-size:19px"><tr><th>Mục tiêu · tập dòng</th><th class=n>Dòng / thửa</th><th class=n>MAE lịch vụ (ngày)</th><th class=n>MAE S1 (ngày)</th><th class=n>Cải thiện [CI 95%]</th><th class=n>Trong ±7 ngày</th></tr>
{srow('Ngày thu hoạch · mọi dòng', 'all_rows', 'HARVESTING_DAY')}{srow('Ngày thu hoạch · dòng có thông tin', 'informative_rows', 'HARVESTING_DAY', 'hl')}
{srow('Ngày sạ · mọi dòng', 'all_rows', 'SOWING_DAY')}{srow('Ngày sạ · dòng có thông tin', 'informative_rows', 'SOWING_DAY', 'hl')}</table>
<div class=grid style="grid-template-columns:1fr 1fr;margin-top:20px">
<div class=card style="background:#FBF0DC;border-color:#E2C48C"><span class="tag a">Vì sao tách hai tập dòng</span><p>Trên mọi dòng, S1 hơn lịch vụ khoảng 2 ngày. Nhưng {pct((nh-ni)/nh,0)} dòng thu hoạch và {pct((nh-nsw)/nh,0)} dòng ngày sạ nằm đúng mốc biên mùa (ngày sạ là ngày 1, ngày thu hoạch là ngày cuối vụ chuẩn); vụ apr-aug có cả 48 dòng cùng một giá trị. Đó là dấu hiệu giá trị mặc định, không phải ngày thật.</p></div>
<div class=card><span class=tag>Kết quả trên dòng có thông tin</span><p>S1 với gradient boosting đơn giản <b>không hơn</b> lịch vụ trung bình: cải thiện gần 0 và CI chứa 0. Chưa chứng minh được SAR dự đoán ngày thật tốt hơn lịch mùa vụ; lịch mùa vụ là baseline rất mạnh vì ngày trong một vụ dồn chặt.</p></div></div>
<p style="font-size:17px;color:#76827A;margin-top:10px">Mô hình đơn giản (chuỗi VV, VH nội suy 6 ngày, đặc trưng thời điểm cực tiểu và cực đại VH); chưa tối ưu. Dữ liệu SICKLE giữ cục bộ, chỉ báo cáo thống kê tổng hợp.</p>""", sec=P3)

# 22 Task A: rice vs non-rice (Earth Engine, Kaggle)
TA = json.load(open(f"{O}/taskA_results.json"))
def tarow(lbl, key, cls=""):
    r = TA[key]; f = r["folds"]; w = r["correct_nonrice_by_worldcover_class"]
    return (f"<tr class={cls}><td>{lbl}</td><td class=n>{pm(f['macro_f1'])}</td><td class=n>{pct(f['prec_rice'][0],1)} / {pct(f['rec_rice'][0],1)}</td>"
            f"<td class=n>{pct(w['40'],0)}</td><td class=n>{pct(w['10'],0)} / {pct(w['30'],0)} / {pct(w['50'],0)} / {pct(w['80'],0)}</td></tr>")
add(head("Lúa so với cây khác: dữ liệu Earth Engine, nhiều nguồn", "1.828 cửa sổ lúa, 2.058 cửa sổ không phải lúa (WorldCover); chia 5 phần theo ô lưới 2 km") + f"""
<table class=cmp style="font-size:18.5px"><tr><th>Đầu vào · mô hình</th><th class=n>Macro-F1</th><th class=n>Precision / Recall lúa</th><th class=n>Đúng: cây khác</th><th class=n>Đúng: rừng / cỏ / đô thị / nước</th></tr>
{tarow('NDVI + NDWI · Random Forest', 'Quang hoc (NDVI+NDWI) | RandomForest')}{tarow('NDVI + NDWI · 1D-CNN', 'Quang hoc (NDVI+NDWI) | CNN')}
{tarow('Chỉ SAR · Random Forest', 'SAR (VV,VH) | RandomForest', 'hl')}{tarow('Chỉ SAR · 1D-CNN', 'SAR (VV,VH) | CNN', 'hl')}
{tarow('Cả bốn kênh · Random Forest', 'Tat ca (NDVI,NDWI,VV,VH) | RandomForest')}{tarow('Cả bốn kênh · 1D-CNN', 'Tat ca (NDVI,NDWI,VV,VH) | CNN')}</table>
<div class=grid style="grid-template-columns:1fr 1fr;margin-top:18px">
<div class=card style="background:#FBF0DC;border-color:#E2C48C"><span class="tag a">Con số quang học không dùng để so sánh với SAR</span><p>Điểm lúa được chọn bằng luật ngưỡng trên NDVI và NDWI, còn điểm cây trồng khác là các pixel không qua luật đó. Phân biệt chúng bằng NDVI là gần như đã biết trước ({pct(TA['Quang hoc (NDVI+NDWI) | CNN']['folds']['macro_f1'][0],1)} phản ánh điều này). <b>Chỉ SAR</b> không dùng kênh nào của luật nên là phép thử công bằng hơn.</p></div>
<div class=card><span class=tag>Đọc kết quả chỉ-SAR</span><p>SAR tách lúa khỏi rừng, đô thị, nước gần như hoàn toàn (96–100%), nhưng chỉ nhận đúng <b>{pct(TA['SAR (VV,VH) | CNN']['correct_nonrice_by_worldcover_class']['40'],0)}</b> cây trồng khác: SAR chưa phân biệt được lúa với hoa màu khác. Đây là phần khó thật của bài toán.</p></div></div>
<p style="font-size:17px;color:#76827A;margin-top:10px">Nhãn lúa là nhãn yếu (luật trên ảnh vệ tinh), chưa xác nhận ngoài đồng. Lớp không phải lúa lấy từ ESA WorldCover 2021.</p>""", sec=P3)

# 23 Limits & next
add(head("Hạn chế và bước tiếp theo", "Những điều chưa được phép khẳng định, và dữ liệu cần thêm") + """
<div class=grid style="grid-template-columns:1fr 1fr;gap:26px">
<div class=card style="border-top:7px solid #C9653F"><h3>Chưa khẳng định được</h3><ul style="font-size:19px;line-height:1.3"><li>Độ chính xác so với ngày gieo, thu hoạch ngoài đồng: không có ground truth thực địa ở Việt Nam.</li><li>Mô hình tốt hơn SOTA: không có SOTA cùng vùng và cùng nhãn để so.</li><li>SAR dự đoán ngày thật tốt hơn lịch mùa vụ: trên SICKLE (dòng có thông tin) chưa thấy cải thiện.</li><li>Hơn kém giữa các mô hình sâu trong biên ±10 ngày của nhãn.</li><li>Hiệu năng khi mây dày: chuỗi đã nội suy khoảng 33% bước và bỏ cửa sổ thiếu quá 50%, nên mùa mưa bị đánh giá nhẹ hơn thực tế.</li><li>Rò rỉ nhỏ: 3 ô lưới, 70 cửa sổ nằm ở nhiều tập; 5 seed dùng chung split.</li></ul></div>
<div class=card style="border-top:7px solid #5FA86B"><h3>Bước tiếp theo và dữ liệu cần thêm</h3><ul style="font-size:19px;line-height:1.3"><li><b>Điểm lúa và cây trồng khác xác nhận ngoài đồng</b>: bài lúa so với cây khác mới dùng nhãn yếu; SAR nhầm gần một nửa cây trồng khác.</li><li><b>Nhãn mắt</b> 150 mẫu, 3 người chấm, tính kappa (Google Earth).</li><li><b>Ground truth ngoài đồng</b>: ngày sạ, thu hoạch thật của 100–150 thửa từ hợp tác xã hoặc Sở Nông nghiệp; kiểm tra ngày mặc định.</li><li>Thêm kênh mặt nạ quan sát làm đầu vào; dự báo sớm EOS từ k bước đầu mùa.</li><li>Chia theo huyện để kiểm tra khái quát hóa không gian.</li></ul></div></div>
<div class=note style="margin-top:14px;font-size:19px">Nếu chưa có ground truth ngoài đồng, kết quả cần gọi đúng tên: <b>dự đoán nhãn vật hậu yếu suy ra từ NDVI bằng chuỗi Sentinel-1 SAR</b>.</div>""", sec=P3)

# 23 Thanks
add("""<div style="padding:200px 0 0 20px"><div style="font-size:78px;font-weight:800">Cảm ơn thầy và các bạn</div><p style="font-size:28px;color:#D3E0D8;margin-top:22px">Câu hỏi và góp ý</p></div>""", "dark")

html = f"<!doctype html><html lang=vi><head><meta charset=utf-8><title>Rice Phenology - Do an mon hoc</title><style>{CSS}</style></head><body>{''.join(slides)}</body></html>"
open(f"{SL}/deck.html", "w", encoding="utf-8").write(html)
pdf = os.path.join(SL, "Slide_giua_ky_Rice_Phenology.pdf")
edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", "file:///" + f"{SL}/deck.html".replace("\\", "/")], check=True, timeout=240)
print("slides", len(slides), "->", pdf)
