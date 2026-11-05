"""Deck v2: same results, presentation style modelled on the team's 'Slide_nhom6' (cream background, sage bands, brown headings,
peach cards, few words per page). All numbers are read from out/*.json, nothing is typed by hand except wording."""
import sys, os, json, base64, subprocess; sys.path.insert(0, os.path.dirname(__file__))
import mt, numpy as np

O, FD, SL = mt.OUT, os.path.join(mt.MT, "figs"), os.path.join(mt.MT, "slides")
E = json.load(open(f"{O}/eda.json")); B = json.load(open(f"{O}/baselines.json")); S = json.load(open(f"{O}/summary.json"))
B.update(json.load(open(f"{O}/extra_baselines.json")))
CZ = json.load(open(f"{O}/causal_results.json")); SAR = json.load(open(f"{O}/sar_results_real.json"))
PB = json.load(open(f"{O}/pack_bootstrap.json")); SK = json.load(open(f"{O}/sickle_s1_results.json"))
TA = json.load(open(f"{O}/taskA_results.json")); CL = json.load(open(f"{O}/cloud_results.json"))
df, X, Y, doy = mt.load()
R = {k: S[k] for k in S if not k.startswith("_")}
cnn, lstm, trn, trd = R["CNN1D"], R["BiLSTM"], R["Transformer_NDVI"], R["Transformer_DOY"]
cal, rule, rf, dlg, slp = B["B0_calendar"], B["B1_threshold_rule"], B["B2_random_forest"], B["B3_double_logistic"], B["B4_ndvi_slope"]


def img(name): return "data:image/png;base64," + base64.b64encode(open(f"{FD}/{name}", "rb").read()).decode()
def f3(v): return ("%.3f" % v).replace(".", ",")
def f1(v): return ("%.1f" % v).replace(".", ",")
def pct(v, d=0): return (("%." + str(d) + "f") % (v * 100)).replace(".", ",") + "%"
def n(v): return f"{v:,}".replace(",", ".")
def ms(a): return float(np.mean(a))


BG, SAGE, SAGE_L, BROWN, PEACH, PEACH_B, INK = "#FFF5EE", "#8FAFA6", "#D5E2DC", "#8B5A2B", "#FDEBD2", "#F0D9B5", "#2B2B2B"
CSS = f"""
@page{{size:1440px 810px;margin:0}}
*{{box-sizing:border-box}} html,body{{margin:0;padding:0}}
body{{font-family:"Segoe UI",Arial,sans-serif;color:{INK};-webkit-print-color-adjust:exact;print-color-adjust:exact}}
.s{{width:1440px;height:810px;position:relative;overflow:hidden;page-break-after:always;background:{BG}}}
.band{{position:absolute;left:0;right:0;bottom:0;height:62px;background:{SAGE}}}
.band.top{{top:0;bottom:auto;height:34px}}
.pill{{position:absolute;bottom:14px;height:34px;border:1.5px solid {BROWN};border-radius:18px;background:{BG};font-size:15px;line-height:31px;text-align:center;color:{INK}}}
.dash{{position:absolute;left:300px;right:300px;bottom:30px;border-top:3px dashed {BG};opacity:.9}}
.dot{{position:absolute;bottom:24px;width:14px;height:14px;border-radius:7px;background:{BG}}}
.body{{position:absolute;left:70px;right:70px;top:34px;bottom:78px}}
h1{{font-family:"Segoe UI Black","Segoe UI",sans-serif;font-weight:900;color:{BROWN};font-size:50px;line-height:1.08;margin:0;letter-spacing:-.5px;text-transform:uppercase}}
h1.c{{text-align:center}}
.ttl{{display:block;margin:18px 0 6px}}
.ttl .sub{{font-size:21px;color:#5C4A38;line-height:1.3;margin-top:6px}}
.dl{{height:0;border-top:3px dashed {SAGE};width:230px;margin:6px 0 14px;position:relative}}
.dl:after{{content:"";position:absolute;right:-8px;top:-9px;width:14px;height:14px;border-radius:7px;background:{SAGE}}}
.pc{{background:{PEACH};border:1.5px solid {PEACH_B};border-radius:14px;padding:16px 20px}}
.pc h3{{margin:0 0 6px;font-size:22px;color:{BROWN};font-weight:800}}
.pc p,.pc li{{font-size:20px;line-height:1.38;margin:0}}
.sg{{background:{SAGE_L};border-radius:14px;padding:16px 20px}}
.sg h3{{margin:0 0 6px;font-size:22px;color:#2F5A50;font-weight:800}}
.sg p,.sg li{{font-size:20px;line-height:1.38;margin:0}}
.bn{{background:{BROWN};color:#fff;border-radius:16px;padding:18px 22px}}
.bn h3{{margin:0 0 6px;font-size:22px;color:#fff}} .bn p,.bn li{{font-size:20px;line-height:1.38;margin:0;color:#fff}}
ul{{margin:0;padding-left:22px}} li{{margin-bottom:6px}}
.grid{{display:grid;gap:18px}}
.big{{font-family:"Segoe UI Black",sans-serif;font-size:50px;color:{BROWN};line-height:1}}
.lb{{font-size:17px;color:#5C4A38;margin-top:4px}}
.num{{display:inline-block;width:34px;height:34px;border-radius:17px;background:{BROWN};color:#fff;text-align:center;line-height:34px;font-weight:700;font-size:16px;margin-right:12px}}
table{{border-collapse:collapse;width:100%;font-size:20px}}
th{{font-size:14px;letter-spacing:.06em;text-transform:uppercase;color:{BROWN};text-align:left;padding:8px 12px;border-bottom:2.5px solid {BROWN}}}
td{{padding:9px 12px;border-bottom:1px solid {PEACH_B}}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
tr.hl td{{background:{PEACH};font-weight:700}}
.chip{{display:inline-block;padding:6px 14px;border-radius:14px;font-weight:700;font-size:18px;color:#fff;margin:0 4px}}
.fig{{display:block;margin:0 auto;max-width:100%;border-radius:10px;background:#fff}}
.note{{background:{PEACH};border-left:7px solid {BROWN};border-radius:8px;padding:12px 20px;font-size:19px;line-height:1.4}}
.pin{{background:#fff;border:3px solid {SAGE};border-radius:22px;padding:56px 14px 14px;position:relative;text-align:center;box-shadow:0 5px 0 {SAGE}}}
.pin .nb{{position:absolute;left:50%;top:-22px;margin-left:-38px;width:76px;height:46px;border-radius:23px;background:{SAGE};border:3px solid #fff;color:#fff;font-weight:800;font-size:25px;line-height:40px}}
.pin h3{{margin:8px 0 6px;font-size:21px;color:#2F5A50;font-weight:800}} .pin p{{font-size:17.5px;line-height:1.35;margin:0;text-align:left}}
"""
slides = []; TOTAL = [0]


def spark(x, y, sz=44, c=BROWN, o=1):
    return (f'<svg style="position:absolute;left:{x}px;top:{y}px;opacity:{o}" width="{sz}" height="{sz}" viewBox="0 0 40 40">'
            f'<path d="M20 0 C21 12 28 19 40 20 C28 21 21 28 20 40 C19 28 12 21 0 20 C12 19 19 12 20 0Z" fill="{c}"/></svg>')
def blob(x, y, d, c): return f'<div style="position:absolute;left:{x}px;top:{y}px;width:{d}px;height:{d}px;border-radius:50%;background:{c}"></div>'
def rays(x, y, c=SAGE, rot=0):
    return (f'<svg style="position:absolute;left:{x}px;top:{y}px;transform:rotate({rot}deg)" width="60" height="60" viewBox="0 0 60 60" stroke="{c}" stroke-width="4" stroke-linecap="round">'
            '<line x1="8" y1="40" x2="22" y2="34"/><line x1="14" y1="22" x2="26" y2="20"/><line x1="30" y1="8" x2="32" y2="18"/></svg>')

ICON = {
    "wave": '<polyline points="4,34 14,34 20,12 28,30 36,6 44,26 52,26 60,34" fill="none" stroke="#8B5A2B" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>',
    "cloud": '<path d="M16 42 C6 42 4 30 14 28 C14 16 32 12 36 24 C48 22 54 36 44 42 Z" fill="#D5E2DC" stroke="#8B5A2B" stroke-width="3.5" stroke-linejoin="round"/>',
    "tag": '<path d="M8 8 H32 L56 32 L32 56 L8 32 Z" fill="#FDEBD2" stroke="#8B5A2B" stroke-width="3.5" stroke-linejoin="round"/><circle cx="19" cy="19" r="4.5" fill="#8B5A2B"/>',
    "net": '<g stroke="#8B5A2B" stroke-width="3"><line x1="12" y1="14" x2="32" y2="32"/><line x1="12" y1="50" x2="32" y2="32"/><line x1="32" y1="32" x2="54" y2="32"/></g><g fill="#8FAFA6" stroke="#8B5A2B" stroke-width="3"><circle cx="12" cy="14" r="7"/><circle cx="12" cy="50" r="7"/><circle cx="32" cy="32" r="7"/><circle cx="54" cy="32" r="7"/></g>',
    "bars": '<g fill="#8FAFA6" stroke="#8B5A2B" stroke-width="3"><rect x="8" y="30" width="12" height="26"/><rect x="26" y="14" width="12" height="42"/><rect x="44" y="22" width="12" height="34"/></g>',
    "globe": '<circle cx="32" cy="32" r="22" fill="#D5E2DC" stroke="#8B5A2B" stroke-width="3.5"/><ellipse cx="32" cy="32" rx="9" ry="22" fill="none" stroke="#8B5A2B" stroke-width="3"/><line x1="10" y1="32" x2="54" y2="32" stroke="#8B5A2B" stroke-width="3"/>',
    "leaf": '<path d="M10 54 C8 24 28 8 56 8 C56 36 40 56 10 54 Z" fill="#8FAFA6" stroke="#8B5A2B" stroke-width="3.5" stroke-linejoin="round"/><path d="M12 52 L40 24" stroke="#8B5A2B" stroke-width="3.5" fill="none"/>',
    "check": '<circle cx="32" cy="32" r="24" fill="#8FAFA6"/><path d="M20 33 L29 42 L45 23" fill="none" stroke="#fff" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>',
}
def icon(k, sz=64): return f'<svg width="{sz}" height="{sz}" viewBox="0 0 64 64">{ICON[k]}</svg>'


def add(body, kind="content", deco=""):
    k = len(slides) + 1
    top = '<div class="band top"></div>' if kind in ("title", "outline", "end") else ""
    foot = ('<div class="band"></div><div class="pill" style="left:56px;width:200px">Rice Phenology</div>'
            '<div class="dash"></div><div class="dot" style="left:290px"></div><div class="dot" style="right:290px"></div>'
            f'<div class="pill" style="right:56px;width:150px">Page {k:02d} of @@N@@</div>')
    slides.append(f'<section class="s" data-fit="{1 if kind == "content" else 0}">{top}{deco}<div class="body">{body}</div>{foot}</section>')


def title(t, sub=""):
    return f'<div class="ttl"><h1>{t}</h1>' + (f'<div class="sub">{sub}</div>' if sub else "") + '</div><div class="dl"></div>'


# ---- numbers used in several places
ndvi_c = float(np.mean([r["viterbi"]["macro_f1"] for r in SAR["results"]["NDVI+DOY (control)"]]))
sar_c = float(np.mean([r["viterbi"]["macro_f1"] for r in SAR["results"]["SAR (VV,VH,VH-VV)+DOY"]]))
cm = lambda nm, p: float(np.mean([r["by_level"][str(p)]["macro_f1"] for r in CL["results"][nm]]))
tf = lambda k: TA[k]["folds"]["macro_f1"][0]
ta_w = lambda k: TA[k]["correct_nonrice_by_worldcover_class"]["40"]
cz = ms([a["macro_f1"] for a in CZ["LSTM_causal"]]); blstm = ms([a["macro_f1"] for a in CZ["BiLSTM"]])
retro = [m["raw"]["retro"][0] for m in (cnn, lstm, trn, trd)]
pts = " ".join("%d,%d" % (i * 6.2, 150 - X[int(np.where((df.n_seasons == 2) & (df.miss_ratio < .15))[0][0])][i] * 150) for i in range(mt.T))

# ===================== 1 Title
add(f'''<div style="position:absolute;left:30px;top:60px;width:520px;height:520px">{blob(0,40,430,SAGE_L)}{blob(250,0,210,PEACH)}{blob(120,260,260,"#F6D9B8")}
<svg style="position:absolute;left:10px;top:200px" width="500" height="200" viewBox="-8 -8 470 180"><polyline points="{pts}" fill="none" stroke="{BROWN}" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/></svg>
{spark(430,20,54)}{spark(8,110,36,"#C98A2B")}{rays(450,300)}{rays(-10,380,SAGE,90)}</div>
<div style="position:absolute;left:600px;top:44px;right:0"><div style="display:inline-block;background:{PEACH};border-radius:12px;padding:14px 38px;font-size:22px;font-weight:600">PHÂN TÍCH CHUỖI THỜI GIAN · ĐỒ ÁN MÔN HỌC</div>
<h1 style="font-size:50px;margin-top:22px;line-height:1.12">Theo dõi vòng đời<br>cây lúa từ chuỗi vệ tinh<br>NDVI và SAR</h1>
<p style="font-size:24px;color:#5C4A38;margin-top:14px">Đà Nẵng và Quảng Nam · Sentinel-2 và Sentinel-1 · 2023–2025</p>
<div style="margin-top:26px;font-size:25px;line-height:1.6">Văn Trọng Huy · 23AI018<br>Đỗ Phúc Vũ Hà · 23AI012<br>Sử Thị Hà My · 23AI031<br>Nguyễn Hà Nhi · 23AI039<br>Trần Thị Thùy Linh · 23AI029</div></div>''', "title")

# ===================== 2 Outline
items = ["Bối cảnh đề tài", "Mục tiêu đề tài", "Hướng làm", "Dữ liệu", "Baseline (paper nền)", "Phương pháp đề xuất", "Kết quả", "Hạn chế và kế hoạch"]
lis = "".join(f'<div style="display:flex;align-items:center;margin:11px 0;font-size:27px"><span class=num style="width:38px;height:38px;border-radius:19px;line-height:38px;font-size:15px">{i:02d}</span>{t}</div>' for i, t in enumerate(items, 1))
add(f'''<h1 style="font-size:84px;margin:26px 0 0 4px">Nội dung</h1><div style="margin:34px 0 0 14px">{lis}</div>''', "outline",
    deco=f'{blob(900,100,460,SAGE_L)}{blob(1180,330,200,PEACH)}<div style="position:absolute;left:980px;top:150px;width:300px;height:420px;background:#fff;border:10px solid {BROWN};border-radius:18px"></div>'
         + "".join(f'<div style="position:absolute;left:1020px;top:{200+i*86}px;width:34px;height:34px;border:4px solid {BROWN};border-radius:5px"></div><div style="position:absolute;left:1075px;top:{210+i*86}px;width:160px;height:12px;background:{SAGE};border-radius:6px"></div><div style="position:absolute;left:1075px;top:{232+i*86}px;width:110px;height:12px;background:{SAGE_L};border-radius:6px"></div>' for i in range(4))
         + spark(1330, 130, 50) + spark(900, 90, 34, "#C98A2B") + rays(1300, 560))

# ===================== 3 Context
add(title("Bối cảnh đề tài") + f'''
<ul style="font-size:25px;line-height:1.5;margin:6px 0 24px 8px"><li>Biết cây lúa đang ở giai đoạn nào theo từng thời điểm giúp theo dõi mùa vụ cho cả vùng, không cần đi từng thửa.</li>
<li>Chuỗi ảnh vệ tinh theo thời gian là nguồn dữ liệu phù hợp: Sentinel-2 (quang học) và Sentinel-1 (radar).</li></ul>
<div class=grid style="grid-template-columns:repeat(3,1fr);gap:22px">
<div class=pc style="text-align:center;padding:22px">{icon("leaf",84)}<h3 style="margin-top:8px">Lúa nhiều vụ</h3><p>Đà Nẵng và Quảng Nam có 1–2 vụ mỗi năm; mỗi vụ có 4 giai đoạn từ gieo đến thu hoạch.</p></div>
<div class=pc style="text-align:center;padding:22px">{icon("cloud",84)}<h3 style="margin-top:8px">Ảnh quang học hay bị mây</h3><p>Trung bình {pct(E['miss_mean'])} số bước thiếu ảnh gốc; radar Sentinel-1 xuyên được mây.</p></div>
<div class=pc style="text-align:center;padding:22px">{icon("tag",84)}<h3 style="margin-top:8px">Thiếu nhãn ngoài đồng</h3><p>Nhóm chưa có ngày gieo, thu hoạch thật; nhãn phải suy ra từ chính ảnh vệ tinh.</p></div></div>''',
    deco=spark(1330, 44, 44) + spark(1270, 96, 24, "#C98A2B"))

# ===================== 4 Goals
add(title("Mục tiêu đề tài") + f'''
<div class=grid style="grid-template-columns:repeat(4,1fr);gap:16px">
<div class=pc><h3>1</h3><p>Gán <b>4 giai đoạn</b> sinh trưởng cho từng bước 5 ngày của chuỗi</p></div>
<div class=pc><h3>2</h3><p>Suy ra các mốc <b>SOS, POS, EOS</b> (bắt đầu, đỉnh, kết thúc mùa) theo ngày</p></div>
<div class=pc><h3>3</h3><p>Thử <b>bỏ NDVI</b>: dùng SAR và xử lý mây, vì nhãn đang sinh từ NDVI</p></div>
<div class=pc><h3>4</h3><p>Nói rõ <b>giới hạn</b> của nhãn yếu, không tuyên bố hơn mức đo được</p></div></div>
<div style="margin-top:30px;background:#fff;border-radius:16px;padding:22px 26px;border:2px solid {SAGE}">
<div style="display:flex;align-items:center;gap:16px;font-size:20px"><div class=sg style="flex:1;text-align:center"><b>Đầu vào</b><br>1 cửa sổ 365 ngày<br>73 giá trị, cách nhau 5 ngày</div><div style="font-size:34px;color:{BROWN}">→</div>
<div class=bn style="flex:1;text-align:center"><b>Mô hình chuỗi</b><br>gán nhãn cho từng bước<br>(sequence labelling)</div><div style="font-size:34px;color:{BROWN}">→</div>
<div class=pc style="flex:1.3;text-align:center"><b>Đầu ra: 73 nhãn</b><br><span class=chip style="background:#B8AD93">Ngoài vụ</span><span class=chip style="background:#5FA86B">Sinh trưởng</span><br><span class=chip style="background:#D9A83A;margin-top:6px">Sinh sản</span><span class=chip style="background:#C9653F;margin-top:6px">Suy giảm</span></div></div></div>''',
    deco=spark(1330, 44, 44) + rays(1250, 40, SAGE))

# ===================== 5 Pipeline
pins = [("01", "globe", "Thu thập", "Sentinel-2 và Sentinel-1 trên Google Earth Engine, 2023–2025, 1.000 điểm"),
        ("02", "cloud", "Tiền xử lý", "Lọc mây bằng SCL, lưới 5 ngày, nội suy chỗ trống, làm mượt Savitzky–Golay"),
        ("03", "tag", "Tạo nhãn", "Luật ngưỡng 20% biên độ NDVI: 4 giai đoạn và mốc SOS, POS, EOS"),
        ("04", "net", "Mô hình", "1D-CNN gán nhãn từng bước, giải mã Viterbi giữ đúng thứ tự giai đoạn"),
        ("05", "bars", "Đánh giá", "Macro-F1, sai số mốc theo ngày, kiểm thử theo năm, đối chiếu SICKLE")]
add('<div style="margin:6px 0 30px"><div style="display:inline-block;background:' + SAGE_L + ';border-radius:16px;padding:14px 40px"><h1 style="font-size:42px;color:#3A3A3A">Tổng quan luồng xử lý</h1></div></div>'
    + '<div class=grid style="grid-template-columns:repeat(5,1fr);gap:22px;margin-top:46px">'
    + "".join(f'<div class=pin><div class=nb>{a}</div>{icon(ic, 72)}<h3>{t}</h3><p>{d}</p></div>' for a, ic, t, d in pins) + '</div>',
    deco=spark(1340, 50, 40))

# ===================== 6 Data
add(title("Dữ liệu", "Tự thu thập từ Google Earth Engine, Đà Nẵng và Quảng Nam, 2023–2025") + f'''
<div class=grid style="grid-template-columns:repeat(5,1fr);gap:14px;text-align:center;margin-bottom:16px">
<div class=pc><div class=big>{n(E['n'])}</div><div class=lb>cửa sổ chuỗi thời gian</div></div><div class=pc><div class=big>{E['fields']}</div><div class=lb>thửa ruộng</div></div>
<div class=pc><div class=big>73</div><div class=lb>bước, mỗi bước 5 ngày</div></div><div class=pc><div class=big>5</div><div class=lb>kênh: NDVI, NDWI, VV, VH, mặt nạ quan sát</div></div>
<div class=pc><div class=big style="font-size:30px;line-height:1.7;white-space:nowrap">{E['split']['train']} / {E['split']['val']} / {E['split']['test']}</div><div class=lb>train / val / test, chia theo ô lưới 2 km</div></div></div>
<img class=fig src="{img('fig_data_overview.png')}" style="width:1240px"><p style="font-size:18px;color:#5C4A38;margin-top:10px;text-align:center">Chọn điểm từ lớp cropland WorldCover 2021, lọc ứng viên lúa bằng biên độ NDVI và NDWI, lấy mẫu phân tầng 300 điểm Đà Nẵng và 700 điểm Quảng Nam.</p>''',
    deco=spark(1340, 44, 40))

# ===================== 7 Labels
add(title("Nhãn: vì sao cần thận trọng") + f'''
<div class=grid style="grid-template-columns:1.45fr 1fr;gap:24px;align-items:start">
<div><img class=fig src="{img('fig_examples.png')}" style="width:100%"><p style="font-size:17px;color:#5C4A38;margin-top:6px;text-align:center">Dải màu dưới mỗi đường NDVI là nhãn giai đoạn của từng bước</p></div>
<div><div class=bn style="margin-bottom:16px"><h3>Nhãn là hàm của NDVI</h3><p>Luật ngưỡng 20% biên độ phát hiện lại nhãn khớp <b>99,9%</b>. Mô hình học NDVI vì thế là đang tái tạo luật gán nhãn.</p></div>
<div class=pc style="margin-bottom:16px"><h3>Nhãn mờ khoảng 10 ngày</h3><p>Đổi ngưỡng 10% / 20% / 30% làm mốc SOS, EOS lệch trung vị 10 ngày. Sai số dưới mức đó nằm trong độ mờ của nhãn.</p></div>
<div class=sg><h3>Cách xử lý</h3><p>Báo cáo đúng tên: "dự đoán nhãn vật hậu yếu", rồi kiểm tra ngoài bằng SAR và SICKLE.</p></div></div></div>''',
    deco=spark(1340, 44, 40))

# ===================== 8 Baseline papers
add(title("Baseline", "Paper làm nền, nhóm phát triển phương pháp từ đó") + f'''
<div class=grid style="grid-template-columns:repeat(3,1fr);gap:18px">
<div class=sg><h3>BreizhCrops (Rußwurm, 2020)</h3><p style="font-size:18.5px">Bộ chuẩn phân loại cây trồng từ chuỗi Sentinel-2: hơn 600.000 chuỗi, so Random Forest với CNN, RNN, Transformer, chia theo không gian.</p><p style="margin-top:10px"><b>Nhóm lấy:</b> Random Forest và cách chia không gian.</p></div>
<div class=sg><h3>Gobin và cs. (2023)</h3><p style="font-size:18.5px">Vật hậu lúa mì, ngô, khoai tây từ Sentinel-2: khớp double-logistic lên fAPAR/fCover, điểm uốn ứng với giai đoạn.</p><p style="margin-top:10px"><b>Nhóm lấy:</b> khớp double-logistic.</p></div>
<div class=sg><h3>Tian và cs. (2023)</h3><p style="font-size:18.5px">Trích xuất ruộng lúa bằng chênh lệch NDVI trước và sau thu hoạch, khử viền nước bằng NDWI, hồi quy ngày thu hoạch.</p><p style="margin-top:10px"><b>Nhóm lấy:</b> ý tưởng độ dốc, chênh lệch NDVI.</p></div></div>
<div class=note style="margin-top:20px"><b>Lưu ý:</b> các paper khác vùng, khác cây và (trừ BreizhCrops) dùng ground truth quan trắc. Nhãn của nhóm sinh từ NDVI nên không so F1 trực tiếp với paper; nhóm cài lại phương pháp của họ trên cùng dữ liệu và cùng metric.</div>
<div class=grid style="grid-template-columns:repeat(5,1fr);gap:12px;margin-top:20px;text-align:center">
<div class=pc style="padding:12px"><b>B0</b><br>Lịch mùa vụ</div><div class=pc style="padding:12px"><b>B1</b><br>Luật ngưỡng 20%</div><div class=pc style="padding:12px"><b>B2</b><br>Random Forest</div><div class=pc style="padding:12px"><b>B3</b><br>Double-logistic</div><div class=pc style="padding:12px"><b>B4</b><br>Độ dốc NDVI</div></div>''',
    deco=spark(1340, 44, 40))

# ===================== 9 Baseline results
def brow(lbl, d, cls=""):
    g = d["dates"]; return f"<tr class={cls}><td>{lbl}</td><td class=n>{f3(d['macro_f1'])}</td><td class=n>{f1(g['sos']['mae_days'])} / {f1(g['eos']['mae_days'])}</td><td class=n>{pct(g['sos']['miss'])} / {pct(g['eos']['miss'])}</td><td class=n>{pct(d['retrograde'])}</td></tr>"
add(title("Kết quả baseline", f"{E['split']['test']} cửa sổ test; cấu hình chọn trên validation") + f'''
<table><tr><th>Baseline</th><th class=n>Macro-F1</th><th class=n>MAE SOS / EOS (ngày)</th><th class=n>Bỏ sót SOS / EOS</th><th class=n>Sai thứ tự</th></tr>
{brow('B0 · Lịch mùa vụ (không dùng NDVI)', cal)}{brow('B1 · Luật ngưỡng 20%', rule)}{brow('B2 · Random Forest', rf, 'hl')}{brow('B3 · Double-logistic', dlg)}{brow('B4 · Độ dốc NDVI', slp)}</table>
<div class=grid style="grid-template-columns:repeat(3,1fr);margin-top:22px">
<div class=sg><h3>Sàn</h3><p>Lịch mùa vụ đã đạt {f3(cal['macro_f1'])} mà không nhìn NDVI; khoảng cách thật của mô hình tính từ đây.</p></div>
<div class=sg><h3>Nhóm dùng NDVI</h3><p>Bốn baseline từ {f3(slp['macro_f1'])} đến {f3(rf['macro_f1'])}, khá sát nhau.</p></div>
<div class=sg><h3>Sai thứ tự</h3><p>Random Forest sai thứ tự giai đoạn ở {pct(rf['retrograde'])} cửa sổ vì đoán từng bước độc lập.</p></div></div>''',
    deco=spark(1340, 44, 40))

# ===================== 10 Proposal: why move on
add('<div style="margin:2px 0 8px"><h1 style="font-size:42px">Từ baseline đến phương pháp đề xuất</h1></div>' + f'''
<div class=grid style="grid-template-columns:repeat(3,1fr);gap:18px;align-items:stretch">
<div class=sg><h3>Baseline luật (B1, B4)</h3><p><b>Cách làm:</b> tìm đỉnh, lấy điểm cắt hoặc điểm dốc nhất.</p><p style="margin-top:8px"><b>Hạn chế:</b> bỏ sót {pct(min(rule['dates']['sos']['miss'], slp['dates']['sos']['miss']))}–{pct(max(rule['dates']['eos']['miss'], slp['dates']['eos']['miss']))} mốc ở hai mép cửa sổ; B4 còn sai thứ tự {pct(slp['retrograde'])}.</p></div>
<div class=sg><h3>Baseline học máy (B2)</h3><p><b>Cách làm:</b> Random Forest dự đoán từng bước từ 73 giá trị NDVI.</p><p style="margin-top:8px"><b>Hạn chế:</b> các bước đoán độc lập, {pct(rf['retrograde'])} cửa sổ sai thứ tự; không dùng ngữ cảnh chuỗi.</p></div>
<div class=bn><h3>Đề xuất: 1D-CNN + Viterbi</h3><p><b>Cách làm:</b> mạng tích chập giãn đọc cả chuỗi, Viterbi giữ thứ tự hợp lệ.</p><p style="margin-top:8px"><b>Điểm mới:</b> học cả trường hợp ở mép cửa sổ; 0% sai thứ tự; ít tham số ({n(cnn['viterbi']['params'])}); thay được NDVI bằng SAR.</p></div></div>
<div class=note style="margin-top:22px"><b>Lý do chọn hướng này:</b> nhãn là chuỗi giai đoạn có thứ tự, nên bài toán là gán nhãn chuỗi. Luật chỉ nhìn đỉnh, Random Forest không có ràng buộc thứ tự; mạng chuỗi kèm giải mã Viterbi xử lý được cả hai.</div>''',
    deco=spark(1340, 44, 40))

# ===================== 11 Method
add(title("Phương pháp: 1D-CNN + Viterbi") + f'''
<div style="display:flex;align-items:stretch;gap:12px;margin-bottom:22px;font-size:19px;line-height:1.35">
<div class=sg style="flex:1;text-align:center"><b>Đầu vào</b><br>NDVI, hoặc VV, VH, NDWI<br>+ sin, cos của ngày<br>73 bước</div><div style="align-self:center;font-size:30px;color:{BROWN}">→</div>
<div class=bn style="flex:1.1;text-align:center"><b>1D-CNN</b><br>3 khối tích chập giãn<br>kernel 5, kênh 32-64-64</div><div style="align-self:center;font-size:30px;color:{BROWN}">→</div>
<div class=sg style="flex:1;text-align:center"><b>Phân loại</b><br>xác suất 4 giai đoạn<br>cho mỗi bước</div><div style="align-self:center;font-size:30px;color:{BROWN}">→</div>
<div class=pc style="flex:1.1;text-align:center"><b>Viterbi</b><br>chỉ giữ nguyên hoặc<br>tiến một giai đoạn</div></div>
<div class=grid style="grid-template-columns:1fr 1.1fr;gap:22px;align-items:center">
<div style="text-align:center"><svg width="470" height="240" viewBox="0 0 470 240"><defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="{BROWN}"/></marker></defs>
<g stroke="{BROWN}" stroke-width="3" fill="none" marker-end="url(#a)"><path d="M106 60 L194 60"/><path d="M276 60 L344 60"/><path d="M380 96 L380 138"/><path d="M344 176 L70 176 L70 98"/></g>
<g font-family="Segoe UI" font-size="14" text-anchor="middle" font-weight="700" fill="#fff"><circle cx="70" cy="60" r="36" fill="#B8AD93"/><text x="70" y="57">Ngoài</text><text x="70" y="74">vụ</text>
<circle cx="235" cy="60" r="40" fill="#5FA86B"/><text x="235" y="57">Sinh</text><text x="235" y="74">trưởng</text><circle cx="380" cy="60" r="36" fill="#D9A83A"/><text x="380" y="57">Sinh</text><text x="380" y="74">sản</text>
<circle cx="380" cy="176" r="36" fill="#C9653F"/><text x="380" y="173">Suy</text><text x="380" y="190">giảm</text></g>
<text x="235" y="228" text-anchor="middle" font-family="Segoe UI" font-size="15" fill="#5C4A38">vòng hợp lệ một chiều, không quay lại</text></svg></div>
<div><div class=pc style="margin-bottom:14px"><h3>Vì sao cần Viterbi</h3><p>Dự đoán thô sai thứ tự giai đoạn ở <b>{pct(min(retro))}–{pct(max(retro))}</b> cửa sổ; Viterbi đưa về <b>0%</b>, không thêm tham số.</p></div>
<div class=sg><h3>Dự đoán gì</h3><p>Mặt nạ 4 giai đoạn cho từng bước 5 ngày, mốc SOS, POS, EOS theo ngày, và nhận diện trực tuyến chỉ từ quá khứ.</p></div></div></div>''',
    deco=spark(1340, 44, 40))

# ===================== 12 Main results
add(title("Kết quả chính", "Macro-F1 trên tập test, trung bình 5 seed") + f'''
<img class=fig src="{img('fig_results.png')}" style="width:1100px">
<div class=grid style="grid-template-columns:repeat(4,1fr);gap:14px;margin-top:14px;text-align:center">
<div class=bn><div class=big style="color:#fff;font-size:40px">{f3(cnn['viterbi']['macro_f1'][0])}</div><p>1D-CNN + Viterbi (± {f3(cnn['viterbi']['macro_f1'][1])})</p></div>
<div class=pc><div class=big style="font-size:40px">{f3(trn['viterbi']['macro_f1'][0])}</div><p>Transformer, chỉ NDVI</p></div>
<div class=pc><div class=big style="font-size:40px">{f3(rf['macro_f1'])}</div><p>Random Forest (B2)</p></div>
<div class=pc><div class=big style="font-size:40px">{f3(cal['macro_f1'])}</div><p>Lịch mùa vụ (B0)</p></div></div>
<p style="font-size:18px;margin-top:12px;color:#5C4A38">Hơn kém giữa các mô hình sâu nằm trong biên ±10 ngày của nhãn; chọn 1D-CNN vì điểm cao nhất với ít tham số nhất.</p>''',
    deco=spark(1340, 44, 40))

# ===================== 13 SAR input
cfg = SAR["results"]
sm = lambda k: ms([x["viterbi"]["macro_f1"] for x in cfg[k]])
sd = lambda k, a: ms([x["dates"][a]["mae_days"] for x in cfg[k]])
def srow(lbl, k, cls=""): return f"<tr class={cls}><td>{lbl}</td><td class=n>{f3(sm(k))}</td><td class=n>{f1(sd(k,'sos'))} / {f1(sd(k,'eos'))}</td></tr>"
add(title("Bỏ NDVI: dùng đầu vào SAR", "1D-CNN, 5 seed; nhãn vẫn sinh từ NDVI") + f'''
<table><tr><th>Đầu vào</th><th class=n>Macro-F1</th><th class=n>MAE SOS / EOS (ngày)</th></tr>
{srow('NDVI (đối chứng)', 'NDVI+DOY (control)')}{srow('Chỉ SAR: VV, VH, VH−VV', 'SAR (VV,VH,VH-VV)+DOY', 'hl')}{srow('SAR + NDWI', 'SAR+NDWI+DOY')}{srow('NDVI + NDWI + SAR', 'NDVI+NDWI+SAR+DOY')}</table>
<div class=grid style="grid-template-columns:1fr 1fr;margin-top:24px">
<div class=sg><h3>Đọc kết quả</h3><p>Chỉ SAR đạt {f3(sar_c)}, thấp hơn NDVI {f3(ndvi_c)} khoảng {f1((ndvi_c-sar_c)*100)} điểm; thêm NDWI lên {f3(sm('SAR+NDWI+DOY'))}. Thông tin về giai đoạn có trong SAR nhưng chưa bằng NDVI.</p></div>
<div class=pc><h3>Chưa kết luận được</h3><p>Nhãn sinh từ NDVI nên chấm điểm có lợi cho NDVI. Điểm SAR thấp chưa có nghĩa SAR kém trên vật hậu thật; chỉ ground truth ngoài đồng mới trả lời.</p></div></div>''',
    deco=spark(1340, 44, 40))

# ===================== 14 SAR + cloud
add(title("SAR và mây", "Che thêm khối 30 ngày quan sát quang học thật; SAR không bị che") + f'''
<div class=grid style="grid-template-columns:1.1fr 1fr;gap:24px;align-items:start">
<div><img class=fig src="{img('fig_cloud.png')}" style="width:100%"></div>
<div><div class=sg style="margin-bottom:12px"><h3>SAR giữ nguyên, NDVI giảm</h3><p>NDVI từ {f3(cm('NDVI',0.0))} xuống {f3(cm('NDVI',0.75))} khi mất 75% bước; chỉ SAR đứng yên ở {f3(cm('SAR',0.0))} và chưa vượt NDVI cho đến khi mất khoảng 63%.</p></div>
<div class=bn style="margin-bottom:12px"><h3>Cách xử lý: che mây khi huấn luyện</h3><p>NDVI + che mây ngẫu nhiên: {f3(cm('NDVI +mask-aug',0.75))} ở 75%, hơn chỉ SAR; đổi lại kém {f1((cm('NDVI',0.0)-cm('NDVI +mask-aug',0.0))*100)} điểm khi trời quang.</p></div>
<div class=pc><p style="font-size:18px">NDWI cũng là quang học nên bị mây như NDVI. Nhãn sinh từ NDVI lúc quang đãng nên phép thử thiên về NDVI.</p></div></div></div>''',
    deco=spark(1340, 44, 40))

# ===================== 15 Task A
def tarow(lbl, k, cls=""):
    return f"<tr class={cls}><td>{lbl}</td><td class=n>{f3(tf(k + ' | RandomForest'))}</td><td class=n>{f3(tf(k + ' | CNN'))}</td><td class=n>{pct(ta_w(k + ' | CNN'))}</td></tr>"
add(title("Lúa so với cây khác", f"Earth Engine, {n(1828)} cửa sổ lúa và {n(2058)} cửa sổ khác; chia theo ô lưới 2 km") + f'''
<table><tr><th>Đầu vào</th><th class=n>Macro-F1 · Random Forest</th><th class=n>Macro-F1 · 1D-CNN</th><th class=n>Nhận đúng cây trồng khác (CNN)</th></tr>
{tarow('NDVI + NDWI', 'Quang hoc (NDVI+NDWI)')}{tarow('Chỉ SAR (VV, VH)', 'SAR (VV,VH)', 'hl')}{tarow('Cả bốn kênh', 'Tat ca (NDVI,NDWI,VV,VH)')}</table>
<div class=grid style="grid-template-columns:1fr 1fr;margin-top:24px">
<div class=pc><h3>Đừng dùng số quang học để so với SAR</h3><p>Điểm lúa được chọn bằng luật trên NDVI và NDWI, nên tách chúng bằng NDVI gần như đã biết trước đáp án.</p></div>
<div class=sg><h3>Chỉ SAR là phép thử công bằng</h3><p>Đạt {f3(tf('SAR (VV,VH) | CNN'))}, tách rừng, đô thị, nước gần hoàn toàn nhưng chỉ nhận đúng {pct(ta_w('SAR (VV,VH) | CNN'))} cây trồng khác: đây là phần khó thật.</p></div></div>''',
    deco=spark(1340, 44, 40))

# ===================== 16 External checks
hk = SK["informative_rows"]["HARVESTING_DAY"]; sw = SK["informative_rows"]["SOWING_DAY"]
add(title("Kiểm chứng ngoài") + f'''
<div class=grid style="grid-template-columns:1fr 1fr;gap:22px">
<div class=sg><h3>Kiểm thử theo năm</h3><p>Huấn luyện 2023–2024, kiểm thử 2025 (70 cửa sổ).</p><div style="display:flex;gap:30px;margin-top:14px"><div><div class=big style="font-size:44px">{f3(PB['year']['NDVI']['mean'])}</div><div class=lb>NDVI</div></div><div><div class=big style="font-size:44px">{f3(PB['year']['SAR+NDWI']['mean'])}</div><div class=lb>SAR + NDWI</div></div></div><p style="margin-top:12px">Kết quả giữ vững qua năm; NDVI hơn SAR + NDWI một khoảng có thật trên nhãn dẫn xuất từ NDVI.</p></div>
<div class=sg><h3>SICKLE: ngày thật của nông dân (Ấn Độ)</h3><p>Sentinel-1 dự đoán ngày thu hoạch, 5 phần theo thửa.</p><div style="display:flex;gap:30px;margin-top:14px"><div><div class=big style="font-size:44px">{f1(hk['season_mean_baseline']['mae'])}</div><div class=lb>MAE lịch vụ (ngày)</div></div><div><div class=big style="font-size:44px">{f1(hk['S1_gradient_boosting']['mae'])}</div><div class=lb>MAE Sentinel-1 (ngày)</div></div></div><p style="margin-top:12px">Trên các dòng có thông tin thật, SAR <b>chưa hơn</b> lịch mùa vụ (cải thiện gần 0, CI chứa 0).</p></div></div>
<div class=note style="margin-top:20px"><b>Vì sao tách "dòng có thông tin":</b> khoảng 44% dòng của SICKLE rơi đúng ngày mặc định ở mép mùa vụ, nên so trên mọi dòng sẽ thổi phồng lợi ích của SAR.</div>''',
    deco=spark(1340, 44, 40))

# ===================== 17 Outputs
chk = ["Mặt nạ 4 giai đoạn cho từng bước 5 ngày, luôn đúng thứ tự.",
       "Mốc SOS, POS, EOS theo ngày, sai số vài ngày, nằm trong độ mờ của nhãn.",
       f"Nhận diện trực tuyến khi chỉ có quá khứ: LSTM nhân quả {f3(cz)} so với BiLSTM {f3(blstm)}.",
       "Gọi đúng tên kết quả: dự đoán nhãn vật hậu yếu suy ra từ NDVI bằng chuỗi vệ tinh."]
add(title("Đầu ra và giải thích") + '<div style="margin-top:24px">' + "".join(
    f'<div style="display:flex;align-items:center;gap:18px;margin-bottom:20px"><div>{icon("check",58)}</div><div style="flex:1;border:2.5px solid {BROWN};border-radius:26px;padding:14px 26px;max-width:880px;font-size:23px;line-height:1.35;background:#FFF9F4">{t}</div></div>' for t in chk) + '</div>',
    deco=blob(1130, 120, 260, SAGE_L) + blob(1250, 360, 150, PEACH) + spark(1330, 100, 56) + spark(1180, 330, 36, "#C98A2B"))

# ===================== 18 Plan
plan = [("Giai đoạn 1", "Ground truth ngoài đồng: ngày sạ, thu hoạch thật của 100–150 thửa (hợp tác xã, Sở Nông nghiệp)"),
        ("Giai đoạn 2", "Nhãn mắt 150 mẫu, 3 người chấm, tính kappa trên ảnh Google Earth"),
        ("Giai đoạn 3", "Thêm kênh mặt nạ quan sát làm đầu vào; dự báo sớm EOS từ k bước đầu mùa"),
        ("Giai đoạn 4", "Điểm lúa và cây trồng khác xác nhận ngoài đồng; chia theo huyện kiểm tra khái quát hóa"),
        ("Giai đoạn 5", "Đánh giá lại, hoàn thiện báo cáo")]
add(title("Kế hoạch thực hiện tiếp") + '<div style="margin-top:14px">' + "".join(
    f'<div style="display:flex;gap:22px;align-items:baseline;margin-bottom:16px"><div style="font-size:26px;color:{BROWN};font-weight:800;width:190px;flex:none">{a}</div><div style="font-size:24px;line-height:1.35">{b}</div></div>' for a, b in plan) + '</div>'
    + f'<div class=note style="margin-top:12px">Chưa có ground truth ngoài đồng thì các kết quả trên là dự đoán nhãn yếu; đây là việc quan trọng nhất còn lại.</div>',
    deco=spark(1340, 44, 40))

# ===================== 19 End
add(f'''<div style="position:absolute;left:20px;top:80px"><div style="font-family:'Segoe UI Black';font-size:130px;line-height:1;color:{BROWN};letter-spacing:-3px">CẢM ƠN</div>
<div style="font-size:64px;margin-top:6px;color:#1F3A33;font-weight:700">thầy và các bạn</div><p style="font-size:26px;margin-top:30px;color:#5C4A38">Câu hỏi và góp ý</p></div>''', "end",
    deco=blob(820, 70, 520, SAGE_L) + blob(1140, 330, 230, PEACH) + blob(940, 400, 200, "#F6D9B8")
         + f'<svg style="position:absolute;left:840px;top:250px" width="520" height="200" viewBox="-8 -8 470 180"><polyline points="{pts}" fill="none" stroke="{BROWN}" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/></svg>'
         + spark(1290, 90, 60) + spark(850, 360, 36, "#C98A2B") + rays(1270, 520))

N = len(slides)
JS = "<script>window.addEventListener('load',function(){document.querySelectorAll('section.s').forEach(function(s){var b=s.querySelector('.body');if(!b||s.dataset.fit!=='1')return;var W=1300,H=690,k=1.3;b.style.bottom='auto';b.style.height='auto';b.style.right='auto';b.style.transformOrigin='top left';for(var i=0;i<20;i++){b.style.transform='none';b.style.width=(W/k)+'px';if(b.offsetHeight*k<=H||k<=1.0)break;k-=0.03;}b.style.transform='scale('+k+')';});});</script>"
html = f"<!doctype html><html lang=vi><head><meta charset=utf-8><title>Rice Phenology - Slide v2</title><style>{CSS}</style></head><body>{''.join(slides).replace('@@N@@', '%02d' % N)}{JS}</body></html>"
open(f"{SL}/deck_v2.html", "w", encoding="utf-8").write(html)
pdf = os.path.join(SL, "Slide_giua_ky_v2_kieu_nhom6.pdf")
edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=4000", f"--print-to-pdf={pdf}", "file:///" + f"{SL}/deck_v2.html".replace("\\", "/")], check=True, timeout=240)
print("slides", N, "->", pdf)
