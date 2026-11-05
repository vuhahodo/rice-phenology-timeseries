import sys, os, json; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import mt, numpy as np
O = mt.OUT
E = json.load(open(f"{O}/eda.json")); B = json.load(open(f"{O}/baselines.json")); S = json.load(open(f"{O}/summary.json")); B.update(json.load(open(f"{O}/extra_baselines.json")))
CZ = json.load(open(f"{O}/causal_results.json")); SAR = json.load(open(f"{O}/sar_results_real.json")); PB = json.load(open(f"{O}/pack_bootstrap.json"))
SK = json.load(open(f"{O}/sickle_s1_results.json")); TA = json.load(open(f"{O}/taskA_results.json")); CL = json.load(open(f"{O}/cloud_results.json"))
R = {k: S[k] for k in S if not k.startswith("_")}; cnn, lstm, trn, trd = R["CNN1D"], R["BiLSTM"], R["Transformer_NDVI"], R["Transformer_DOY"]
bl = lambda d: dict(f1=d["macro_f1"], sos=d["dates"]["sos"]["mae_days"], eos=d["dates"]["eos"]["mae_days"], miss_sos=d["dates"]["sos"]["miss"], miss_eos=d["dates"]["eos"]["miss"], retro=d["retrograde"])
cm = lambda nm: [float(np.mean([r["by_level"][str(p)]["macro_f1"] for r in CL["results"][nm]])) for p in CL["levels"]]
cfg = SAR["results"]; ms = lambda a: float(np.mean(a))
sar = lambda k: dict(f1=ms([x["viterbi"]["macro_f1"] for x in cfg[k]]), sos=ms([x["dates"]["sos"]["mae_days"] for x in cfg[k]]), eos=ms([x["dates"]["eos"]["mae_days"] for x in cfg[k]]))
tf = lambda k: TA[k]["folds"]["macro_f1"][0]; w40 = lambda k: TA[k]["correct_nonrice_by_worldcover_class"]["40"]
hk = SK["informative_rows"]["HARVESTING_DAY"]
D = dict(
    n=E["n"], fields=E["fields"], split=E["split"], miss_mean=E["miss_mean"],
    cnn=dict(f1=cnn["viterbi"]["macro_f1"], params=cnn["viterbi"]["params"]), trn=dict(f1=trn["viterbi"]["macro_f1"]), trd=dict(f1=trd["viterbi"]["macro_f1"]), lstm=dict(f1=lstm["viterbi"]["macro_f1"]),
    retro_min=min(m["raw"]["retro"][0] for m in (cnn, lstm, trn, trd)), retro_max=max(m["raw"]["retro"][0] for m in (cnn, lstm, trn, trd)),
    B0=bl(B["B0_calendar"]), B1=bl(B["B1_threshold_rule"]), B2=bl(B["B2_random_forest"]), B3=bl(B["B3_double_logistic"]), B4=bl(B["B4_ndvi_slope"]),
    sar=dict(ndvi=sar("NDVI+DOY (control)"), sar=sar("SAR (VV,VH,VH-VV)+DOY"), sarndwi=sar("SAR+NDWI+DOY"), all=sar("NDVI+NDWI+SAR+DOY")),
    cloud=dict(levels=CL["levels"], ndvi=cm("NDVI"), ndvi_aug=cm("NDVI +mask-aug"), sar=cm("SAR"), sarndwi=cm("SAR+NDWI"), all=cm("NDVI+NDWI+SAR"), all_aug=cm("NDVI+NDWI+SAR +mask-aug")),
    taskA=dict(opt_rf=tf("Quang hoc (NDVI+NDWI) | RandomForest"), opt_cnn=tf("Quang hoc (NDVI+NDWI) | CNN"), sar_rf=tf("SAR (VV,VH) | RandomForest"), sar_cnn=tf("SAR (VV,VH) | CNN"),
               all_rf=tf("Tat ca (NDVI,NDWI,VV,VH) | RandomForest"), all_cnn=tf("Tat ca (NDVI,NDWI,VV,VH) | CNN"),
               w_opt=w40("Quang hoc (NDVI+NDWI) | CNN"), w_sar=w40("SAR (VV,VH) | CNN"), w_all=w40("Tat ca (NDVI,NDWI,VV,VH) | CNN")),
    year=dict(ndvi=PB["year"]["NDVI"]["mean"], sarndwi=PB["year"]["SAR+NDWI"]["mean"]),
    sickle=dict(base=hk["season_mean_baseline"]["mae"], s1=hk["S1_gradient_boosting"]["mae"]),
    causal=dict(lstm=ms([a["macro_f1"] for a in CZ["LSTM_causal"]]), bilstm=ms([a["macro_f1"] for a in CZ["BiLSTM"]])),
)
json.dump(D, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json"), "w"), indent=1, default=float)
print("data.json ok", D["cnn"]["f1"], D["cloud"]["ndvi_aug"])
