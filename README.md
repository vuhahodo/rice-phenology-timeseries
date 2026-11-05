# Rice Phenology from Sentinel-1/2 Time Series

Midterm project for the Time Series course. We track the rice growth cycle (fallow → vegetative → reproductive → ripening) in Đà Nẵng and Quảng Nam, Vietnam (2023–2025) from satellite time series pulled with Google Earth Engine:

- **Sentinel-2 SR**: NDVI, NDWI (optical, blocked by cloud)
- **Sentinel-1 GRD**: VV, VH (radar, sees through cloud)
- **ESA WorldCover 2021**: to sample rice vs. non-rice points

Labels (4 phenology stages) come from a 20%-of-amplitude threshold on the clean NDVI curve; we are upfront about this being a weak/circular label source and we validate externally against the SICKLE dataset's farmer-reported sowing/harvest dates.

## What's here

| Folder | Contents |
|---|---|
| `scripts/` | All experiment code: baselines (B0 calendar, B1 rule, B2 random forest, B3 double-logistic, B4 NDVI-slope), the 1D-CNN/BiLSTM/Transformer deep models + cyclic Viterbi decoding, cloud-robustness simulation, SAR-only and SAR+NDWI experiments, year hold-out, SICKLE cross-check |
| `out/` | Every number used on the slides, as JSON — nothing is hand-typed |
| `figs/` | Generated result figures |
| `data_v32.csv` | Main windowed dataset (1828 windows × 73 steps) |
| `sar/` | Multi-channel npz (NDVI, NDWI, VV, VH, obs-mask) used for the SAR and cloud-robustness experiments |
| `kaggle_kernel/` | Notebook used to pull data from Earth Engine and run the full pipeline on Kaggle |
| `kaggle_dataset/`, `kaggle_out/` | Exported dataset and downloaded run results from the Kaggle run |
| `pptx/`, `slides/` | Slide deck generator (pptxgenjs) and the built decks (PDF/PPTX) |
| `colab/` | Earth Engine extraction helper |

## Headline results (test set, macro-F1 on the 4 stages after Viterbi decoding)

| Model | Macro-F1 |
|---|---|
| B0 calendar | 0.642 |
| B1 rule | 0.752 |
| B3 double-logistic | 0.759 |
| B4 NDVI-slope | 0.746 |
| B2 random forest | 0.767 |
| **1D-CNN + cyclic Viterbi** | **0.891 ± 0.004** |

Radar-only (VV/VH, no optical at all) reaches 0.711; adding NDWI (still cloud-robust) brings it to 0.850 — useful when optical is lost to persistent cloud.

## Reproducing

```
python -I scripts/run_baselines.py        # B0, B1, B2 -> out/baselines.json
python -I scripts/run_extra_baselines.py  # B3, B4      -> out/extra_baselines.json
python -I scripts/run_dl.py               # CNN1D / BiLSTM / Transformer, 5 seeds, Viterbi, masked-input robustness -> out/dl_results.json
python -I scripts/run_sar.py              # needs sar/rice_phenology_multi_v3.npz -> out/sar_results_real.json
python -I scripts/run_cloud.py            # cloud-block simulation, mask-augmented training -> out/cloud_results.json
python -I scripts/run_pack.py             # year hold-out, tolerance sweep, bootstrap CI -> out/pack_*.json
python -I scripts/run_sickle_s1.py        # external check against SICKLE farmer-reported dates -> out/sickle_s1_results.json
python -I scripts/make_results.py         # aggregate + figures -> out/summary.json, figs/
```

Model classes referenced by `scripts/*.py` (`NDVICNN1D`, `NDVIBiLSTM`, `NDVITransformer`) live under `src/models/`; add that folder next to `scripts/` before rerunning training from scratch. All result JSON in `out/` is already included, so the slides and figures can be rebuilt (`make_results.py`) without retraining.

## Data sources / citations

- Sentinel-2 SR Harmonized, Sentinel-1 GRD, ESA WorldCover 2021 — via [Google Earth Engine](https://doi.org/10.1016/j.rse.2017.06.031)
- Baseline B2 (random forest) methodology: Rußwurm et al. (2020), *BreizhCrops: A Time Series Dataset for Crop Type Mapping*, [arXiv:1905.11893](https://arxiv.org/abs/1905.11893) — PDF: [`papers/B2_russwurm2020_breizhcrops.pdf`](papers/B2_russwurm2020_breizhcrops.pdf)
- Baseline B3 (double-logistic fit): Gobin et al. (2023), *Crop Phenology Modelling Using Proximal and Satellite Sensor Data*, [Remote Sensing 15(8), 2090](https://www.mdpi.com/2072-4292/15/8/2090) — PDF: [`papers/B3_gobin2023_crop_phenology_modelling.pdf`](papers/B3_gobin2023_crop_phenology_modelling.pdf)
- Baseline B4 (NDVI difference/slope): Tian et al. (2023), *Research on Rice Fields Extraction by NDVI Difference Method Based on Sentinel Data*, [Sensors 23(13), 5876](https://www.mdpi.com/1424-8220/23/13/5876) — PDF: [`papers/B4_tian2023_rice_ndvi_difference.pdf`](papers/B4_tian2023_rice_ndvi_difference.pdf)
- External validation: Sani et al. (2023), *SICKLE: A Multi-Sensor Satellite Imagery Dataset Annotated with Multiple Key Cropping Parameters*, [arXiv:2312.00069](https://arxiv.org/abs/2312.00069)

## Notes

- `kaggle_kernel/kernel-metadata.json` references a private Earth Engine service-account-key dataset (`ee-service-key`) used only to run the notebook under this account. It is **not included** in this repo — anyone reproducing the Earth Engine extraction step should authenticate with their own Google account or service account.
