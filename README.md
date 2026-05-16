# EEG Seizure Regularisation Study

Logistic regression for EEG seizure detection across **three real datasets**,
studying preprocessing order, L1/L2/Elastic-Net regularisation, class-imbalance
handling, and cross-dataset generalisation.

Machine Learning — Semester Major Assignment
Author: Faisal Hakimi (faisalh5556@gmail.com)

## Datasets (downloaded reproducibly at runtime via `kagglehub`)

| ID  | Dataset | Kaggle source | True origin |
|-----|---------|---------------|-------------|
| DS1 | Epileptic Seizure Recognition | `harunshimanto/epileptic-seizure-recognition` | UCI ML Repository |
| DS2 | CHB-MIT Scalp EEG | `adibadea/chbmitseizuredataset` | CHB-MIT / PhysioNet |
| DS3 | Bonn University EEG | `peimandaii/epilepsy-diagnosis-dataset` | Univ. of Bonn (Andrzejak 2001) |

No data is vendored or fabricated; datasets are fetched anonymously on run.

## Key finding

Logistic regression on **raw EEG amplitude** is degenerate (ROC-AUC ~0.50).
A unified time-domain + spectral feature extractor restores it (UCI/Bonn ~0.99,
CHB-MIT ~0.69 — a genuinely hard real task). Holding that methodology fixed,
the regularisation **penalty choice is a second-order effect**; feature
extraction and imbalance handling are the first-order levers. Every Q1–Q4
conclusion is computed from results and guarded by `assert` checks.

## Repository contents

| File | Purpose |
|------|---------|
| `notebook00a7a8e509.ipynb` | Main analysis notebook (runs end-to-end; results + figures embedded) |
| `Seizure_Prediction_Report_IEEE.docx` | IEEE-format report |
| `Seizure_Prediction_Presentation.pptx` | Presentation deck (9 slides) |
| `figures/` | Generated figures (PNG) |
| `build_notebook.py` | Regenerates the notebook |
| `make_report.py` | Recomputes metrics → `results.json` |
| `gen_docx.py` / `gen_pptx.py` | Build the report / presentation from `results.json` |

## How to run

Requires Python 3.11+ and internet access (no Kaggle credentials needed).

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn kagglehub nbconvert python-docx python-pptx

# Execute the notebook end-to-end
python -m nbconvert --to notebook --execute --inplace notebook00a7a8e509.ipynb

# (optional) regenerate report + presentation
python make_report.py && python gen_docx.py && python gen_pptx.py
```

> The notebook downloads data at runtime; it needs internet on a clean run.
> Saved cell outputs are already embedded, so the results are viewable offline.
