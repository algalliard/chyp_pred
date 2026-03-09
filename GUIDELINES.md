# Project Guidelines: Trio-Based Variant Pathogenicity Classification

> This document is intended to be used as a working guide together with a
> bioinformatics/genetics AI agent (GitHub Copilot / Claude Sonnet).
> Each section ends with a **"Agent Action"** block describing the task to
> delegate to the agent.

---

## 1. Project Overview

### Goal

Build a machine-learning pipeline that, given whole-exome/genome variant
annotation files for a **family trio** (proband + father + mother), predicts
whether a variant in the proband is **pathogenic** (or likely pathogenic)
while leveraging:

- Variant-level annotation features (CADD, REVEL, PolyPhen2, SIFT, zygosity…)
- Family inheritance patterns (de novo, compound heterozygous, homozygous
  recessive, X-linked…)
- Patient phenotype (HPO terms mapped to disease associations)

### Clinical context

Samples come from a hearing-disorder cohort.  The two index HPO terms used
so far are:
- `HP:0000407` — Sensorineural hearing impairment
- `HP:0008619` — Bilateral sensorineural hearing impairment

The pipeline can generalise to other phenotypic groups by swapping phenotype
files.

### Current state

| Item | Status |
|------|--------|
| Real proband input CSV (`CHYP30_proband.csv`) | ✅ available |
| Synthetic variant files (`synthetic_1…50.csv`) | ✅ generated (50 × 300 rows) |
| `generate_synthetic.py` script | ✅ available |
| EDA notebook (`input_eda.ipynb`) | ✅ done |
| Classification notebook (`synthetic_classification.ipynb`) | ✅ proof-of-concept |
| **Trio definition file** | ✅ `data/trios.tsv` (16 trios, validated) |
| **Phenotypic annotation file** | ✅ `data/phenotypes.tsv` (48 samples, seed=42) |
| Proper per-role sample naming (`_proband`, `_father`, `_mother`) | ✅ 48 files renamed in `data/` |
| Trio-aware feature engineering | ❌ not started |
| Final model with phenotype integration | ❌ not started |

---

## 2. File & Directory Conventions

```
predtest/
├── input/                          ← real patient CSVs (never modify)
│   └── CHYP30_proband.csv
├── data/                           ← generated/synthetic working data
│   ├── trios.tsv                   ← trio definition file  (Step 3)
│   ├── phenotypes.tsv              ← per-sample phenotype  (Step 4)
│   └── CHYP30_proband.csv          ← symlink / copy of input
│   └── SAMPLENAME_{proband,father,mother}.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_trio_assembly.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modelling.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── generate_synthetic.py       ← existing utility
│   ├── trio_loader.py              ← Step 5
│   ├── features.py                 ← Step 6
│   └── model.py                    ← Step 7
├── models/                         ← serialised models
├── reports/                        ← figures and summary CSVs
└── GUIDELINES.md                   ← this file
```

### Sample file naming rule

Every variant CSV **must** follow the pattern:

```
{SAMPLE_ID}_{ROLE}.csv
```

where `ROLE` ∈ `{proband, father, mother}` and `SAMPLE_ID` is arbitrary
(e.g. `CHYP30`, `T1_CHILD`, etc.).

---

## 3. Step 1 — Reorganise Synthetic Data into Trio-Named Files

The 50 existing `synthetic_N.csv` files need to be:

1. **Grouped** into 16 trios (≈ 48 files) + 2 spare singletons.
2. **Renamed** following the `{SAMPLE_ID}_{ROLE}.csv` convention.

Suggested mapping (16 complete trios, IDs `SYNTH01`–`SYNTH16`):

| Trio | Proband file | Father file | Mother file |
|------|-------------|-------------|-------------|
| T01 | SYNTH01_proband | SYNTH01_father | SYNTH01_mother |
| … | … | … | … |
| T16 | SYNTH16_proband | SYNTH16_father | SYNTH16_mother |

> **Agent Action**
> ```
> rename synthetic_1.csv → data/SYNTH01_proband.csv
> rename synthetic_2.csv → data/SYNTH01_father.csv
> rename synthetic_3.csv → data/SYNTH01_mother.csv
> … (continue for all 50 files)
> ```
> Also update the `Sample` column inside each file to match the new name.

---

## 4. Step 2 — Create the Trio Definition File (`trios.tsv`)

**Recommended format: TSV** (tab-separated) — avoids confusion with commas
inside sample IDs and is trivially loadable with `pd.read_csv(..., sep='\t')`.

### Schema

```
trio_id	proband	father	mother	phenotype_group
T01	SYNTH01_proband	SYNTH01_father	SYNTH01_mother	hearing_loss
T02	SYNTH02_proband	SYNTH02_father	SYNTH02_mother	hearing_loss
…
```

Column definitions:

| Column | Type | Description |
|--------|------|-------------|
| `trio_id` | string | Unique trio identifier (T01, T02, …) |
| `proband` | string | Base name of the proband variant CSV (no extension) |
| `father` | string | Base name of the father variant CSV |
| `mother` | string | Base name of the mother variant CSV |
| `phenotype_group` | string | Clinical cohort label (optional but useful for stratified splitting) |

> **Agent Action**
> Generate `data/trios.tsv` programmatically from the list of renamed files.
> Validate that every sample ID mentioned in `trios.tsv` has a matching CSV
> in `data/`.

---

## 5. Step 3 — Create the Phenotypic Annotation File (`phenotypes.tsv`)

Because real phenotype data are not available for all samples, generate a
**synthetic phenotype file** that realistically mirrors clinical practice.

### Schema (recommended)

```
sample_id	hpo_terms	affected	age_of_onset_years	sex	notes
SYNTH01_proband	HP:0000407;HP:0008619	1	3.5	F	congenital bilateral SNHL
SYNTH01_father	HP:0000407	0	45.0	M	mild unilateral
SYNTH01_mother		0		F	unaffected
…
```

| Column | Description |
|--------|-------------|
| `sample_id` | Must match values in `trios.tsv` |
| `hpo_terms` | Semicolon-separated HPO codes |
| `affected` | Binary (1 = clinically affected proband / carrier) |
| `age_of_onset_years` | Age at first symptom; `NaN` for unaffected |
| `sex` | M / F / Other |
| `notes` | Free-text reminder |

### Synthetic generation rules

- Probands: always `affected=1`.  Sample 2–3 HPO terms from a curated list
  (predominantly hearing-related).
- Fathers / mothers: 30 % carrier (`affected=0`), 10 % affected.
- Age of onset: gamma-distributed (shape=2, scale=5) for probands.
- Sex: random 50/50 for children, realistic for parents.

> **Agent Action**
> Write `src/generate_phenotypes.py` that produces `data/phenotypes.tsv`
> for all sample IDs found in `trios.tsv`.  Seed with `--seed` for
> reproducibility.  HPO term pool to use for hearing disorders:
> `HP:0000407, HP:0008619, HP:0000365, HP:0001751, HP:0000370, HP:0000377, HP:0000407`.

---

## 6. Step 4 — Trio-Aware Data Loading (`src/trio_loader.py`)

Build a loader module that:

1. Reads `trios.tsv` and `phenotypes.tsv`.
2. For each trio, loads the three variant CSVs.
3. **Joins** proband variants with father and mother variants on a shared key
   (`Gene` + `HGVS_Coding`) to determine inheritance mode.
4. Returns a flat variant-level DataFrame with extra columns:

| New column | Description |
|-----------|-------------|
| `trio_id` | From trios.tsv |
| `role` | proband / father / mother |
| `inheritance_mode` | Inferred (see below) |
| `proband_zygosity` | From proband file |
| `father_has_variant` | Boolean |
| `mother_has_variant` | Boolean |
| `proband_phenotype_hpo` | HPO terms string |
| `affected` | Binary label from phenotypes.tsv |

### Inheritance mode inference rules

| proband zyg. | father_has | mother_has | Inferred mode |
|---|---|---|---|
| Heterozygous | False | False | de_novo |
| Heterozygous | True | False | paternal |
| Heterozygous | False | True | maternal |
| Heterozygous | True | True | biparental_het |
| Homozygous | True | True | autosomal_recessive |
| Homozygous | False | False | de_novo_hom |

> **Agent Action**
> Implement `TrioLoader` class in `src/trio_loader.py` with method
> `load_all() -> pd.DataFrame`.  Include unit tests in `tests/test_trio_loader.py`.

---

## 7. Step 5 — Feature Engineering (`src/features.py`)

Starting from the loaded DataFrame, build the ML feature matrix.

### Parsed / derived features

| Feature | Source column | Treatment |
|---------|--------------|-----------|
| `cadd_score` | `Functional` (parse `CADD:xx.x`) | float |
| `revel_score` | `Functional` (parse `REVEL:xx.x`) | float |
| `papi_score` | `Functional` (parse `PaPI:xx.x`) | float |
| `polyphen2_score` | `Functional` | float |
| `sift_score` | `Functional` | float |
| `gnomad_af` | `Population` (parse `gnomAD:xx%`) | float |
| `pathogenicity_score` | `Pathogenicity_Score` | float (already numeric) |
| `sample_coverage` | `Sample_Coverage` | float |
| `alt_depth` | `Sample_Coverage_Alternate` | float |
| `vaf` | alt / total coverage | float |
| `effect_severity` | `Effect` | ordinal encode |
| `zygosity` | `Sample_Zygosity` | label encode |
| `inheritance_mode` | derived | one-hot |
| `condition_inheritance` | `Condition_Inheritance` | one-hot |
| `criteria_score` | `Criteria` (count ACMG criteria) | int |
| `father_has_variant` | derived | bool |
| `mother_has_variant` | derived | bool |
| `hpo_overlap_score` | HPO overlap between proband HPO and disease HPO | float |

### Target variable

```
y = 1  if Classification ∈ {"Pathogenic", "Likely pathogenic"}
y = 0  otherwise
```

> **Agent Action**
> Implement `build_features(df) -> (X: pd.DataFrame, y: pd.Series)` in
> `src/features.py`.  Parse all sub-fields from `Functional` and `Population`
> columns with regex.  Include the `hpo_overlap_score` using an HPO ontology
> library (e.g. `pyhpo`).

---

## 8. Step 6 — Model Development (`src/model.py` + `notebooks/04_modelling.ipynb`)

### Recommended pipeline

```
ColumnTransformer
    ├── numeric        → SimpleImputer(median) → StandardScaler
    └── categorical    → SimpleImputer(constant='missing') → OneHotEncoder

↓

Class balancing: SMOTE or class_weight='balanced'

↓

Classifier (compare):
    - RandomForestClassifier
    - GradientBoostingClassifier (XGBoost / LightGBM)
    - LogisticRegression (baseline)

↓

Calibration: CalibratedClassifierCV (for probability output)
```

### Experiment tracking

Use `mlflow` to log parameters, metrics, and artefacts.  Each run:
- Logs `f1_macro`, `roc_auc`, `average_precision` on the test fold.
- Saves the pipeline as a `.joblib` artefact.

### Cross-validation strategy

Use **GroupKFold** with `groups = trio_id` to prevent data leakage between
family members across folds.

> **Agent Action**
> Scaffold `src/model.py` with `train(X, y, groups, config) -> pipeline`
> and `evaluate(pipeline, X_test, y_test) -> dict`.
> Add MLflow logging.  Suggest a reasonable hyperparameter grid for grid search.

---

## 9. Step 7 — Evaluation Notebook (`notebooks/05_evaluation.ipynb`)

This notebook should:

1. Load the best-saved pipeline from `models/`.
2. Run on a hold-out test set (last 20 % of trios, chronological order).
3. Plot:
   - ROC curve per class
   - Precision-recall curve
   - Per-gene pathogenicity heatmap
   - Feature importance bar chart (SHAP values preferred)
   - Confusion matrix
4. Summarise performance per `inheritance_mode` to identify where the model
   struggles.

> **Agent Action**
> Create `notebooks/05_evaluation.ipynb` with all plots, using the pipeline
> saved in `models/best_pipeline.joblib`.  Use SHAP (v0.45+) for
> feature importance.

---

## 10. Step 8 — Generating More Realistic Synthetic Data

The current `generate_synthetic.py` perturbs numeric columns with Gaussian
noise and randomly reassigns genes.  For trio-aware analysis we also need:

1. **Three correlated files per trio** — variants must overlap between family
   members at the expected rates (see inheritance mode table above).
2. **More realistic VAF** — homozygous variants: VAF ≈ 0.95–1.0; heterozygous
   VAF ≈ 0.45–0.55.
3. **Classification labels corrupted by realistic noise** — 5 % of VUS
   flipped to Pathogenic to simulate reclassification.

> **Agent Action**
> Extend `generate_synthetic.py` with a `--trio` flag and a `--trio-count`
> argument that generates linked proband/father/mother triplets in one call
> and places them in `data/` with the correct naming convention.
> Also update `trios.tsv` and `phenotypes.tsv` automatically.

---

## 11. Step 9 — Integrate Real Data When Available

### Prerequisites

The pipeline has been hardened to accept both file formats transparently:
- **Synthetic files** (`data/SYNTHXX_*.csv`) — plain CSV, `Sample` column present.
- **Real eVAI files** (`input/CHYPXX_*.csv`) — 5 quoted `##…` metadata lines at
  the top, no `Sample` column.

`TrioLoader._load_sample()` detects and handles both automatically.

---

### Step-by-step checklist

**Step 9.1 — Copy the new files into `data/`**

> **Never modify files in `input/`.  Always copy them.**

```bash
cp input/CHYPXX_proband.csv  data/CHYPXX_proband.csv
cp input/CHYPXX_father.csv   data/CHYPXX_father.csv
cp input/CHYPXX_mother.csv   data/CHYPXX_mother.csv
```

If you only have the proband file (father/mother not yet sequenced), create
placeholder empty files so `TrioLoader` doesn't raise `FileNotFoundError`:

```bash
head -1 data/SYNTH01_proband.csv > data/CHYPXX_father.csv   # header only
head -1 data/SYNTH01_proband.csv > data/CHYPXX_mother.csv
```

---

**Step 9.2 — Add a row to `data/trios.tsv`**

Append one line (tab-separated):

```
TREAL01	CHYPXX_proband	CHYPXX_father	CHYPXX_mother	hearing_loss
```

You can also run:

```bash
echo -e "TREAL01\tCHYPXX_proband\tCHYPXX_father\tCHYPXX_mother\thearing_loss" \
    >> data/trios.tsv
```

> `trio_id` must be unique.  Suggested convention: `TREAL01`, `TREAL02`, … for
> real trios vs `T01`–`T16` for synthetic ones.

---

**Step 9.3 — Add rows to `data/phenotypes.tsv`**

Append three rows (tab-separated), one per family member.  Use real HPO terms
from the eVAI metadata line `"##Phenotypes:…"` at the top of the proband file.

```
CHYPXX_proband	HP:0000407;HP:0008619	1	<age>	<M/F>	<clinical notes>
CHYPXX_father	HP:0000407	0		M	mild carrier
CHYPXX_mother		0		F	unaffected
```

Or let the script regenerate the whole phenotypes file:

```bash
python src/generate_phenotypes.py \
    --trios   data/trios.tsv \
    --output  data/phenotypes.tsv \
    --seed    42
```

> **Note:** re-running `generate_phenotypes.py` overwrites phenotypes for all
> samples.  Prefer manual appending if you want to preserve manually edited
> real-patient phenotype data.

---

**Step 9.4 — Optionally flag synthetic vs real rows**

To let the model distinguish real patient variants from synthetic ones, add an
`is_synthetic` boolean feature.  Run the notebooks with this column present:

```bash
# For all existing synthetic files:
for f in data/SYNTH*.csv; do
    python -c "
import pandas as pd, sys
df = pd.read_csv('$f')
df['is_synthetic'] = 1
df.to_csv('$f', index=False)
"
done

# The real file will naturally have is_synthetic=0 (missing → imputed to 0)
```

The preprocessor in `src/model.py` treats missing numeric values with median
imputation, so you can skip this step and rely on NaN → 0 imputation instead.

---

**Step 9.5 — Re-run the notebooks in order**

No code changes are needed.  Simply re-execute:

| # | Notebook | What happens |
|---|----------|-------------|
| 1 | `01_eda.ipynb` | Now includes the new real-patient proband in the cohort plots |
| 2 | `02_trio_assembly.ipynb` | `TrioLoader` picks up the new trio from `trios.tsv` |
| 3 | `03_feature_engineering.ipynb` | Features are built for all trios including the new one |
| 4 | `04_modelling.ipynb` | Retrains on synthetic + real data combined; saves new `models/best_pipeline.joblib` |
| 5 | `05_evaluation.ipynb` | Evaluates on the updated hold-out set |

> The hold-out split in notebooks 4 and 5 is based on the **last 20 % of
> `trio_id` values sorted lexicographically**.  If `TREAL01` sorts after `T16`,
> it will land in the test set; add more real trios before re-running to keep
> it in training.

---

### Quick-start copy-paste (single real trio, proband only)

```bash
# 1. Copy files
cp input/CHYP30_proband.csv  data/CHYP30_proband.csv
head -1 data/SYNTH01_proband.csv > data/CHYP30_father.csv
head -1 data/SYNTH01_proband.csv > data/CHYP30_mother.csv

# 2. Append to trios.tsv
echo -e "TREAL01\tCHYP30_proband\tCHYP30_father\tCHYP30_mother\thearing_loss" \
    >> data/trios.tsv

# 3. Append to phenotypes.tsv  (HPO terms from the ##Phenotypes line)
printf "CHYP30_proband\tHP:0000407;HP:0008619\t1\t\tM\treal patient\n" >> data/phenotypes.tsv
printf "CHYP30_father\t\t0\t\tM\tunknown\n" >> data/phenotypes.tsv
printf "CHYP30_mother\t\t0\t\tF\tunknown\n" >> data/phenotypes.tsv

# 4. Re-run notebooks 01 → 05 in order
```

> **Recommended practice**: keep `input/` read-only and always copy/symlink
> files into `data/` before processing.

---

## 12. Dependency Checklist

```bash
# core
pip install pandas numpy scikit-learn matplotlib seaborn

# boosting
pip install xgboost lightgbm

# imbalanced learning
pip install imbalanced-learn

# HPO ontology
pip install pyhpo

# explainability
pip install shap

# experiment tracking
pip install mlflow

# testing
pip install pytest
```

A `requirements.txt` should be generated and committed alongside this file.

> **Agent Action**
> Run `pip freeze > requirements.txt` after installing all dependencies and
> confirm all imports resolve correctly.

---

## 13. Recommended Order of Execution

```
Step 1  →  Rename synthetic files
Step 2  →  Create trios.tsv
Step 3  →  Create phenotypes.tsv  (generate_phenotypes.py)
Step 4  →  Implement TrioLoader   (trio_loader.py)
Step 5  →  Feature engineering   (features.py)
Step 6  →  EDA with trio context  (01_eda.ipynb  — refactor existing notebook)
Step 7  →  Model training         (model.py  +  04_modelling.ipynb)
Step 8  →  Evaluation             (05_evaluation.ipynb)
Step 9  →  Generate richer synth  (generate_synthetic.py --trio)
Step 10 →  Swap in real data      (when available)
```

---

## 14. Open Questions / Design Decisions

| # | Question | Current decision |
|---|----------|-----------------|
| 1 | How to handle variants present in a sample but not annotated (missing gene/HGVS)? | Impute as "unknown", exclude from inheritance join |
| 2 | Compound heterozygous detection across two CSVs? | Match by Gene only; verify distinct HGVS_Coding |
| 3 | X-linked inheritance detection? | Defer until sex chromosome annotations confirmed present |
| 4 | Multi-allelic variants? | Treat each row independently for now |
| 5 | HPO overlap scoring without ontology lib? | Fall back to string set intersection |
| 6 | VUS classification — include or exclude? | Include as `y=0` but track separately |

---

*Last updated: 2026-03-09.  Generated with GitHub Copilot (Claude Sonnet 4.6).*

---

## 15. Progress Log

| Date | Step | Status |
|------|------|--------|
| 2026-03-09 | Step 1 — Rename 50 synthetic files into `data/SYNTHXX_{role}.csv` | ✅ done |
| 2026-03-09 | Step 2 — Generate & validate `data/trios.tsv` | ✅ done |
| 2026-03-09 | Step 3 — `src/generate_phenotypes.py` + `data/phenotypes.tsv` | ✅ done |
| 2026-03-09 | Step 4 — `src/trio_loader.py` + `tests/test_trio_loader.py` (30/30 tests pass) | ✅ done |
| 2026-03-09 | Step 5 — `src/features.py` + `tests/test_features.py` (44/44 tests pass) | ✅ done |
| 2026-03-09 | Step 6 — `src/model.py` + `tests/test_model.py` (34/34 tests pass) | ✅ done |
| 2026-03-09 | Step 7 — `notebooks/05_evaluation.ipynb` (10 sections, all plots + per-inheritance breakdown) | ✅ done |
| 2026-03-09 | Step 8 — `generate_synthetic.py --trio / --trio-count` with correlated parent carriage, realistic VAF, 5 % VUS noise, auto-update of `trios.tsv` + `phenotypes.tsv` | ✅ done |
| 2026-03-09 | Notebooks — created `notebooks/01_eda.ipynb`, `02_trio_assembly.ipynb`, `03_feature_engineering.ipynb`, `04_modelling.ipynb` | ✅ done |
| 2026-03-09 | Step 9 — expanded with full real-data checklist; fixed `TrioLoader._load_sample` to auto-skip eVAI `##` metadata and inject missing `Sample` column | ✅ done |
