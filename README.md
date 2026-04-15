# Trio-Based Variant Pathogenicity Classification

A machine-learning pipeline that predicts whether a variant in a proband is
**pathogenic** (or likely pathogenic) by combining:

- Variant-level annotations from **eVAI** (CADD, REVEL, PolyPhen2, SIFT, zygosity…)
- Family **inheritance patterns** inferred from proband / father / mother trio files
- Patient **HPO phenotype** terms matched to disease associations via the HPO ontology

Designed for hearing-disorder cohorts but generalisable to any eVAI-annotated trio dataset.

---

## Requirements

Python ≥ 3.10 is required.

```bash
pip install -r requirements.txt
```

Key dependencies: `pandas`, `scikit-learn`, `xgboost`, `lightgbm`,
`imbalanced-learn`, `pyhpo`, `shap`, `mlflow`, `pytest`.

---

## Directory Layout

```
├── src/
│   ├── trio_loader.py          ← loads & joins trio CSVs, infers inheritance mode
│   ├── features.py             ← builds ML feature matrix (X, y)
│   ├── model.py                ← trains & evaluates classifiers via MLflow
│   └── generate_phenotypes.py  ← generates synthetic phenotypes.tsv
├── notebooks/
│   ├── 01_eda.ipynb            ← exploratory data analysis
│   ├── 02_trio_assembly.ipynb  ← TrioLoader demo & trio statistics
│   ├── 03_feature_engineering.ipynb ← build and inspect feature matrix
│   ├── 04_modelling.ipynb      ← train / compare / save best model
│   ├── 05_evaluation.ipynb     ← ROC, SHAP, confusion matrix, per-inheritance breakdown
│   ├── 06_vus_dataset_and_features.ipynb ← trusted labels vs VUS + enrichment features
│   ├── 07_vus_model_training.ipynb       ← train VUS ranker on trusted labels only
│   ├── 08_vus_prioritization.ipynb       ← score and rank VUS candidates
│   └── 09_trio_clustering_and_enrichment.ipynb ← trio-level unsupervised clusters + enrichment
├── tests/                      ← pytest unit tests (108 tests total)
├── generate_synthetic.py       ← generate synthetic eVAI trio CSVs for testing
├── data/
│   ├── trios.tsv               ← trio manifest (edit / replace with your data)
│   └── phenotypes.tsv          ← per-sample HPO phenotypes (edit / replace)
├── input/                      ← place real eVAI CSVs here (git-ignored)
├── models/                     ← saved model pipelines (git-ignored)
└── reports/                    ← generated figures and summary CSVs (git-ignored)
```

---

## Input Data Format

Each variant file must be an **eVAI-exported CSV** following the naming convention:

```
{SAMPLE_ID}_{ROLE}.csv      where ROLE ∈ {proband, father, mother}
```

Both file formats are handled automatically:

| Format | Description |
|--------|-------------|
| **Real eVAI files** | 5 quoted `##…` metadata lines at the top, no `Sample` column |
| **Synthetic files** | Plain CSV, `Sample` column present as first column |

### Trio manifest — `data/trios.tsv`

```
trio_id	proband	father	mother	phenotype_group
T01	SAMPLE01_proband	SAMPLE01_father	SAMPLE01_mother	hearing_loss
```

### Phenotype file — `data/phenotypes.tsv`

```
sample_id	hpo_terms	affected	age_of_onset_years	sex	notes
SAMPLE01_proband	HP:0000407;HP:0008619	1	3.5	F	bilateral SNHL
SAMPLE01_father		0		M	unaffected
SAMPLE01_mother		0		F	unaffected
```

HPO terms are semicolon-separated. Leave blank for unaffected/unknown individuals.

---

## Quick Start — Synthetic Data (for testing)

```bash
# 1. Generate 5 synthetic trios (proband + father + mother each)
python generate_synthetic.py \
    --trio \
    --trio-count 5 \
    --data-dir data \
    --seed 42

# 2. Run notebooks in order
jupyter lab   # then open and run 01 → 05 in sequence
```

This also auto-populates `data/trios.tsv` and `data/phenotypes.tsv`.

---

## Quick Start — Real Data

See `REAL_DATA_ONLY.txt` for a full walkthrough. Summary:

```bash
# 1. Copy real eVAI files into data/  (never modify input/)
cp input/YOURPATIENT_proband.csv  data/
cp input/YOURPATIENT_father.csv   data/
cp input/YOURPATIENT_mother.csv   data/

# 2. Append to trios.tsv
echo -e "TREAL01\tYOURPATIENT_proband\tYOURPATIENT_father\tYOURPATIENT_mother\thearing_loss" \
    >> data/trios.tsv

# 3. Append to phenotypes.tsv
printf "YOURPATIENT_proband\tHP:0000407;HP:0008619\t1\t\tM\tnotes\n" >> data/phenotypes.tsv
printf "YOURPATIENT_father\t\t0\t\tM\tunknown\n"  >> data/phenotypes.tsv
printf "YOURPATIENT_mother\t\t0\t\tF\tunknown\n"  >> data/phenotypes.tsv

# 4. Re-run notebooks 01 → 05
```

---

## VUS Prioritization Workflow

When your goal is to rank uncertain variants (instead of standard pathogenic vs non-pathogenic classification), run:

```bash
# Prerequisite: run notebooks 01 -> 05 once to ensure assembled/features artifacts exist

# VUS workflow
06_vus_dataset_and_features.ipynb
07_vus_model_training.ipynb
08_vus_prioritization.ipynb
09_trio_clustering_and_enrichment.ipynb
```

Outputs produced:
- `data/vus_trusted_X.parquet`, `data/vus_trusted_y.csv`, `data/vus_trusted_groups.csv`
- `data/vus_unlabeled_X.parquet`, `data/vus_unlabeled_meta.csv`
- `models/vus_ranker.joblib`
- `data/vus_priority_ranking.csv`
- `reports/vus_top10_per_trio.csv`, `reports/vus_top5_per_gene.csv`
- `reports/trio_cluster_assignments.csv`, `reports/trio_embedding_pca.csv`
- `reports/cluster_gene_enrichment.csv`, `reports/cluster_hpo_enrichment.csv`

Training labels in this workflow:
- Positive: `Pathogenic`, `Likely pathogenic`
- Negative: `Benign`, `Likely benign`
- Unlabeled ranking target: `Uncertain significance`

---

## Inheritance Mode Inference

| Proband zygosity | Father has variant | Mother has variant | Inferred mode |
|---|---|---|---|
| Heterozygous | No | No | `de_novo` |
| Heterozygous | Yes | No | `paternal` |
| Heterozygous | No | Yes | `maternal` |
| Heterozygous | Yes | Yes | `biparental_het` |
| Homozygous | Yes | Yes | `autosomal_recessive` |
| Homozygous | No | No | `de_novo_hom` |

---

## Features Used

| Feature | Source |
|---------|--------|
| `cadd_score`, `revel_score`, `papi_score`, `polyphen2_score`, `sift_score` | Parsed from `Functional` column |
| `gnomad_af` | Parsed from `Population` column |
| `vaf` | alt depth / total coverage |
| `criteria_score` | Count of ACMG criteria from `Criteria` column |
| `hpo_overlap_score` | HPO ontology overlap between proband and disease terms |
| `inheritance_mode` (one-hot) | Inferred from trio join |
| `condition_inheritance` (one-hot) | From `Condition_Inheritance` column |
| `effect_severity` | Ordinal-encoded `Effect` column |
| `zygosity_code` | Label-encoded `Sample_Zygosity` |
| `father_has_variant`, `mother_has_variant` | From trio join |

**Target**: `y = 1` if `Classification` ∈ {Pathogenic, Likely pathogenic}, else `y = 0`.

---

## Model Training

The pipeline uses **GroupKFold** cross-validation with `groups = trio_id` to
prevent data leakage between family members across folds.

Classifiers compared:
- `RandomForestClassifier`
- `XGBClassifier`
- `LGBMClassifier`
- `LogisticRegression` (baseline)

Class imbalance is handled with `class_weight='balanced'` and optional SMOTE.
All runs are tracked with **MLflow** (`mlflow ui` to browse experiments).

### Adjust for small cohorts

Edit `DEFAULT_CONFIG` in `src/model.py` if you have fewer than 5 trios:

```python
"cv_folds": 3,   # must be ≤ number of trios
```

---

## Running Tests

```bash
pytest tests/ -v
```

108 unit tests covering `trio_loader`, `features`, and `model`.

---

## HPO Terms (Hearing Disorders)

Default pool used for synthetic phenotype generation and HPO overlap scoring:

| HPO code | Description |
|----------|-------------|
| `HP:0000407` | Sensorineural hearing impairment |
| `HP:0008619` | Bilateral sensorineural hearing impairment |
| `HP:0000365` | Hearing impairment |
| `HP:0001751` | Vestibular dysfunction |
| `HP:0000370` | Abnormality of the middle ear |
| `HP:0000377` | Abnormality of the pinna |

Replace or extend these in `src/generate_phenotypes.py` and `src/features.py`
for other clinical cohorts.

---

## Generating More Synthetic Data

```bash
# Generate 10 additional trios (auto-increments trio IDs, appends TSV files)
python generate_synthetic.py \
    --trio \
    --trio-count 10 \
    -i input/YOUR_PROBAND.csv \   # use a real file as template for column structure
    -n 300 \                       # variants per file
    --data-dir data \
    --seed 123
```

---

## Contributing

1. Fork this repository.
2. Place your eVAI CSV files in `data/` and update `trios.tsv` / `phenotypes.tsv`.
3. Run `pytest tests/` to confirm all tests pass with your data.
4. Submit a pull request with code changes only — **never commit patient data**.

---

## Privacy & Ethics

- The `input/` directory is git-ignored. **Never commit real patient files.**
- Synthetic data in `data/SYNTH*.csv` is also git-ignored (can be regenerated).
- Only the manifest TSV files (`trios.tsv`, `phenotypes.tsv`) with synthetic IDs
  are tracked as reference examples.
- This tool is intended for research use only.
