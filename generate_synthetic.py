#!/usr/bin/env python3
"""
Utility for creating synthetic variant report CSV files, including trio-linked
(proband + father + mother) generation.

Single-file mode (original behaviour):
    python generate_synthetic.py \\
        -i input/YOUR_PROBAND.csv \\
        -o synthetic.csv \\
        -n 1000

    python generate_synthetic.py \\
        -i input/YOUR_PROBAND.csv \\
        -o synthetic_%d.csv \\
        -n 300 -m 5

Trio mode:
    python generate_synthetic.py \\
        --trio \\
        --trio-count 5 \\
        -i input/YOUR_PROBAND.csv \\
        -n 300 \\
        --data-dir data \\
        --seed 42

Trio mode generates linked proband/father/mother CSVs with:
  - Per-variant inheritance mode drawn from a realistic distribution.
  - Correlated father/mother variant carriage derived from that mode.
  - Realistic VAF: homozygous rows → 0.95–1.00, heterozygous → 0.45–0.55.
  - 5 % of VUS variants flipped to "Pathogenic" as label-noise simulation.
  - Automatic append of new rows to data/trios.tsv and data/phenotypes.tsv.
"""

import argparse
import os
import random
import re
from io import StringIO

import numpy as np
import pandas as pd


META_PREFIX = "##"

# ---------------------------------------------------------------------------
# Inheritance mode distribution (per variant in the proband)
# ---------------------------------------------------------------------------
_IMODE_MODES = [
    "de_novo",
    "paternal",
    "maternal",
    "biparental_het",
    "autosomal_recessive",
    "de_novo_hom",
    "homozygous_partial",
]
_IMODE_WEIGHTS = [0.40, 0.20, 0.20, 0.10, 0.05, 0.03, 0.02]

# ---------------------------------------------------------------------------
# HPO term pool — mirrors src/generate_phenotypes.py
# ---------------------------------------------------------------------------
_HPO_POOL = [
    "HP:0000407", "HP:0008619", "HP:0000365", "HP:0001751",
    "HP:0000370", "HP:0000377", "HP:0000411", "HP:0000405", "HP:0000358",
]
_PROBAND_CORE_HPO = ["HP:0000407", "HP:0008619", "HP:0000365"]
_PROBAND_NOTES = [
    "congenital bilateral SNHL",
    "early-onset sensorineural hearing loss",
    "profound bilateral hearing loss, cochlear implant candidate",
    "moderate-severe bilateral SNHL",
    "bilateral hearing loss, unknown onset",
]
_PARENT_UNAFFECTED_NOTES = [
    "unaffected, obligate carrier suspected",
    "normal hearing, no reported symptoms",
    "unaffected",
    "clinically unaffected",
]
_PARENT_AFFECTED_NOTES = [
    "mild unilateral hearing loss",
    "age-related hearing impairment",
    "mild bilateral SNHL",
    "hearing aid user",
]


# ===========================================================================
# Shared helpers
# ===========================================================================

def load_table(path: str) -> pd.DataFrame:
    """Return DataFrame from a variant CSV, skipping ##-prefixed metadata lines."""
    data_lines = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.lstrip('"').startswith(META_PREFIX):
                continue
            data_lines.append(line)
    return pd.read_csv(StringIO("".join(data_lines)))


def _add_sample_column(df: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    """Set (or insert) the 'Sample' column to *sample_id*."""
    df = df.copy()
    if "Sample" in df.columns:
        df["Sample"] = sample_id
    else:
        df.insert(0, "Sample", sample_id)
    return df


def perturb_value(val):
    """Add small Gaussian noise to numeric values; leave others unchanged."""
    if pd.isna(val):
        return val
    if isinstance(val, (int, float, np.integer, np.floating)):
        if np.isfinite(val):
            scale = max(abs(val) * 0.01, 0.1)
            return val + random.gauss(0, scale)
    return val


def synthesize(df: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    """Build *n* synthetic rows by sampling with replacement and perturbing."""
    indices = rng.choices(df.index.tolist(), k=n)
    new_rows = []
    for idx in indices:
        row = df.loc[idx].copy()
        for col in df.select_dtypes(include=["number"]).columns:
            row[col] = perturb_value(row[col])
        if "Gene" in df.columns:
            row["Gene"] = rng.choice(df["Gene"].tolist())
        new_rows.append(row)
    return pd.DataFrame(new_rows, columns=df.columns)


# ===========================================================================
# Trio synthesis helpers
# ===========================================================================

def _assign_inheritance_modes(n: int, rng: random.Random) -> list:
    """Draw *n* inheritance modes from the canonical distribution."""
    return rng.choices(_IMODE_MODES, weights=_IMODE_WEIGHTS, k=n)


def _assign_parent_carriage(inh_modes: list, rng: random.Random) -> tuple:
    """Return (father_has, mother_has) boolean lists — one bool per variant.

    For *homozygous_partial* exactly one parent is chosen at random, so
    the two lists remain correlated (one True, one False per row).
    """
    father_has, mother_has = [], []
    for mode in inh_modes:
        if mode == "de_novo":
            f, m = False, False
        elif mode == "paternal":
            f, m = True, False
        elif mode == "maternal":
            f, m = False, True
        elif mode in ("biparental_het", "autosomal_recessive"):
            f, m = True, True
        elif mode == "de_novo_hom":
            f, m = False, False
        elif mode == "homozygous_partial":
            f = rng.random() < 0.5
            m = not f
        else:
            f, m = False, False
        father_has.append(f)
        mother_has.append(m)
    return father_has, mother_has


def _apply_vaf(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """Adjust coverage columns so VAF matches the assigned zygosity.

    Homozygosity → VAF 0.95–1.00; Heterozygosity → VAF 0.45–0.55.
    Modifies *Sample_Coverage_Alternate* and *Sample_Coverage_Reference* in
    place (on a copy).
    """
    df = df.copy().reset_index(drop=True)
    for i in range(len(df)):
        is_hom = str(df.at[i, "Sample_Zygosity"]) == "Homozygosity"
        cov = df.at[i, "Sample_Coverage"]
        try:
            cov = float(cov)
            if not np.isfinite(cov):
                cov = 30.0
        except (TypeError, ValueError):
            cov = 30.0
        cov = max(cov, 5.0)
        target_vaf = rng.uniform(0.95, 1.00) if is_hom else rng.uniform(0.45, 0.55)
        alt = max(0.0, round(cov * target_vaf + rng.gauss(0, 0.3), 2))
        ref = round(cov - alt + rng.gauss(0, 0.3), 2)
        df.at[i, "Sample_Coverage_Alternate"] = alt
        df.at[i, "Sample_Coverage_Reference"] = ref
    return df


def _flip_vus_to_pathogenic(df: pd.DataFrame, rate: float,
                             rng: random.Random) -> pd.DataFrame:
    """Flip *rate* fraction of VUS rows to 'Pathogenic' as label noise."""
    if "Classification" not in df.columns:
        return df
    df = df.copy()
    vus_idx = df.index[
        df["Classification"].str.contains("Uncertain", case=False, na=False)
    ].tolist()
    n_flip = max(0, int(len(vus_idx) * rate))
    if n_flip:
        df.loc[rng.sample(vus_idx, n_flip), "Classification"] = "Pathogenic"
    return df


def _synthesize_proband(base_df: pd.DataFrame, n: int, rng: random.Random,
                         sample_id: str) -> tuple:
    """Synthesize proband DataFrame with zygosity + VAF per inheritance mode.

    Returns (proband_df, inh_modes).
    """
    synth = synthesize(base_df, n, rng).reset_index(drop=True)
    inh_modes = _assign_inheritance_modes(n, rng)

    for i, mode in enumerate(inh_modes):
        is_hom = mode in {"autosomal_recessive", "de_novo_hom", "homozygous_partial"}
        synth.at[i, "Sample_Zygosity"] = "Homozygosity" if is_hom else "Heterozygosity"

    synth = _apply_vaf(synth, rng)
    synth = _add_sample_column(synth, sample_id)
    synth = _flip_vus_to_pathogenic(synth, 0.05, rng)
    return synth, inh_modes


def _synthesize_parent(proband_df: pd.DataFrame, parent_has: list,
                        sample_id: str, rng: random.Random) -> pd.DataFrame:
    """Build a parent CSV from proband rows selected by *parent_has* booleans.

    Selected variants are reset to Heterozygosity with realistic VAF.
    """
    proband_df = proband_df.reset_index(drop=True)
    keep = [i for i, has in enumerate(parent_has) if has]

    if keep:
        parent_df = proband_df.iloc[keep].copy().reset_index(drop=True)
        parent_df["Sample_Zygosity"] = "Heterozygosity"
        parent_df = _apply_vaf(parent_df, rng)
    else:
        parent_df = pd.DataFrame(columns=proband_df.columns)

    parent_df = _add_sample_column(parent_df, sample_id)
    return parent_df


# ===========================================================================
# Phenotype generation helpers (mirrors src/generate_phenotypes.py)
# ===========================================================================

def _sample_hpo(role: str, affected: int, rng: random.Random) -> str:
    if role == "proband":
        core = rng.sample(_PROBAND_CORE_HPO, k=min(2, len(_PROBAND_CORE_HPO)))
        extra = rng.sample(
            [h for h in _HPO_POOL if h not in core], k=rng.choice([0, 1])
        )
        return ";".join(core + extra)
    if affected == 1:
        return ";".join(rng.sample(_HPO_POOL, k=rng.randint(1, 2)))
    return rng.choice(_HPO_POOL) if rng.random() < 0.30 else ""


def _sample_age(role: str, affected: int,
                np_rng: np.random.Generator) -> str:
    if role == "proband":
        return f"{max(float(np_rng.gamma(2, 5)), 0.0):.1f}"
    if affected == 1:
        return f"{max(float(np_rng.gamma(5, 8)), 18.0):.1f}"
    return ""


def _phenotype_row(sample_id: str, role: str,
                   rng: random.Random, np_rng: np.random.Generator) -> dict:
    affected = 1 if role == "proband" else (1 if rng.random() < 0.10 else 0)
    sex = {"father": "M", "mother": "F"}.get(role, rng.choice(["M", "F"]))
    notes_pool = (
        _PROBAND_NOTES if role == "proband"
        else (_PARENT_AFFECTED_NOTES if affected else _PARENT_UNAFFECTED_NOTES)
    )
    return {
        "sample_id":          sample_id,
        "hpo_terms":          _sample_hpo(role, affected, rng),
        "affected":           affected,
        "age_of_onset_years": _sample_age(role, affected, np_rng),
        "sex":                sex,
        "notes":              rng.choice(notes_pool),
    }


# ===========================================================================
# TSV manifest helpers
# ===========================================================================

def _next_trio_number(trios_path: str) -> int:
    """Return the next integer trio index by inspecting the existing trios.tsv."""
    if not os.path.exists(trios_path):
        return 1
    trios = pd.read_csv(trios_path, sep="\t")
    if trios.empty:
        return 1
    nums = trios["trio_id"].str.extract(r"(\d+)$").astype(float).iloc[:, 0].dropna()
    return int(nums.max()) + 1 if not nums.empty else 1


def _next_sample_number(data_dir: str, prefix: str) -> int:
    """Return the next integer sample index by scanning *data_dir* for *prefix* files."""
    try:
        files = os.listdir(data_dir)
    except FileNotFoundError:
        return 1
    nums = []
    for name in files:
        if name.startswith(prefix) and name.endswith(".csv"):
            m = re.search(r"(\d+)", name[len(prefix):])
            if m:
                nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


def _append_to_trios(trios_path: str, new_rows: list) -> None:
    new_df = pd.DataFrame(new_rows, columns=[
        "trio_id", "proband", "father", "mother", "phenotype_group"
    ])
    if os.path.exists(trios_path):
        combined = pd.concat(
            [pd.read_csv(trios_path, sep="\t"), new_df], ignore_index=True
        )
    else:
        combined = new_df
    combined.to_csv(trios_path, sep="\t", index=False)


def _append_to_phenotypes(phenotypes_path: str, new_rows: list) -> None:
    cols = ["sample_id", "hpo_terms", "affected", "age_of_onset_years", "sex", "notes"]
    new_df = pd.DataFrame(new_rows, columns=cols)
    if os.path.exists(phenotypes_path):
        combined = pd.concat(
            [pd.read_csv(phenotypes_path, sep="\t"), new_df], ignore_index=True
        )
    else:
        combined = new_df
    combined.to_csv(phenotypes_path, sep="\t", index=False)


# ===========================================================================
# Trio generation orchestrator
# ===========================================================================

def generate_trios(args) -> None:
    """Generate *--trio-count* linked proband/father/mother CSV triplets."""
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    os.makedirs(args.data_dir, exist_ok=True)
    base_df = load_table(args.input)

    prefix = args.sample_prefix
    trio_start   = _next_trio_number(args.trios_tsv)
    sample_start = _next_sample_number(args.data_dir, prefix)

    new_trio_rows  = []
    new_pheno_rows = []

    for i in range(args.trio_count):
        trio_num   = trio_start   + i
        sample_num = sample_start + i

        trio_id        = f"T{trio_num:02d}"
        sample_base    = f"{prefix}{sample_num:02d}"
        proband_id     = f"{sample_base}_proband"
        father_id      = f"{sample_base}_father"
        mother_id      = f"{sample_base}_mother"

        print(f"\n[{trio_id}] generating {proband_id} …")

        proband_df, inh_modes = _synthesize_proband(
            base_df, args.nrows, rng, proband_id
        )
        father_has, mother_has = _assign_parent_carriage(inh_modes, rng)
        father_df = _synthesize_parent(proband_df, father_has, father_id, rng)
        mother_df = _synthesize_parent(proband_df, mother_has, mother_id, rng)

        for df_role, sid in [
            (proband_df, proband_id),
            (father_df,  father_id),
            (mother_df,  mother_id),
        ]:
            out_path = os.path.join(args.data_dir, f"{sid}.csv")
            df_role.to_csv(out_path, index=False)
            print(f"  wrote {out_path}  ({len(df_role)} rows)")

        new_trio_rows.append({
            "trio_id":         trio_id,
            "proband":         proband_id,
            "father":          father_id,
            "mother":          mother_id,
            "phenotype_group": "hearing_loss",
        })
        for role, sid in [
            ("proband", proband_id),
            ("father",  father_id),
            ("mother",  mother_id),
        ]:
            new_pheno_rows.append(_phenotype_row(sid, role, rng, np_rng))

    _append_to_trios(args.trios_tsv, new_trio_rows)
    print(f"\nUpdated {args.trios_tsv}  (+{len(new_trio_rows)} trio rows)")

    _append_to_phenotypes(args.phenotypes_tsv, new_pheno_rows)
    print(f"Updated {args.phenotypes_tsv}  (+{len(new_pheno_rows)} phenotype rows)")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic variant CSV files (single-file or trio mode)"
    )

    # shared
    parser.add_argument("-i", "--input", required=True,
                        help="source variant CSV to synthesise from")
    parser.add_argument("-n", "--nrows", type=int, default=300,
                        help="rows per synthesised file (default: 300)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for reproducibility")

    # single-file mode
    parser.add_argument("-o", "--output", default=None,
                        help="output path (single-file mode); supports %%d for numbered files")
    parser.add_argument("-m", "--count", type=int, default=1,
                        help="number of single files to produce (single-file mode)")

    # trio mode
    parser.add_argument("--trio", action="store_true",
                        help="enable trio mode: generate linked proband+father+mother triplets")
    parser.add_argument("--trio-count", type=int, default=1,
                        help="number of trios to generate (trio mode, default: 1)")
    parser.add_argument("--data-dir", default="data",
                        help="output directory for trio CSVs (default: data)")
    parser.add_argument("--sample-prefix", default="SYNTH",
                        help="prefix for auto-named sample IDs (default: SYNTH)")
    parser.add_argument("--trios-tsv", default="data/trios.tsv",
                        help="trios manifest to update (default: data/trios.tsv)")
    parser.add_argument("--phenotypes-tsv", default="data/phenotypes.tsv",
                        help="phenotypes manifest to update (default: data/phenotypes.tsv)")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.trio:
        print(f"Trio mode: generating {args.trio_count} trio(s) into '{args.data_dir}/' …")
        generate_trios(args)
        return

    # --- original single-file path ---
    if args.output is None:
        parser.error("-o/--output is required in single-file mode")

    df = load_table(args.input)

    for i in range(args.count):
        out_path = args.output if args.count == 1 else args.output % (i + 1)
        name = os.path.splitext(os.path.basename(out_path))[0]
        synth = synthesize(df, args.nrows, random)
        synth = _add_sample_column(synth, name)
        synth.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(synth)} rows) from {args.input}")


if __name__ == "__main__":
    main()
