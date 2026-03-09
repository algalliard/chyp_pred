#!/usr/bin/env python3
"""
Generate a synthetic phenotype annotation file (phenotypes.tsv) for every
sample listed in a trio definition file (trios.tsv).

The output mirrors what a real clinical annotation sheet would look like:
  - Probands are always clinically affected (affected=1).
  - Parents are affected with ~10 % probability; carriers have no explicit
    flag here (affected=0 covers both unaffected and silent carriers).
  - HPO terms are sampled from a curated hearing-disorder pool.
  - Age of onset follows a Gamma(shape=2, scale=5) distribution for probands
    and a Gamma(shape=5, scale=8) for affected parents.

Usage
-----
    python src/generate_phenotypes.py \\
        --trios   data/trios.tsv \\
        --output  data/phenotypes.tsv \\
        --seed    42

Arguments
---------
--trios   Path to the trio TSV file (default: data/trios.tsv)
--output  Output path for the phenotype TSV (default: data/phenotypes.tsv)
--seed    Integer random seed for full reproducibility (optional)
"""

import argparse
import random
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# HPO term pool — hearing-disorder relevant terms
# ---------------------------------------------------------------------------
HPO_POOL = [
    "HP:0000407",   # Sensorineural hearing impairment
    "HP:0008619",   # Bilateral sensorineural hearing impairment
    "HP:0000365",   # Impaired hearing
    "HP:0001751",   # Vertigo
    "HP:0000370",   # Abnormality of the middle ear
    "HP:0000377",   # Abnormal pinna morphology
    "HP:0000411",   # Protruding ear
    "HP:0000405",   # Conductive hearing impairment
    "HP:0000358",   # Posteriorly rotated ears
    "HP:0000remote",# placeholder — removed below
]
# remove placeholder
HPO_POOL = [h for h in HPO_POOL if "remote" not in h]

# Hearing-specific "core" HPO terms that probands are most likely to carry
PROBAND_CORE_HPO = ["HP:0000407", "HP:0008619", "HP:0000365"]

# Effect-variant notes per role
PROBAND_NOTES = [
    "congenital bilateral SNHL",
    "early-onset sensorineural hearing loss",
    "profound bilateral hearing loss, cochlear implant candidate",
    "moderate-severe bilateral SNHL",
    "bilateral hearing loss, unknown onset",
]
PARENT_UNAFFECTED_NOTES = [
    "unaffected, obligate carrier suspected",
    "normal hearing, no reported symptoms",
    "unaffected",
    "clinically unaffected",
]
PARENT_AFFECTED_NOTES = [
    "mild unilateral hearing loss",
    "age-related hearing impairment",
    "mild bilateral SNHL",
    "hearing aid user",
]


def sample_hpo_terms(role: str, affected: int, rng: random.Random) -> str:
    """Return a semicolon-separated string of HPO terms appropriate for the
    given role and affected status."""
    if role == "proband":
        # always at least 2 core terms; optionally add 1 extra
        core = rng.sample(PROBAND_CORE_HPO, k=min(2, len(PROBAND_CORE_HPO)))
        extra = rng.sample(
            [h for h in HPO_POOL if h not in core],
            k=rng.choice([0, 1])
        )
        return ";".join(core + extra)
    elif affected == 1:
        # affected parent: 1-2 mild HPO terms
        return ";".join(rng.sample(HPO_POOL, k=rng.randint(1, 2)))
    else:
        # unaffected parent: 30 % chance of a single mild carrier term
        if rng.random() < 0.30:
            return rng.choice(HPO_POOL)
        return ""


def sample_age_of_onset(role: str, affected: int, rng: random.Random,
                        np_rng: np.random.Generator) -> str:
    """Return age of onset in years as a string (rounded to 1 dp), or ''."""
    if role == "proband":
        age = float(np_rng.gamma(shape=2, scale=5))
        return f"{max(age, 0.0):.1f}"
    elif affected == 1:
        age = float(np_rng.gamma(shape=5, scale=8))
        return f"{max(age, 18.0):.1f}"
    return ""


def sample_sex(role: str, rng: random.Random) -> str:
    if role == "father":
        return "M"
    elif role == "mother":
        return "F"
    return rng.choice(["M", "F"])


def sample_notes(role: str, affected: int, rng: random.Random) -> str:
    if role == "proband":
        return rng.choice(PROBAND_NOTES)
    elif affected == 1:
        return rng.choice(PARENT_AFFECTED_NOTES)
    return rng.choice(PARENT_UNAFFECTED_NOTES)


def is_affected(role: str, rng: random.Random) -> int:
    """Probands always affected; parents affected with ~10 % probability."""
    if role == "proband":
        return 1
    return 1 if rng.random() < 0.10 else 0


def generate_phenotypes(trios_path: str, output_path: str, seed: int | None):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    trios = pd.read_csv(trios_path, sep="\t")

    rows = []
    for _, trio in trios.iterrows():
        for role in ("proband", "father", "mother"):
            sample_id = trio[role]
            affected = is_affected(role, rng)
            hpo = sample_hpo_terms(role, affected, rng)
            age = sample_age_of_onset(role, affected, rng, np_rng)
            sex = sample_sex(role, rng)
            notes = sample_notes(role, affected, rng)
            rows.append({
                "sample_id": sample_id,
                "hpo_terms": hpo,
                "affected": affected,
                "age_of_onset_years": age,
                "sex": sex,
                "notes": notes,
            })

    df = pd.DataFrame(rows, columns=[
        "sample_id", "hpo_terms", "affected",
        "age_of_onset_years", "sex", "notes",
    ])
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Wrote {len(df)} rows → {output_path}")

    # quick summary
    for role in ("proband", "father", "mother"):
        subset = df[df["sample_id"].str.endswith(f"_{role}")]
        n_aff = subset["affected"].sum()
        print(f"  {role:8s}: {len(subset)} samples, {n_aff} affected")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic phenotype annotations for trio samples"
    )
    parser.add_argument(
        "--trios", default="data/trios.tsv",
        help="path to trios TSV (default: data/trios.tsv)"
    )
    parser.add_argument(
        "--output", default="data/phenotypes.tsv",
        help="output TSV path (default: data/phenotypes.tsv)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="integer random seed for reproducibility"
    )
    args = parser.parse_args()
    generate_phenotypes(args.trios, args.output, args.seed)


if __name__ == "__main__":
    main()
