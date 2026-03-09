"""
src/trio_loader.py
------------------
Load and assemble proband-centric variant DataFrames for every family trio
defined in data/trios.tsv, enriched with phenotype data from
data/phenotypes.tsv and family-inheritance context inferred by joining
variant keys (Gene + HGVS_Coding) across the three family members.

Typical usage
-------------
    from src.trio_loader import TrioLoader

    loader = TrioLoader(
        trios_path="data/trios.tsv",
        phenotypes_path="data/phenotypes.tsv",
        data_dir="data",
    )
    df = loader.load_all()   # returns a proband-level variant DataFrame
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns used to match a variant across family members
JOIN_KEYS = ["Gene", "HGVS_Coding"]

# Zygosity strings considered "homozygous" (case-insensitive prefix match)
HOMOZYGOUS_PREFIXES = ("homozygous", "hom")

INHERITANCE_COLUMNS = [
    "trio_id",
    "inheritance_mode",
    "proband_zygosity",
    "father_has_variant",
    "mother_has_variant",
    "proband_phenotype_hpo",
    "affected",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _is_homozygous(zygosity: str | float) -> bool:
    """Return True if the zygosity string indicates a homozygous call."""
    if pd.isna(zygosity):
        return False
    return str(zygosity).strip().lower().startswith(HOMOZYGOUS_PREFIXES)


def _infer_inheritance(zygosity: str | float,
                       father_has: bool,
                       mother_has: bool) -> str:
    """Infer inheritance mode from proband zygosity and parental presence.

    Rules (from GUIDELINES §6):
        Heterozygous + 0/0   → de_novo
        Heterozygous + 1/0   → paternal
        Heterozygous + 0/1   → maternal
        Heterozygous + 1/1   → biparental_het
        Homozygous   + 1/1   → autosomal_recessive
        Homozygous   + 0/0   → de_novo_hom
        Homozygous   + other → homozygous_partial   (one parent only)
    """
    hom = _is_homozygous(zygosity)
    if hom:
        if father_has and mother_has:
            return "autosomal_recessive"
        if not father_has and not mother_has:
            return "de_novo_hom"
        return "homozygous_partial"
    else:
        # heterozygous / unknown
        if not father_has and not mother_has:
            return "de_novo"
        if father_has and not mother_has:
            return "paternal"
        if not father_has and mother_has:
            return "maternal"
        return "biparental_het"


def _build_variant_set(df: pd.DataFrame) -> set[tuple]:
    """Return a set of (Gene, HGVS_Coding) tuples present in a DataFrame.

    Rows where either key is NaN / empty are excluded from the set so they
    don't produce false-positive inheritance matches.
    """
    valid = df[JOIN_KEYS].dropna()
    valid = valid[valid["Gene"].str.strip() != ""]
    valid = valid[valid["HGVS_Coding"].str.strip() != ""]
    return set(zip(valid["Gene"], valid["HGVS_Coding"]))


# ---------------------------------------------------------------------------
# TrioLoader
# ---------------------------------------------------------------------------

class TrioLoader:
    """Load all trios and return a single proband-variant DataFrame.

    Parameters
    ----------
    trios_path : str or Path
        Path to the TSV file defining trios (columns: trio_id, proband,
        father, mother, phenotype_group).
    phenotypes_path : str or Path
        Path to the TSV file with per-sample phenotype data (columns:
        sample_id, hpo_terms, affected, age_of_onset_years, sex, notes).
    data_dir : str or Path
        Directory that contains the per-sample variant CSV files.
    """

    def __init__(
        self,
        trios_path: str | Path = "data/trios.tsv",
        phenotypes_path: str | Path = "data/phenotypes.tsv",
        data_dir: str | Path = "data",
    ):
        self.trios_path = Path(trios_path)
        self.phenotypes_path = Path(phenotypes_path)
        self.data_dir = Path(data_dir)

        self._trios: pd.DataFrame | None = None
        self._phenotypes: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> pd.DataFrame:
        """Load every trio and return a proband-level variant DataFrame.

        The returned DataFrame contains one row per proband variant.  Each
        row is annotated with:
            - trio_id, inheritance_mode, proband_zygosity
            - father_has_variant, mother_has_variant
            - proband_phenotype_hpo, affected (from phenotypes.tsv)

        Returns
        -------
        pd.DataFrame
            Proband variants with family-context columns appended.
        """
        self._trios = pd.read_csv(self.trios_path, sep="\t")
        self._phenotypes = pd.read_csv(
            self.phenotypes_path, sep="\t", dtype=str
        ).fillna("")

        frames: list[pd.DataFrame] = []
        for _, trio_row in self._trios.iterrows():
            try:
                trio_df = self._load_trio(trio_row)
                frames.append(trio_df)
            except FileNotFoundError as exc:
                print(f"[TrioLoader] WARNING – skipping trio "
                      f"{trio_row['trio_id']}: {exc}")

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        return result

    def load_trio(self, trio_id: str) -> pd.DataFrame:
        """Load a single trio by its ID.

        Convenience wrapper around ``load_all`` for interactive use.
        """
        if self._trios is None:
            self._trios = pd.read_csv(self.trios_path, sep="\t")
        if self._phenotypes is None:
            self._phenotypes = pd.read_csv(
                self.phenotypes_path, sep="\t", dtype=str
            ).fillna("")

        trio_row = self._trios[self._trios["trio_id"] == trio_id]
        if trio_row.empty:
            raise ValueError(f"Trio '{trio_id}' not found in {self.trios_path}")
        return self._load_trio(trio_row.iloc[0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _csv_path(self, sample_id: str) -> Path:
        return self.data_dir / f"{sample_id}.csv"

    def _load_sample(self, sample_id: str) -> pd.DataFrame:
        path = self._csv_path(sample_id)
        if not path.exists():
            raise FileNotFoundError(f"Sample CSV not found: {path}")

        # Support both plain CSVs (synthetic) and eVAI files that start with
        # quoted "##…" metadata lines.  Strip those lines before parsing so
        # real patient files work identically to synthetic ones.
        from io import StringIO
        data_lines = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.lstrip('"').startswith("##"):
                    continue
                data_lines.append(line)
        df = pd.read_csv(StringIO("".join(data_lines)), dtype=str).fillna("")

        # Ensure a 'Sample' column exists; use sample_id as fallback so
        # real eVAI files (which have no Sample column) work transparently.
        if "Sample" not in df.columns:
            df.insert(0, "Sample", sample_id)

        # normalise join-key whitespace
        for col in JOIN_KEYS:
            if col in df.columns:
                df[col] = df[col].str.strip()
        return df

    def _get_phenotype(self, sample_id: str) -> dict:
        """Return phenotype dict for a sample, empty strings if not found."""
        row = self._phenotypes[
            self._phenotypes["sample_id"] == sample_id
        ]
        if row.empty:
            return {"hpo_terms": "", "affected": ""}
        r = row.iloc[0]
        return {
            "hpo_terms": r.get("hpo_terms", ""),
            "affected": r.get("affected", ""),
        }

    def _load_trio(self, trio_row: pd.Series) -> pd.DataFrame:
        """Load one trio and return an annotated proband DataFrame."""
        trio_id = trio_row["trio_id"]
        proband_id = trio_row["proband"]
        father_id = trio_row["father"]
        mother_id = trio_row["mother"]

        proband_df = self._load_sample(proband_id)
        father_df = self._load_sample(father_id)
        mother_df = self._load_sample(mother_id)

        father_variants = _build_variant_set(father_df)
        mother_variants = _build_variant_set(mother_df)

        proband_pheno = self._get_phenotype(proband_id)

        # annotate proband rows
        proband_df = proband_df.copy()
        proband_df["trio_id"] = trio_id

        def _has_variant(row: pd.Series, variant_set: set) -> bool:
            gene = row.get("Gene", "").strip()
            hgvs = row.get("HGVS_Coding", "").strip()
            if not gene or not hgvs:
                return False
            return (gene, hgvs) in variant_set

        proband_df["father_has_variant"] = proband_df.apply(
            lambda r: _has_variant(r, father_variants), axis=1
        )
        proband_df["mother_has_variant"] = proband_df.apply(
            lambda r: _has_variant(r, mother_variants), axis=1
        )
        proband_df["proband_zygosity"] = proband_df.get(
            "Sample_Zygosity", pd.Series("", index=proband_df.index)
        )
        proband_df["inheritance_mode"] = proband_df.apply(
            lambda r: _infer_inheritance(
                r["proband_zygosity"],
                r["father_has_variant"],
                r["mother_has_variant"],
            ),
            axis=1,
        )
        proband_df["proband_phenotype_hpo"] = proband_pheno["hpo_terms"]
        proband_df["affected"] = proband_pheno["affected"]

        return proband_df
