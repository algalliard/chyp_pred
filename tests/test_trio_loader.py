"""
tests/test_trio_loader.py
-------------------------
Unit tests for src.trio_loader.TrioLoader and its helpers.

Tests use a fully in-memory fixture (no real files required for most tests)
plus a lightweight on-disk integration test that writes temporary CSV/TSV
files and exercises the full load_all() path.
"""

from __future__ import annotations

import os
import sys
import textwrap
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Ensure the project root is on sys.path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trio_loader import (
    TrioLoader,
    _infer_inheritance,
    _is_homozygous,
    _build_variant_set,
)


# ---------------------------------------------------------------------------
# Unit tests — pure helper functions
# ---------------------------------------------------------------------------

class TestIsHomozygous:
    def test_homozygosity_exact(self):
        assert _is_homozygous("Homozygosity") is True

    def test_homozygosity_lower(self):
        assert _is_homozygous("homozygosity") is True

    def test_hom_prefix(self):
        assert _is_homozygous("hom") is True

    def test_heterozygosity(self):
        assert _is_homozygous("Heterozygosity") is False

    def test_empty_string(self):
        assert _is_homozygous("") is False

    def test_nan(self):
        assert _is_homozygous(float("nan")) is False

    def test_none_like(self):
        assert _is_homozygous(None) is False


class TestInferInheritance:
    @pytest.mark.parametrize("zyg,father,mother,expected", [
        ("Heterozygosity", False, False, "de_novo"),
        ("Heterozygosity", True,  False, "paternal"),
        ("Heterozygosity", False, True,  "maternal"),
        ("Heterozygosity", True,  True,  "biparental_het"),
        ("Homozygosity",   True,  True,  "autosomal_recessive"),
        ("Homozygosity",   False, False, "de_novo_hom"),
        ("Homozygosity",   True,  False, "homozygous_partial"),
        ("Homozygosity",   False, True,  "homozygous_partial"),
        # unknown/empty zygosity treated as heterozygous
        ("",               False, False, "de_novo"),
        ("",               True,  True,  "biparental_het"),
    ])
    def test_all_combos(self, zyg, father, mother, expected):
        assert _infer_inheritance(zyg, father, mother) == expected


class TestBuildVariantSet:
    def _make_df(self, rows):
        return pd.DataFrame(rows, columns=["Gene", "HGVS_Coding"])

    def test_basic(self):
        df = self._make_df([("GJB2", "c.35delG"), ("OTOF", "c.100A>T")])
        vs = _build_variant_set(df)
        assert ("GJB2", "c.35delG") in vs
        assert ("OTOF", "c.100A>T") in vs

    def test_excludes_empty_gene(self):
        df = self._make_df([("", "c.35delG"), ("GJB2", "c.35delG")])
        vs = _build_variant_set(df)
        assert ("", "c.35delG") not in vs
        assert ("GJB2", "c.35delG") in vs

    def test_excludes_nan(self):
        df = self._make_df([(None, "c.35delG"), ("GJB2", None)])
        vs = _build_variant_set(df)
        assert len(vs) == 0

    def test_whitespace_stripped(self):
        df = self._make_df([("  GJB2  ", "  c.35delG  ")])
        # strip happens in _load_sample; set should still contain stripped keys
        # but here the DataFrame has not been through _load_sample, so keys
        # are already stripped before we call _build_variant_set
        df["Gene"] = df["Gene"].str.strip()
        df["HGVS_Coding"] = df["HGVS_Coding"].str.strip()
        vs = _build_variant_set(df)
        assert ("GJB2", "c.35delG") in vs


# ---------------------------------------------------------------------------
# Integration test — temporary on-disk fixture
# ---------------------------------------------------------------------------

def _write_variant_csv(path: Path, sample_id: str, rows: list[dict]):
    """Write a minimal variant CSV in the format expected by TrioLoader."""
    base_cols = [
        "Sample", "Gene", "Classification", "HGVS_Coding",
        "Sample_Zygosity", "Sample_Coverage",
        "Sample_Coverage_Alternate", "Sample_Coverage_Reference",
        "Pathogenicity_Score", "Effect", "Criteria",
        "Condition_Inheritance", "Functional", "Population",
    ]
    records = []
    for r in rows:
        rec = {c: r.get(c, "") for c in base_cols}
        rec["Sample"] = sample_id
        records.append(rec)
    pd.DataFrame(records, columns=base_cols).to_csv(path, index=False)


@pytest.fixture()
def trio_tmpdir(tmp_path):
    """Build a minimal on-disk trio dataset and return path info."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # --- proband variants ------------------------------------------------
    proband_rows = [
        {"Gene": "GJB2", "HGVS_Coding": "c.35delG",
         "Sample_Zygosity": "Homozygosity", "Classification": "Pathogenic",
         "Pathogenicity_Score": "5.0"},
        {"Gene": "OTOF", "HGVS_Coding": "c.100A>T",
         "Sample_Zygosity": "Heterozygosity", "Classification": "Likely pathogenic",
         "Pathogenicity_Score": "4.5"},
        {"Gene": "CDH23", "HGVS_Coding": "c.200G>A",
         "Sample_Zygosity": "Heterozygosity", "Classification": "Uncertain significance",
         "Pathogenicity_Score": "3.0"},
    ]
    _write_variant_csv(data_dir / "P01_proband.csv", "P01_proband", proband_rows)

    # --- father variants (carries GJB2 and CDH23) -----------------------
    father_rows = [
        {"Gene": "GJB2", "HGVS_Coding": "c.35delG",
         "Sample_Zygosity": "Heterozygosity"},
        {"Gene": "CDH23", "HGVS_Coding": "c.200G>A",
         "Sample_Zygosity": "Heterozygosity"},
    ]
    _write_variant_csv(data_dir / "P01_father.csv", "P01_father", father_rows)

    # --- mother variants (carries GJB2 only) ----------------------------
    mother_rows = [
        {"Gene": "GJB2", "HGVS_Coding": "c.35delG",
         "Sample_Zygosity": "Heterozygosity"},
    ]
    _write_variant_csv(data_dir / "P01_mother.csv", "P01_mother", mother_rows)

    # --- trios.tsv -------------------------------------------------------
    trios_path = data_dir / "trios.tsv"
    trios_path.write_text(
        "trio_id\tproband\tfather\tmother\tphenotype_group\n"
        "T01\tP01_proband\tP01_father\tP01_mother\thearing_loss\n"
    )

    # --- phenotypes.tsv --------------------------------------------------
    pheno_path = data_dir / "phenotypes.tsv"
    pheno_path.write_text(
        "sample_id\thpo_terms\taffected\tage_of_onset_years\tsex\tnotes\n"
        "P01_proband\tHP:0000407;HP:0008619\t1\t3.5\tF\ttest proband\n"
        "P01_father\t\t0\t\tM\tunaffected\n"
        "P01_mother\t\t0\t\tF\tunaffected\n"
    )

    return {
        "data_dir": data_dir,
        "trios_path": trios_path,
        "pheno_path": pheno_path,
    }


class TestTrioLoaderIntegration:
    def _loader(self, fixture):
        return TrioLoader(
            trios_path=fixture["trios_path"],
            phenotypes_path=fixture["pheno_path"],
            data_dir=fixture["data_dir"],
        )

    def test_load_all_returns_dataframe(self, trio_tmpdir):
        df = self._loader(trio_tmpdir).load_all()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # three proband variants

    def test_trio_id_column_present(self, trio_tmpdir):
        df = self._loader(trio_tmpdir).load_all()
        assert "trio_id" in df.columns
        assert (df["trio_id"] == "T01").all()

    def test_inheritance_gjb2_autosomal_recessive(self, trio_tmpdir):
        df = self._loader(trio_tmpdir).load_all()
        gjb2 = df[df["Gene"] == "GJB2"].iloc[0]
        assert gjb2["inheritance_mode"] == "autosomal_recessive"
        assert gjb2["father_has_variant"] == True
        assert gjb2["mother_has_variant"] == True

    def test_inheritance_otof_de_novo(self, trio_tmpdir):
        df = self._loader(trio_tmpdir).load_all()
        otof = df[df["Gene"] == "OTOF"].iloc[0]
        assert otof["inheritance_mode"] == "de_novo"
        assert otof["father_has_variant"] == False
        assert otof["mother_has_variant"] == False

    def test_inheritance_cdh23_paternal(self, trio_tmpdir):
        df = self._loader(trio_tmpdir).load_all()
        cdh23 = df[df["Gene"] == "CDH23"].iloc[0]
        assert cdh23["inheritance_mode"] == "paternal"
        assert cdh23["father_has_variant"] == True
        assert cdh23["mother_has_variant"] == False

    def test_phenotype_columns_populated(self, trio_tmpdir):
        df = self._loader(trio_tmpdir).load_all()
        assert "proband_phenotype_hpo" in df.columns
        assert "affected" in df.columns
        assert df["proband_phenotype_hpo"].iloc[0] == "HP:0000407;HP:0008619"
        assert df["affected"].iloc[0] == "1"

    def test_load_single_trio(self, trio_tmpdir):
        loader = self._loader(trio_tmpdir)
        df = loader.load_trio("T01")
        assert len(df) == 3

    def test_load_trio_invalid_id_raises(self, trio_tmpdir):
        loader = self._loader(trio_tmpdir)
        with pytest.raises(ValueError, match="not found"):
            loader.load_trio("T99")

    def test_missing_csv_warns_and_skips(self, trio_tmpdir):
        # Remove father file — trio should be skipped with a warning
        (trio_tmpdir["data_dir"] / "P01_father.csv").unlink()
        df = self._loader(trio_tmpdir).load_all()
        assert len(df) == 0  # trio skipped
