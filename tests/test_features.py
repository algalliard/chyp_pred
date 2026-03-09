"""
tests/test_features.py
----------------------
Unit tests for src/features.py
"""
import math
import pytest
import numpy as np
import pandas as pd

from src.features import (
    _parse_functional,
    _parse_population,
    _encode_effect,
    _encode_zygosity,
    _count_acmg_criteria,
    _compute_hpo_overlap,
    build_features,
    EFFECT_SEVERITY,
)


# ---------------------------------------------------------------------------
# _parse_functional
# ---------------------------------------------------------------------------

class TestParseFunctional:
    def test_all_fields_present(self):
        v = "PaPI:0.601|CADD:21.4|REVEL:0.021|PolyPhen2:0.019|SIFT:0.04"
        r = _parse_functional(v)
        assert math.isclose(r["papi_score"],      0.601, rel_tol=1e-5)
        assert math.isclose(r["cadd_score"],      21.4,  rel_tol=1e-5)
        assert math.isclose(r["revel_score"],     0.021, rel_tol=1e-5)
        assert math.isclose(r["polyphen2_score"], 0.019, rel_tol=1e-5)
        assert math.isclose(r["sift_score"],      0.04,  rel_tol=1e-5)

    def test_only_cadd(self):
        r = _parse_functional("CADD:39.0")
        assert math.isclose(r["cadd_score"], 39.0)
        assert r["revel_score"] is None
        assert r["sift_score"] is None

    def test_null_input(self):
        for val in (None, "", float("nan"), 42):
            r = _parse_functional(val)  # type: ignore[arg-type]
            assert all(v is None for v in r.values())

    def test_scientific_notation(self):
        r = _parse_functional("CADD:1.5e-2")
        assert math.isclose(r["cadd_score"], 0.015, rel_tol=1e-5)

    def test_partial_fields(self):
        v = "PaPI:1.0|CADD:24.0"
        r = _parse_functional(v)
        assert r["papi_score"] == 1.0
        assert r["cadd_score"] == 24.0
        assert r["revel_score"] is None


# ---------------------------------------------------------------------------
# _parse_population
# ---------------------------------------------------------------------------

class TestParsePopulation:
    def test_normal_value(self):
        v = "gnomAD:6.84067E-5%|ExAC:NA|1000GP:0%|ESP:0%"
        r = _parse_population(v)
        # 6.84067E-5 % / 100 ≈ 6.84067E-7
        assert r["gnomad_af"] == pytest.approx(6.84067e-5 / 100, rel=1e-5)

    def test_zero_af(self):
        v = "gnomAD:0%|ExAC:0%|1000GP:0%|ESP:0%"
        r = _parse_population(v)
        assert r["gnomad_af"] == 0.0

    def test_na_gnomad(self):
        v = "gnomAD:NA|ExAC:NA"
        r = _parse_population(v)
        assert r["gnomad_af"] is None

    def test_null_input(self):
        for val in (None, "", float("nan")):
            r = _parse_population(val)  # type: ignore[arg-type]
            assert r["gnomad_af"] is None

    def test_decimal_af(self):
        v = "gnomAD:0.013873800000000002%|ExAC:0%|1000GP:0%|ESP:0%"
        r = _parse_population(v)
        assert r["gnomad_af"] == pytest.approx(0.013873800000000002 / 100)


# ---------------------------------------------------------------------------
# _encode_effect
# ---------------------------------------------------------------------------

class TestEncodeEffect:
    def test_stop_gained_greater_than_missense(self):
        assert _encode_effect("stop_gained") > _encode_effect("missense_variant")

    def test_missense_greater_than_synonymous(self):
        assert _encode_effect("missense_variant") > _encode_effect("synonymous_variant")

    def test_pipe_separated_takes_max(self):
        combined = _encode_effect("missense_variant|stop_gained")
        solo     = _encode_effect("stop_gained")
        assert combined == solo

    def test_unknown_effect_returns_minus1(self):
        assert _encode_effect("fairy_tale_variant") == -1

    def test_empty_string(self):
        assert _encode_effect("") == -1

    def test_none(self):
        assert _encode_effect(None)  == -1  # type: ignore

    def test_frameshift_high_severity(self):
        assert _encode_effect("frameshift_variant") >= _encode_effect("missense_variant")


# ---------------------------------------------------------------------------
# _encode_zygosity
# ---------------------------------------------------------------------------

class TestEncodeZygosity:
    def test_homozygosity(self):
        assert _encode_zygosity("Homozygosity") == 1

    def test_heterozygosity(self):
        assert _encode_zygosity("Heterozygosity") == 0

    def test_case_insensitive(self):
        assert _encode_zygosity("homozygosity") == 1
        assert _encode_zygosity("Heterozygous") == 0

    def test_unknown(self):
        assert _encode_zygosity("compound_het") == -1
        assert _encode_zygosity("") == -1
        assert _encode_zygosity(None) == -1  # type: ignore


# ---------------------------------------------------------------------------
# _count_acmg_criteria
# ---------------------------------------------------------------------------

class TestCountAcmgCriteria:
    def test_two_criteria(self):
        assert _count_acmg_criteria("PVS1-very strong|PM2-supporting") == 2

    def test_three_criteria(self):
        assert _count_acmg_criteria("PM2-moderate|PP3-moderate|BP1-supporting") == 3

    def test_single(self):
        assert _count_acmg_criteria("PM2-moderate") == 1

    def test_empty(self):
        assert _count_acmg_criteria("") == 0
        assert _count_acmg_criteria(None) == 0  # type: ignore


# ---------------------------------------------------------------------------
# _compute_hpo_overlap
# ---------------------------------------------------------------------------

class TestComputeHpoOverlap:
    def test_no_proband_terms_returns_zero(self):
        assert _compute_hpo_overlap("", "") == 0.0
        assert _compute_hpo_overlap(None, "OMIM:601544") == 0.0  # type: ignore

    def test_known_hearing_term_nonzero(self):
        # HP:0000407 is in FALLBACK_DISEASE_HPO; should yield > 0 even with
        # no Condition_ID (which triggers fallback set)
        score = _compute_hpo_overlap("HP:0000407", "")
        assert score > 0.0

    def test_returns_float(self):
        score = _compute_hpo_overlap("HP:0000407;HP:0008619", "")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_semicolon_and_comma_separated(self):
        s1 = _compute_hpo_overlap("HP:0000407;HP:0000365", "")
        s2 = _compute_hpo_overlap("HP:0000407,HP:0000365", "")
        assert math.isclose(s1, s2, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# build_features — integration tests
# ---------------------------------------------------------------------------

def _make_minimal_df(n: int = 4) -> pd.DataFrame:
    """Create a minimal DataFrame mimicking TrioLoader output."""
    rng = np.random.default_rng(0)

    records = []
    for i in range(n):
        records.append({
            "Gene": f"GENE{i}",
            "HGVS_Coding": f"c.{i}A>T",
            "Classification": ["Pathogenic", "Likely pathogenic",
                                "Uncertain significance", "Benign"][i % 4],
            "Functional":  "PaPI:0.9|CADD:30.0|REVEL:0.8|PolyPhen2:0.9|SIFT:0.0",
            "Population":  "gnomAD:1.0E-5%|ExAC:NA|1000GP:0%|ESP:0%",
            "Effect":      ["stop_gained", "missense_variant",
                            "synonymous_variant", "frameshift_variant"][i % 4],
            "Sample_Zygosity": ["Homozygosity", "Heterozygosity"][i % 2],
            "Condition_Inheritance": "Autosomal recessive",
            "Criteria":    "PVS1-very strong|PM2-supporting",
            "Condition_ID": "",
            "Sample_Coverage":           int(rng.integers(30, 200)),
            "Sample_Coverage_Alternate": int(rng.integers(5, 80)),
            "Sample_Coverage_Reference": int(rng.integers(5, 80)),
            "Pathogenicity_Score":       round(float(rng.random()), 4),
            "inheritance_mode":          "de_novo",
            "father_has_variant":        False,
            "mother_has_variant":        False,
            "proband_phenotype_hpo":     "HP:0000407;HP:0008619",
            "trio_id":                   f"T0{i+1}",
        })
    return pd.DataFrame(records)


class TestBuildFeatures:
    def test_returns_tuple_of_two(self):
        df = _make_minimal_df()
        result = build_features(df)
        assert isinstance(result, tuple) and len(result) == 2

    def test_X_and_y_same_length(self):
        df = _make_minimal_df(8)
        X, y = build_features(df)
        assert len(X) == len(y) == 8

    def test_y_is_binary(self):
        df = _make_minimal_df(8)
        _, y = build_features(df)
        assert set(y.unique()).issubset({0, 1})

    def test_y_pathogenic_rows_labelled_1(self):
        df = _make_minimal_df(4)
        _, y = build_features(df)
        # Row 0 → "Pathogenic", row 1 → "Likely pathogenic"
        assert y.iloc[0] == 1
        assert y.iloc[1] == 1
        assert y.iloc[2] == 0  # Uncertain significance
        assert y.iloc[3] == 0  # Benign

    def test_cadd_parsed_correctly(self):
        df = _make_minimal_df(2)
        X, _ = build_features(df)
        assert "cadd_score" in X.columns
        assert X["cadd_score"].iloc[0] == pytest.approx(30.0)

    def test_gnomad_af_is_proportion(self):
        df = _make_minimal_df(2)
        X, _ = build_features(df)
        assert "gnomad_af" in X.columns
        assert X["gnomad_af"].iloc[0] == pytest.approx(1.0e-5 / 100)

    def test_vaf_in_zero_one(self):
        df = _make_minimal_df(4)
        X, _ = build_features(df)
        valid = X["vaf"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_inh_onehot_present(self):
        df = _make_minimal_df(4)
        X, _ = build_features(df)
        inh_cols = [c for c in X.columns if c.startswith("inh_")]
        assert len(inh_cols) >= 1

    def test_ci_onehot_present(self):
        df = _make_minimal_df(4)
        X, _ = build_features(df)
        ci_cols = [c for c in X.columns if c.startswith("ci_")]
        assert len(ci_cols) >= 1

    def test_hpo_overlap_score_in_range(self):
        df = _make_minimal_df(4)
        X, _ = build_features(df)
        assert "hpo_overlap_score" in X.columns
        s = X["hpo_overlap_score"]
        assert (s >= 0).all() and (s <= 1).all()

    def test_criteria_score_correct(self):
        df = _make_minimal_df(4)
        X, _ = build_features(df)
        # All rows have "PVS1-very strong|PM2-supporting" → 2 criteria
        assert (X["criteria_score"] == 2).all()

    def test_zygosity_code_values(self):
        df = _make_minimal_df(4)
        X, _ = build_features(df)
        assert set(X["zygosity_code"].unique()).issubset({-1, 0, 1})

    def test_no_unexpected_string_columns(self):
        df = _make_minimal_df(4)
        X, _ = build_features(df)
        string_cols = [c for c in X.columns if X[c].dtype == object]
        assert string_cols == [], f"Unexpected object columns: {string_cols}"

    def test_father_mother_flags_int(self):
        df = _make_minimal_df(4)
        X, _ = build_features(df)
        assert X["father_has_variant"].dtype in (np.int32, np.int64, int, np.int8)
        assert X["mother_has_variant"].dtype in (np.int32, np.int64, int, np.int8)

    def test_missing_optional_columns_handled(self):
        """build_features must not crash when optional columns are absent."""
        df = _make_minimal_df(4)
        df = df.drop(columns=["Pathogenicity_Score", "Sample_Coverage_Alternate",
                               "proband_phenotype_hpo", "Condition_ID",
                               "Condition_Inheritance"])
        X, y = build_features(df)
        assert len(X) == 4
