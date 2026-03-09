"""
src/features.py
---------------
Feature engineering for the trio-aware variant pathogenicity classifier.

Public API
----------
    from src.features import build_features

    X, y = build_features(df)   # df from TrioLoader.load_all()

Features produced
-----------------
Numeric (parsed/derived):
    cadd_score, revel_score, papi_score, polyphen2_score, sift_score,
    gnomad_af, pathogenicity_score, sample_coverage, alt_depth,
    sample_coverage_reference, vaf, criteria_score, hpo_overlap_score,
    father_has_variant (bool→int), mother_has_variant (bool→int)

Ordinal encoded:
    effect_severity   (missense < splice < frameshift/stop < unknown)
    zygosity_code     (heterozygous=0, homozygous=1, other=-1)

One-hot encoded (returned as individual columns):
    inheritance_mode_*   (de_novo, paternal, maternal, …)
    condition_inheritance_*

Target
------
    y = 1  if Classification contains "Pathogenic" or "Likely pathogenic"
    y = 0  otherwise

HPO overlap
-----------
Uses pyhpo if available; falls back to normalised set-intersection otherwise.
The score measures similarity between the proband's HPO terms and the HPO
terms associated with the variant's Condition_ID (via pyhpo disease lookup).
"""

from __future__ import annotations

import re
import warnings
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Optional pyhpo import — graceful fallback to set-intersection
# ---------------------------------------------------------------------------

try:
    import pyhpo
    pyhpo.Ontology()          # initialise once at import time
    _PYHPO_AVAILABLE = True
except Exception:
    _PYHPO_AVAILABLE = False
    warnings.warn(
        "pyhpo not available or failed to initialise. "
        "HPO overlap will be computed via simple set-intersection.",
        ImportWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered from least to most severe for ordinal encoding
EFFECT_ORDER = [
    "synonymous_variant",
    "intron_variant",
    "splice_region_variant",
    "missense_variant",
    "conservative_inframe_deletion",
    "conservative_inframe_insertion",
    "disruptive_inframe_deletion",
    "disruptive_inframe_insertion",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "frameshift_variant",
    "stop_gained",
    "start_lost",
    "stop_lost",
]
EFFECT_SEVERITY: dict[str, int] = {e: i for i, e in enumerate(EFFECT_ORDER)}

PATHOGENIC_LABELS = {"Pathogenic", "Likely pathogenic"}

# Hearing-disorder HPO terms used as disease reference when no Condition_ID
# disease mapping is resolvable
FALLBACK_DISEASE_HPO = {
    "HP:0000407",  # Sensorineural hearing impairment
    "HP:0008619",  # Bilateral sensorineural hearing impairment
    "HP:0000365",  # Impaired hearing
    "HP:0001751",  # Vertigo
    "HP:0000370",  # Abnormality of the middle ear
    "HP:0000377",  # Abnormal pinna morphology
    "HP:0000405",  # Conductive hearing impairment
}

# ---------------------------------------------------------------------------
# Regex helpers — parse structured sub-fields
# ---------------------------------------------------------------------------

def _parse_functional(value: str) -> dict[str, Optional[float]]:
    """Parse pipe-delimited key:value pairs from the Functional column.

    Example input:
        "PaPI:1.0|CADD:30.0|REVEL:0.954|PolyPhen2:1.0|SIFT:0.0"

    Returns a dict with keys:
        cadd_score, revel_score, papi_score, polyphen2_score, sift_score
    """
    result: dict[str, Optional[float]] = {
        "cadd_score": None,
        "revel_score": None,
        "papi_score": None,
        "polyphen2_score": None,
        "sift_score": None,
    }
    if not isinstance(value, str) or not value.strip():
        return result

    patterns = {
        "cadd_score":      r"CADD:([0-9Ee.+\-]+)",
        "revel_score":     r"REVEL:([0-9Ee.+\-]+)",
        "papi_score":      r"PaPI:([0-9Ee.+\-]+)",
        "polyphen2_score": r"PolyPhen2:([0-9Ee.+\-]+)",
        "sift_score":      r"SIFT:([0-9Ee.+\-]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, value)
        if m:
            try:
                result[key] = float(m.group(1))
            except ValueError:
                pass
    return result


def _parse_population(value: str) -> dict[str, Optional[float]]:
    """Parse allele frequency from the Population column.

    Example input:
        "gnomAD:0.00111516%|ExAC:NA|1000GP:0%|ESP:0%"

    Returns:
        gnomad_af  (as a plain proportion 0-1, not percentage)
    """
    result: dict[str, Optional[float]] = {"gnomad_af": None}
    if not isinstance(value, str) or not value.strip():
        return result

    m = re.search(r"gnomAD:([0-9Ee.+\-]+)%", value)
    if m:
        try:
            result["gnomad_af"] = float(m.group(1)) / 100.0
        except ValueError:
            pass
    return result


def _encode_effect(effect: str) -> int:
    """Return an ordinal severity score for the most severe effect in the
    pipe-delimited Effect field (e.g. "missense_variant|splice_region_variant").

    Unknown effects receive a mid-range score.
    """
    if not isinstance(effect, str) or not effect.strip():
        return -1
    parts = [e.strip() for e in effect.split("|")]
    scores = [EFFECT_SEVERITY.get(p, -1) for p in parts]
    return max(scores)


def _encode_zygosity(zyg: str) -> int:
    """Return 1 for homozygous, 0 for heterozygous, -1 for unknown."""
    if not isinstance(zyg, str):
        return -1
    low = zyg.strip().lower()
    if low.startswith("hom"):
        return 1
    if low.startswith("het"):
        return 0
    return -1


def _count_acmg_criteria(criteria: str) -> int:
    """Count the number of ACMG/AMP criteria listed in the Criteria field.

    Example:
        "PM5-moderate|PP3-moderate|PP5-supporting"  →  3
    """
    if not isinstance(criteria, str) or not criteria.strip():
        return 0
    return len([c for c in criteria.split("|") if c.strip()])


# ---------------------------------------------------------------------------
# HPO overlap scoring
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _get_hpo_obj(hp_id: str):
    """Return the pyhpo HPOTerm object for an HPO ID, or None."""
    try:
        return pyhpo.Ontology.get_hpo_object(hp_id)
    except Exception:
        return None


def _hpo_set_intersection_score(proband_terms: set[str],
                                 disease_terms: set[str]) -> float:
    """Normalised Jaccard-like overlap: |A ∩ B| / |A ∪ B|."""
    if not proband_terms or not disease_terms:
        return 0.0
    intersection = len(proband_terms & disease_terms)
    union = len(proband_terms | disease_terms)
    return intersection / union if union else 0.0


def _pyhpo_similarity(proband_terms: set[str],
                       disease_terms: set[str]) -> float:
    """Compute pairwise max-similarity between two HPO term sets using pyhpo.

    Uses Resnik similarity averaged over all proband terms.
    Falls back to set-intersection on any pyhpo error.
    """
    try:
        proband_objs = [_get_hpo_obj(t) for t in proband_terms if t]
        disease_objs = [_get_hpo_obj(t) for t in disease_terms if t]
        proband_objs = [o for o in proband_objs if o is not None]
        disease_objs = [o for o in disease_objs if o is not None]
        if not proband_objs or not disease_objs:
            return _hpo_set_intersection_score(proband_terms, disease_terms)

        scores = []
        for p_obj in proband_objs:
            row_max = max(
                (p_obj.similarity_score(d_obj) for d_obj in disease_objs),
                default=0.0,
            )
            scores.append(row_max)
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return _hpo_set_intersection_score(proband_terms, disease_terms)


@lru_cache(maxsize=256)
def _disease_hpo_terms(condition_id: str) -> frozenset[str]:
    """Look up HPO terms associated with a disease OMIM/ORPHA ID via pyhpo.

    Returns an empty frozenset if the disease is not found or pyhpo is
    unavailable.
    """
    if not _PYHPO_AVAILABLE or not isinstance(condition_id, str):
        return frozenset()
    cid = condition_id.strip()
    if not cid:
        return frozenset()
    try:
        # pyhpo stores diseases; try direct lookup by OMIM id (numeric part)
        numeric = re.sub(r"[^0-9]", "", cid)
        if not numeric:
            return frozenset()
        diseases = pyhpo.Ontology.omim_diseases
        disease = next(
            (d for d in diseases if str(d.id) == numeric), None
        )
        if disease is None:
            return frozenset()
        return frozenset(str(t.id) for t in disease.hpo)
    except Exception:
        return frozenset()


def _compute_hpo_overlap(proband_hpo_str: str, condition_id: str) -> float:
    """Return HPO overlap score between proband phenotype and disease HPO.

    If pyhpo disease lookup yields nothing, fall back to the curated
    FALLBACK_DISEASE_HPO hearing-disorder set.
    """
    proband_terms = set(
        t.strip()
        for t in re.split(r"[;,]", proband_hpo_str or "")
        if t.strip().startswith("HP:")
    )
    if not proband_terms:
        return 0.0

    disease_terms = set(_disease_hpo_terms(str(condition_id or "")))
    if not disease_terms:
        disease_terms = FALLBACK_DISEASE_HPO

    if _PYHPO_AVAILABLE:
        return _pyhpo_similarity(proband_terms, disease_terms)
    return _hpo_set_intersection_score(proband_terms, disease_terms)


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a feature matrix and target vector from a TrioLoader DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``TrioLoader.load_all()``.  Must contain at minimum the
        columns produced by TrioLoader (variant columns + trio context).

    Returns
    -------
    X : pd.DataFrame
        Numeric + encoded feature matrix, one row per variant.
    y : pd.Series
        Binary target (1 = Pathogenic / Likely pathogenic, 0 = otherwise).
    """
    df = df.copy()

    feat: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # 1. Parse Functional sub-fields
    # ------------------------------------------------------------------
    func_parsed = df["Functional"].apply(
        lambda v: _parse_functional(v) if isinstance(v, str) else _parse_functional("")
    )
    for key in ("cadd_score", "revel_score", "papi_score",
                "polyphen2_score", "sift_score"):
        feat[key] = func_parsed.apply(lambda d, k=key: d[k]).astype(float)

    # ------------------------------------------------------------------
    # 2. Parse Population sub-fields
    # ------------------------------------------------------------------
    pop_parsed = df["Population"].apply(
        lambda v: _parse_population(v) if isinstance(v, str) else _parse_population("")
    )
    feat["gnomad_af"] = pop_parsed.apply(lambda d: d["gnomad_af"]).astype(float)

    # ------------------------------------------------------------------
    # 3. Direct numeric columns
    # ------------------------------------------------------------------
    for col, out in [
        ("Pathogenicity_Score",        "pathogenicity_score"),
        ("Sample_Coverage",            "sample_coverage"),
        ("Sample_Coverage_Alternate",  "alt_depth"),
        ("Sample_Coverage_Reference",  "sample_coverage_reference"),
    ]:
        feat[out] = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")

    # VAF = alt / total coverage
    total = feat["sample_coverage"]
    alt   = feat["alt_depth"]
    feat["vaf"] = (alt / total.replace(0, np.nan)).astype(float)

    # ------------------------------------------------------------------
    # 4. Ordinal encoding
    # ------------------------------------------------------------------
    feat["effect_severity"] = df.get(
        "Effect", pd.Series("", index=df.index)
    ).apply(_encode_effect).astype(int)

    feat["zygosity_code"] = df.get(
        "Sample_Zygosity", pd.Series("", index=df.index)
    ).apply(_encode_zygosity).astype(int)

    # ------------------------------------------------------------------
    # 5. Inheritance-mode one-hot
    # ------------------------------------------------------------------
    if "inheritance_mode" in df.columns:
        inh_dummies = pd.get_dummies(
            df["inheritance_mode"].fillna("unknown"),
            prefix="inh"
        )
        for col in inh_dummies.columns:
            feat[col] = inh_dummies[col].astype(int)

    # ------------------------------------------------------------------
    # 6. Condition_Inheritance one-hot
    # ------------------------------------------------------------------
    if "Condition_Inheritance" in df.columns:
        # normalise: take first inheritance mode when multiple are listed
        ci = df["Condition_Inheritance"].fillna("unknown").apply(
            lambda v: str(v).split("|")[0].strip()
        )
        ci_dummies = pd.get_dummies(ci, prefix="ci")
        for col in ci_dummies.columns:
            feat[col] = ci_dummies[col].astype(int)

    # ------------------------------------------------------------------
    # 7. ACMG criteria count
    # ------------------------------------------------------------------
    feat["criteria_score"] = df.get(
        "Criteria", pd.Series("", index=df.index)
    ).apply(_count_acmg_criteria).astype(int)

    # ------------------------------------------------------------------
    # 8. Family boolean flags
    # ------------------------------------------------------------------
    feat["father_has_variant"] = (
        df.get("father_has_variant", pd.Series(False, index=df.index))
        .fillna(False).astype(int)
    )
    feat["mother_has_variant"] = (
        df.get("mother_has_variant", pd.Series(False, index=df.index))
        .fillna(False).astype(int)
    )

    # ------------------------------------------------------------------
    # 9. HPO overlap score
    # ------------------------------------------------------------------
    proband_hpo = df.get(
        "proband_phenotype_hpo", pd.Series("", index=df.index)
    ).fillna("")
    condition_id = df.get(
        "Condition_ID", pd.Series("", index=df.index)
    ).fillna("")

    feat["hpo_overlap_score"] = pd.Series(
        [
            _compute_hpo_overlap(str(hpo), str(cid))
            for hpo, cid in zip(proband_hpo, condition_id)
        ],
        index=df.index,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # 10. Assemble X
    # ------------------------------------------------------------------
    X = pd.DataFrame(feat, index=df.index)

    # ------------------------------------------------------------------
    # 11. Target variable y
    # ------------------------------------------------------------------
    classification = df.get(
        "Classification", pd.Series("", index=df.index)
    ).fillna("")
    y = classification.apply(
        lambda c: int(any(label in c for label in PATHOGENIC_LABELS))
    ).rename("is_pathogenic")

    return X, y


# ---------------------------------------------------------------------------
# Convenience: feature names grouped by type (for ColumnTransformer)
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "cadd_score", "revel_score", "papi_score", "polyphen2_score", "sift_score",
    "gnomad_af", "pathogenicity_score", "sample_coverage", "alt_depth",
    "sample_coverage_reference", "vaf", "criteria_score", "hpo_overlap_score",
    "father_has_variant", "mother_has_variant",
]

ORDINAL_FEATURES = ["effect_severity", "zygosity_code"]


def feature_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    """Return a dict mapping group name → list of column names in X.

    Useful for building a sklearn ColumnTransformer.
    """
    numeric = [c for c in NUMERIC_FEATURES + ORDINAL_FEATURES if c in X.columns]
    onehot  = [c for c in X.columns if c.startswith(("inh_", "ci_"))]
    return {"numeric": numeric, "onehot": onehot}
