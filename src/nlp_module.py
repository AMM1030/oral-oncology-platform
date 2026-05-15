"""
Clinical Note NLP Analyzer
---------------------------
Extracts oral-cancer-suspicious findings from free-text dental notes.

Design choices:
  * Lexicon + regex over deep learning -> transparent and clinically explainable.
  * Each term has a severity weight; the urgency score is the weighted sum.
  * Negation handling: terms preceded by 'no', 'without', 'denies', 'wnl',
    'unremarkable' in the same sentence are suppressed.
  * The module is a pure function -> easy to call from the Streamlit app.

Public API:
    analyze_note(text: str) -> dict
    analyze_notes_dataframe(df, text_col='note_text') -> pd.DataFrame
"""

from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd


# -----------------------------------------------------------------------------
# Clinical lexicon. Weights reflect severity (higher = more concerning).
# Designed in consultation with oral oncology screening guidelines.
# -----------------------------------------------------------------------------

HIGH_URGENCY_TERMS = {
    r"non[- ]healing ulcer":            5,
    r"persistent ulcer":                 5,
    r"leukoplakia":                      5,
    r"erythroplakia":                    5,
    r"concerning for malignancy":        5,
    r"suspicion(?: for malignancy)?":    4,
    r"indurated (?:lesion|border)":      4,
    r"rolled border":                    4,
    r"ulcerated lesion":                 4,
    r"non[- ]tender mass":               4,
    r"lump in mouth":                    4,
    r"firm[, ].*?mass":                  3,
    r"neck swelling":                    4,
    r"lymphadenopathy":                  4,
    r"difficulty swallowing":            4,
    r"dysphagia":                        4,
    r"irregular margins":                3,
    r"urgent referral":                  3,
}

MEDIUM_URGENCY_TERMS = {
    r"white patch":                          3,
    r"red patch":                            3,
    r"mixed red and white":                  3,
    r"present (?:for )?more than two weeks": 3,
    r"present for approximately":            2,
    r"hasn['’]?t healed":                    3,
    r"not resolving":                        3,
    r"no resolution":                        3,
    r"bleeding lesion":                      2,
    r"bleed.*on (?:brushing|probing)":       2,
    r"bleeds easily":                        2,
    r"unexplained oral pain":                2,
    r"numbness":                             2,
    r"burning sensation":                    1,
}

# Low-urgency terms are explicitly NOT scored — they're benign findings we
# want to detect to *suppress* false positives in downstream summaries.
BENIGN_TERMS = {
    r"aphthous ulcer",
    r"frictional keratosis",
    r"geographic tongue",
    r"cheek bite",
    r"benign finding",
}

# Negation cues. If any of these appear within ~5 words BEFORE a matched
# term, the match is suppressed.
# Negation cues. Each one is a regex pattern with word boundaries so
# "no" doesn't match inside "non-healing" and "not" doesn't match inside
# "noted". This was an important bug fix during development.
NEGATION_CUES = [
    r"\bno\b", r"\bnot\b", r"\bwithout\b",
    r"\bdenies\b", r"\bdenied\b",
    r"\bnegative for\b", r"\babsent\b",
    r"\bwnl\b", r"\bunremarkable\b",
    r"\bno abnormalit\w*",   # 'no abnormalities noted'
    r"\bruled out\b",
]

NEGATION_WINDOW = 35  # characters to look back


# -----------------------------------------------------------------------------
# Core analyzer
# -----------------------------------------------------------------------------

def _is_negated(text: str, start: int) -> bool:
    """
    Look back NEGATION_WINDOW chars and check for a negation cue.
    Uses regex with word boundaries so 'no' doesn't match inside 'non-' and
    'not' doesn't match inside 'noted'.
    """
    preceding = text[max(0, start - NEGATION_WINDOW): start].lower()
    return any(re.search(pattern, preceding) for pattern in NEGATION_CUES)

def analyze_note(text: str) -> Dict:
    """Analyze one clinical note. Returns a dict of structured findings."""
    if not isinstance(text, str) or not text.strip():
        return _empty_result()

    text_lower = text.lower()
    extracted: List[Dict] = []
    score = 0

    # Score high- and medium-urgency terms
    for term_dict in (HIGH_URGENCY_TERMS, MEDIUM_URGENCY_TERMS):
        for pattern, weight in term_dict.items():
            for m in re.finditer(pattern, text_lower):
                if _is_negated(text_lower, m.start()):
                    continue
                extracted.append({
                    "term": m.group(),
                    "weight": weight,
                    "position": m.start(),
                })
                score += weight

    # Detect benign mentions (informational only)
    benign_found = [
        re.search(p, text_lower).group()
        for p in BENIGN_TERMS
        if re.search(p, text_lower)
    ]

    # Bucket score
    if score >= 10:
        category = "High"
    elif score >= 5:
        category = "Medium"
    elif score >= 1:
        category = "Low"
    else:
        category = "None"

    return {
        "extracted_terms":      [e["term"] for e in extracted],
        "n_suspicious_terms":   len(extracted),
        "urgency_score":        score,
        "suspicious_flag":      int(score > 0),
        "note_risk_category":   category,
        "benign_terms":         benign_found,
    }


def _empty_result() -> Dict:
    return {
        "extracted_terms":    [],
        "n_suspicious_terms": 0,
        "urgency_score":      0,
        "suspicious_flag":    0,
        "note_risk_category": "None",
        "benign_terms":       [],
    }


def analyze_notes_dataframe(df: pd.DataFrame, text_col: str = "note_text"
                            ) -> pd.DataFrame:
    """Apply analyze_note to every row in a dataframe; returns enriched copy."""
    results = df[text_col].apply(analyze_note).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), results], axis=1)


# -----------------------------------------------------------------------------
# CLI for quick smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "Pt presents for routine prophylaxis. No complaints. Soft tissue exam unremarkable.",
        "Noted persistent white patch (leukoplakia) on lateral tongue. Cannot be wiped off.",
        "Patient c/o 6 weeks of non-healing ulcer. Bleeds easily on probing. Urgent referral made.",
        "Aphthous ulcer on hard palate - resolved at exam. No referral needed.",
    ]
    for i, s in enumerate(samples, 1):
        r = analyze_note(s)
        print(f"\n[{i}] {s[:80]}...")
        print(f"    score={r['urgency_score']:>2}  category={r['note_risk_category']:<6}  terms={r['extracted_terms']}")