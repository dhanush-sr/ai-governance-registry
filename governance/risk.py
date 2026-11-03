"""EU AI Act (Regulation (EU) 2024/1689) risk-tier classification.

The tier is computed from the card's `risk` answers, never taken on trust.
The card's `declared_tier` must match, so a model cannot dodge high-risk
obligations by labelling itself "minimal".
"""

# Art. 5(1): prohibited practices -> cannot be deployed at all.
PROHIBITED = {
    "subliminal_manipulation": "Art. 5(1)(a)",
    "exploiting_vulnerabilities": "Art. 5(1)(b)",
    "social_scoring": "Art. 5(1)(c)",
    "predictive_policing_profiling": "Art. 5(1)(d)",
    "untargeted_facial_scraping": "Art. 5(1)(e)",
    "emotion_recognition_workplace_education": "Art. 5(1)(f)",
    "biometric_categorisation_sensitive": "Art. 5(1)(g)",
    "realtime_remote_biometric_id_public": "Art. 5(1)(h)",
}

# Art. 6: high-risk systems (Annex I safety components + Annex III use cases).
HIGH_RISK = {
    "product_safety_component": "Art. 6(1), Annex I",
    "biometric_identification": "Annex III 1",
    "critical_infrastructure": "Annex III 2",
    "education": "Annex III 3",
    "employment": "Annex III 4",
    "public_benefits": "Annex III 5(a)",
    "creditworthiness": "Annex III 5(b)",
    "insurance_pricing": "Annex III 5(c)",
    "emergency_dispatch": "Annex III 5(d)",
    "law_enforcement": "Annex III 6",
    "migration_border": "Annex III 7",
    "justice_democracy": "Annex III 8",
}

TIERS = ("unacceptable", "high", "limited", "minimal")
USE_CASES = set(PROHIBITED) | set(HIGH_RISK) | {"other"}


def classify(risk: dict) -> tuple[str, str]:
    """Return (tier, legal basis) for a card's `risk` block."""
    use_case = risk.get("use_case")
    if use_case not in USE_CASES:
        raise ValueError(f"unknown use_case {use_case!r}; expected one of {sorted(USE_CASES)}")
    if use_case in PROHIBITED:
        return "unacceptable", f"{PROHIBITED[use_case]} prohibited practice ({use_case})"
    if use_case in HIGH_RISK:
        return "high", f"{HIGH_RISK[use_case]} ({use_case})"
    if risk.get("interacts_with_people") or risk.get("generates_synthetic_content"):
        return "limited", "Art. 50 transparency obligations"
    return "minimal", "no Art. 5, Art. 6 or Art. 50 trigger"
