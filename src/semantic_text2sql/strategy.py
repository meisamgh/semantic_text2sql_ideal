"""Deterministic hints inspired by the upstream exact/fuzzy/semantic router."""

from __future__ import annotations

import re
from typing import Literal

from semantic_text2sql.models import StrategyHints


def route_question(question: str) -> StrategyHints:
    lowered = question.casefold()
    semantic_phrases = [
        phrase
        for phrase in ("similar to", "like this", "about", "theme", "recommend", "concept")
        if phrase in lowered
    ]
    fuzzy_phrases = [
        phrase
        for phrase in ("approximately", "maybe spelled", "spelled wrong", "sounds like", "typo")
        if phrase in lowered
    ]
    quoted = re.findall(r"['\"]([^'\"]{2,80})['\"]", question)
    fuzzy_terms = quoted if fuzzy_phrases else []
    semantic_terms = quoted if semantic_phrases else semantic_phrases
    if semantic_phrases and fuzzy_phrases:
        mode: Literal["exact", "fuzzy", "semantic", "hybrid"] = "hybrid"
    elif semantic_phrases:
        mode = "semantic"
    elif fuzzy_phrases:
        mode = "fuzzy"
    else:
        mode = "exact"
    reasons = []
    if semantic_phrases:
        reasons.append("Question contains conceptual or similarity language.")
    if fuzzy_phrases:
        reasons.append("Question explicitly requests typo-tolerant matching.")
    if not reasons:
        reasons.append("Question can be represented with exact relational operations.")
    return StrategyHints(
        mode=mode,
        fuzzy_terms=fuzzy_terms,
        semantic_terms=semantic_terms,
        reasons=reasons,
    )
