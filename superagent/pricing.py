"""Auditable model-price configuration for course-constrained LLM calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    model_id: str
    input_rmb_per_million: float
    output_rmb_per_million: float
    source: str
    verified_release_before_cutoff: bool


# Prices transcribed from the user-provided HPC AI Computing Platform images
# on 2026-11-13. Prices are RMB per 1M Tokens; cache discounts are excluded.
# The course's release-date rule remains independent from this price table.
CATALOG: dict[str, ModelPricing] = {
    "deepseek-v4-pro": ModelPricing(
        "DeepSeek-V4-Pro", 2.25, 6.75,
        "HPC AI Computing Platform price table, image supplied 2026-11-13", True,
    ),
    "deepseek-v4.1-flash": ModelPricing(
        "DeepSeek-V4.1-Flash", 0.50, 2.00,
        "Provisional price from 2026-11-13 screenshot; API name corrected per AGENTS.md 2026-11-18", True,
    ),
    "qwen3.8-27b-w8a8": ModelPricing(
        "Qwen3.8-27B-w8a8", 0.10, 0.34,
        "HPC AI Computing Platform price table, image supplied 2026-11-13", False,
    ),
}


def resolve_pricing(model: str) -> ModelPricing:
    """Return a configured price only for a named model with known evidence."""
    normalized = model.casefold().strip()
    aliases = {
        "deepseek-v4-pro": "deepseek-v4-pro",
        "deepseek-v4.1-flash": "deepseek-v4.1-flash",
        # Retain old configuration compatibility, but never send the old ID.
        "deepseek-v4.1-flash-1017": "deepseek-v4.1-flash",
        "deepseek-v4-1-flash-1017": "deepseek-v4.1-flash",
        "qwen3.8-27b-w8a8": "qwen3.8-27b-w8a8",
    }
    try:
        pricing = CATALOG[aliases[normalized]]
    except KeyError as error:
        supported = ", ".join(entry.model_id for entry in CATALOG.values())
        raise ValueError(f"No price configuration for {model!r}. Supported models: {supported}") from error
    if not pricing.verified_release_before_cutoff:
        raise ValueError(
            f"{pricing.model_id} has a price in the supplied table, but its release-date "
            "eligibility has not yet been recorded. Select a verified model or add evidence first."
        )
    return pricing
