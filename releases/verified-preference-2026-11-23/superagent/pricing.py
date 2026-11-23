"""Frozen official off-peak rates for course accounting, not account debits."""

from __future__ import annotations

from dataclasses import dataclass


DEEPSEEK_PRICE_SOURCE = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"
DEEPSEEK_PRICE_VERSION = "deepseek-off-peak-2026-11-16"
DEEPSEEK_PRICE_BASIS = "official_off_peak_no_cache_discount"


@dataclass(frozen=True)
class ModelPricing:
    model_id: str
    input_rmb_per_million: float
    output_rmb_per_million: float
    source: str
    verified_release_before_cutoff: bool
    price_basis: str = "unverified"
    price_version: str = "unverified"
    verified_on: str | None = None
    official_model_version: str | None = None

    def course_cost(self, input_tokens: int, output_tokens: int) -> float:
        """All prompt tokens use the uncached rate, independent of call time.

        output_tokens is the provider's billed completion total. Do not add a
        reasoning-token breakdown again if it is already included in that total.
        """
        if any(type(value) is not int or value < 0 for value in (input_tokens, output_tokens)):
            raise ValueError("Course accounting requires nonnegative integer token totals.")
        return (input_tokens * self.input_rmb_per_million
                + output_tokens * self.output_rmb_per_million) / 1_000_000

    def metadata(self) -> dict:
        return {
            "catalog_model": self.model_id,
            "official_model_version": self.official_model_version,
            "input_rmb_per_million": self.input_rmb_per_million,
            "output_rmb_per_million": self.output_rmb_per_million,
            "source_url": self.source,
            "price_basis": self.price_basis,
            "price_version": self.price_version,
            "verified_on": self.verified_on,
            "call_time_affects_course_rate": False,
            "cache_discount_applied": False,
            "release_cutoff": "2026-10-21",
            "verified_release_before_cutoff": self.verified_release_before_cutoff,
        }


# Per the user's clarified rule, DeepSeek uses updated official off-peak
# prices even when the request is sent during peak hours. These are the rates
# checked on 2026-11-16, not a claimed historical snapshot from 2026-10-21.
# The model-release cutoff remains independent. Platform charges, discounts,
# and free credits do not reduce course costs. Qwen uses separately verified
# official original-model Beijing standard prices, not the school gateway tariff.
CATALOG: dict[str, ModelPricing] = {
    "deepseek-v4-pro": ModelPricing(
        "DeepSeek-V4-Pro", 4.5, 13.5, DEEPSEEK_PRICE_SOURCE, True,
        DEEPSEEK_PRICE_BASIS, DEEPSEEK_PRICE_VERSION, "2026-11-16", "DeepSeek-V4-Pro-0813",
    ),
    "deepseek-v4.1-flash-1017": ModelPricing(
        "DeepSeek-V4.1-Flash-1017", 1.0, 4.0, DEEPSEEK_PRICE_SOURCE, True,
        DEEPSEEK_PRICE_BASIS, DEEPSEEK_PRICE_VERSION, "2026-11-16", "DeepSeek-V4.1-Flash-1017",
    ),
    "qwen3.8-27b-w8a8": ModelPricing(
        "Qwen3.8-27B-w8a8", 3.0, 12.0,
        "https://help.aliyun.com/zh/model-studio/qwen3-8-27b", True,
        "official_standard_beijing_no_cache_discount", "qwen-standard-2026-11-17",
        "2026-11-17", "Qwen3.8-27B",
    ),
}


def resolve_pricing(model: str) -> ModelPricing:
    """Return a configured price only for a named model with known evidence."""
    normalized = model.casefold().strip()
    aliases = {
        "deepseek-v4-pro": "deepseek-v4-pro",
        "deepseek-v4-pro-0813": "deepseek-v4-pro",
        "deepseek-flash": "deepseek-v4.1-flash-1017",
        "deepseek-v4.1-flash": "deepseek-v4.1-flash-1017",
        "deepseek-v4.1-flash-1017": "deepseek-v4.1-flash-1017",
        "deepseek-v4-1-flash-1017": "deepseek-v4.1-flash-1017",
        "qwen3.8-27b-w8a8": "qwen3.8-27b-w8a8",
        "qwen3.8-27b": "qwen3.8-27b-w8a8",
        "qwen": "qwen3.8-27b-w8a8",
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
