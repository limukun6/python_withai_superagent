"""Local and LLM-backed planners constrained to an auditable candidate set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import requests

from .events import RunLogger
from .pricing import ModelPricing, resolve_pricing


class Planner(Protocol):
    def choose(self, profile: dict[str, Any], completed: list[dict[str, Any]], available: list[str]) -> tuple[str, str]: ...


class LocalPlanner:
    """Deterministic fallback used for no-cost development and API failures."""

    def choose(self, profile: dict[str, Any], completed: list[dict[str, Any]], available: list[str]) -> tuple[str, str]:
        if not completed:
            return available[0], "Start with the lowest-complexity candidate to verify the data and metric path."
        return available[0], "Evaluate the next schema-appropriate candidate after observing earlier validation results."


class OpenAICompatiblePlanner:
    """One decision per experiment, with strict JSON parsing and cost records."""

    def __init__(
        self,
        logger: RunLogger,
        api_url: str,
        api_key_file: str,
        model: str,
        pricing: ModelPricing,
        budget_rmb: float,
        max_output_tokens: int = 350,
        timeout_seconds: int = 60,
    ) -> None:
        self.logger = logger
        self.api_url = api_url
        self.api_key = Path(api_key_file).read_text(encoding="utf-8").strip()
        if not self.api_key:
            raise ValueError("API key file is empty.")
        self.model = pricing.model_id
        self.pricing = pricing
        self.budget_rmb = budget_rmb
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.cost_rmb = 0.0

    def _estimate_cost(self, prompt_text: str) -> float:
        # A Chinese-heavy payload can approach one token per character. Add the
        # fixed system prompt allowance so the budget guard errs conservatively.
        estimated_input_tokens = len(prompt_text) + 300
        return (
            estimated_input_tokens * self.pricing.input_rmb_per_million
            + self.max_output_tokens * self.pricing.output_rmb_per_million
        ) / 1_000_000

    def choose(self, profile: dict[str, Any], completed: list[dict[str, Any]], available: list[str]) -> tuple[str, str]:
        compact_history = [
            {key: row[key] for key in ("family", "status", "validation_accuracy", "reason")}
            for row in completed
        ]
        prompt = {
            "role": "experiment planner",
            "instruction": (
                "Choose exactly one next candidate from allowed_candidates. Use the data profile and completed "
                "experiments. Return strict JSON only: {\"candidate\": string, \"reason\": string}. "
                "Do not propose code, hyperparameters outside the candidate, or access to test data."
            ),
            "data_profile": profile,
            "completed_experiments": compact_history,
            "allowed_candidates": available,
        }
        prompt_text = json.dumps(prompt, ensure_ascii=False)
        estimated_cost = self._estimate_cost(prompt_text)
        remaining = self.budget_rmb - self.cost_rmb
        if estimated_cost > remaining:
            raise RuntimeError(
                f"Estimated LLM call cost {estimated_cost:.6f} RMB exceeds remaining "
                f"budget {remaining:.6f} RMB."
            )
        response = requests.post(
            self.api_url,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": self.max_output_tokens,
                "messages": [
                    {"role": "system", "content": "You are a cautious machine-learning experiment planner."},
                    {"role": "user", "content": prompt_text},
                ],
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        call_cost = (
            input_tokens * self.pricing.input_rmb_per_million
            + output_tokens * self.pricing.output_rmb_per_million
        ) / 1_000_000
        self.cost_rmb += call_cost
        self.logger.usage(
            event="llm_call", provider_url=self.api_url, model=self.model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            input_rmb_per_million=self.pricing.input_rmb_per_million,
            output_rmb_per_million=self.pricing.output_rmb_per_million,
            price_source=self.pricing.source,
            estimated_pre_call_cost_rmb=estimated_cost,
            cost_rmb=call_cost,
            cumulative_cost_rmb=self.cost_rmb,
            budget_rmb=self.budget_rmb,
        )
        content = body["choices"][0]["message"]["content"]
        try:
            decision = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError(f"Planner did not return valid JSON: {error}") from error
        candidate = decision.get("candidate")
        reason = decision.get("reason")
        if candidate not in available or not isinstance(reason, str) or not reason.strip():
            raise ValueError("Planner selected an unavailable candidate or omitted its reason.")
        return candidate, reason.strip()


def make_planner(args: Any, logger: RunLogger) -> Planner:
    if args.planner == "local":
        return LocalPlanner()
    required = [args.api_url, args.api_key_file, args.model]
    if any(value is None for value in required):
        raise ValueError("LLM planner requires API URL, key file and model.")
    pricing = resolve_pricing(args.model)
    return OpenAICompatiblePlanner(
        logger=logger,
        api_url=args.api_url,
        api_key_file=args.api_key_file,
        model=args.model,
        pricing=pricing,
        budget_rmb=args.llm_budget_rmb,
    )
