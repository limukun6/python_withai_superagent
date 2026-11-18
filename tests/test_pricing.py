import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM
from superagent.planner import OpenAICompatiblePlanner
from superagent.pricing import DEEPSEEK_PRICE_BASIS, DEEPSEEK_PRICE_VERSION, resolve_pricing


class OfficialPricingTests(unittest.TestCase):
    def test_million_input_and_output_costs(self):
        self.assertEqual(resolve_pricing("DeepSeek-V4-Pro").course_cost(1_000_000, 1_000_000), 18)
        self.assertEqual(resolve_pricing("DeepSeek-V4.1-Flash-1017").course_cost(1_000_000, 1_000_000), 5)

    def test_catalog_matches_checked_source_record(self):
        record_path = Path(__file__).resolve().parents[1] / "docs" / "deepseek_pricing_2026-11-16.json"
        snapshot = json.loads(record_path.read_text())
        for model in snapshot["models"]:
            price = resolve_pricing(model["model"])
            self.assertEqual(price.input_rmb_per_million, model["input_rmb_per_million"])
            self.assertEqual(price.output_rmb_per_million, model["output_rmb_per_million"])
            self.assertEqual(price.source, snapshot["source_url"])
            self.assertEqual(price.price_version, snapshot["price_version"])
            self.assertEqual(price.verified_on, snapshot["checked_on"])
            self.assertFalse(price.metadata()["call_time_affects_course_rate"])
            self.assertFalse(price.metadata()["cache_discount_applied"])

    def test_official_aliases_and_separately_verified_qwen_standard_price(self):
        self.assertEqual(resolve_pricing("deepseek-flash"), resolve_pricing("DeepSeek-V4.1-Flash-1017"))
        self.assertEqual(resolve_pricing("deepseek-v4-pro-0813"), resolve_pricing("DeepSeek-V4-Pro"))
        qwen=resolve_pricing("Qwen3.8-27B-w8a8")
        self.assertEqual(qwen,resolve_pricing("Qwen3.8-27B"))
        self.assertEqual((qwen.input_rmb_per_million,qwen.output_rmb_per_million),(3.,12.))
        self.assertTrue(qwen.verified_release_before_cutoff)
        self.assertEqual(qwen.price_basis,"official_standard_beijing_no_cache_discount")

    def test_unknown_or_invalid_usage_cannot_be_calculated_as_zero(self):
        price = resolve_pricing("deepseek-flash")
        for bad in (None, -1, True, 1.5, float("nan"), "1000"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                price.course_cost(bad, 100)
            with self.subTest(value=bad), self.assertRaises(ValueError):
                price.course_cost(100, bad)


class PricingLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.key = self.root / "key.txt"
        self.key.write_text("mock-only-not-a-real-key")
        self.logger = RunLogger(self.root)
        self.logger.initialize_usage_ledger()

    def test_course_rate_ignores_cache_and_clock_keeps_actual_charge_unknown(self):
        client = ExperimentLLM("https://example.invalid/api", str(self.key), "DeepSeek-V4-Pro", 20)
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1000,
                 "prompt_cache_hit_tokens": 900_000, "prompt_cache_miss_tokens": 100_000,
                 "completion_tokens_details": {"reasoning_tokens": 400}}
        with patch("superagent.experiment_llm.requests.post") as post:
            post.return_value.ok = True
            post.return_value.json.return_value = {"model": "DeepSeek-V4-Pro-0813", "usage": usage,
                                                   "choices": [{"message": {"content": "{}"}}]}
            for clock in ("2026-11-16T02:00:00+00:00", "2026-11-16T15:00:00+00:00"):
                with patch("superagent.events.utc_now", return_value=clock):
                    client.ask({}, self.logger, "planner")
        records = [json.loads(row) for row in self.logger.usage_path.read_text().splitlines()][1:]
        self.assertEqual(len(records), 2)
        for record in records:
            # Every prompt token costs the off-peak cache-miss rate. The 400
            # reasoning tokens are already in completion_tokens, not added again.
            self.assertAlmostEqual(record["course_cost_rmb"], 4.5135)
            self.assertEqual(record["cost_rmb"], record["course_cost_rmb"])
            self.assertIsNone(record["actual_cost_rmb"])
            self.assertEqual(record["raw_usage"], usage)
            self.assertEqual(record["price_status"], DEEPSEEK_PRICE_BASIS)
            self.assertEqual(record["pricing"]["price_version"], DEEPSEEK_PRICE_VERSION)

    def test_pre_call_estimate_uses_updated_rates(self):
        client = ExperimentLLM("https://example.invalid/api", str(self.key), "DeepSeek-V4-Pro", .02)
        with patch("superagent.experiment_llm.requests.post") as post:
            with self.assertRaises(RuntimeError):
                client.ask({}, self.logger, "planner")
            post.assert_not_called()

    def test_legacy_client_uses_same_frozen_course_prices(self):
        price = resolve_pricing("DeepSeek-V4.1-Flash-1017")
        planner = OpenAICompatiblePlanner(self.logger, "https://example.invalid/api", str(self.key),
                                          price.model_id, price, 20)
        with patch("superagent.planner.requests.post") as post:
            post.return_value.json.return_value = {
                "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1000,
                          "prompt_cache_hit_tokens": 900_000},
                "choices": [{"message": {"content": '{"candidate":"a","reason":"fixture"}'}}],
            }
            self.assertEqual(planner.choose({}, [], ["a"])[0], "a")
        record = json.loads(self.logger.usage_path.read_text().splitlines()[-1])
        self.assertAlmostEqual(record["course_cost_rmb"], 1.004)
        self.assertIsNone(record["actual_cost_rmb"])
        self.assertEqual(record["pricing"]["input_rmb_per_million"], 1.0)

    def test_missing_usage_records_unknown_cost_with_price_version(self):
        client = ExperimentLLM("https://example.invalid/api", str(self.key), "deepseek-flash", 1)
        with patch("superagent.experiment_llm.requests.post") as post:
            post.return_value.ok = True
            post.return_value.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
            with self.assertRaises(RuntimeError):
                client.ask({}, self.logger, "planner")
        record = json.loads(self.logger.usage_path.read_text().splitlines()[-1])
        self.assertIsNone(record["course_cost_rmb"])
        self.assertIsNone(record["actual_cost_rmb"])
        self.assertEqual(record["pricing"]["price_version"], DEEPSEEK_PRICE_VERSION)


if __name__ == "__main__":
    unittest.main()
