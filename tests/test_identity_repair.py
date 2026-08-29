from __future__ import annotations

import unittest

from muddywater.text_simplification.identity_repair import (
    PairRecord,
    analyze_identity_rates,
    evaluate_candidate,
    extract_numbers,
    topic_flags,
)


class IdentityRepairQualityTests(unittest.TestCase):
    def test_accepts_conservative_simplification(self) -> None:
        decision = evaluate_candidate(
            "由于近期连续出现强降雨天气，相关部门决定暂时关闭部分山区景区。",
            "近期连续强降雨，相关部门决定暂时关闭部分山区景区。",
        )
        self.assertTrue(decision.accepted)

    def test_rejects_number_loss(self) -> None:
        decision = evaluate_candidate(
            "该计划于2026年8月15日启动，共覆盖200个站点。",
            "该计划于2026年启动，覆盖多个站点。",
        )
        self.assertFalse(decision.accepted)
        self.assertIn("number_loss", decision.reasons)

    def test_rejects_generation_limit(self) -> None:
        decision = evaluate_candidate("这是一段需要简化的完整句子。", "这是一段简化句子。", truncated=True)
        self.assertFalse(decision.accepted)
        self.assertIn("generation_length_limit", decision.reasons)

    def test_extracts_dates_percentages_and_times(self) -> None:
        self.assertEqual(
            extract_numbers("2026-08-15在14:35达到7.3%。"),
            ("2026-08-15", "14:35", "7.3%"),
        )

    def test_political_analysis_is_explicitly_heuristic(self) -> None:
        records = [
            PairRecord("train", 1, "国务院发布通知。", "国务院发布通知。"),
            PairRecord("train", 2, "天气很好。", "今天天气很好。"),
        ]
        self.assertIn("government_law", topic_flags(records[0].source))
        report = analyze_identity_rates(records)
        self.assertEqual(report["identity_rows"], 1)
        self.assertEqual(report["any_political"]["matched_identity_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
