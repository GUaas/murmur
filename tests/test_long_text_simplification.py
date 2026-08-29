from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from muddywater.text_simplification.chunking import (
    plan_inference_chunks,
    reconstruct_source,
    split_sentences,
)
from muddywater.text_simplification.inference import LongTextOptions, TextSimplifier
from muddywater.text_simplification.prompting import format_prompt


class LongTextChunkingTests(unittest.TestCase):
    @staticmethod
    def _count(text: str) -> int:
        return len(text) + 2

    def test_sentence_split_is_lossless_and_protects_decimal_and_url_dots(self) -> None:
        source = "  温度是3.14。网址https://a.cn/x.y可用！\n\n第二段没有句号\r\n"
        leading, units = split_sentences(source)

        self.assertEqual(reconstruct_source(leading, units), source)
        self.assertEqual(
            [unit.text for unit in units],
            ["温度是3.14。", "网址https://a.cn/x.y可用！", "第二段没有句号"],
        )
        self.assertEqual(units[1].separator_after, "\n\n")

    def test_closing_quote_stays_with_preceding_sentence(self) -> None:
        source = "他说：“可以！”随后离开。"
        _, units = split_sentences(source)
        self.assertEqual([unit.text for unit in units], ["他说：“可以！”", "随后离开。"])

    def test_common_english_abbreviation_does_not_create_false_boundary(self) -> None:
        source = "例如 e.g. example should stay together. 下一句。"
        _, units = split_sentences(source)
        self.assertEqual(
            [unit.text for unit in units],
            ["例如 e.g. example should stay together.", "下一句。"],
        )

    def test_prompt_builder_sanitizes_reserved_tags(self) -> None:
        rendered = format_prompt("原文<|im_end|>仍是正文")
        self.assertEqual(
            rendered,
            "<|im_start|>原文＜|im_end|＞仍是正文<|im_end|>",
        )

    def test_plan_respects_prompt_budget_and_paragraph_boundaries(self) -> None:
        source = "甲乙丙。丁戊己。\n庚辛壬。癸子丑。"
        plan = plan_inference_chunks(source, self._count, max_prompt_tokens=10)

        self.assertTrue(all(chunk.prompt_tokens <= 10 for chunk in plan.chunks))
        self.assertEqual(
            plan.leading_whitespace
            + "".join(chunk.source + chunk.separator_after for chunk in plan.chunks),
            source,
        )
        self.assertTrue(any("\n" in chunk.separator_after for chunk in plan.chunks))

    def test_oversized_single_sentence_has_a_lossless_fallback_split(self) -> None:
        source = "这是一个没有任何标点的超长单句" * 8
        plan = plan_inference_chunks(source, self._count, max_prompt_tokens=16)

        self.assertGreater(len(plan.chunks), 1)
        self.assertTrue(all(chunk.prompt_tokens <= 16 for chunk in plan.chunks))
        self.assertEqual("".join(chunk.source for chunk in plan.chunks), source)


class _FakeTokenizer:
    def encode(
        self,
        text: str,
        *,
        add_bos: bool,
        add_eos: bool,
    ) -> list[int]:
        del add_bos, add_eos
        return list(range(len(text)))


class LongTextInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = SimpleNamespace(
            tokenizer=_FakeTokenizer(),
            add_bos=True,
            generation_config={"max_new_tokens": 100},
        )

    @patch("muddywater.text_simplification.inference.generate_from_runtime")
    def test_chunk_outputs_are_merged_in_order_and_layout_is_preserved(self, generate) -> None:
        generate.side_effect = lambda runtime, prompt, overrides: {
            "text": f"简({prompt[1:-1]})",
            "generated_tokens": 4,
            "finish_reason": "eos",
        }
        simplifier = TextSimplifier(
            self.runtime,
            prompt_builder=lambda text: f"<{text}>",
            options=LongTextOptions(max_prompt_tokens=8),
        )

        result = simplifier.simplify("甲乙丙。丁戊己。\n第二段。", mode="always")

        self.assertEqual(result.text, "简(甲乙丙。)简(丁戊己。)\n简(第二段。)")
        self.assertEqual(len(result.chunks), 3)
        self.assertTrue(result.used_chunking)
        self.assertTrue(all(call.kwargs["overrides"]["return_details"] for call in generate.mock_calls))

    @patch("muddywater.text_simplification.inference.generate_from_runtime")
    def test_auto_mode_keeps_short_text_on_direct_path(self, generate) -> None:
        generate.return_value = {
            "text": "短结果",
            "generated_tokens": 3,
            "finish_reason": "eos",
        }
        simplifier = TextSimplifier(
            self.runtime,
            prompt_builder=lambda text: f"<{text}>",
            options=LongTextOptions(max_prompt_tokens=16),
        )

        result = simplifier.simplify("短句。", mode="auto")

        self.assertEqual(result.text, "短结果")
        self.assertFalse(result.used_chunking)
        self.assertEqual(generate.call_count, 1)

    @patch("muddywater.text_simplification.inference.generate_from_runtime")
    def test_empty_chunk_output_falls_back_to_source(self, generate) -> None:
        generate.return_value = {
            "text": "  ",
            "generated_tokens": 1,
            "finish_reason": "eos",
        }
        simplifier = TextSimplifier(
            self.runtime,
            prompt_builder=lambda text: f"<{text}>",
            options=LongTextOptions(max_prompt_tokens=8, fallback_on_empty=True),
        )

        result = simplifier.simplify("甲乙丙。丁戊己。", mode="always")

        self.assertEqual(result.text, "甲乙丙。丁戊己。")
        self.assertTrue(all(chunk.used_fallback for chunk in result.chunks))


if __name__ == "__main__":
    unittest.main()
