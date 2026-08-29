from __future__ import annotations

import unittest

from muddywater.assessment.data import (
    _even_sample,
    adapt_arc,
    adapt_ceval,
    adapt_winogrande,
)
from muddywater.assessment.generation import (
    _parse_json_output,
    _three_item_format,
    repetition_ratio,
)


class AssessmentDataTests(unittest.TestCase):
    def test_even_sample_keeps_endpoints(self) -> None:
        self.assertEqual(_even_sample(list(range(10)), 3), [0, 4, 9])

    def test_arc_answer_label_mapping(self) -> None:
        examples = adapt_arc(
            [
                {
                    "row_idx": 7,
                    "row": {
                        "id": "x",
                        "question": "Q?",
                        "choices": {"text": ["one", "two"], "label": ["1", "2"]},
                        "answerKey": "2",
                    },
                }
            ],
            "arc_test",
        )
        self.assertEqual(examples[0].answer_index, 1)
        self.assertEqual(examples[0].primary_metric, "accuracy_norm")

    def test_winogrande_builds_full_continuations(self) -> None:
        examples = adapt_winogrande(
            [
                {
                    "row_idx": 0,
                    "row": {
                        "sentence": "The _ ran home.",
                        "option1": "dog",
                        "option2": "table",
                        "answer": "1",
                    },
                }
            ]
        )
        self.assertEqual(examples[0].context, "The ")
        self.assertEqual(examples[0].options, ("dog ran home.", "table ran home."))

    def test_ceval_uses_constrained_letter_continuations(self) -> None:
        examples = adapt_ceval(
            [
                {
                    "row_idx": 0,
                    "row": {
                        "question": "二加二等于？",
                        "A": "3",
                        "B": "4",
                        "C": "5",
                        "D": "6",
                        "answer": "B",
                    },
                }
            ],
            "math",
            1,
        )
        self.assertEqual(examples[0].answer_index, 1)
        self.assertEqual(examples[0].options, ("A", "B", "C", "D"))
        self.assertTrue(examples[0].context.endswith("答案："))


class AssessmentGenerationTests(unittest.TestCase):
    def test_repetition_ratio_detects_collapse(self) -> None:
        self.assertGreater(repetition_ratio("水网密布" * 20), 0.8)
        self.assertEqual(repetition_ratio("甲乙丙丁戊己庚辛"), 0.0)

    def test_format_checks(self) -> None:
        self.assertTrue(_parse_json_output('{"name": "苹果", "count": 3}'))
        self.assertFalse(_parse_json_output("name=苹果"))
        self.assertTrue(_three_item_format("红色，绿色，蓝色"))
        self.assertFalse(_three_item_format("红色，绿色，蓝色，白色"))


if __name__ == "__main__":
    unittest.main()
