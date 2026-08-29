from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from muddywater.dataset import LabeledLanguageModelingDataset, load_labeled_texts
from muddywater.source_target import SourceTargetTemplate


class _CharacterSpanTokenizer:
    bos_id = 2
    eos_id = 3
    pad_id = 0

    def encode_with_char_spans(self, text, add_bos=True, add_eos=True):
        ids = [self.bos_id] if add_bos else []
        spans = [None] if add_bos else []
        for index, character in enumerate(text):
            ids.append(10 + ord(character))
            spans.append((index, index + 1))
        if add_eos:
            ids.append(self.eos_id)
            spans.append(None)
        return ids, spans


class SourceTargetTemplateTests(unittest.TestCase):
    def test_compact_format_and_mask(self) -> None:
        template = SourceTargetTemplate()
        text, mask = template.format_pair("较长原句。", "精简句。")
        prompt = "较长原句。<|im_start|>"
        self.assertEqual(text, prompt + "精简句。")
        self.assertEqual(mask, "0" * len(prompt) + "1" * len("精简句。"))
        self.assertEqual(template.generation_prompt("较长原句。"), prompt)

    def test_reserved_separator_is_rejected(self) -> None:
        template = SourceTargetTemplate()
        with self.assertRaisesRegex(ValueError, "reserved label"):
            template.format_pair("坏<|im_start|>输入", "输出")

    def test_two_reserved_labels_wrap_the_source(self) -> None:
        template = SourceTargetTemplate(
            source_label="<|im_start|>",
            target_label="<|im_end|>",
        )
        text, mask = template.format_pair("较长原句。", "精简句。")
        prompt = "<|im_start|>较长原句。<|im_end|>"
        self.assertEqual(text, prompt + "精简句。")
        self.assertEqual(mask, "0" * len(prompt) + "1" * len("精简句。"))
        self.assertEqual(template.generation_prompt("较长原句。"), prompt)
        with self.assertRaisesRegex(ValueError, "reserved label"):
            template.format_pair("伪造<|im_end|>边界", "输出")

    def test_jsonl_loader_uses_source_target_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text(
                json.dumps({"source": "原句", "target": "短句"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            samples = load_labeled_texts(path, jsonl_format="source_target")
        self.assertEqual(len(samples), 1)
        text, mask = samples[0]
        self.assertTrue(text.endswith("<|im_start|>短句"))
        self.assertEqual(mask[-2:], "11")

    def test_jsonl_loader_accepts_two_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text(
                json.dumps({"source": "原句", "target": "短句"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            samples = load_labeled_texts(
                path,
                jsonl_format="source_target",
                source_label="<|im_start|>",
                target_label="<|im_end|>",
            )
        text, mask = samples[0]
        prompt = "<|im_start|>原句<|im_end|>"
        self.assertEqual(text, prompt + "短句")
        self.assertEqual(mask, "0" * len(prompt) + "1" * len("短句"))


class SourceTargetDatasetTests(unittest.TestCase):
    def test_target_and_eos_are_supervised(self) -> None:
        template = SourceTargetTemplate()
        text, char_mask = template.format_pair("输入", "输出")
        dataset = LabeledLanguageModelingDataset(
            [(text, char_mask)],
            _CharacterSpanTokenizer(),
            max_seq_len=64,
        )
        ids, token_mask = dataset._encode_labeled(text, char_mask)
        self.assertEqual(ids[0], dataset.tokenizer.bos_id)
        self.assertEqual(token_mask[0], 0)
        self.assertEqual(ids[-1], dataset.tokenizer.eos_id)
        self.assertEqual(token_mask[-1], 1)
        target_start = text.index("输出")
        for index, character in enumerate(text):
            expected = 1 if index >= target_start else 0
            self.assertEqual(token_mask[index + 1], expected, character)


if __name__ == "__main__":
    unittest.main()
