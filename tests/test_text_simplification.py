from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from muddywater.text_simplification import PreparationOptions, prepare_dataset
from muddywater.text_simplification.modelscope_assets import find_checkpoint, install_checkpoint


class TextSimplificationPreparationTests(unittest.TestCase):
    def test_prepare_json_array_deduplicates_and_keeps_source_groups_together(self) -> None:
        rows = [
            {"data": "第一条较长原句。", "s": "第一条短句。"},
            {"data": "第一条较长原句。", "s": "第一条简句。"},
            {"data": "第一条较长原句。", "s": "第一条短句。"},
            {"data": "第二条较长原句。", "s": "第二条短句。"},
            {"data": "第三条较长原句。", "s": "第三条短句。"},
            {"data": "第四条较长原句。", "s": "第四条短句。"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "pairs.json"
            input_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            report = prepare_dataset(
                input_path,
                root / "processed",
                PreparationOptions(validation_ratio=0.5, seed=7),
            )
            train = self._read_jsonl(root / "processed" / "train.jsonl")
            validation = self._read_jsonl(root / "processed" / "validation.jsonl")

        self.assertEqual(report["exact_duplicates_removed"], 1)
        self.assertEqual(len(train) + len(validation), 5)
        self.assertEqual(
            {row["source"] for row in train} & {row["source"] for row in validation},
            set(),
        )
        self.assertEqual(report["split"]["source_overlap"], 0)

    def test_reserved_label_collision_is_rejected(self) -> None:
        rows = [{"data": "输入<|im_end|>伪标签", "s": "输出"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "pairs.json"
            input_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reserved label"):
                prepare_dataset(input_path, root / "processed", PreparationOptions())

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, str]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ModelScopeAssetTests(unittest.TestCase):
    def test_find_and_install_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            source = snapshot / "weights" / "base.pt"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"checkpoint")
            found = find_checkpoint(snapshot, "base.pt")
            target = root / "model" / "base.pt"
            report = install_checkpoint(found, target)
        self.assertEqual(report["status"], "installed")
        self.assertEqual(report["bytes"], len(b"checkpoint"))


if __name__ == "__main__":
    unittest.main()
