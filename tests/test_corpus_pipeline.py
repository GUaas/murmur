from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from muddywater.corpus_text import clean_text, dedupe_digest
from muddywater.hf_dataset import (
    build_download_manifest,
    combine_completed_download_manifests,
    select_evenly_spaced,
    select_evenly_spaced_extension,
)
from muddywater.parquet_corpus import (
    CorpusCleaningPolicy,
    clean_parquet_corpus,
    load_jsonl_dedupe_seed,
)


class DatasetSelectionTests(unittest.TestCase):
    def test_even_selection_spans_the_full_sorted_file_list(self) -> None:
        paths = [f"4_5/{index:06d}.parquet" for index in range(100)]
        selected = select_evenly_spaced(paths, count=10)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(set(selected)), 10)
        self.assertEqual(selected[0], "4_5/000005.parquet")
        self.assertEqual(selected[-1], "4_5/000095.parquet")

    def test_manifest_estimates_selected_token_fraction(self) -> None:
        all_paths = [f"4_5/{index:06d}.parquet" for index in range(100)]
        selected = select_evenly_spaced(all_paths, count=25)
        manifest = build_download_manifest(
            repo_id="owner/dataset",
            revision="a" * 40,
            subset="4_5",
            all_paths=all_paths,
            selected_paths=selected,
            upstream_subset_tokens=40_000,
        )
        self.assertEqual(
            manifest["selection"]["estimated_selected_tokens_before_local_cleaning"],
            10_000,
        )

    def test_extension_is_balanced_and_does_not_overlap_base(self) -> None:
        paths = [f"4_5/{index:06d}.parquet" for index in range(100)]
        base = select_evenly_spaced(paths, count=30)
        extension = select_evenly_spaced_extension(
            paths,
            base,
            target_total_count=55,
        )
        self.assertEqual(len(extension), 25)
        self.assertFalse(set(base) & set(extension))
        self.assertEqual(len(set(base) | set(extension)), 55)
        self.assertLess(paths.index(extension[0]), 5)
        self.assertGreater(paths.index(extension[-1]), 94)

    def test_completed_manifests_can_be_combined(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = {
                "repo_id": "owner/dataset",
                "revision": "a" * 40,
                "subset": "4_5",
            }

            def manifest(repo_path: str, size_bytes: int) -> dict:
                return {
                    "status": "completed",
                    "dataset": dataset,
                    "selection": {
                        "available_files": 10,
                        "upstream_subset_tokens": 10_000,
                        "estimated_selected_tokens_before_local_cleaning": 1_000,
                    },
                    "selected_repo_paths": [repo_path],
                    "completed_files": [
                        {
                            "repo_path": repo_path,
                            "local_path": str(root / Path(repo_path).name),
                            "size_bytes": size_bytes,
                            "sha256": "b" * 64,
                        }
                    ],
                    "completed_bytes": size_bytes,
                }

            first = root / "first.json"
            second = root / "second.json"
            first.write_text(
                json.dumps(manifest("4_5/000001.parquet", 100)),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(manifest("4_5/000005.parquet", 200)),
                encoding="utf-8",
            )
            combined = combine_completed_download_manifests(
                [first, second],
                root / "combined.json",
            )
            self.assertEqual(combined["completed_count"], 2)
            self.assertEqual(combined["completed_bytes"], 300)
            self.assertEqual(
                combined["selection"][
                    "estimated_selected_tokens_before_local_cleaning"
                ],
                2_000,
            )


class CorpusTextTests(unittest.TestCase):
    def test_cleaning_preserves_chinese_punctuation_without_nfkc(self) -> None:
        text = "第一句。\u200b  第二句！\r\n第三句？"
        cleaned = clean_text(text, normalize_unicode=False)
        self.assertEqual(cleaned, "第一句。 第二句！\n第三句？")

    def test_normalized_dedupe_ignores_spacing_and_punctuation(self) -> None:
        left = dedupe_digest("中文，测试！", "normalized")
        right = dedupe_digest("中文 测试", "normalized")
        self.assertEqual(left, right)


class ParquetCleaningTests(unittest.TestCase):
    def test_score_threshold_must_match_declared_scale(self) -> None:
        CorpusCleaningPolicy(min_score=0.8, score_scale_max=1.0).validate()
        with self.assertRaisesRegex(
            ValueError,
            "min_score must be between 0 and score_scale_max",
        ):
            CorpusCleaningPolicy(min_score=4.0, score_scale_max=1.0).validate()

    def test_jsonl_seed_supports_incremental_cross_corpus_dedupe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cleaned = root / "cleaned"
            cleaned.mkdir()
            shard = cleaned / "clean_00000.jsonl"
            shard.write_text(
                json.dumps(
                    {"text": "一段用于测试增量语料去重的中文文本。"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            seed, stats = load_jsonl_dedupe_seed(
                cleaned,
                dedupe_mode="normalized",
                expected_records=1,
                expected_shard_names=[shard.name],
                progress_interval=0,
            )

            self.assertEqual(stats["records"], 1)
            self.assertEqual(stats["unique_digests"], 1)
            self.assertIn(
                dedupe_digest("一段用于测试增量语料去重的中文文本", "normalized"),
                seed,
            )

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("pyarrow") is not None,
        "pyarrow is optional",
    )
    def test_small_parquet_is_filtered_chunked_and_provenance_is_kept(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.parquet"
            good = "这是用于预训练的高质量中文段落，包含完整的上下文和自然语言表达。" * 4
            long_text = "".join(
                f"第{index}段长篇中文内容需要按照自然边界进行分块，以保留连续语义。"
                for index in range(35)
            )
            table = pa.table(
                {
                    "text": [
                        good,
                        good,
                        "This document is almost entirely English and should be removed." * 3,
                        "分数不足的中文内容。" * 20,
                        long_text,
                    ],
                    "score": [4.5, 4.5, 4.5, 3.5, 4.8],
                    "source": ["CCI3", "CCI3", "Other", "WuDao", "Wanjuan"],
                }
            )
            pq.write_table(table, input_path)
            policy = CorpusCleaningPolicy(
                min_chars=50,
                max_chars=300,
                min_cjk_ratio=0.25,
                max_records_per_shard=2,
                batch_size=2,
            )
            report = clean_parquet_corpus(
                input_paths=[input_path],
                output_dir=root / "cleaned",
                report_path=root / "cleaning_report.json",
                source_manifest={
                    "dataset": {"repo_id": "owner/dataset", "revision": "a" * 40},
                    "selection": {"selected_files": 1},
                },
                policy=policy,
                progress_interval=0,
            )

            self.assertEqual(report["raw_documents"], 5)
            self.assertEqual(report["drop_reasons"]["duplicate_normalized"], 1)
            self.assertEqual(report["drop_reasons"]["low_cjk_ratio"], 1)
            self.assertEqual(report["drop_reasons"]["score_below_minimum"], 1)
            self.assertGreater(report["chunked_documents"], 0)
            self.assertGreaterEqual(len(report["output_shards"]), 2)

            output_records = []
            for path in sorted((root / "cleaned").glob("clean_*.jsonl")):
                output_records.extend(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                )
            self.assertEqual(len(output_records), report["kept_records"])
            self.assertTrue(
                all(
                    {"text", "source", "score", "source_file", "source_row", "chunk_index"}
                    <= set(record)
                    for record in output_records
                )
            )

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("pyarrow") is not None,
        "pyarrow is optional",
    )
    def test_incremental_cleaning_drops_text_already_present_in_seed(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "extension.parquet"
            existing = "这是已经存在于基础语料中的中文内容，增量清洗时应当被全局去重。" * 4
            novel = "这是新增语料中的不同中文内容，质量合格并且应当被保留下来用于训练。" * 4
            pq.write_table(
                pa.table(
                    {
                        "text": [existing, novel],
                        "score": [0.9, 0.9],
                        "source": ["CCI3", "CCI3"],
                    }
                ),
                input_path,
            )
            policy = CorpusCleaningPolicy(
                min_score=0.8,
                score_scale_max=1.0,
                min_chars=50,
                max_chars=500,
                min_cjk_ratio=0.25,
                max_records_per_shard=10,
                batch_size=2,
            )
            report = clean_parquet_corpus(
                input_paths=[input_path],
                output_dir=root / "cleaned_extension",
                report_path=root / "cleaning_extension_report.json",
                source_manifest={
                    "dataset": {"repo_id": "owner/dataset", "revision": "a" * 40},
                    "selection": {"selected_files": 1},
                },
                policy=policy,
                progress_interval=0,
                seed_digests={dedupe_digest(existing, "normalized")},
            )

            self.assertEqual(report["kept_records"], 1)
            self.assertEqual(
                report["drop_reasons"]["duplicate_against_seed_normalized"],
                1,
            )
            self.assertEqual(report["dedupe_seed_unique_digests"], 1)


if __name__ == "__main__":
    unittest.main()
