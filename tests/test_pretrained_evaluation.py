from __future__ import annotations

import math
import unittest

import torch

from muddywater.pretrained_evaluation.artifacts import audit_state_tensors
from muddywater.pretrained_evaluation.reporting import json_safe


class PretrainedArtifactAuditTests(unittest.TestCase):
    def test_state_audit_detects_nonfinite_and_tied_serialization(self) -> None:
        embedding = torch.tensor([[1.0, 0.0], [2.0, -3.0]])
        state = {
            "transformer.wte.weight": embedding,
            "lm_head.weight": embedding.clone(),
            "bad.weight": torch.tensor([float("nan")]),
            "zero.weight": torch.zeros(3),
        }

        audit = audit_state_tensors(state)

        self.assertFalse(audit["all_finite"])
        self.assertEqual(audit["nonfinite_by_tensor"], {"bad.weight": 1})
        self.assertEqual(audit["all_zero_tensors"], ["zero.weight"])
        self.assertTrue(audit["serialized_embedding_equals_lm_head"])

    def test_json_report_replaces_nonfinite_values(self) -> None:
        payload = {"finite": 1.5, "nan": math.nan, "nested": [math.inf]}

        self.assertEqual(
            json_safe(payload),
            {"finite": 1.5, "nan": None, "nested": [None]},
        )


if __name__ == "__main__":
    unittest.main()
