from __future__ import annotations

import unittest

from torch.utils.data import DataLoader, SequentialSampler

from muddywater.sampling import (
    DeterministicEpochSampler,
    ResumeOffsetSampler,
    set_loader_start_batch,
)


class ResumeOffsetSamplerTests(unittest.TestCase):
    def test_offset_preserves_deterministic_epoch_order(self) -> None:
        dataset = list(range(40))
        baseline = DeterministicEpochSampler(dataset, seed=17, shuffle=True)
        baseline.set_epoch(3)
        expected = list(baseline)

        resumed_base = DeterministicEpochSampler(dataset, seed=17, shuffle=True)
        resumed = ResumeOffsetSampler(resumed_base)
        resumed.set_epoch(3)
        resumed.set_start_index(13)

        self.assertEqual(list(resumed), expected[13:])
        self.assertEqual(len(resumed), len(expected) - 13)

    def test_loader_starts_at_completed_batch_boundary(self) -> None:
        dataset = list(range(20))
        sampler = ResumeOffsetSampler(SequentialSampler(dataset))
        loader = DataLoader(dataset, batch_size=4, sampler=sampler)

        self.assertTrue(set_loader_start_batch(loader, 3))
        self.assertEqual(len(loader), 2)
        self.assertEqual([batch.tolist() for batch in loader], [[12, 13, 14, 15], [16, 17, 18, 19]])

    def test_legacy_loader_uses_safe_fallback(self) -> None:
        loader = DataLoader(list(range(8)), batch_size=2, shuffle=False)
        self.assertFalse(set_loader_start_batch(loader, 2))


if __name__ == "__main__":
    unittest.main()
