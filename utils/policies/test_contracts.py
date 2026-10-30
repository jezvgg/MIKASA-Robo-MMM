"""CPU adapter tests; real-data checks enabled by POLICY_DATA_ROOT."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from utils.policies.data import FetchDataset, epoch_plan
from utils.policies.sampling import DiagnosticSampler, EpisodeSampler
from utils.policies.memory import CausalMemory
from utils.policies.workspace import contained, environment, require_space, usage_bytes


class SamplingTests(unittest.TestCase):
    def test_exact_epoch(self):
        class Source:
            episodes = [object(), object(), object()]
            ends = [3, 8, 10]
            def __len__(self):
                return 10
        sampler = EpisodeSampler(Source(), 42)
        for _ in range(3):
            indices = list(sampler)
            self.assertEqual(sorted(indices), list(range(10)))
            for start, stop in ((0, 3), (3, 8), (8, 10)):
                self.assertEqual([x for x in indices if start <= x < stop], list(range(start, stop)))

    def test_partial_batch_not_dropped(self):
        plan = epoch_plan(1_483_491, 32)
        self.assertEqual(plan["samples_per_epoch"], 1_483_491)
        self.assertEqual(plan["final_batch_valid"], 3)
        self.assertEqual(plan["steps_per_epoch"], 46_360)


class CausalMemoryTests(unittest.TestCase):
    def setUp(self):
        class Buffer:
            def __init__(self):
                self.frames = {}
            def clear(self):
                self.frames.clear()
            def add_buffer(self, images, states, indices):
                for index in indices:
                    self.frames[index] = states[0].copy()
            def get_history_feats(self, index):
                return self.frames[index]
        self.memory = CausalMemory.__new__(CausalMemory)
        self.memory.buffer = Buffer()
        self.memory.episode, self.memory.frame = None, -1
        self.memory.ingested, self.memory.resets = 0, 0
        self.image = np.zeros((2, 2, 3), dtype=np.uint8)
        self.state = np.ones(12, dtype=np.float32)

    def test_prefix_and_future_guard(self):
        memory = self.memory
        memory.ingest(0, 0, self.image, self.state)
        self.assertEqual(set(memory.gather([0], 0)), {0})
        with self.assertRaisesRegex(ValueError, "future/past-episode"):
            memory.gather([1], 0)
        with self.assertRaisesRegex(ValueError, "non-causal"):
            memory.ingest(0, 2, self.image, self.state)

    def test_episode_and_restart_clear(self):
        memory = self.memory
        memory.ingest(0, 0, self.image, self.state)
        memory.ingest(0, 1, self.image, self.state)
        memory.ingest(1, 0, self.image, self.state * 2)
        self.assertEqual(set(memory.buffer.frames), {0})
        with self.assertRaisesRegex(ValueError, "episode boundary"):
            memory.gather([0], 0)
        memory.ingest(1, 0, self.image, self.state * 3)
        np.testing.assert_array_equal(memory.gather([0], 1)[0], self.state * 3)
        self.assertEqual(memory.resets, 3)


class DiagnosticTests(unittest.TestCase):
    def test_three_microsteps_include_both_tasks(self):
        from types import SimpleNamespace
        source = SimpleNamespace(tasks=("takeitback", "veggie"), ends=[5, 12],
            episodes=[SimpleNamespace(task="takeitback", global_index=0),
                      SimpleNamespace(task="veggie", global_index=1)])
        self.assertEqual(list(DiagnosticSampler(source, 3)), [0, 1, 5])


class WorkspaceTests(unittest.TestCase):
    def test_destinations_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            root.mkdir()
            env = environment(root)
            for name in ("HOME", "TMPDIR", "HF_HOME", "UV_CACHE_DIR", "OPENPI_DATA_HOME",
                         "TORCH_HOME", "PYTHONPYCACHEPREFIX", "WANDB_DIR"):
                self.assertTrue(Path(env[name]).resolve().is_relative_to(root))
            outside = Path(temporary) / "outside"
            outside.mkdir()
            (root / "escape").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                contained(root, root / "escape" / "output")
            before = usage_bytes(root)
            (outside / "large-input").write_bytes(b"0" * 100_000)
            self.assertEqual(usage_bytes(root), before)
            self.assertGreaterEqual(require_space(root, 0), 0)


@unittest.skipUnless(os.environ.get("POLICY_DATA_ROOT"), "real dataset root not set")
class RealDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = FetchDataset(os.environ["POLICY_DATA_ROOT"])

    def test_task_counts_and_tail_boundaries(self):
        dataset = self.dataset
        self.assertEqual(len(dataset), 1_483_491)
        self.assertEqual(len(dataset.episodes), 2000)
        for task in dataset.tasks:
            episode = next(e for e in dataset.episodes if e.task == task)
            end = dataset.ends[episode.global_index] - 1
            sample = dataset.sample(end, 20, images=False)
            self.assertEqual(sample["task"], task)
            self.assertEqual(sample["state"].shape, (12,))
            self.assertEqual(sample["actions"].shape, (20, 13))
            rows = dataset.episode_rows(episode.global_index)
            np.testing.assert_array_equal(sample["actions"],
                np.repeat(np.asarray(rows["action"][-1], dtype=np.float32)[None], 20, axis=0))

    def test_full_epoch_sampler(self):
        dataset = self.dataset
        indices = np.fromiter(iter(EpisodeSampler(dataset, 42)), dtype=np.int64, count=len(dataset))
        np.testing.assert_array_equal(np.sort(indices), np.arange(len(dataset)))


if __name__ == "__main__":
    unittest.main()
