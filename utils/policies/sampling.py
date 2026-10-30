"""Exact-pass episode sampling for bounded memory-feature caches."""
from __future__ import annotations

import numpy as np


class DiagnosticSampler:
    """Tiny explicit sample, balanced across tasks, chronological within episodes.

    This is deliberately NOT an epoch. It lets even three optimizer steps see
    both datasets without requiring memory access to unseen earlier frames.
    """
    def __init__(self, source, anchors: int):
        if anchors < len(source.tasks):
            raise ValueError("diagnostic needs at least one anchor per task")
        self.indices = []
        for task_index, task in enumerate(source.tasks):
            episode = next(e for e in source.episodes if e.task == task)
            start = int(source.ends[episode.global_index - 1]) if episode.global_index else 0
            count = anchors // len(source.tasks) + (task_index < anchors % len(source.tasks))
            end = int(source.ends[episode.global_index])
            if start + count > end:
                raise ValueError("diagnostic exceeds first episode length")
            self.indices.extend(range(start, start + count))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class EpisodeSampler:
    """Shuffle episodes, retain chronological anchors; visit every frame once.

    Chronological anchors permit causal TokenDrop updates and cache eviction at
    episode boundaries. Task weighting remains exactly proportional to frames.
    """
    def __init__(self, source, seed: int):
        self.source = source
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.source.episodes))
        self.epoch += 1
        for episode_index in order:
            start = self.source.ends[episode_index - 1] if episode_index else 0
            yield from range(start, self.source.ends[episode_index])
