"""
PortfolioReplayBuffer
=====================

A replay memory that supports the unified portfolio-level TD3 training
described in Contribution 2: a single shared buffer that accumulates
transitions from multiple representative medoid buildings (B1, B5).

Each transition stores the 5-tuple expected by the existing TD3 agent
(state, action, reward, next_state, done) PLUS parallel source metadata
(building_id, cycle, episode_step, global_step). The 5-tuple is stored
identically to the existing `ReplayBuffer` in `agents.rl` so that the
existing TD3 gradient update code can sample without modification.

The buffer is a FIFO ring; transitions are silently overwritten when
capacity is reached. With capacity = 1,000,000 and the unified experiment
generating 525,600 transitions, eviction never happens.

This class is independent of `agents.rl.ReplayBuffer` (not a subclass) so
that modifications to the existing TD3 path remain zero.
"""

from __future__ import annotations

import random
import numpy as np


class PortfolioReplayBuffer:
    """
    Shared multi-building replay buffer with parallel source metadata.

    Storage layout
    --------------
    `self.buffer`   : list of 5-tuples (state, action, reward, next_state, done)
                      aligned 1-to-1 with `self._meta`.
    `self._meta`    : list of 4-tuples (building_id, cycle, episode_step, global_step)
                      aligned 1-to-1 with `self.buffer`.
    `self._source_counts` : dict {building_id: int}
    """

    def __init__(self, capacity: int):
        if not isinstance(capacity, int):
            capacity = int(capacity)
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.buffer: list = []
        self._meta: list = []
        self.position: int = 0
        self._source_counts: dict = {}

    # ------------------------------------------------------------------ push
    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        building_id: int,
        cycle: int,
        episode_step: int,
        global_step: int,
    ) -> None:
        """Insert one transition with source metadata."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self._meta.append(None)

        idx = self.position
        self.buffer[idx] = (state, action, reward, next_state, done)
        self._meta[idx] = (int(building_id), int(cycle), int(episode_step), int(global_step))

        # Update source counter
        bid = int(building_id)
        self._source_counts[bid] = self._source_counts.get(bid, 0) + 1

        self.position = (self.position + 1) % self.capacity

    # ------------------------------------------------------------------ sample
    def sample(self, batch_size: int):
        """
        Sample a random batch. Returns (state, action, reward, next_state, done)
        with the exact same shape and order as the existing `ReplayBuffer.sample`
        so the existing TD3.add_to_buffer code works without modification.

        For source-tagged sampling (e.g. to verify mixed B1+B5 batches), use
        `sample_with_meta(batch_size)` instead.
        """
        if len(self.buffer) == 0:
            raise RuntimeError("Cannot sample from an empty buffer")
        if batch_size > len(self.buffer):
            raise ValueError(
                f"batch_size={batch_size} exceeds buffer size={len(self.buffer)}"
            )

        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]

        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_with_meta(self, batch_size: int):
        """
        Like `sample`, but also returns `meta_arr` of shape (batch_size, 4)
        with columns (building_id, cycle, episode_step, global_step).
        """
        if len(self.buffer) == 0:
            raise RuntimeError("Cannot sample from an empty buffer")
        if batch_size > len(self.buffer):
            raise ValueError(
                f"batch_size={batch_size} exceeds buffer size={len(self.buffer)}"
            )

        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]
        meta_batch = [self._meta[i] for i in indices]

        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        meta_arr = np.asarray(meta_batch, dtype=np.int64)
        return state, action, reward, next_state, done, meta_arr

    # ------------------------------------------------------------------ stats
    def __len__(self) -> int:
        return len(self.buffer)

    def source_counts(self) -> dict:
        """Return a copy of the per-building transition counts."""
        return dict(self._source_counts)

    def source_fractions(self) -> dict:
        """Return a copy of the per-building transition fractions."""
        total = sum(self._source_counts.values())
        if total == 0:
            return {bid: 0.0 for bid in self._source_counts}
        return {bid: c / total for bid, c in self._source_counts.items()}

    def summary(self) -> dict:
        """Return a human-readable summary of the buffer composition."""
        return {
            "total_transitions": len(self.buffer),
            "capacity": self.capacity,
            "fill_fraction": len(self.buffer) / self.capacity,
            "position": self.position,
            "source_counts": self.source_counts(),
            "source_fractions": self.source_fractions(),
        }

    # ------------------------------------------------------------------ serialisation
    def state_dict(self) -> dict:
        """Return a serialisable snapshot of the buffer (lists, not numpy)."""
        return {
            "capacity": self.capacity,
            "position": self.position,
            "buffer": self.buffer,
            "meta": self._meta,
            "source_counts": dict(self._source_counts),
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore the buffer from a `state_dict` snapshot."""
        self.capacity = int(sd["capacity"])
        self.position = int(sd["position"])
        self.buffer = list(sd["buffer"])
        self._meta = list(sd["meta"])
        self._source_counts = dict(sd["source_counts"])
