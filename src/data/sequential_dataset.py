"""
Sequential dataset for world-model training.

Yields fixed-length trajectories ``(frames, actions)`` of shape
``([T, C, H, W], [T, A])`` from either an in-memory tensor source or from
a directory of per-episode files. Designed to feed an action-conditioned
predictor for LeWM-style training.

Two backends are provided:

1. ``SyntheticSequentialDataset``: generates random trajectories on the fly
   for unit tests and smoke tests — useful when no real environment data is
   available.
2. ``EpisodeFileSequentialDataset``: loads ``.pt`` files where each file is
   a dict ``{"frames": [T, C, H, W], "actions": [T, A]}`` and produces
   length-``seq_len`` windows from those episodes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class SyntheticSequentialDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Synthetic random trajectories. Useful only for tests and smoke runs.

    Returns dicts of ``{"frames": [T, C, H, W], "actions": [T, A]}``.
    """

    def __init__(
        self,
        num_episodes: int,
        seq_len: int,
        image_size: int = 64,
        channels: int = 3,
        action_dim: int = 4,
        seed: int | None = None,
    ):
        self.num_episodes = num_episodes
        self.seq_len = seq_len
        self.image_size = image_size
        self.channels = channels
        self.action_dim = action_dim
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Use a per-index seed so __getitem__ is deterministic when seed is set.
        local_gen = torch.Generator()
        local_gen.manual_seed(int(self.generator.initial_seed() + idx))
        frames = torch.rand(
            self.seq_len,
            self.channels,
            self.image_size,
            self.image_size,
            generator=local_gen,
        )
        actions = torch.rand(self.seq_len, self.action_dim, generator=local_gen) * 2 - 1
        return {"frames": frames, "actions": actions}


class EpisodeFileSequentialDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Loads trajectories from episode files on disk and yields windowed samples.

    Each ``.pt`` file under ``root_dir`` must contain a dict with keys:
        - ``"frames"``: tensor of shape ``[L, C, H, W]`` (or convertible)
        - ``"actions"``: tensor of shape ``[L, A]`` (or convertible)
    where ``L`` is the episode length (must be ``>= seq_len``).

    Args:
        root_dir: Directory containing per-episode ``.pt`` files.
        seq_len: Window length to sample from each episode.
        transform: Optional per-frame transform applied to the frames tensor.
        random_window: If True, sample a random window each call; otherwise
            return overlapping windows in order.
    """

    def __init__(
        self,
        root_dir: str | Path,
        seq_len: int,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        random_window: bool = True,
    ):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir does not exist: {self.root_dir}")
        self.seq_len = seq_len
        self.transform = transform
        self.random_window = random_window

        self.episode_files = sorted(self.root_dir.glob("*.pt"))
        if not self.episode_files:
            raise FileNotFoundError(f"No .pt files found under {self.root_dir}")

        # Build a window index: list of (file_idx, start_offset) tuples.
        # For random_window mode, each episode contributes one virtual sample.
        # For deterministic mode, each episode contributes (L - seq_len + 1) windows.
        self._window_index: list[tuple[int, int]] = []
        for file_idx, path in enumerate(self.episode_files):
            data = torch.load(path, map_location="cpu", weights_only=False)
            length = data["frames"].shape[0]
            if length < seq_len:
                continue  # skip too-short episodes
            if random_window:
                self._window_index.append((file_idx, 0))
            else:
                for start in range(length - seq_len + 1):
                    self._window_index.append((file_idx, start))

    def __len__(self) -> int:
        return len(self._window_index)

    def _load_episode(self, file_idx: int) -> dict[str, torch.Tensor]:
        path = self.episode_files[file_idx]
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[no-any-return]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_idx, start = self._window_index[idx]
        episode = self._load_episode(file_idx)
        frames = episode["frames"]
        actions = episode["actions"]
        L = frames.shape[0]

        if self.random_window:
            start = int(torch.randint(0, L - self.seq_len + 1, (1,)).item())

        end = start + self.seq_len
        frames_window = frames[start:end].float()
        actions_window = actions[start:end].float()

        if self.transform is not None:
            frames_window = self.transform(frames_window)

        return {"frames": frames_window, "actions": actions_window}


def build_sequential_dataloader(
    dataset: Dataset[dict[str, torch.Tensor]],
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = False,
) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
    """Convenience wrapper to build a DataLoader for sequential datasets."""

    def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        frames = torch.stack([item["frames"] for item in batch], dim=0)
        actions = torch.stack([item["actions"] for item in batch], dim=0)
        return {"frames": frames, "actions": actions}

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
    )


def _build_synthetic_from_config(
    config: dict[str, Any],
) -> SyntheticSequentialDataset:
    """Helper to build a SyntheticSequentialDataset from a config dict."""
    return SyntheticSequentialDataset(
        num_episodes=config.get("num_episodes", 100),
        seq_len=config.get("seq_len", 16),
        image_size=config.get("image_size", 64),
        channels=config.get("channels", 3),
        action_dim=config.get("action_dim", 4),
        seed=config.get("seed", None),
    )
