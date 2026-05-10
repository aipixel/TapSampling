from __future__ import annotations

import h5py
import torch
from torch.utils.data import Dataset


class ActionDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        window_size: int = 10,
        split: str | None = None,
        train_split_ratio: float = 0.8,
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if split not in {None, "train", "val"}:
            raise ValueError(f"Unsupported split: {split}")

        self.window_size = window_size
        self.data: dict[str, torch.Tensor] = {}
        self.lengths: dict[str, int] = {}
        self.keys: list[str] = []
        self.slice_index: list[tuple[str, int]] = []
        self.feature_dim = 0

        with h5py.File(h5_path, "r") as handle:
            for key in handle.keys():
                array = handle[key][...]
                tensor = torch.from_numpy(array).to(torch.float32)
                if tensor.ndim != 2:
                    raise ValueError(f"Expected 2D action array for {key}, got shape {tuple(tensor.shape)}")
                if self.feature_dim == 0:
                    self.feature_dim = int(tensor.shape[1])
                elif int(tensor.shape[1]) != self.feature_dim:
                    raise ValueError(
                        f"Inconsistent feature dimension for {key}: expected {self.feature_dim}, got {tensor.shape[1]}"
                    )
                self.data[key] = tensor
                self.lengths[key] = int(tensor.shape[0])
                self.keys.append(key)

        if split is not None:
            split_idx = int(len(self.keys) * train_split_ratio)
            if split == "train":
                self.keys = self.keys[:split_idx]
            else:
                self.keys = self.keys[split_idx:]

        for key in self.keys:
            length = self.lengths[key]
            if length < window_size:
                continue
            for start in range(length - window_size + 1):
                self.slice_index.append((key, start))

        approx_mb = sum(value.numel() for value in self.data.values()) * 4 / 1024**2
        print(
            f"[ActionDataset] loaded {len(self.keys)} sequences, "
            f"{len(self.slice_index)} total slices, "
            f"~{approx_mb:.1f} MB"
        )

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, idx: int):
        key, start = self.slice_index[idx]
        return {"actions": self.data[key][start : start + self.window_size]}
