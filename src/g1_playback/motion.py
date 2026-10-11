from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class MotionClip:
    """Canonical playback representation.

    qpos is MuJoCo-native on purpose: this is the terminal representation for
    the visualization harness. Generator-specific adapters should convert into
    this type.
    """

    qpos: np.ndarray
    fps: float
    target_pos: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.qpos = np.asarray(self.qpos, dtype=np.float64)
        if self.qpos.ndim != 2:
            raise ValueError(f"qpos must have shape [T, nq], got {self.qpos.shape}")

        if self.fps <= 0:
            raise ValueError("fps must be positive")

        if self.target_pos is not None:
            self.target_pos = np.asarray(self.target_pos, dtype=np.float64)
            if self.target_pos.shape != (len(self.qpos), 3):
                raise ValueError(
                    "target_pos must have shape [T, 3] with the same T as qpos; "
                    f"got {self.target_pos.shape} for T={len(self.qpos)}"
                )

    @property
    def nframes(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def duration(self) -> float:
        return self.nframes / self.fps

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "qpos": self.qpos,
            "fps": np.array(self.fps, dtype=np.float64),
        }
        if self.target_pos is not None:
            payload["target_pos"] = self.target_pos
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "MotionClip":
        with np.load(path) as data:
            target = data["target_pos"] if "target_pos" in data else None
            return cls(
                qpos=data["qpos"],
                fps=float(data["fps"]),
                target_pos=target,
            )
