"""Active local-to-world rotations, XYZ positions, XYZW quaternions."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result.copy()


def quaternion(value, name: str = "quaternion") -> np.ndarray:
    q = vector(value, 4, name)
    if abs(np.linalg.norm(q) - 1.0) > 1e-4:
        raise ValueError(f"{name} must be unit length, in [x,y,z,w] order")
    q /= np.linalg.norm(q)
    return q


def rotation(q) -> np.ndarray:
    return Rotation.from_quat(quaternion(q)).as_matrix()


def yaw_quaternion(yaw: float) -> np.ndarray:
    return Rotation.from_rotvec([0.0, yaw, 0.0]).as_quat()


def same_orientation(a, b, tolerance: float = 1e-5) -> bool:
    return abs(float(np.dot(quaternion(a), quaternion(b)))) >= 1.0 - tolerance


def tip_from_wrist(wrist_position, wrist_orientation, wrist_to_tip_local) -> np.ndarray:
    """A rigid point transform: the offset rotates with the wrist."""
    return np.asarray(wrist_position) + rotation(wrist_orientation) @ np.asarray(wrist_to_tip_local)
