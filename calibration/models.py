"""Core data types for camera intrinsic calibration.

Coordinate conventions
----------------------
* Object/board points are 3D points in the calibration-board frame, Z=0 plane.
* Image points are 2D pixel coordinates ``(u, v)`` with the origin at the
  top-left pixel of the image, ``u`` growing right, ``v`` growing down.
* The pinhole model is::

      x = X/Z, y = Y/Z                      (normalised camera coords)
      r2 = x*x + y*y
      x_d = x*(1 + k1*r2 + k2*r2^2) + 2*p1*x*y + p2*(r2 + 2*x*x)
      y_d = y*(1 + k1*r2 + k2*r2^2) + p1*(r2 + 2*y*y) + 2*p2*x*y
      u = fx * x_d + cx
      v = fy * y_d + cy

  with radial distortion ``(k1, k2)`` and tangential distortion ``(p1, p2)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameObservation:
    """2D corner observations of one frame.

    ``frame_id`` uniquely identifies the frame; duplicate submissions with the
    same id are applied exactly once.  ``object_points`` are the board-frame
    3D coordinates (Z=0) and ``image_points`` the matching pixel detections.
    """

    frame_id: str
    object_points: tuple[tuple[float, float, float], ...]
    image_points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RejectedFrame:
    """A frame that was received but not accepted for calibration."""

    frame_id: str
    reason: str


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverConfig:
    """Numerical configuration of the bundle-adjustment solver."""

    max_iterations: int = 50
    convergence_tolerance: float = 1e-9
    # Levenberg-Marquardt damping, kept fixed for determinism.
    initial_damping: float = 1e-3


@dataclass(frozen=True)
class TaskConfig:
    """Configuration of a calibration task."""

    image_width: int
    image_height: int
    solver: SolverConfig = field(default_factory=SolverConfig)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraIntrinsics:
    """Estimated intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float


@dataclass(frozen=True)
class FrameError:
    """Reprojection statistics of a single frame."""

    frame_id: str
    mean_error: float
    max_error: float


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of one calibration run."""

    intrinsics: CameraIntrinsics
    frame_errors: tuple[FrameError, ...]
    mean_reprojection_error: float
    iterations: int
    converged: bool
    # Sorted ids of the frames that actually took part in the optimisation.
    used_frame_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# Mapping policy
# ---------------------------------------------------------------------------


class BorderPolicy:
    """How undistortion handles pixels mapped outside the source image."""

    CLAMP = "clamp"  # clamp coordinates to the image border
    SKIP = "skip"  # leave the destination pixel at the fill value (0)


def sorted_frames(frames: Sequence[FrameObservation]) -> list[FrameObservation]:
    """Return frames sorted by ``frame_id`` (canonical processing order)."""
    return sorted(frames, key=lambda f: f.frame_id)
