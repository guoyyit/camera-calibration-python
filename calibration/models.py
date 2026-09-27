"""Data models for camera intrinsic calibration.

Coordinate conventions
----------------------
- Object/board points are 2D points on a planar calibration target (z = 0),
  given in the target's own metric coordinate frame, units of metres or
  millimetres (consistent within a task).
- Image points are pixel coordinates with the origin at the top-left pixel
  centre, x to the right, y down.
- The camera model is the standard pinhole model with radial-tangential
  distortion (Brown-Conrady):

      x_d = x_u * (1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x_u y_u + p2 (r^2 + 2 x_u^2)
      y_d = y_u * (1 + k1 r^2 + k2 r^4 + k3 r^6) + p1 (r^2 + 2 y_u^2) + 2 p2 x_u y_u

  where (x_u, y_u) are normalised ideal coordinates, r^2 = x_u^2 + y_u^2,
  and pixel coordinates are u = fx * x_d + cx, v = fy * y_d + cy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class FrameStatus(str, Enum):
    """Lifecycle status of a submitted frame."""

    ACCEPTED = "accepted"      # valid, participates in calibration
    DUPLICATE = "duplicate"    # same frame_id already stored; ignored
    CORRUPT = "corrupt"        # failed validation; ignored
    ABANDONED = "abandoned"    # explicitly excluded from calibration


@dataclass(frozen=True)
class FrameObservation:
    """One frame of 2D corner observations.

    ``object_points`` and ``image_points`` are (N, 2) arrays with N >= 4
    corresponding points.  ``frame_id`` uniquely identifies the frame;
    frames with a repeated id are treated as duplicates and ignored.
    """

    frame_id: str
    object_points: np.ndarray  # (N, 2) float64
    image_points: np.ndarray   # (N, 2) float64
    image_size: tuple[int, int]  # (width, height) of the source image


@dataclass(frozen=True)
class CalibrationConfig:
    """Tunable solver configuration."""

    max_iterations: int = 50
    convergence_tolerance: float = 1e-10
    # Damping (Levenberg-Marquardt initial lambda) kept fixed for
    # reproducibility; the solver is deterministic given the same inputs.
    use_tangential: bool = True
    use_k3: bool = True


@dataclass
class CalibrationResult:
    """Outcome of one calibration run."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray          # (5,) -> [k1, k2, p1, p2, k3]
    rms_error: float                # overall RMS reprojection error (pixels)
    per_frame_error: dict[str, float]  # frame_id -> RMS reprojection error
    iterations: int                 # iterations actually performed
    converged: bool
    frame_ids: list[str]            # sorted ids of frames that participated
    config: CalibrationConfig = field(default_factory=CalibrationConfig)

    def camera_matrix(self) -> np.ndarray:
        """Return the 3x3 intrinsic matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


class TaskStatus(str, Enum):
    CREATED = "created"
    CALIBRATED = "calibrated"


@dataclass
class TaskInfo:
    """Metadata about a calibration task."""

    task_id: str
    status: TaskStatus
    frame_count: int          # number of ACCEPTED frames
    result: Optional[CalibrationResult] = None
