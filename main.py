"""Application entrypoint.

Run with:  uv run uvicorn main:app
The store path can be overridden with the CALIBRATION_DB env var.
"""

import os

from calibration.api import create_app

app = create_app(os.environ.get("CALIBRATION_DB", "calibration.db"))
