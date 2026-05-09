"""Reflex config for the FinSage chat app.

Databricks Apps exposes a single port (DATABRICKS_APP_PORT, default 8000).
In `--env prod` Reflex serves the compiled frontend bundle and the backend
RPC API from the same port, which fits the platform's single-port contract.

For local dev (no DATABRICKS_APP_PORT) Reflex falls back to its conventional
3000/8000 split.
"""

from __future__ import annotations

import os

import reflex as rx

_APP_PORT = int(os.getenv("DATABRICKS_APP_PORT", "8000"))

config = rx.Config(
    app_name="finsage_app",
    backend_port=_APP_PORT,
    frontend_port=_APP_PORT,
    api_url=os.getenv("DATABRICKS_APP_URL", f"http://localhost:{_APP_PORT}"),
    tailwind=None,  # use Reflex's built-in theme system, not Tailwind
    show_built_with_reflex=False,
)
