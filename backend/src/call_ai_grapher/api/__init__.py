"""HTTP API exposing the handwriting improvement pipeline to the web UI."""

from call_ai_grapher.api.server import create_app

__all__ = ["create_app"]
