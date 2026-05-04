"""Logging helpers for the IDP system."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(f"idp_system.{name}")
