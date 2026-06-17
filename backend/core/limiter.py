"""Shared slowapi rate-limiter instance.

Import this module in ``main.py`` (to attach to app.state) and in route
modules (to apply ``@limiter.limit(...)`` decorators).  Keeping the
``Limiter`` in its own module avoids circular imports between main.py and
the router files.
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
