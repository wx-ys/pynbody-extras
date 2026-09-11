"""Helpers for safely accessing pynbody simulation arrays.

pynbody snapshots expose two kinds of arrays:

- **stored** arrays, listed by ``sim.keys()``; and
- **derived** arrays, computed on demand and accessible via ``sim[name]`` but
  *not* listed by ``sim.keys()``.

Because derived arrays are invisible to ``keys()``, a membership test such as
``name in sim.keys()`` is an unreliable existence check — it reports false
negatives and silently changes behaviour.  Probe ``sim[name]`` directly via
:func:`get_array` / :func:`has_array` instead, so every call site handles both
kinds of array the same way.

Example
-------
::

    softening = get_array(sim, "smooth")  # array or None, derived arrays included
"""

from __future__ import annotations

from typing import Any

__all__ = ["get_array", "has_array"]


def get_array(sim: Any, name: str, default: Any = None) -> Any:
    """Return ``sim[name]`` if the array is available, otherwise *default*.

    Unlike ``name in sim.keys()``, this probes ``sim[name]`` directly, so
    pynbody *derived* arrays (which are not listed by ``keys()``) are returned
    rather than wrongly treated as missing.
    """
    try:
        return sim[name]
    except KeyError:
        return default


def has_array(sim: Any, name: str) -> bool:
    """Whether ``sim[name]`` is available (including derived arrays)."""
    try:
        sim[name]
    except KeyError:
        return False
    return True
