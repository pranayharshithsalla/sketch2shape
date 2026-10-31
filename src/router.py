"""Retry/backoff helper shared by the image and mesh provider routers."""
from __future__ import annotations

import json
import logging
import re
import time

from . import config

log = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    pass


# Errors that retrying will not fix (quota, auth, bad request) -> fail fast to the next provider
_FATAL_HINTS = ("quota", "401", "403", "unauthor", "invalid state", "not found", "404",
                "billing", "payment", "credit")


def with_hf_tokens(call):
    """Run call(token) with each configured HF token until one has ZeroGPU quota left."""
    last = None
    for i, tok in enumerate(config.HF_TOKENS or [None]):
        try:
            return call(tok)
        except Exception as e:
            last = e
            if "quota" not in str(e).lower():
                raise
            log.warning("HF token #%d out of ZeroGPU quota, trying next", i + 1)
    raise last


# Providers that hit quota / auth errors are skipped until this timestamp, so later runs
# do not waste ~20 s per call rediscovering that a free quota is exhausted. Persisted so
# restarts and batch runs remember too.
_COOLDOWN_FILE = config.OUTPUTS / ".provider_cooldown.json"
try:
    _cooldown: dict[str, float] = json.loads(_COOLDOWN_FILE.read_text())
except Exception:
    _cooldown = {}


def _save_cooldown():
    try:
        _COOLDOWN_FILE.write_text(json.dumps(_cooldown))
    except OSError:
        pass


def reset_cooldowns():
    _cooldown.clear()
    _save_cooldown()


def _cooldown_seconds(msg: str) -> float:
    m = re.search(r"try again in (\d+):(\d+):(\d+)", msg)
    if m:
        h, mi, s = map(int, m.groups())
        return max(60.0, h * 3600 + mi * 60 + s)
    if any(h in msg for h in ("401", "unauthor", "402", "payment", "billing")):
        return 6 * 3600.0
    if "quota" in msg or "429" in msg or "runs limit" in msg:
        return 30 * 60.0
    if any(h in msg for h in ("503", "high demand", "unavailable", "timed out", "timeout")):
        return 10 * 60.0  # overloaded service: try again later, not on every click
    return 0.0


def in_cooldown(name: str) -> float:
    return max(0.0, _cooldown.get(name, 0.0) - time.time())


def with_retries(fn, *args, name: str = "provider", retries: int | None = None, **kwargs):
    retries = config.MAX_RETRIES if retries is None else retries
    left = in_cooldown(name)
    if left:
        raise ProviderError(f"{name}: skipped, quota/auth cooldown for another {left / 60:.0f} min")
    last = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            msg = str(e).lower()
            log.warning("%s attempt %d failed: %s", name, attempt + 1, str(e)[:300])
            cd = _cooldown_seconds(msg)
            if cd:
                _cooldown[name] = time.time() + cd
                _save_cooldown()
            if any(h in msg for h in _FATAL_HINTS) or attempt == retries:
                break
            time.sleep(2 ** attempt * 2)
    raise ProviderError(f"{name}: {type(last).__name__}: {str(last)[:300]}")
