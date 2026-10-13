"""Settings read from environment variables.

Every knob is read as ``STEERCA_<NAME>``, falling back to ``CARL_<NAME>``: the
code was developed under the name CARL, and the commands in the papers'
reproducibility tables use that prefix. Either spelling works; ``STEERCA_``
wins when both are set.
"""
import os

PREFIXES = ("STEERCA_", "CARL_")


def setting(name, default=None):
    """The value of ``STEERCA_<name>`` or ``CARL_<name>``, else ``default``."""
    for prefix in PREFIXES:
        value = os.environ.get(prefix + name)
        if value is not None:
            return value
    return default


def env_flag(name, default=False):
    """A boolean setting: unset gives ``default``; ``0``/``false``/``no``/empty is False."""
    value = setting(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def models_dir():
    """Where training writes and resumes checkpoints (``STEERCA_MODELS``, default ``./models``)."""
    return setting("MODELS", "./models")
