import logging
import sys

logger = logging.getLogger("pynext")

# Expose names for import *
__all__ = ["logger", "setlevel", "set_color"]

class DuplicateFilter(logging.Filter):
    """A filter that removes duplicated successive log entries."""

    # source    #yt
    # https://stackoverflow.com/questions/44691558/suppress-multiple-messages-with-same-content-in-python-logging-module-aka-log-co
    def filter(self, record):
        current_log = (record.module, record.levelno, record.msg, record.args)
        if current_log != getattr(self, "last_log", None):
            self.last_log = current_log
            return True
        return False

class BlankLineFormatter(logging.Formatter):

    def format(self, record):
        if record.msg == "" and not record.args: # blank line
            return ""
        return super().format(record)

class _Ansi:
    RESET = "\033[0m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"

class Fore:
    RED = _Ansi.RED
    YELLOW = _Ansi.YELLOW
    GREEN = _Ansi.GREEN
    CYAN = _Ansi.CYAN
    MAGENTA = _Ansi.MAGENTA
    BLUE = _Ansi.BLUE

class Style:
    RESET_ALL = _Ansi.RESET

# default format strings
ufstring = "%(name)-3s: [%(levelname)-9s] %(asctime)s %(message)s"
cfstring = "%(name)-3s: [%(levelname)-18s] %(asctime)s %(message)s"

# config holder to avoid rebinding module-level names (avoids `global` usage)
_config = {
    "colors_enabled": False,  # None -> auto (detect TTY), True/False -> explicit
}

# color palette (mutable dict — we will mutate in-place to avoid `global`)
_color_palette = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.MAGENTA,
}

class ColoredFormatter(BlankLineFormatter):
    """Formatter that injects color codes into the formatted message based on level."""
    def __init__(self, fmt=None, use_colors=None):
        super().__init__(fmt)
        self._use_colors = use_colors

    def _colors_on(self):
        # determine final on/off: explicit override -> use it; else auto-detect TTY
        enabled = _config.get("colors_enabled")
        if enabled is not None:
            return bool(enabled)
        if self._use_colors is not None:
            return bool(self._use_colors)
        try:
            return getattr(sys.stdout, "isatty", lambda: False)()
        except Exception:
            return False

    def format(self, record):
        base = super().format(record)
        if not self._colors_on():
            return base
        color = _color_palette.get(record.levelno, "")
        reset = Style.RESET_ALL if hasattr(Style, "RESET_ALL") else "\033[0m"
        # color only the levelname inside the formatted string
        # we color the whole formatted line for more visible effect
        return f"{color}{base}{reset}"

# configure logger defaults (safe to call multiple times)
def _ensure_handler():
    # avoid adding duplicate stream handlers if module reloaded
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler):
            return h
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(ufstring))
    logger.addHandler(handler)
    return handler

logger.setLevel(logging.INFO)
logger.addFilter(DuplicateFilter())
_ensure_handler()

# Public API functions

def setlevel(level: int | str = logging.INFO) -> None:
    """
    Set the logger level.

    Parameters
    ----------
    level : int, optional
        Logging level to set for the logger. Valid values are:
            - logging.DEBUG: 10
            - logging.INFO: 20
            - logging.WARNING: 30
            - logging.ERROR: 40
            - logging.CRITICAL: 50
        Default is logging.INFO.
    """
    if isinstance(level, str):
        level = level.upper()
        if hasattr(logging, level):
            level = getattr(logging, level)
        else:
            raise ValueError(f"Unknown logging level: {level}")
    logger.setLevel(level)

def set_color(enabled: bool = True, palette: dict | None = None) -> None:
    """
    Enable or disable colored output and optionally set a custom palette.

    Parameters
    ----------
    enabled : bool or None, optional
        True to force colors on, False to force off,
        None to auto-detect TTY (default None).
    palette : dict or None, optional
        Optional mapping from logging level ints to color codes.
        Color codes can come from `colorama.Fore` or raw ANSI strings.
    """
    # mutate config dictionary instead of rebinding module variables
    _config["colors_enabled"] = enabled

    if palette:
        # update palette in-place (avoids rebinding the module name)
        _color_palette.clear()
        _color_palette.update({
            logging.DEBUG: Fore.CYAN,
            logging.INFO: Fore.GREEN,
            logging.WARNING: Fore.YELLOW,
            logging.ERROR: Fore.RED,
            logging.CRITICAL: Fore.MAGENTA,
        })
        # then overlay provided palette entries
        _color_palette.update(palette)

    # update formatters on stream handlers
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler):
            # replace with a ColoredFormatter respecting the current global flag
            h.setFormatter(ColoredFormatter(ufstring))
