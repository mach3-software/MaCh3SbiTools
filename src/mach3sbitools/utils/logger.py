"""
Rich-backed logging utilities for mach3sbitools.
"""

import logging
from pathlib import Path
from typing import ClassVar

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

_THEME = Theme(
    {
        "logging.level.debug": "dim cyan",
        "logging.level.info": "bold green",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
        "repr.number": "bold cyan",
        "repr.str": "green",
    }
)

#: Shared :class:`~rich.console.Console` instance used for all log output.
console = Console(theme=_THEME)


class MaCh3Logger:
    """
    Rich-backed logger for mach3sbitools.

    Wraps :class:`logging.Logger` with a :class:`~rich.logging.RichHandler`
    for coloured console output and an optional plain-text file handler.

    Usage in application code::

        logger = MaCh3Logger("mach3sbi", log_file="run.log", level="INFO")

    Usage in submodules::

        from mach3sbitools.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Loaded [bold]50k[/] pairs")
    """

    LEVELS: ClassVar[dict] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    FILE_FORMAT = "[%(asctime)s] [%(levelname)-8s] %(name)s: %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        name: str = "mach3sbi",
        level: str = "INFO",
        log_file: Path | None = None,
        file_level: str | None = None,
        show_path: bool = False,
    ):
        """
        :param name: Logger name shown in every log line.
        :param level: Console log level (``DEBUG`` / ``INFO`` / ``WARNING`` /
            ``ERROR`` / ``CRITICAL``).
        :param log_file: Optional path to write plain-text logs alongside the
            Rich console output.
        :param file_level: Log level for the file handler. Defaults to ``DEBUG``.
        :param show_path: Show source file and line number in console output.
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        # Handlers accumulate otherwise: constructing this twice (a notebook,
        # a test session, the CLI) would duplicate every log line.
        for handler in list(self._logger.handlers):
            self._logger.removeHandler(handler)
            handler.close()

        rich_handler = RichHandler(
            console=console,
            level=self.LEVELS.get(level.upper(), logging.INFO),
            show_time=True,
            show_level=True,
            show_path=show_path,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
            log_time_format=self.DATEFMT,
        )
        self._logger.addHandler(rich_handler)

        if log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setLevel(self.LEVELS.get((file_level or "DEBUG").upper(), logging.DEBUG))
            fh.setFormatter(logging.Formatter(self.FILE_FORMAT, datefmt=self.DATEFMT))
            self._logger.addHandler(fh)
            self._logger.info(f"Logging to file: [cyan]{log_file}[/]")

    def debug(self, msg: str, *args, **kwargs) -> None:
        """
        Log *msg* at DEBUG level.

        :param msg: Message, optionally with ``%``-style placeholders.
        :param args: Values substituted into *msg*.
        :param kwargs: Forwarded to the stdlib logger.
        """
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """
        Log *msg* at INFO level.

        :param msg: Message, optionally with ``%``-style placeholders.
        :param args: Values substituted into *msg*.
        :param kwargs: Forwarded to the stdlib logger.
        """
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """
        Log *msg* at WARNING level.

        :param msg: Message, optionally with ``%``-style placeholders.
        :param args: Values substituted into *msg*.
        :param kwargs: Forwarded to the stdlib logger.
        """
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """
        Log *msg* at ERROR level.

        :param msg: Message, optionally with ``%``-style placeholders.
        :param args: Values substituted into *msg*.
        :param kwargs: Forwarded to the stdlib logger.
        """
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """
        Log *msg* at CRITICAL level.

        :param msg: Message, optionally with ``%``-style placeholders.
        :param args: Values substituted into *msg*.
        :param kwargs: Forwarded to the stdlib logger.
        """
        self._logger.critical(msg, *args, **kwargs)

    def set_level(self, level: str) -> None:
        """
        Adjust the console handler log level at runtime.

        :param level: New log level string.
        """
        for handler in self._logger.handlers:
            if isinstance(handler, RichHandler):
                handler.setLevel(self.LEVELS.get(level.upper(), logging.INFO))

    @property
    def logger(self) -> logging.Logger:
        """
        :returns: The underlying :class:`logging.Logger`, for stdlib
            compatibility.
        """
        return self._logger


def get_logger(name: str = "mach3sbi") -> logging.Logger:
    """
    Return a named child logger for use in submodules.

    :param name: Logger name, typically ``__name__``.
    :returns: A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
