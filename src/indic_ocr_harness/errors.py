"""Exceptions raised by the harness."""


class HarnessError(Exception):
    """Base class for harness errors."""


class ModelUnavailableError(HarnessError):
    """The Bodhan snapshot cannot be found or downloaded, or it does not have the expected layout."""


class IncompatibleEnvironmentError(HarnessError):
    """An installed dependency would make the recognizer produce wrong output (e.g. a tokenizer swap)."""


class CacheMissError(HarnessError):
    """A read was not in the cache and the recognizer is disabled (``allow_compute=False``)."""
