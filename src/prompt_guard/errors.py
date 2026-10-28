"""Stable API errors. Messages never include submitted text."""


class ScanError(Exception):
    """A request failed in a way the API can explain without a score."""

    def __init__(self, code: str, message: str, status_code: int, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


class FatalConfigError(Exception):
    """Startup configuration cannot succeed by retrying the download."""


def unauthorized() -> ScanError:
    return ScanError(
        "unauthorized",
        "Missing or invalid API credentials.",
        401,
        False,
    )


def invalid_request(message: str) -> ScanError:
    return ScanError("invalid_request", message, 422, False)


def request_too_large() -> ScanError:
    return ScanError(
        "request_too_large",
        "The request exceeds the configured size limit.",
        413,
        False,
    )


def input_too_long() -> ScanError:
    return ScanError(
        "input_too_long",
        "The input exceeds the configured token limit.",
        413,
        False,
    )


def too_many_windows() -> ScanError:
    return ScanError(
        "too_many_windows",
        "The input requires more windows than the configured limit.",
        413,
        False,
    )


def queue_full() -> ScanError:
    return ScanError("queue_full", "The inference queue is full.", 429, True)


def not_ready() -> ScanError:
    return ScanError("not_ready", "The model is not ready.", 503, True)


def scan_timeout() -> ScanError:
    return ScanError(
        "scan_timeout",
        "The scan did not complete within the configured deadline.",
        504,
        True,
    )


def internal_error() -> ScanError:
    return ScanError("internal_error", "The scan failed.", 500, False)


def error_payload(request_id: str, error: ScanError) -> dict[str, object]:
    return {
        "request_id": request_id,
        "status": "error",
        "error": {
            "code": error.code,
            "message": error.message,
            "retryable": error.retryable,
        },
    }
