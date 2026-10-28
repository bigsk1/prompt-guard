"""Bearer authentication for /v1 routes.

Comparison hashes both values first so a length mismatch does not exit early.
"""

import hashlib
import hmac


def credentials_match(authorization: str | None, api_key: str) -> bool:
    if not api_key:
        return False
    presented = _bearer_token(authorization)
    return hmac.compare_digest(_digest(presented), _digest(api_key))


def _bearer_token(authorization: str | None) -> str:
    if authorization is None:
        return ""
    scheme, separator, token = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or token == "" or " " in token:
        return ""
    return token


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()
