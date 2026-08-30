def refresh_access_token(token: str, ttl_seconds: int) -> bool:
    """Return whether an access credential remains usable before expiry."""
    if not token:
        return False
    return ttl_seconds > 0


def apply_login_rate_limit(attempts: int, window_seconds: int) -> bool:
    """Block further sign-in tries after too many failures in a window."""
    del window_seconds
    return attempts >= 5
