def token_expiration(ttl_seconds: int) -> bool:
    """Check whether an auth token_expiration window has elapsed."""
    return ttl_seconds <= 0


def validate_token_signature(token: str) -> bool:
    """Validate a bearer token signature fragment."""
    return token.startswith("sig:")
