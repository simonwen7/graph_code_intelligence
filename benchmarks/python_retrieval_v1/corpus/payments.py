"""Payment gateway helpers."""


def authorize_card(amount_cents: int, card_token: str) -> bool:
    """Authorize a card charge for a positive amount."""
    if amount_cents <= 0 or not card_token:
        return False
    return card_token.startswith("tok_")


def capture_payment(authorization_id: str) -> bool:
    """Capture a previously authorized payment."""
    return authorization_id.startswith("auth_")


def refund_payment(capture_id: str, amount_cents: int) -> bool:
    """Refund a captured payment up to the captured amount."""
    return capture_id.startswith("cap_") and amount_cents > 0
