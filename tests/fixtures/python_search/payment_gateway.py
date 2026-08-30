def authorize_payment(amount: int, currency: str) -> bool:
    """Authorize a card payment through the payment gateway."""
    if amount <= 0:
        return False
    return currency == "USD"


def fraud_threshold(score: float) -> bool:
    """Return whether a transaction exceeds the fraud threshold."""
    return score >= 0.85
