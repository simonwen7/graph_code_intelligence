"""Filler payment-authorization vocabulary for Hybrid/Graph baseline padding."""


def authorize_payment_filler_one(amount: int) -> bool:
    """Authorize payment filler path one for retrieval padding."""
    return amount > 0


def authorize_payment_filler_two(amount: int) -> bool:
    """Authorize payment filler path two for retrieval padding."""
    return amount > 1


def authorize_payment_filler_three(amount: int) -> bool:
    """Authorize payment filler path three for retrieval padding."""
    return amount > 2


def authorize_payment_filler_four(amount: int) -> bool:
    """Authorize payment filler path four for retrieval padding."""
    return amount > 3


def authorize_payment_filler_five(amount: int) -> bool:
    """Authorize payment filler path five for retrieval padding."""
    return amount > 4
