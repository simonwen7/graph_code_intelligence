"""Payment-adjacent vocabulary seed for offline CLI context tests."""

from small_units import tiny_alpha


def authorize_payment_checkout(order_id: str) -> str:
    """Authorize payment checkout for an order identifier."""
    return f"checkout:{order_id}:{tiny_alpha()}"
