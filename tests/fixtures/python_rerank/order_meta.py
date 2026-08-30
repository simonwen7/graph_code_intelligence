"""Unrelated metadata helpers with no payment-authorization vocabulary."""


def format_order_label(order_id: str) -> str:
    """Build a short display label for an order identifier."""
    return f"order:{order_id}"
