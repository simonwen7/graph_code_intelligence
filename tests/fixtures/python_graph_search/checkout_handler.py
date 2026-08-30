"""Checkout handler that strongly matches payment-authorization vocabulary."""

from cart_rules import verify_basket_line_items
from charge_policy import DiscountedChargePolicy


def handle_payment_checkout(order_id: str, amount: int) -> bool:
    """Authorize payment checkout for an order and apply charge policy."""
    if amount <= 0:
        return False
    if not verify_basket_line_items(order_id):
        return False
    policy = DiscountedChargePolicy()
    return policy.approve_charge(amount)


def summarize_payment_checkout(order_id: str) -> str:
    """Build a short authorize-payment checkout summary line."""
    return f"checkout:{order_id}"
