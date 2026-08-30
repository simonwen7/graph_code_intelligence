"""Payment transfer entrypoint with strong authorization vocabulary."""

from line_checks import verify_line_bundle
from policy_tree import SpecialChargePolicy


def authorize_payment_transfer(order_id: str, amount: int) -> bool:
    """Authorize payment transfer after validating the order payload."""
    if amount <= 0:
        return False
    if not verify_line_bundle(order_id):
        return False
    return SpecialChargePolicy().approve_transfer(amount)


def describe_payment_transfer(order_id: str) -> str:
    """Describe an authorize-payment transfer summary for operators."""
    return f"transfer:{order_id}"
