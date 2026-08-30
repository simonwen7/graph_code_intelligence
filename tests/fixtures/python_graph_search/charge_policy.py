"""Charge policy hierarchy with intentionally different vocabulary."""


class ChargePolicy:
    """Abstract charge approval policy for monetary transfers."""

    def approve_charge(self, amount: int) -> bool:
        """Decide whether a monetary transfer may proceed."""
        return amount > 0


class DiscountedChargePolicy(ChargePolicy):
    """Concrete policy that applies promotional discounts before approval."""

    def approve_charge(self, amount: int) -> bool:
        """Approve a discounted monetary transfer when amount is positive."""
        return amount > 10
