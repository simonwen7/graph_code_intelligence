"""Charge policy hierarchy with distinct vocabulary from payment transfer."""


class ChargePolicy:
    """Abstract policy for monetary transfer approval."""

    def approve_transfer(self, amount: int) -> bool:
        """Decide whether a monetary transfer may proceed."""
        return amount > 0


class SpecialChargePolicy(ChargePolicy):
    """Concrete policy applying promotional adjustments before approval."""

    def approve_transfer(self, amount: int) -> bool:
        """Approve a discounted monetary transfer when amount is large enough."""
        return amount > 10
