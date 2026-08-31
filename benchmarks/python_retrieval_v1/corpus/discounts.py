"""Discount policies with simple inheritance."""

from pricing import line_total_cents


class DiscountPolicy:
    """Base discount policy."""

    def discount_cents(self, subtotal_cents: int) -> int:
        """Return cents to subtract from a subtotal."""
        return 0


class PercentageDiscount(DiscountPolicy):
    """Subtract a fixed percentage of the subtotal."""

    def __init__(self, percent: int) -> None:
        self.percent = percent

    def discount_cents(self, subtotal_cents: int) -> int:
        if self.percent <= 0:
            return 0
        return (subtotal_cents * self.percent) // 100


class LoyaltyDiscount(PercentageDiscount):
    """Increase the percentage discount for loyalty members."""

    def __init__(self, percent: int, bonus_percent: int) -> None:
        super().__init__(percent)
        self.bonus_percent = bonus_percent

    def discount_cents(self, subtotal_cents: int) -> int:
        boosted = PercentageDiscount.discount_cents(self, subtotal_cents)
        return boosted + (subtotal_cents * self.bonus_percent) // 100


def discounted_line_total(unit_cents: int, quantity: int, policy: DiscountPolicy) -> int:
    """Compute a line total after applying a discount policy."""
    raw = line_total_cents(unit_cents, quantity)
    return max(0, raw - policy.discount_cents(raw))
