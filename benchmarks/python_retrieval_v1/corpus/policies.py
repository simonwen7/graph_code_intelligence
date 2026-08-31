"""Charge approval policies."""

from payments import authorize_card


class ChargePolicy:
    """Base charge approval policy."""

    def approve(self, amount_cents: int, card_token: str) -> bool:
        """Approve a charge request."""
        return authorize_card(amount_cents, card_token)


class StrictChargePolicy(ChargePolicy):
    """Reject small or oversized charges."""

    def __init__(self, minimum_cents: int, maximum_cents: int) -> None:
        self.minimum_cents = minimum_cents
        self.maximum_cents = maximum_cents

    def approve(self, amount_cents: int, card_token: str) -> bool:
        if amount_cents < self.minimum_cents or amount_cents > self.maximum_cents:
            return False
        return ChargePolicy.approve(self, amount_cents, card_token)


class PrepaidChargePolicy(StrictChargePolicy):
    """Strict policy that also requires prepaid tokens."""

    def approve(self, amount_cents: int, card_token: str) -> bool:
        if not card_token.startswith("tok_prepaid_"):
            return False
        return StrictChargePolicy.approve(self, amount_cents, card_token)
