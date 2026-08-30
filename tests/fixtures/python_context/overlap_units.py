"""Parent class / child method overlap for context packing."""


class BasketService:
    """Shopping basket service with nested helpers."""

    def verify_line(self, order_id: str) -> bool:
        """Ensure a shopping basket line identifier is present."""

        def helper() -> str:
            """Build a local basket marker."""
            return f"line:{order_id}"

        return bool(helper())
