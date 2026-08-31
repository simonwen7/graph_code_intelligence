"""Catalog pricing primitives."""


def unit_price_cents(sku: str, base_cents: int) -> int:
    """Return the list unit price in cents for a SKU."""
    if base_cents < 0:
        raise ValueError("base_cents must be non-negative")
    return base_cents


def line_total_cents(unit_cents: int, quantity: int) -> int:
    """Multiply unit price by quantity."""
    if quantity < 0:
        raise ValueError("quantity must be non-negative")
    return unit_cents * quantity


def apply_tax_cents(subtotal_cents: int, tax_bps: int) -> int:
    """Add tax to a subtotal using basis points."""
    if tax_bps < 0:
        raise ValueError("tax_bps must be non-negative")
    return subtotal_cents + (subtotal_cents * tax_bps) // 10_000
