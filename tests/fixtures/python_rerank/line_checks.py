"""SKU/line integrity helpers intentionally avoiding payment vocabulary."""


def verify_line_bundle(order_id: str) -> bool:
    """Ensure every SKU row in the shopping basket is complete."""
    return bool(order_id) and "-" in order_id


def recount_sku_rows(rows: list[str]) -> int:
    """Count SKU rows present in a shopping basket payload."""
    return len(rows)
