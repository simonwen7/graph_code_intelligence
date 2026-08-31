"""Warehouse inventory checks."""


class InventoryItem:
    """A stocked SKU with available quantity."""

    def __init__(self, sku: str, quantity: int) -> None:
        self.sku = sku
        self.quantity = quantity

    def has_stock(self, requested: int) -> bool:
        """Return True when enough units are available."""
        return requested > 0 and self.quantity >= requested


def reserve_units(item: InventoryItem, requested: int) -> bool:
    """Attempt to reserve inventory for an order line."""
    if not item.has_stock(requested):
        return False
    item.quantity -= requested
    return True


def restock_units(item: InventoryItem, incoming: int) -> int:
    """Increase on-hand quantity and return the new total."""
    if incoming > 0:
        item.quantity += incoming
    return item.quantity
