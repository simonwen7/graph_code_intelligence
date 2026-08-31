"""Order validation rules."""

from inventory import InventoryItem, reserve_units
from orders import Order
from users import is_active_customer


def validate_order_customer(order: Order) -> bool:
    """Ensure the order belongs to an active customer."""
    return is_active_customer(order.customer)


def validate_order_lines(order: Order) -> bool:
    """Reject empty orders or non-positive quantities."""
    if not order.lines:
        return False
    return all(line.quantity > 0 and bool(line.sku) for line in order.lines)


def validate_inventory_for_order(order: Order, catalog: dict[str, InventoryItem]) -> bool:
    """Confirm each line can be reserved from inventory."""
    if not validate_order_lines(order):
        return False
    for line in order.lines:
        item = catalog.get(line.sku)
        if item is None or not item.has_stock(line.quantity):
            return False
    return True


def reserve_order_inventory(order: Order, catalog: dict[str, InventoryItem]) -> bool:
    """Reserve inventory for every line when validation succeeds."""
    if not validate_inventory_for_order(order, catalog):
        return False
    for line in order.lines:
        reserve_units(catalog[line.sku], line.quantity)
    return True
