"""Checkout orchestration across pricing, inventory, and payments."""

from discounts import DiscountPolicy, discounted_line_total
from inventory import InventoryItem
from notifications import notify_order_confirmed
from orders import Order
from policies import ChargePolicy
from pricing import apply_tax_cents, unit_price_cents
from store import save_order
from validation import reserve_order_inventory, validate_order_customer


def compute_order_subtotal(
    order: Order,
    price_table: dict[str, int],
    policy: DiscountPolicy,
) -> int:
    """Sum discounted line totals for an order."""
    total = 0
    for line in order.lines:
        unit = unit_price_cents(line.sku, price_table[line.sku])
        total += discounted_line_total(unit, line.quantity, policy)
    return total


def finalize_checkout(
    order: Order,
    catalog: dict[str, InventoryItem],
    price_table: dict[str, int],
    discount: DiscountPolicy,
    charge_policy: ChargePolicy,
    card_token: str,
    tax_bps: int,
) -> bool:
    """Validate, reserve inventory, charge, persist, and notify."""
    if not validate_order_customer(order):
        return False
    if not reserve_order_inventory(order, catalog):
        return False
    subtotal = compute_order_subtotal(order, price_table, discount)
    due = apply_tax_cents(subtotal, tax_bps)
    if not charge_policy.approve(due, card_token):
        return False
    save_order(order)
    notify_order_confirmed(order)
    return True
