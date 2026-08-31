"""In-memory order persistence."""

from orders import Order

_ORDERS: dict[str, Order] = {}


def save_order(order: Order) -> None:
    """Persist an order in the in-memory store."""
    _ORDERS[order.order_id] = order


def load_order(order_id: str) -> Order | None:
    """Load a previously saved order."""
    return _ORDERS.get(order_id)


def delete_order(order_id: str) -> bool:
    """Delete a saved order if present."""
    return _ORDERS.pop(order_id, None) is not None


def list_order_ids() -> list[str]:
    """Return saved order identifiers."""
    return sorted(_ORDERS)
