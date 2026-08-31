"""Order models and helpers."""

from users import Customer


class OrderLine:
    """One SKU quantity on an order."""

    def __init__(self, sku: str, quantity: int) -> None:
        self.sku = sku
        self.quantity = quantity


class Order:
    """A customer order with line items."""

    def __init__(self, order_id: str, customer: Customer) -> None:
        self.order_id = order_id
        self.customer = customer
        self.lines: list[OrderLine] = []

    def add_line(self, sku: str, quantity: int) -> None:
        """Append a line item to the order."""
        self.lines.append(OrderLine(sku, quantity))

    def total_quantity(self) -> int:
        """Sum quantities across all lines."""
        return sum(line.quantity for line in self.lines)


def empty_order(order_id: str, customer: Customer) -> Order:
    """Create an order with no lines yet."""
    return Order(order_id, customer)
