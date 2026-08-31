"""Customer notification helpers."""

from orders import Order
from users import Customer, normalize_email


def build_confirmation_subject(order: Order) -> str:
    """Build an email subject for order confirmation."""
    return f"Order {order.order_id} confirmed"


def build_confirmation_body(order: Order) -> str:
    """Build a short confirmation body."""
    return f"Thanks {order.customer.display_name()} for order {order.order_id}"


def notify_order_confirmed(order: Order) -> str:
    """Compose and return a confirmation message (no network I/O)."""
    recipient = normalize_email(order.customer.email)
    subject = build_confirmation_subject(order)
    body = build_confirmation_body(order)
    return f"to={recipient};subject={subject};body={body}"


def notify_customer_welcome(customer: Customer) -> str:
    """Compose a welcome message for a new customer."""
    return f"welcome:{normalize_email(customer.email)}"
