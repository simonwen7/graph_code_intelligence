"""Customer identity helpers used across checkout."""


class Customer:
    """A storefront customer account."""

    def __init__(self, customer_id: str, email: str) -> None:
        self.customer_id = customer_id
        self.email = email

    def display_name(self) -> str:
        """Return a short customer label for receipts."""
        return f"customer:{self.customer_id}"


def normalize_email(raw: str) -> str:
    """Normalize an email address for lookups."""
    return raw.strip().lower()


def is_active_customer(customer: Customer) -> bool:
    """Return True when a customer account may place orders."""
    return bool(customer.customer_id) and "@" in customer.email
