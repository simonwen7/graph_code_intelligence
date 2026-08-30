"""Reference and call edges with mixed resolution honesty."""

from cart_rules import recount_basket_sku_rows


def annotate_order_metadata(order_id: str) -> str:
    """Attach metadata using a referenced basket counter helper."""
    count = recount_basket_sku_rows([order_id])
    return f"meta:{count}"


class MetadataWorker:
    """Worker that issues probable and unresolved calls."""

    def refresh(self) -> None:
        """Invoke dynamic and unknown callees for resolution coverage."""
        self.refresh()
        unknown_external_hook()
