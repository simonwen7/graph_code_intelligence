def export_rows_csv(rows: list[str]) -> str:
    """Serialize table rows into a comma-separated report payload."""
    return ",".join(rows)


def render_markdown_table(rows: list[str]) -> str:
    """Render rows as a simple markdown table body."""
    return "| " + " | ".join(rows) + " |"
