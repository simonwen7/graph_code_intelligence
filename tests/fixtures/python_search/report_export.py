def csv_export(rows: list[str]) -> str:
    """Serialize rows into a csv_export payload string."""
    return ",".join(rows)


def report_summary(title: str) -> str:
    """Build a short report_summary title line."""
    return f"summary:{title}"
