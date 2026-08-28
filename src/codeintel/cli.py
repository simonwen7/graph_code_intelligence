"""Command-line interface for the code intelligence engine."""

from __future__ import annotations

import typer

from codeintel import __version__

app = typer.Typer(
    name="aicode",
    help="Graph-Augmented Code Intelligence Engine.",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Graph-Augmented Code Intelligence Engine."""


@app.command()
def version() -> None:
    """Display the installed project version."""
    typer.echo(f"aicode {__version__}")


def main() -> None:
    """Run the command-line application."""
    app()


if __name__ == "__main__":
    main()
