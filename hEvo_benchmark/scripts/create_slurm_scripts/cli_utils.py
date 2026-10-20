from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover
    Console = None
    Panel = None
    Table = None


console = Console() if Console is not None else None


def _stringify(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def print_banner(title: str, body: str) -> None:
    if console and Panel is not None:
        console.print(Panel.fit(body, title=title, border_style="cyan"))
        return
    print(f"[{title}] {body}")


def print_settings(
    title: str,
    settings: Mapping[str, Any],
    value_styles: Mapping[str, str] | None = None,
) -> None:
    value_styles = value_styles or {}
    if console and Table is not None:
        table = Table(title=title, show_lines=False)
        table.add_column("Key", style="bold cyan")
        table.add_column("Value", style="white")
        for key, value in settings.items():
            rendered_value = _stringify(value)
            style = value_styles.get(key)
            if style:
                rendered_value = f"[{style}]{rendered_value}[/{style}]"
            table.add_row(key, rendered_value)
        console.print(table)
        return

    print(title)
    for key, value in settings.items():
        print(f"  - {key}: {_stringify(value)}")


def print_directory(title: str, path: Path) -> None:
    if console and Table is not None:
        table = Table(title=title, show_header=True)
        table.add_column("Item", style="bold cyan")
        table.add_column("Path", style="white")
        table.add_row("Root", str(path))
        console.print(table)
        return

    print(title)
    print(f"  Root: {path}")


def print_task_model_summary(
    title: str,
    task_model_counts: Mapping[str, int],
) -> None:
    if console and Table is not None:
        table = Table(title=title, show_header=True)
        table.add_column("subtasks", style="bold cyan")
        table.add_column("models nums", style="white", justify="right")
        for task_name, count in task_model_counts.items():
            table.add_row(task_name, str(count))
        console.print(table)
        return

    print(title)
    for task_name, count in task_model_counts.items():
        print(f"  {task_name}: {count}")
