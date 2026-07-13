"""
Typer usage example -- a persistent chat-style CLI (simulated agent reply, no real API call)
Uses rich for styling: startup title, bordered input prompt, colored user/AI messages

Usage:
    python typer_demo.py chat
    python typer_demo.py chat --resume
    python typer_demo.py clear
    python typer_demo.py --help
"""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

app = typer.Typer(help="A simple persistent chat CLI example")
console = Console()

SESSION_FILE = Path.cwd() / ".my_agent" / "session.json"


def fake_agent_reply(user_input: str) -> str:
    """
    Simulates a single agent reply. In a real project, replace this with
    your actual agentic loop:
        response = client.messages.create(...)
    then handle the tool_use loop and extract the final text block.
    """
    if "file" in user_input.lower() or "directory" in user_input.lower():
        return "(simulated) current directory contains main.py, README.md, tools/"
    return f"(simulated) received your message: {user_input}"


def print_title():
    console.print(
        Panel.fit(
            "[bold cyan]MyAgent[/bold cyan] - a simple terminal AI assistant\n"
            "[dim]Type 'exit' or press Ctrl+C to quit[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )
    )


@app.command()
def chat(
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume the last session"),
):
    """Start a persistent chat session. Type 'exit' to quit."""
    messages = []
    console.clear()
    print_title()

    if resume and SESSION_FILE.exists():
        messages = json.loads(SESSION_FILE.read_text())
        console.print(f"[dim]Resumed last session ({len(messages)} messages)[/dim]\n")

    while True:
        try:
            # Bordered input effect: wrap the prompt with a divider line above and below
            console.rule(style="dim")
            user_input = Prompt.ask("[bold yellow]You[/bold yellow]")
            console.rule(style="dim")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if user_input.strip().lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})

        # === In a real project, replace this with your agentic loop ===
        reply = fake_agent_reply(user_input)
        console.print(f"[bold green]Claude[/bold green]: [green]{reply}[/green]\n")

        messages.append({"role": "assistant", "content": reply})

    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(json.dumps(messages, ensure_ascii=False, indent=2))
    console.print(f"[dim]Session saved ({len(messages)} messages) to {SESSION_FILE}[/dim]")


@app.command()
def clear():
    """Clear the saved session file."""
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()
        typer.echo("Session cleared")
    else:
        typer.echo("No session file found")


if __name__ == "__main__":
    app()