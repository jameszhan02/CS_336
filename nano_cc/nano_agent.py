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
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

app = typer.Typer(help="A simple persistent chat CLI example")
console = Console() # form rich console | chalk function form here

SESSION_FILE = Path.cwd() / ".my_agent" / "session.json"
WORKSPACE_ROOT = Path.cwd().resolve()

load_dotenv()

client = Anthropic()

TOOLS = [
    {
        "name": "read_file",
        "description": "Read a text file inside the current workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the workspace root.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "replace_in_file",
        "description": "Replace text in a file inside the current workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the workspace root.",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to replace.",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Whether to replace all matches. Defaults to false.",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
]


def workspace_path(path: str) -> Path:
    """Resolve a user/model path and keep it inside this repo."""
    full_path = (WORKSPACE_ROOT / path).resolve()
    if not full_path.is_relative_to(WORKSPACE_ROOT):
        raise ValueError(f"Path escapes workspace: {path}")
    return full_path


def read_file(path: str) -> str:
    full_path = workspace_path(path)
    if not full_path.exists():
        return f"File not found: {path}"
    if not full_path.is_file():
        return f"Not a file: {path}"

    text = full_path.read_text(encoding="utf-8")
    if len(text) > 20_000:
        return text[:20_000] + "\n\n[truncated after 20000 characters]"
    return text


def replace_in_file(
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
) -> str:
    full_path = workspace_path(path)
    if not full_path.exists():
        return f"File not found: {path}"
    if not full_path.is_file():
        return f"Not a file: {path}"

    text = full_path.read_text(encoding="utf-8")
    if old_text not in text:
        return f"No match found in {path}"

    count = text.count(old_text) if replace_all else 1
    updated = text.replace(old_text, new_text, -1 if replace_all else 1)
    full_path.write_text(updated, encoding="utf-8")
    return f"Replaced {count} occurrence(s) in {path}"


def run_tool(name: str, tool_input: dict[str, Any]) -> str:
    try:
        if name == "read_file":
            return read_file(**tool_input)
        if name == "replace_in_file":
            return replace_in_file(**tool_input)
        return f"Unknown tool: {name}"
    except Exception as exc:
        return f"Tool error: {type(exc).__name__}: {exc}"


def agent_reply(messages: list[dict[str, Any]]) -> str:
    """
    A minimal agentic loop:
    1. Send conversation + tool schemas to the model.
    2. If the model asks for tool_use, run the local Python function.
    3. Send tool_result back to the model.
    4. Repeat until the model returns final text.
    """
    response = client.messages.create(
        model="deepseek-v4-pro",
        max_tokens=1024,
        system=(
            "You are a concise coding assistant. You can read and modify "
            "files in the current workspace using tools. Before editing, "
            "read the relevant file first."
        ),
        messages=messages,
        tools=TOOLS,
    )

    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = run_tool(block.name, block.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model="deepseek-v4-pro",
            max_tokens=1024,
            system=(
                "You are a concise coding assistant. You can read and modify "
                "files in the current workspace using tools. Before editing, "
                "read the relevant file first."
            ),
            messages=messages,
            tools=TOOLS,
        )

    messages.append({"role": "assistant", "content": response.content})
    final_text = next(block for block in response.content if block.type == "text")
    return final_text.text


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

        reply = agent_reply(messages)
        console.print(f"[bold green]Assistant[/bold green]: [green]{reply}[/green]\n")

    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(
        json.dumps(
            messages,
            ensure_ascii=False,
            indent=2,
            default=lambda obj: obj.model_dump() if hasattr(obj, "model_dump") else str(obj),
        )
    )
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
