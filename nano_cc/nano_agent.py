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
import os
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

EXTRA_ROOTS = [
    Path(path).expanduser().resolve()
    for path in os.environ.get("AGENT_EXTRA_ROOTS", "").split(":")
    if path
]
ALLOWED_ROOTS = [WORKSPACE_ROOT, *EXTRA_ROOTS]

client = Anthropic()

SYSTEM_PROMPT = (
    "You are a concise coding assistant. You can read and modify files only inside "
    f"these allowed roots: {', '.join(str(root) for root in ALLOWED_ROOTS)}. "
    "Prefer relative paths from the current workspace. Absolute paths are allowed "
    "only when they are inside an allowed root. Never invent paths. If you do not "
    "know the correct path, call list_files or search_files first. For large files, "
    "read focused line ranges with read_file. Before editing, read the relevant file "
    "first."
)

SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache"}

TOOLS = [
    {
        "name": "list_files",
        "description": (
            "List files and directories inside the current workspace. Use this "
            "before read_file or replace_in_file when you do not know the exact "
            "relative path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Directory path. Use a relative path from the current workspace, "
                        "or an absolute path inside an allowed root."
                    ),
                }
            },
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a text file inside the current workspace. Supports line ranges "
            "so you can inspect large files in chunks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "File path. Use a relative path from the current workspace, "
                        "or an absolute path inside an allowed root."
                    ),
                },
                "start_line": {
                    "type": "integer",
                    "description": "1-based line number to start reading from. Defaults to 1.",
                    "minimum": 1,
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to return. Defaults to 200.",
                    "minimum": 1,
                    "maximum": 1000,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_files",
        "description": (
            "Search text files for a literal query inside the current workspace. "
            "Use this to find relevant files, functions, classes, or call sites."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Literal text to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search. Defaults to the workspace root.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum matching lines to return. Defaults to 50.",
                    "minimum": 1,
                    "maximum": 200,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "replace_in_file",
        "description": "Replace text in a file inside the current workspace. Use relative paths only.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "File path. Use a relative path from the current workspace, "
                        "or an absolute path inside an allowed root."
                    ),
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
    """Resolve a user/model path and keep it inside configured roots."""
    raw_path = Path(path).expanduser()
    full_path = raw_path.resolve() if raw_path.is_absolute() else (WORKSPACE_ROOT / raw_path).resolve()

    if not any(full_path.is_relative_to(root) for root in ALLOWED_ROOTS):
        allowed = ", ".join(str(root) for root in ALLOWED_ROOTS)
        raise ValueError(f"Path is outside allowed roots. Allowed roots: {allowed}")
    return full_path


def list_files(path: str = ".") -> str:
    full_path = workspace_path(path)
    if not full_path.exists():
        return f"Directory not found: {path}"
    if not full_path.is_dir():
        return f"Not a directory: {path}"

    entries = []
    for child in sorted(full_path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        if child.name in SKIP_DIRS:
            continue
        try:
            relative = child.relative_to(WORKSPACE_ROOT)
        except ValueError:
            relative = child
        suffix = "/" if child.is_dir() else ""
        entries.append(f"{relative}{suffix}")

    if not entries:
        return f"No files found in {path}"

    return "\n".join(entries[:200])


def read_file(path: str, start_line: int = 1, max_lines: int = 200) -> str:
    full_path = workspace_path(path)
    if not full_path.exists():
        return f"File not found: {path}"
    if not full_path.is_file():
        return f"Not a file: {path}"
    if start_line < 1:
        return "start_line must be >= 1"
    if max_lines < 1 or max_lines > 1000:
        return "max_lines must be between 1 and 1000"

    lines = full_path.read_text(encoding="utf-8").splitlines()
    if start_line > len(lines):
        return f"{path} has {len(lines)} lines; start_line {start_line} is past the end"

    start_index = start_line - 1
    end_index = min(start_index + max_lines, len(lines))
    numbered = [
        f"{line_no:>5} | {line}"
        for line_no, line in enumerate(lines[start_index:end_index], start=start_line)
    ]

    header = f"{path}: lines {start_line}-{end_index} of {len(lines)}"
    if end_index < len(lines):
        header += f" (next start_line: {end_index + 1})"
    return header + "\n" + "\n".join(numbered)


def search_files(query: str, path: str = ".", max_results: int = 50) -> str:
    if not query:
        return "query must not be empty"
    if max_results < 1 or max_results > 200:
        return "max_results must be between 1 and 200"

    full_path = workspace_path(path)
    if not full_path.exists():
        return f"Path not found: {path}"

    candidates = [full_path] if full_path.is_file() else full_path.rglob("*")
    results = []

    for candidate in candidates:
        if len(results) >= max_results:
            break
        if any(part in SKIP_DIRS for part in candidate.parts):
            continue
        if not candidate.is_file():
            continue

        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue

        for line_no, line in enumerate(lines, start=1):
            if query in line:
                try:
                    display_path = candidate.relative_to(WORKSPACE_ROOT)
                except ValueError:
                    display_path = candidate
                results.append(f"{display_path}:{line_no}: {line.strip()}")
                if len(results) >= max_results:
                    break

    if not results:
        return f"No matches found for {query!r} in {path}"
    return "\n".join(results)


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
        if name == "list_files":
            return list_files(**tool_input)
        if name == "read_file":
            return read_file(**tool_input)
        if name == "search_files":
            return search_files(**tool_input)
        if name == "replace_in_file":
            return replace_in_file(**tool_input)
        return f"Unknown tool: {name}"
    except Exception as exc:
        return f"Tool error: {type(exc).__name__}: {exc}"


def block_value(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def content_block_to_dict(block: Any) -> dict[str, Any]:
    """Convert Anthropic SDK content blocks into plain message-history dicts."""
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    raise TypeError(f"Unsupported content block type: {type(block).__name__}")


def response_content_to_dicts(content: Any) -> list[dict[str, Any]]:
    return [content_block_to_dict(block) for block in content]


def tool_use_ids(message: dict[str, Any]) -> list[str]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [
        block_value(block, "id")
        for block in content
        if block_value(block, "type") == "tool_use" and block_value(block, "id")
    ]


def tool_result_ids(message: dict[str, Any]) -> set[str]:
    content = message.get("content")
    if not isinstance(content, list):
        return set()
    return {
        block_value(block, "tool_use_id")
        for block in content
        if block_value(block, "type") == "tool_result" and block_value(block, "tool_use_id")
    }


def trim_incomplete_tool_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop history from the first assistant tool_use that lacks immediate results."""
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        expected_ids = tool_use_ids(message)
        if not expected_ids:
            continue

        next_message = messages[index + 1] if index + 1 < len(messages) else None
        actual_ids = tool_result_ids(next_message) if next_message else set()
        if not set(expected_ids).issubset(actual_ids):
            return messages[:index]

    return messages


def trim_incomplete_tool_history_in_place(messages: list[dict[str, Any]]) -> int:
    trimmed = trim_incomplete_tool_history(messages)
    removed_count = len(messages) - len(trimmed)
    if removed_count:
        messages[:] = trimmed
    return removed_count


def agent_reply(messages: list[dict[str, Any]]) -> str:
    """
    A minimal agentic loop:
    1. Send conversation + tool schemas to the model.
    2. If the model asks for tool_use, run the local Python function.
    3. Send tool_result back to the model.
    4. Repeat until the model returns final text.
    """
    removed_count = trim_incomplete_tool_history_in_place(messages)
    if removed_count:
        console.print(f"[dim]Trimmed incomplete tool history before request (-{removed_count})[/dim]")

    response = client.messages.create(
        model="DeepSeek-V4-Flash",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=TOOLS,
    )

    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            block_type = block_value(block, "type")
            if block_type == "text":
                console.print(f"[dim]Assistant step: {block_value(block, 'text')}[/dim]")
                continue

            if block_type != "tool_use":
                continue

            # console.print(
            #     f"[dim]Tool call: {block.name}({json.dumps(block.input, ensure_ascii=False)})[/dim]"
            # )
            result = run_tool(block_value(block, "name"), block_value(block, "input"))
            console.print(f"[dim]Tool result: {result}[/dim]")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block_value(block, "id"),
                    "content": result,
                }
            )

        messages.append({"role": "assistant", "content": response_content_to_dicts(response.content)})
        messages.append({"role": "user", "content": tool_results})

        removed_count = trim_incomplete_tool_history_in_place(messages)
        if removed_count:
            console.print(f"[dim]Trimmed incomplete tool history before request (-{removed_count})[/dim]")

        response = client.messages.create(
            model="DeepSeek-V4-Flash",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=TOOLS,
        )

    messages.append({"role": "assistant", "content": response_content_to_dicts(response.content)})
    text_blocks = [
        block_value(block, "text")
        for block in response.content
        if block_value(block, "type") == "text" and block_value(block, "text")
    ]
    if text_blocks:
        return "\n".join(text_blocks)

    content_types = ", ".join(str(block_value(block, "type")) for block in response.content) or "empty"
    return (
        "Model returned no text response. "
        f"stop_reason={response.stop_reason}, content_types={content_types}"
    )


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
        original_count = len(messages)
        trim_incomplete_tool_history_in_place(messages)
        if len(messages) != original_count:
            SESSION_FILE.write_text(json.dumps(messages, ensure_ascii=False, indent=2))
            console.print(
                f"[dim]Trimmed incomplete tool history "
                f"({original_count} -> {len(messages)} messages)[/dim]"
            )
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

        removed_count = trim_incomplete_tool_history_in_place(messages)
        if removed_count:
            console.print(f"[dim]Trimmed incomplete tool history before new input (-{removed_count})[/dim]")

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
