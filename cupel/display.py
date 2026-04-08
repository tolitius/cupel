"""cupel.display — terminal display (rich scoreboard)."""

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


SCORE_COLORS = {0: "red", 1: "yellow", 2: "green", 3: "bold green"}


def score_symbol(status, phase="eval"):
    if status is None:
        return "[dim]·[/]"
    if status == "running":
        return "[yellow]⏳[/]"
    if status == "judging":
        return "[cyan]⚖[/]"
    if status == "skip":
        return "[dim]skip[/]"
    if status == "error":
        return "[red]ERR[/]"
    if isinstance(status, tuple):
        score, elapsed = status
        color = SCORE_COLORS.get(score, "white")
        return f"[{color}]{score}[/] [dim]({elapsed}s)[/]"
    return f"[green]{status}[/]"


def shorten_model(m):
    s = m.replace("Qwen3.5-", "Q").replace("Nemotron-Cascade-2-", "Nemo·")
    s = s.replace("-4bit", "·4b").replace("-8bit", "·8b").replace("-bf16", "·bf16")
    return s[:14]


def build_table(prompts, models, state, title_prefix="⚡", api_url=None, phase="eval"):
    title = f"{title_prefix} Local LLM Eval"
    if api_url:
        host = api_url.split("//")[-1].split("/")[0]
        title += f"  [dim]@ {host}[/]"

    table = Table(title=title, show_lines=False, pad_edge=True, expand=False)
    table.add_column("#", width=3, justify="right", style="dim")
    table.add_column("Prompt", min_width=24, max_width=38, no_wrap=True)

    w = 14 if phase == "judge" else 10
    for m in models:
        table.add_column(shorten_model(m), width=w, justify="center")

    for p in prompts:
        pid = p["id"]
        row = [str(pid), p["title"][:37]]
        for m in models:
            row.append(score_symbol(state.get((m, pid)), phase))
        table.add_row(*row)

    table.add_section()
    if phase == "judge":
        summary_row = ["", "[bold]Score[/]"]
        for m in models:
            total = sum(v[0] for p in prompts if isinstance((v := state.get((m, p["id"]))), tuple))
            count = sum(1 for p in prompts if isinstance(state.get((m, p["id"])), tuple))
            mx = len(prompts) * 3
            summary_row.append(f"[bold]{total}[/]/{mx}" if count > 0 else "[dim]·[/]")
        table.add_row(*summary_row)
    else:
        summary_row = ["", "[bold]Done[/]"]
        for m in models:
            done = sum(1 for p in prompts if state.get((m, p["id"])) not in (None, "running"))
            summary_row.append(f"[bold]{done}[/]/{len(prompts)}")
        table.add_row(*summary_row)

    return table
