"""cupel eval results formatter.

Converts Cupel eval result JSON files into either:
- a readable Markdown report
- a filtered/self-contained JSON summary

Also supports listing available result files with ``--list``.

Input files are expected to have the shape::

    {
      "model": "...",
      "timestamp": "YYYYMMDD_HHMMSS",
      "notes": "",
      "results": [ ... ]
    }

Eval result files usually live in ``~/.cupel/eval-results/``.
"""

import glob as globmod
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


RESULT_COLUMNS = [
    "title",
    "elapsed_seconds",
    "completion_tokens",
    "thinking_tokens",
    "score",
    "judge_reason",
    "criteria_results",
]


def markdown_escape(value: Any) -> str:
    """Convert a value to Markdown-table-safe text."""
    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    return (
        value.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "<br>")
    )


def format_criteria_results(criteria_results: Any, mode: str = "compact") -> str:
    """
    Format nested criteria_results for a Markdown table cell.

    compact:
        P1: ✓; P2: ✗; N1: ✓

    full:
        P1: ✓ — quote text<br>P2: ✗ — quote text

    json:
        Raw compact JSON string.
    """
    if criteria_results is None:
        return ""

    if mode == "json":
        return json.dumps(criteria_results, ensure_ascii=False, separators=(",", ":"))

    if not isinstance(criteria_results, list):
        return str(criteria_results)

    parts: list[str] = []

    for item in criteria_results:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue

        criterion_id = item.get("id", "")
        met = item.get("met", None)

        if met is True:
            status = "✓"
        elif met is False:
            status = "✗"
        else:
            status = "?"

        if mode == "full":
            quote = item.get("quote", "")
            if quote:
                parts.append(f"{criterion_id}: {status} — {quote}")
            else:
                parts.append(f"{criterion_id}: {status}")
        else:
            parts.append(f"{criterion_id}: {status}")

    separator = "<br>" if mode == "full" else "; "
    return separator.join(parts)


def result_to_markdown_row(result: dict[str, Any], criteria_mode: str) -> list[str]:
    row = []

    for column in RESULT_COLUMNS:
        if column == "criteria_results":
            value = format_criteria_results(result.get(column), criteria_mode)
        else:
            value = result.get(column, "")

        row.append(markdown_escape(value))

    return row


def make_markdown_table(results: list[dict[str, Any]], criteria_mode: str) -> str:
    lines = []

    header = "| " + " | ".join(RESULT_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(RESULT_COLUMNS)) + " |"

    lines.append(header)
    lines.append(separator)

    for result in results:
        row = result_to_markdown_row(result, criteria_mode)
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def make_markdown_output(eval_files: list[dict[str, Any]], criteria_mode: str) -> str:
    """
    Return Markdown in this shape:

    # cupel bench results

    **Model:** ...
    **Timestamp:** ...
    **Notes:** ...

    | title | ... |
    """
    if not eval_files:
        metadata = {
            "model": None,
            "timestamp": None,
            "notes": "",
            "results": [],
        }
    else:
        metadata = eval_files[0]

    model = markdown_escape(metadata.get("model", ""))
    timestamp = markdown_escape(metadata.get("timestamp", ""))
    notes = markdown_escape(metadata.get("notes", ""))

    all_results = [
        result
        for eval_file in eval_files
        for result in eval_file["results"]
    ]

    lines = [
        "# cupel bench results",
        "",
        f"**Model:** {model}  ",
        f"**Timestamp:** {timestamp}  ",
    ]

    if notes:
        lines.append(f"**Notes:** {notes}")
    else:
        lines.append("**Notes:**")

    lines.extend(
        [
            "",
            make_markdown_table(all_results, criteria_mode),
        ]
    )

    return "\n".join(lines)


def result_to_json_object(result: dict[str, Any]) -> dict[str, Any]:
    """Return only the fields represented in the Markdown table."""
    return {
        "title": result.get("title"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "completion_tokens": result.get("completion_tokens"),
        "thinking_tokens": result.get("thinking_tokens"),
        "score": result.get("score"),
        "judge_reason": result.get("judge_reason"),
        "criteria_results": result.get("criteria_results"),
    }


def load_eval_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results")

    if not isinstance(results, list):
        raise ValueError(f"{path}: expected top-level key 'results' to be a list")

    return {
        "model": data.get("model"),
        "timestamp": data.get("timestamp"),
        "notes": data.get("notes", ""),
        "results": [r for r in results if isinstance(r, dict)],
    }


def make_json_output(eval_files: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Return JSON in this shape:

    {
      "model": "...",
      "timestamp": "...",
      "notes": "...",
      "results": [...]
    }

    If multiple input files are provided, results are combined. The shared metadata
    is taken from the first file.
    """
    if not eval_files:
        return {
            "model": None,
            "timestamp": None,
            "notes": "",
            "results": [],
        }

    first = eval_files[0]

    return {
        "model": first.get("model"),
        "timestamp": first.get("timestamp"),
        "notes": first.get("notes", ""),
        "results": [
            result_to_json_object(result)
            for eval_file in eval_files
            for result in eval_file["results"]
        ],
    }


def load_results_index(results_dir: Path) -> list[dict[str, Any]]:
    """Load all eval result files and return numbered entries sorted oldest-first."""
    entries = []

    for path in sorted(results_dir.glob("eval_*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            model = data.get("model", "")
            raw_ts = data.get("timestamp", "")
            try:
                ts = datetime.strptime(raw_ts, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                ts = raw_ts
            # Compute score/total/accuracy from results
            results = data.get("results", [])
            score = 0
            scored_count = 0
            for r in results:
                if isinstance(r, dict) and r.get("score") is not None:
                    score += r["score"]
                    scored_count += 1
            total = scored_count * 3
            if total > 0:
                accuracy = round(score / total * 100, 1)
            else:
                accuracy = None

            entries.append({
                "model": model, "timestamp": ts, "file": path.name,
                "score": score, "total": total, "accuracy": accuracy,
            })
        except (json.JSONDecodeError, OSError):
            continue

    # Sort oldest first (latest at bottom)
    entries.sort(key=lambda e: e["timestamp"])

    # Assign stable numbers starting from 1
    for i, e in enumerate(entries, 1):
        e["n"] = i

    return entries


def format_results_list(entries: list[dict[str, Any]], fmt: str) -> str:
    """Format a list of result entries as Markdown or JSON."""
    if fmt == "json":
        return json.dumps(entries, ensure_ascii=False, indent=2)

    # Markdown table
    lines = [
        "| # | model | timestamp | score | accuracy | file |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for e in entries:
        if e.get("total", 0) > 0:
            score_str = f"{e['score']}/{e['total']}"
            acc_str = f"{e['accuracy']}%"
        else:
            score_str = ""
            acc_str = ""
        lines.append(f"| {e['n']} | {markdown_escape(e['model'])} | {e['timestamp']} | {score_str} | {acc_str} | {e['file']} |")
    return "\n".join(lines)


def cmd_results(args):
    """Format eval result files as Markdown or JSON, or list available results."""
    from cupel.config import resolve_path

    pick_numbers = None
    if getattr(args, "n", None):
        pick_numbers = [int(x.strip()) for x in args.n.split(",")]

    # -n mode: select specific numbered results and format them
    if pick_numbers:
        results_dir = resolve_path("./eval-results")
        if not results_dir.is_dir():
            print(f"  No results directory found: {results_dir}", file=sys.stderr)
            sys.exit(1)
        all_entries = load_results_index(results_dir)
        by_number = {e["n"]: e for e in all_entries}
        eval_files = []
        for num in pick_numbers:
            if num in by_number:
                path = results_dir / by_number[num]["file"]
                eval_files.append(load_eval_file(path))
            else:
                print(f"  ⚠ no result #{num}", file=sys.stderr)

        if not eval_files:
            print("  ✘ No matching result files found", file=sys.stderr)
            sys.exit(1)

        if args.format == "json":
            output_data = [make_json_output([ef]) for ef in eval_files]
            output = json.dumps(output_data, ensure_ascii=False, indent=2)
        else:
            sections = [make_markdown_output([ef], args.criteria_mode) for ef in eval_files]
            output = "\n\n".join(sections)

        if args.output:
            Path(args.output).write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
        return

    # List mode: no files given, or --list, or --last
    if not (args.files):
        results_dir = resolve_path("./eval-results")
        if not results_dir.is_dir():
            print(f"  No results directory found: {results_dir}", file=sys.stderr)
            sys.exit(1)
        entries = load_results_index(results_dir)
        if getattr(args, "last", None):
            entries = entries[-args.last:]
        output = format_results_list(entries, args.format)
        if args.output:
            Path(args.output).write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
        return

    # Format mode: positional files provided
    input_files = []
    for pattern in args.files:
        expanded = globmod.glob(pattern)
        if expanded:
            input_files.extend(expanded)
        elif Path(pattern).exists():
            input_files.append(pattern)
        else:
            print(f"  ⚠ no match: {pattern}", file=sys.stderr)
    input_files = sorted(set(input_files))

    if not input_files:
        print("  ✘ No matching result files found", file=sys.stderr)
        sys.exit(1)

    eval_files = [load_eval_file(Path(f)) for f in input_files]

    if args.format == "json":
        output_data = make_json_output(eval_files)
        output = json.dumps(output_data, ensure_ascii=False, indent=2)
    else:
        output = make_markdown_output(eval_files, args.criteria_mode)

    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
