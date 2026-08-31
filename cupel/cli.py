"""cupel.cli — command-line interface (run, judge, ui, init)."""

import glob
import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

from cupel import __version__


def cmd_run(args):
    """Collect responses from models. No judging."""
    from cupel.config import (
        load_config, load_dotenv, get_api_config, parse_prompt_ids,
        resolve_path,
    )
    from cupel.eval import find_image, run_prompt
    from cupel.schema import eval_set_meta
    from cupel.discovery import detect_hardware
    from cupel.display import HAS_RICH, build_table

    dotenv_path = load_dotenv(args.env_file)
    api_url, api_key = get_api_config()
    cfg, config_path = load_config(args.config)

    if args.models:
        cfg["models"] = [m.strip() for m in args.models.split(",")]
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.thinking is not None:
        cfg["_thinking_budget"] = args.thinking
    elif "thinking" in cfg and cfg["thinking"] is not None:
        cfg["_thinking_budget"] = int(cfg["thinking"])

    models = cfg["models"]
    if not models:
        print("\n  ✘ No models configured. Add them to config.yml or use --models\n")
        sys.exit(1)

    eval_set_path = resolve_path(cfg["eval_set"], config_path)
    if not eval_set_path.exists():
        print(f"\n  ✘ Eval set not found: {eval_set_path}\n")
        sys.exit(1)

    with open(eval_set_path) as f:
        eval_set = json.load(f)

    output_dir = resolve_path(cfg["output_dir"], config_path)

    print()
    if config_path:
        print(f"  config:    {config_path}")
    if dotenv_path:
        print(f"  .env:      {dotenv_path}")
    host = api_url.split("//")[-1].split("/")[0]
    print(f"  endpoint:  {host}")
    print(f"  eval set:  {eval_set['name']} ({len(eval_set['prompts'])} prompts)")
    print(f"  models:    {', '.join(models)}")
    tb = cfg.get("_thinking_budget")
    print(f"  thinking:  {tb if tb is not None else 'model default'}")
    print(f"  output:    {output_dir}")

    image_b64 = find_image(cfg["image_filename"], args.image_dir, config_path)
    print()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    thinking_budget = cfg.get("_thinking_budget")

    prompts = eval_set["prompts"]

    if args.prompts:
        prompt_ids = parse_prompt_ids(args.prompts)
        prompts = [p for p in prompts if p["id"] in prompt_ids]
        if not prompts:
            print(f"\n  ✘ No prompts match IDs: {args.prompts}\n")
            sys.exit(1)
        print(f"  prompts:   {', '.join(str(p['id']) for p in prompts)} ({len(prompts)} of {len(eval_set['prompts'])})")

    # Repeat the whole collection N times. Confidence intervals built from a single
    # run only capture prompt-to-prompt variation; the larger source of noise is
    # run-to-run sampling, and the only way to measure that is to run it again.
    repeat = max(1, getattr(args, "repeat", 1) or 1)
    hw = detect_hardware()
    t_label = f"_think{thinking_budget}" if thinking_budget is not None else ""
    saved_files = []
    all_results = {}

    for rep in range(repeat):
        if repeat > 1:
            print(f"\n  \u2500\u2500 run {rep + 1} of {repeat} \u2500\u2500")
        state = {}
        all_results = {m: [] for m in models}
        # distinct suffix per repetition: second-resolution timestamps collide
        rep_label = f"_r{rep + 1}" if repeat > 1 else ""

        if HAS_RICH:
            from rich.console import Console
            from rich.live import Live

            console = Console()
            with Live(
                build_table(prompts, models, state, api_url=api_url),
                console=console, refresh_per_second=4,
            ) as live:
                for model in models:
                    for p in prompts:
                        state[(model, p["id"])] = "running"
                        live.update(build_table(prompts, models, state, api_url=api_url))
                        result, status = run_prompt(api_url, api_key, model, p, cfg, image_b64)
                        state[(model, p["id"])] = status
                        all_results[model].append(result)
                        live.update(build_table(prompts, models, state, api_url=api_url))
            console.print()
            console.print(build_table(prompts, models, state, api_url=api_url))
        else:
            for model in models:
                print(f"\n{'='*60}\n  MODEL: {model}\n{'='*60}")
                for p in prompts:
                    print(f"  [{p['id']:2d}] {p['title'][:40]}...", end=" ", flush=True)
                    result, status = run_prompt(api_url, api_key, model, p, cfg, image_b64)
                    all_results[model].append(result)
                    if status == "skip":
                        print("SKIPPED")
                    elif status == "error":
                        print(f"ERROR: {result.get('error','')[:80]}")
                    else:
                        ttok = result.get('thinking_tokens', 0)
                        think_str = f"  🧠 {ttok} think tok" if ttok > 0 else ""
                        print(f"done ({status}, {result.get('completion_tokens','?')} tok{think_str})")

        # Save per-model JSONs
        for model in models:
            safe = model.replace("/", "_").replace(" ", "_")
            out = output_dir / f"eval_{safe}{t_label}_{timestamp}{rep_label}.json"
            with open(out, "w") as f:
                json.dump({
                    "model": model, "api_url": api_url,
                    "thinking_budget": thinking_budget,
                    "timestamp": timestamp,
                    # full identity, not just the name: re-judging must load the rubrics
                    # this run actually used, matched by hash
                    "eval_set": eval_set_meta(eval_set_path, eval_set),
                    "temperature": cfg.get("temperature"),
                    "notes": eval_set_path.stem,
                    "hardware": hw,
                    "results": all_results[model],
                }, f, indent=2)
            saved_files.append(str(out))
            print(f"  Saved: {out}")

    # Print errors
    for m in models:
        errs = [r for r in all_results[m] if r.get("error")]
        if errs:
            unique = set(r["error"] for r in errs)
            print(f"\n  ⚠ {m}: {len(errs)} errors")
            for e in sorted(unique):
                print(f"    → {e[:100]}")

    print(f"\n✅ Run complete. To score:")
    print(f"  python eval.py judge {' '.join(saved_files)}\n")


def _print_judge_agreement(all_data, prompts):
    """Report where judges disagreed.

    A prompt several judges score differently is usually a prompt whose rubric is
    ambiguous, not a prompt the model half-answered. That makes this the most
    direct signal available about which parts of an eval set need rewriting.
    """
    title_by_id = {p["id"]: p.get("title", "") for p in prompts}
    rows = []
    for _, data in all_data:
        for result in data.get("results", []):
            judgments = [j for j in result.get("judgments", []) if j.get("score") is not None]
            if len(judgments) < 2:
                continue
            scores = [j["score"] for j in judgments]
            spread = max(scores) - min(scores)
            if spread:
                rows.append((spread, data.get("model", ""), result["id"], judgments))

    if not rows:
        print("\n  judges agreed on every prompt\n")
        return

    rows.sort(reverse=True, key=lambda r: r[0])
    print(f"\n  judge disagreement — {len(rows)} prompt(s), widest first:\n")
    for spread, model, pid, judgments in rows:
        title = title_by_id.get(pid, "")[:38]
        print(f"    #{pid:<3} {title:<38} spread {spread}")
        for j in judgments:
            print(f"         {j['judge_model'][:28]:28s} → {j['score']}  {j.get('reason', '')[:60]}")
    print()


def _resolve_judge_endpoint(cfg, judge_model, args):
    """Endpoint and key for one judge.

    Explicit CLI flags win; otherwise the model is looked up among configured and
    discovered providers, so judges from different providers in a --judges list each
    reach their own endpoint.
    """
    from cupel.config import get_judge_config, get_providers_config, resolve_api_key_for_port
    from cupel.discovery import discover_providers

    if args.judge_url:
        url = args.judge_url
        key = (os.environ.get(args.judge_key_env, "no-key") if args.judge_key_env
               else get_judge_config(cfg)[2])
        return url, key

    for p in get_providers_config(cfg):
        if judge_model in (p.get("models") or []):
            return (p.get("api_url", ""),
                    os.environ.get(p.get("api_key_env", "LLM_API_KEY"), "no-key"))

    for p in discover_providers():
        if judge_model in (p.get("models") or []):
            base = (p.get("url") or "").rstrip("/")
            return base + "/v1/chat/completions", resolve_api_key_for_port(p.get("port", 0))

    _, url, key = get_judge_config(cfg)
    if args.judge_key_env:
        key = os.environ.get(args.judge_key_env, "no-key")
    return url, key


def _cmd_drop_judge(input_files, judge_model):
    """Remove one judge's judgments from result files and recompute scores."""
    from cupel.schema import (
        normalize_run, refresh_judges, drop_judge, judge_models, repair_mirror,
    )

    for fpath in input_files:
        with open(fpath) as f:
            data = normalize_run(json.load(f))

        before = sum(r["score"] for r in data["results"] if r.get("score") is not None)
        touched = sum(1 for r in data["results"] if drop_judge(r, judge_model))
        # Also recompute every other result's mirror. A failed judgment writes no
        # judgment at all but did overwrite judge_reason, so those results are
        # unreachable by drop_judge and only a full recompute restores them.
        repaired = repair_mirror(data)
        if not touched and not repaired:
            print(f"  — {Path(fpath).name}: no judgments from {judge_model}")
            continue

        refresh_judges(data)
        after = sum(r["score"] for r in data["results"] if r.get("score") is not None)
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)

        remaining = ", ".join(judge_models(data)) or "none"
        print(f"  ✓ {Path(fpath).name}: dropped {judge_model} from {touched} result(s)")
        extra = f", repaired {repaired} mirror(s)" if repaired else ""
        print(f"      score {before} → {after}{extra}   remaining judges: {remaining}")
    print()


def cmd_judge(args):
    """Score existing result JSONs with a judge model."""
    from cupel.config import (
        load_config, load_dotenv, get_judge_config, resolve_path,
    )
    from cupel.eval import judge_one, rubrics_for_run
    from cupel.schema import (
        normalize_run, refresh_judges, judge_models,
    )
    from cupel.display import HAS_RICH, build_table

    dotenv_path = load_dotenv(args.env_file)
    cfg, config_path = load_config(args.config)

    # Resolve judge config — one judge, or several in a single pass
    if getattr(args, "judges", None):
        judge_list = [m.strip() for m in args.judges.split(",") if m.strip()]
    elif args.judge_model:
        judge_list = [args.judge_model]
    else:
        configured, _, _ = get_judge_config(cfg)
        judge_list = [configured] if configured else []
    if not judge_list:
        print("\n  ✘ No judge model configured.")
        print("    Set judge.model in config.yml or use --judge-model\n")
        sys.exit(1)
    # endpoints are resolved per judge inside the scoring loop

    # Resolve input files (expand globs)
    input_files = []
    for pattern in args.files:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(expanded)
        elif os.path.exists(pattern):
            input_files.append(pattern)
        else:
            print(f"  ⚠ no match: {pattern}")
    input_files = sorted(set(input_files))

    if not input_files:
        print("\n  ✘ No result files found. Usage:")
        print("    python eval.py judge eval_results/eval_*.json\n")
        sys.exit(1)

    # --drop-judge is an undo, not a scoring pass — handle it and return
    if getattr(args, "drop_judge", None):
        print(f"\n  dropping judgments by: {args.drop_judge}\n")
        _cmd_drop_judge(input_files, args.drop_judge)
        return

    # Each run is judged against the eval set it actually used — recorded, else
    # identified from its prompts. --eval-set forces one explicitly. The configured
    # set is NOT a fallback: silently using it is what graded Clojure answers
    # against Python rubrics.
    forced_eval_set = None
    if getattr(args, "eval_set", None):
        if not args.eval_set.exists():
            print(f"\n  ✘ Eval set not found: {args.eval_set}\n")
            sys.exit(1)
        with open(args.eval_set) as f:
            forced_eval_set = json.load(f)
        print(f"\n  eval set:  {args.eval_set} (forced)")

    # Load all result files
    all_data = []
    for fpath in input_files:
        with open(fpath) as f:
            # bring pre-judgments files into the current shape before adding to them
            all_data.append((fpath, normalize_run(json.load(f))))

    # Resolve rubrics per file
    rubrics = {}
    for fpath, data in all_data:
        r_by_id, p_by_id, file_prompts, warning = rubrics_for_run(data, forced_eval_set)
        if warning:
            print(f"  ⚠ {Path(fpath).name}: {warning}")
        if not r_by_id:
            print(f"\n  ✘ Cannot judge {Path(fpath).name}: {warning or 'no rubrics available'}\n")
            sys.exit(1)
        rubrics[fpath] = (r_by_id, p_by_id, file_prompts)

    # the display table and summary follow the first file's eval set
    _, _, prompts = rubrics[all_data[0][0]]

    models = [d["model"] for _, d in all_data]

    judge_label = ", ".join(judge_list)
    print()
    if config_path:
        print(f"  config:    {config_path}")
    if dotenv_path:
        print(f"  .env:      {dotenv_path}")
    print(f"  judge:     {judge_label}")
    print(f"  scoring:   {len(input_files)} file(s), {len(prompts)} prompts each")
    for fpath, _ in all_data:
        print(f"    → {fpath}")
    print()

    # Score each result, once per judge. Judgments accumulate, so scoring with
    # several judges leaves every judgment on the record and their disagreement
    # becomes measurable rather than being overwritten.
    for judge_idx, judge_model in enumerate(judge_list):
        # --replace clears prior judgments once, on the first judge only
        replace_this = args.replace and judge_idx == 0
        # Resolve the endpoint per judge. Resolving once up front sent every judge in
        # a --judges list to the first judge's provider with the first judge's key.
        judge_url, judge_key = _resolve_judge_endpoint(cfg, judge_model, args)
        if len(judge_list) > 1:
            host = judge_url.split("//")[-1].split("/")[0]
            print(f"\n  judge {judge_idx + 1}/{len(judge_list)}: {judge_model} @ {host}\n")
        # Score each result
        score_state = {}

        if HAS_RICH:
            from rich.console import Console
            from rich.live import Live

            console = Console()
            with Live(
                build_table(prompts, models, score_state, title_prefix="⚖",
                            api_url=judge_url, phase="judge"),
                console=console, refresh_per_second=4,
            ) as live:
                for fpath, data in all_data:
                    model = data["model"]
                    rubric_by_id, prompt_by_id, _ = rubrics[fpath]
                    for result in data["results"]:
                        pid = result["id"]
                        score_state[(model, pid)] = "judging"
                        live.update(build_table(prompts, models, score_state,
                                                title_prefix="⚖", api_url=judge_url, phase="judge"))

                        status, score = judge_one(
                            result, rubric_by_id.get(pid), prompt_by_id.get(pid, ""),
                            judge_url, judge_key, judge_model, replace=replace_this,
                        )
                        score_state[(model, pid)] = (
                            (score, result.get("elapsed_seconds", "")) if status == "scored"
                            else status
                        )
                        live.update(build_table(prompts, models, score_state,
                                                title_prefix="⚖", api_url=judge_url, phase="judge"))

            console.print()
            console.print(build_table(prompts, models, score_state,
                                       title_prefix="⚖", api_url=judge_url, phase="judge"))
        else:
            for fpath, data in all_data:
                model = data["model"]
                rubric_by_id, prompt_by_id, _ = rubrics[fpath]
                print(f"\n  Judging: {model}")
                for result in data["results"]:
                    pid = result["id"]
                    print(f"    [{pid:2d}] {result['title'][:35]}...", end=" ", flush=True)
                    status, score = judge_one(
                        result, rubric_by_id.get(pid), prompt_by_id.get(pid, ""),
                        judge_url, judge_key, judge_model, replace=replace_this,
                    )
                    if status == "scored":
                        print(f"→ {score}/3  {(result.get('judge_reason') or '')[:50]}")
                    elif status == "skip":
                        print("→ skipped")
                    else:
                        last = (result.get("judge_errors") or [{}])[-1].get("error", "")
                        print(f"→ ERROR: {last[:60]}")

    # Save scores back into the result files
    for fpath, data in all_data:
        refresh_judges(data)
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)
        judges = judge_models(data)
        extra = f"  (judges: {', '.join(judges)})" if len(judges) > 1 else ""
        print(f"  Updated: {fpath}{extra}")

    # Where judges disagree, the rubric is ambiguous — that is the useful signal
    if args.show_agreement or len(judge_list) > 1:
        _print_judge_agreement(all_data, prompts)

    # Write scoring summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    judge_safe = "+".join(m.replace("/", "_").replace(" ", "_")[:20] for m in judge_list)
    models_safe = "+".join(m.replace("/", "_").replace(" ", "_")[:20] for m in models)
    output_dir = resolve_path(cfg["output_dir"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / f"scoring_{models_safe}_by_{judge_safe}_{timestamp}.md"

    with open(summary, "w") as f:
        f.write(f"# Eval Scoring Summary — {timestamp}\n\n")
        f.write(f"**Judge:** {judge_label}\n")
        f.write(f"**Files:** {len(input_files)}\n")
        f.write(f"\nScoring: 0=wrong · 1=partial · 2=correct/shallow · 3=correct/insightful\n\n")

        cols = ["#", "Title"] + [m[:22] for m in models]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")

        totals = {m: 0 for m in models}
        counts = {m: 0 for m in models}

        for p in prompts:
            row = [str(p["id"]), p["title"][:28]]
            for fpath, data in all_data:
                m = data["model"]
                r = next((x for x in data["results"] if x["id"] == p["id"]), None)
                if r and r.get("skipped"):
                    row.append("SKIP")
                elif r and r.get("error"):
                    row.append("ERR")
                elif r and r.get("score") is not None:
                    s = r["score"]
                    t = r.get("elapsed_seconds", "")
                    totals[m] += s
                    counts[m] += 1
                    row.append(f"**{s}** ({t}s)" if t else f"**{s}**")
                else:
                    row.append("  ")
            f.write("| " + " | ".join(row) + " |\n")

        # Totals — each model is scored out of what it actually completed, so a
        # model that errored on a prompt is not silently charged a zero for it.
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        total_row = ["", "**Total**"]
        for m in models:
            total_row.append(f"**{totals[m]}/{counts[m] * 3}**" if counts[m] > 0 else "—")
        f.write("| " + " | ".join(total_row) + " |\n")

        # Reasoning details
        f.write(f"\n---\n\n## Judge Reasoning ({judge_label})\n\n")
        for p in prompts:
            has_any = False
            for _, data in all_data:
                r = next((x for x in data["results"] if x["id"] == p["id"]), None)
                if r and r.get("score") is not None:
                    if not has_any:
                        f.write(f"### {p['id']}. {p['title']}\n\n")
                        has_any = True
                    f.write(f"- **{data['model'][:20]}** → {r['score']}/3: {r.get('judge_reason','')}\n")
            if has_any:
                f.write("\n")

    print(f"\n  Summary: {summary}")

    # Terminal score summary
    print()
    n_prompts = len(prompts)
    for m in models:
        if counts[m] > 0:
            unscored = f"  ({n_prompts - counts[m]} unscored)" if counts[m] < n_prompts else ""
            print(f"  {m[:35]:35s}  {totals[m]:2d}/{counts[m] * 3}{unscored}")
    print(f"\n✅ Judging complete.\n")


def main():
    parser = argparse.ArgumentParser(
        description="cupel — custom benchmarks to determine precise LLM gold content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version",
                        version=f"cupel {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="")

    # ── run ──
    run_parser = sub.add_parser("run", help="Collect responses from models",
        epilog="""
examples:
  cupel run
  cupel run --models "Qwen3.5-27B-8bit"
  cupel run --thinking 4096
        """, formatter_class=argparse.RawDescriptionHelpFormatter)
    run_parser.add_argument("--config", default=None, type=Path)
    run_parser.add_argument("--models", default=None)
    run_parser.add_argument("--prompts", default=None, help="Prompt IDs to run (e.g. 18-22 or 1,18-22)")
    run_parser.add_argument("--thinking", type=int, default=None, metavar="BUDGET")
    run_parser.add_argument("--repeat", type=int, default=1, metavar="N",
        help="Run the whole set N times — the only way to measure run-to-run variance")
    run_parser.add_argument("--image-dir", default=None, type=Path)
    run_parser.add_argument("--output-dir", default=None)
    run_parser.add_argument("--env-file", default=None, type=Path)

    # ── judge ──
    judge_parser = sub.add_parser("judge", help="Score existing result files",
        epilog="""
examples:
  cupel judge eval-results/eval_*.json
  cupel judge eval-results/eval_Qwen*.json --judge-model gpt-4o
  cupel judge eval-results/*.json --judge-url https://api.openai.com/v1/chat/completions --judge-key-env OPENAI_API_KEY
        """, formatter_class=argparse.RawDescriptionHelpFormatter)
    judge_parser.add_argument("files", nargs="+", help="Result JSON file(s) or glob patterns")
    judge_parser.add_argument("--judge-model", default=None, help="Override judge model")
    judge_parser.add_argument("--judges", default=None, metavar="M1,M2",
        help="Score with several judges in one pass — their disagreement is the diagnostic")
    judge_parser.add_argument("--replace", action="store_true",
        help="Discard existing judgments instead of appending (destructive)")
    judge_parser.add_argument("--show-agreement", action="store_true",
        help="Print the per-prompt spread between judges")
    judge_parser.add_argument("--drop-judge", default=None, metavar="MODEL",
        help="Undo: remove MODEL's judgments and recompute scores from what remains")
    judge_parser.add_argument("--eval-set", default=None, type=Path,
        help="Score against this eval set instead of inferring it from the run")
    judge_parser.add_argument("--judge-url", default=None, help="Override judge API URL")
    judge_parser.add_argument("--judge-key-env", default=None, help="Env var name for judge API key")
    judge_parser.add_argument("--config", default=None, type=Path)
    judge_parser.add_argument("--env-file", default=None, type=Path)

    # ── ui ──
    ui_parser = sub.add_parser("ui", help="Open the web dashboard")
    ui_parser.add_argument("--port", type=int, default=8042)
    ui_parser.add_argument("--host", default="127.0.0.1",
        help="Bind address. Defaults to loopback; the API is unauthenticated, "
             "so 0.0.0.0 exposes it to your whole network")

    # ── init ──
    sub.add_parser("init", help="Create config.yml + eval-set.json in current directory")

    # ── results ──
    results_parser = sub.add_parser("results",
        help="Format eval result files as Markdown or JSON",
        epilog="""
examples:
  cupel results --list
  cupel results --list --format json
  cupel results eval-results/eval_*.json
  cupel results eval-results/eval_*.json -o results.md
  cupel results eval-results/eval_*.json --format json
  cupel results eval-results/eval_*.json --criteria-mode full
        """, formatter_class=argparse.RawDescriptionHelpFormatter)
    results_parser.add_argument("files", nargs="*",
        help="JSON result file(s) to format")
    results_parser.add_argument("--list", action="store_true",
        help="List available result files in ~/.cupel/eval-results/")
    results_parser.add_argument("-o", "--output",
        help="Write output to file instead of stdout")
    results_parser.add_argument("--format", choices=["markdown", "json"],
        default="markdown", help="Output format (default: markdown)")
    results_parser.add_argument("--criteria-mode", choices=["compact", "full", "json"],
        default="compact",
        help="How to render criteria_results in Markdown (default: compact)")
    results_parser.add_argument("--last", type=int, default=None, metavar="N",
        help="Show only the last N results")
    results_parser.add_argument("-n", default=None, metavar="NUMBERS",
        help="Show results by number (e.g. -n 42,12,87)")

    # ── compare ──
    compare_parser = sub.add_parser("compare",
        help="Paired comparison of two runs — is A actually better than B?",
        epilog="""
examples:
  cupel compare eval-results/eval_A_*.json eval-results/eval_B_*.json
  cupel compare a.json b.json --metric score
  cupel compare a.json b.json --format json
        """, formatter_class=argparse.RawDescriptionHelpFormatter)
    compare_parser.add_argument("files", nargs="+",
        help="Exactly two result JSON files")
    compare_parser.add_argument("--metric", choices=["auto", "check_score", "score"],
        default="auto",
        help="What to compare on (default: auto — the criteria vector when available)")
    compare_parser.add_argument("--format", choices=["text", "json"], default="text")
    compare_parser.add_argument("-o", "--output", help="Write to a file instead of stdout")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "judge":
        cmd_judge(args)
    elif args.command == "ui":
        from cupel.server import _start_ui
        _start_ui(port=args.port, host=args.host)
    elif args.command == "results":
        from cupel.tools.results import cmd_results
        cmd_results(args)
    elif args.command == "compare":
        from cupel.tools.compare import cmd_compare
        cmd_compare(args)
    elif args.command == "init":
        _cmd_init()
    else:
        # No subcommand — always open the dashboard
        from cupel.server import _start_ui
        _start_ui()


def _cmd_init():
    """Create config.yml + eval-set.json in ~/.cupel/."""
    import shutil
    import yaml
    from cupel.discovery import detect_hardware, discover_providers

    cupel_home = Path.home() / ".cupel"
    cfg_path = cupel_home / "config.yml"
    es_path = cupel_home / "eval-sets" / "eval-set.json"

    if cfg_path.exists():
        print(f"  config.yml already exists, skipping")
    else:
        hw = detect_hardware()
        providers = discover_providers()
        online_models = []
        for p in providers:
            if p["status"] == "online":
                online_models.extend(p["models"])
        cfg = {
            "models": online_models[:5] if online_models else ["your-model-here"],
            "eval_set": "eval-sets/eval-set.json",
            "output_dir": "./eval-results",
            "max_tokens": 16384,
            "thinking": None,
            "judge": {"model": "", "api_url": "", "api_key_env": ""},
        }
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  created: {cfg_path}")

    if es_path.exists():
        print(f"  eval-set.json already exists, skipping")
    else:
        es_path.parent.mkdir(parents=True, exist_ok=True)
        full = Path(__file__).parent / "data" / "starter-eval-set.json"
        if full.exists():
            shutil.copy2(full, es_path)
            print(f"  created: {es_path} (starter eval set)")
        else:
            print(f"  eval-set.json not found in package")
    print()
