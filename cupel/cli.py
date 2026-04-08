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
    )
    from cupel.eval import find_image, run_prompt
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

    eval_set_path = Path(cfg["eval_set"])
    if not eval_set_path.exists() and config_path:
        alt = Path(config_path).parent / eval_set_path
        if alt.exists():
            eval_set_path = alt
    if not eval_set_path.exists():
        print(f"\n  ✘ Eval set not found: {cfg['eval_set']}\n")
        sys.exit(1)

    with open(eval_set_path) as f:
        eval_set = json.load(f)

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
    print(f"  output:    {cfg['output_dir']}")

    image_b64 = find_image(cfg["image_filename"], args.image_dir)
    print()

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    thinking_budget = cfg.get("_thinking_budget")

    state = {}
    all_results = {m: [] for m in models}
    prompts = eval_set["prompts"]

    if args.prompts:
        prompt_ids = parse_prompt_ids(args.prompts)
        prompts = [p for p in prompts if p["id"] in prompt_ids]
        if not prompts:
            print(f"\n  ✘ No prompts match IDs: {args.prompts}\n")
            sys.exit(1)
        print(f"  prompts:   {', '.join(str(p['id']) for p in prompts)} ({len(prompts)} of {len(eval_set['prompts'])})")

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
                    print(f"done ({status}, {result.get('completion_tokens','?')} tok)")

    # Save per-model JSONs
    t_label = f"_think{thinking_budget}" if thinking_budget is not None else ""
    saved_files = []
    for model in models:
        safe = model.replace("/", "_").replace(" ", "_")
        out = output_dir / f"eval_{safe}{t_label}_{timestamp}.json"
        with open(out, "w") as f:
            json.dump({
                "model": model, "api_url": api_url,
                "thinking_budget": thinking_budget,
                "timestamp": timestamp,
                "eval_set": eval_set["name"],
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


def cmd_judge(args):
    """Score existing result JSONs with a judge model."""
    from cupel.config import (
        load_config, load_dotenv, get_judge_config,
    )
    from cupel.eval import score_one, _prompt_text_for_judge
    from cupel.display import HAS_RICH, build_table

    dotenv_path = load_dotenv(args.env_file)
    cfg, config_path = load_config(args.config)

    # Resolve judge config
    if args.judge_model:
        judge_model = args.judge_model
    else:
        judge_model, _, _ = get_judge_config(cfg)
    if not judge_model:
        print("\n  ✘ No judge model configured.")
        print("    Set judge.model in config.yml or use --judge-model\n")
        sys.exit(1)

    if args.judge_url:
        judge_url = args.judge_url
    else:
        _, judge_url, _ = get_judge_config(cfg)

    if args.judge_key_env:
        judge_key = os.environ.get(args.judge_key_env, "no-key")
    else:
        _, _, judge_key = get_judge_config(cfg)

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

    # Load eval set for rubrics
    eval_set_path = Path(cfg["eval_set"])
    if not eval_set_path.exists() and config_path:
        alt = Path(config_path).parent / eval_set_path
        if alt.exists():
            eval_set_path = alt
    if not eval_set_path.exists():
        print(f"\n  ✘ Eval set not found: {cfg['eval_set']} (needed for rubrics)\n")
        sys.exit(1)

    with open(eval_set_path) as f:
        eval_set = json.load(f)

    rubric_by_id = {p["id"]: p.get("rubric", {}) for p in eval_set["prompts"]}
    prompt_by_id = {p["id"]: _prompt_text_for_judge(p) for p in eval_set["prompts"]}
    prompts = eval_set["prompts"]

    # Load all result files
    all_data = []
    for fpath in input_files:
        with open(fpath) as f:
            all_data.append((fpath, json.load(f)))

    models = [d["model"] for _, d in all_data]

    judge_host = judge_url.split("//")[-1].split("/")[0]
    print()
    if config_path:
        print(f"  config:    {config_path}")
    if dotenv_path:
        print(f"  .env:      {dotenv_path}")
    print(f"  judge:     {judge_model} @ {judge_host}")
    print(f"  scoring:   {len(input_files)} file(s), {len(prompts)} prompts each")
    for fpath, _ in all_data:
        print(f"    → {fpath}")
    print()

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
                for result in data["results"]:
                    pid = result["id"]

                    has_response = result.get("response") or any(r for r in result.get("responses", []))
                    if result.get("skipped") or result.get("error") or not has_response:
                        score_state[(model, pid)] = "skip" if result.get("skipped") else "error"
                        live.update(build_table(prompts, models, score_state,
                                                title_prefix="⚖", api_url=judge_url, phase="judge"))
                        continue

                    score_state[(model, pid)] = "judging"
                    live.update(build_table(prompts, models, score_state,
                                            title_prefix="⚖", api_url=judge_url, phase="judge"))

                    try:
                        score, reason = score_one(
                            judge_url, judge_key, judge_model,
                            prompt_by_id.get(pid, ""),
                            rubric_by_id.get(pid, {}),
                            result["response"],
                            responses=result.get("responses"),
                        )
                        if score is not None:
                            result["score"] = score
                            result["judge_reason"] = reason
                            result["judge_model"] = judge_model
                            elapsed = result.get("elapsed_seconds", "")
                            score_state[(model, pid)] = (score, elapsed)
                        else:
                            result["judge_reason"] = reason
                            score_state[(model, pid)] = "error"
                    except Exception as e:
                        result["judge_reason"] = f"judge error: {e}"
                        score_state[(model, pid)] = "error"

                    live.update(build_table(prompts, models, score_state,
                                            title_prefix="⚖", api_url=judge_url, phase="judge"))

        console.print()
        console.print(build_table(prompts, models, score_state,
                                   title_prefix="⚖", api_url=judge_url, phase="judge"))
    else:
        for fpath, data in all_data:
            model = data["model"]
            print(f"\n  Judging: {model}")
            for result in data["results"]:
                pid = result["id"]
                if result.get("skipped") or result.get("error") or not result.get("response"):
                    continue
                print(f"    [{pid:2d}] {result['title'][:35]}...", end=" ", flush=True)
                try:
                    score, reason = score_one(
                        judge_url, judge_key, judge_model,
                        prompt_by_id.get(pid, ""),
                        rubric_by_id.get(pid, {}),
                        result["response"],
                        responses=result.get("responses"),
                    )
                    if score is not None:
                        result["score"] = score
                        result["judge_reason"] = reason
                        result["judge_model"] = judge_model
                        print(f"→ {score}/3  {reason[:50]}")
                    else:
                        result["judge_reason"] = reason
                        print(f"→ parse error")
                except Exception as e:
                    result["judge_reason"] = f"judge error: {e}"
                    print(f"→ ERROR: {str(e)[:60]}")

    # Save scores back into the result files
    for fpath, data in all_data:
        data["judge"] = judge_model
        data["judge_url"] = judge_url
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Updated: {fpath}")

    # Write scoring summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    judge_safe = judge_model.replace("/", "_").replace(" ", "_")[:20]
    models_safe = "+".join(m.replace("/", "_").replace(" ", "_")[:20] for m in models)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / f"scoring_{models_safe}_by_{judge_safe}_{timestamp}.md"

    with open(summary, "w") as f:
        f.write(f"# Eval Scoring Summary — {timestamp}\n\n")
        f.write(f"**Judge:** {judge_model} @ {judge_host}\n")
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

        # Totals
        mx = len(prompts) * 3
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        total_row = ["", "**Total**"]
        for m in models:
            total_row.append(f"**{totals[m]}/{mx}**" if counts[m] > 0 else "—")
        f.write("| " + " | ".join(total_row) + " |\n")

        # Reasoning details
        f.write(f"\n---\n\n## Judge Reasoning ({judge_model})\n\n")
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
    for m in models:
        if counts[m] > 0:
            print(f"  {m[:35]:35s}  {totals[m]:2d}/{mx}")
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
    judge_parser.add_argument("--judge-url", default=None, help="Override judge API URL")
    judge_parser.add_argument("--judge-key-env", default=None, help="Env var name for judge API key")
    judge_parser.add_argument("--config", default=None, type=Path)
    judge_parser.add_argument("--env-file", default=None, type=Path)

    # ── ui ──
    sub.add_parser("ui", help="Open the web dashboard")

    # ── init ──
    sub.add_parser("init", help="Create config.yml + eval-set.json in current directory")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "judge":
        cmd_judge(args)
    elif args.command == "ui":
        from cupel.server import _start_ui
        _start_ui()
    elif args.command == "init":
        _cmd_init()
    else:
        # No subcommand — always open the dashboard
        from cupel.server import _start_ui
        _start_ui()


def _cmd_init():
    """Create config.yml + eval-set.json in current directory."""
    import shutil
    import yaml
    from cupel.discovery import detect_hardware, discover_providers

    cfg_path = Path.cwd() / "config.yml"
    es_path = Path.cwd() / "eval-sets" / "eval-set.json"

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
            "temperature": 0,
            "max_tokens": 16384,
            "thinking": None,
            "judge": {"model": "", "api_url": "", "api_key_env": ""},
        }
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
