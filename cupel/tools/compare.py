"""cupel compare — paired comparison of two eval runs.

The leaderboard answers "how did each model do". This answers the sharper
question: "is A actually better than B, or is the gap noise?"

Two runs of the same eval set are compared task by task. Pairing cancels task
difficulty, which is the dominant source of variance in a small eval set, so this
resolves differences that two independent confidence intervals cannot.

Where a criteria rubric was used, the comparison runs on `check_score` — the
continuous value from the criteria vector — rather than the 0-3 collapse. That
collapse is what makes a well-built eval set look like it cannot separate models.
"""

import json
import sys
from pathlib import Path

from cupel.stats import (
    paired_bootstrap, score_pct_ci, overlaps, criterion_discrimination,
)


def load_run(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data.get("results"), list):
        raise ValueError(f"{path}: expected a 'results' list")
    return data


def pick_metric(runs, requested="auto"):
    """Choose what to compare on.

    `check_score` when every run has it (criteria rubrics), else the 0-3 `score`.
    """
    if requested != "auto":
        return requested
    for run in runs:
        scored = [r for r in run["results"] if r.get("score") is not None]
        if not scored or not all("check_score" in r for r in scored):
            return "score"
    return "check_score"


def scores_by_id(run, metric):
    return {
        r["id"]: r[metric]
        for r in run["results"]
        if r.get(metric) is not None and r.get("score") is not None
    }


def criteria_rows(run):
    """Per-task {criterion_id: met} for every task that has a criteria vector."""
    rows = []
    for r in run["results"]:
        cr = r.get("criteria_results")
        if isinstance(cr, list) and cr:
            rows.append({c["id"]: bool(c.get("met")) for c in cr if "id" in c})
    return rows


def _pct_ci(run, metric):
    """Interval for one run, in whatever units the metric uses."""
    vals = list(scores_by_id(run, metric).values())
    if not vals:
        return (0.0, 0.0, 0.0)
    max_per = 3 if metric == "score" else 1
    lo, hi = score_pct_ci(vals, max_per_item=max_per)
    pct = 100.0 * sum(vals) / (len(vals) * max_per)
    return (pct, lo, hi)


def format_comparison(run_a, run_b, metric, name_a="", name_b="") -> str:
    a_name = name_a or run_a.get("model", "A")
    b_name = name_b or run_b.get("model", "B")

    a_pct, a_lo, a_hi = _pct_ci(run_a, metric)
    b_pct, b_lo, b_hi = _pct_ci(run_b, metric)

    out = []
    out.append(f"\n  comparing on: {metric}"
               + ("   (criteria vector — the discriminating signal)"
                  if metric == "check_score" else
                  "   (0-3 collapse — low power; use a criteria rubric to improve it)"))
    out.append("")
    width = max(len(a_name), len(b_name))
    out.append(f"  A  {a_name:<{width}}  {a_pct:5.1f}%   [{a_lo:5.1f}, {a_hi:5.1f}]")
    out.append(f"  B  {b_name:<{width}}  {b_pct:5.1f}%   [{b_lo:5.1f}, {b_hi:5.1f}]")
    out.append("")

    if overlaps((a_lo, a_hi), (b_lo, b_hi)):
        out.append("  independent intervals OVERLAP — this alone does not resolve the ordering")
    else:
        out.append("  independent intervals are separated")

    pb = paired_bootstrap(scores_by_id(run_a, metric), scores_by_id(run_b, metric))
    out.append("")
    if pb["n"] == 0:
        out.append("  no shared tasks — nothing to pair")
        return "\n".join(out) + "\n"

    same_model = run_a.get("model") and run_a.get("model") == run_b.get("model")

    lo, hi = pb["ci"]
    header = ("  NOISE FLOOR — same model, two runs:" if same_model
              else f"  paired over {pb['n']} shared tasks:")
    out.append(header)
    out.append(f"    mean difference (A - B)   {pb['mean_diff']:+.3f}   CI [{lo:+.3f}, {hi:+.3f}]")
    out.append(f"    A better on {pb['a_wins']}, B better on {pb['b_wins']}, tied on {pb['ties']}")

    if same_model:
        # Identical model, so any detected difference is run-to-run variance. Reporting
        # it as a win would be exactly the error this command exists to prevent.
        out.append("")
        out.append("    Both runs are the SAME model, so this is not a comparison — it measures")
        out.append(f"    how much cupel's own numbers move between runs: {abs(pb['mean_diff']):.3f}")
        out.append(f"    on {pb['a_wins'] + pb['b_wins']} of {pb['n']} tasks.")
        out.append("    Treat a difference between two DIFFERENT models as real only when it")
        out.append("    clearly exceeds this floor.")
    elif pb["significant"]:
        winner = a_name if pb["mean_diff"] > 0 else b_name
        out.append(f"    -> {winner} is ahead: the interval excludes zero")
    else:
        out.append("    -> NOT RESOLVED: the interval includes zero, so the ordering is not supported")

    if not same_model:
        out.append("")
        out.append("    caveat: one run per side. This interval covers task sampling but NOT")
        out.append("    run-to-run variance. Pin temperature and compare repeat runs of the same")
        out.append("    model to establish the noise floor before trusting a narrow win.")

    # Which criteria are doing the work
    rows = criteria_rows(run_a) + criteria_rows(run_b)
    if rows:
        disc = criterion_discrimination(rows)
        carrying = sorted((d["discrimination"], cid) for cid, d in disc.items()
                          if d["discrimination"] > 0)
        dead = sorted(cid for cid, d in disc.items() if d["discrimination"] == 0)
        out.append("")
        if carrying:
            out.append("  criteria separating the models (higher = more informative):")
            for score, cid in sorted(carrying, reverse=True)[:10]:
                d = disc[cid]
                out.append(f"    {cid:<6} {score:.2f}   passed {d['met']}/{d['n']}")
        if dead:
            out.append(f"  dead weight — every run scored these identically: {', '.join(dead)}")
            out.append("    (drop them or make them harder; they cost tokens and add no signal)")

    return "\n".join(out) + "\n"


def cmd_compare(args):
    paths = [Path(p) for p in args.files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"  ✘ not found: {p}", file=sys.stderr)
        sys.exit(1)
    if len(paths) != 2:
        print("  ✘ compare takes exactly two result files", file=sys.stderr)
        sys.exit(1)

    try:
        run_a, run_b = (load_run(p) for p in paths)
    except ValueError as e:
        print(f"  ✘ {e}", file=sys.stderr)
        sys.exit(1)

    metric = pick_metric([run_a, run_b], args.metric)

    if args.format == "json":
        a_pct, a_lo, a_hi = _pct_ci(run_a, metric)
        b_pct, b_lo, b_hi = _pct_ci(run_b, metric)
        pb = paired_bootstrap(scores_by_id(run_a, metric), scores_by_id(run_b, metric))
        payload = {
            "metric": metric,
            "a": {"model": run_a.get("model"), "pct": round(a_pct, 2),
                  "ci": [round(a_lo, 2), round(a_hi, 2)]},
            "b": {"model": run_b.get("model"), "pct": round(b_pct, 2),
                  "ci": [round(b_lo, 2), round(b_hi, 2)]},
            "paired": {**pb, "ci": [round(pb["ci"][0], 4), round(pb["ci"][1], 4)]},
            "criteria": criterion_discrimination(criteria_rows(run_a) + criteria_rows(run_b)),
        }
        output = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        output = format_comparison(run_a, run_b, metric)

    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
