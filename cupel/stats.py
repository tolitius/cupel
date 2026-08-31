"""cupel.stats — uncertainty and comparison for eval scores.

Two ideas drive this module:

1. A single eval run is one sample. Repeat runs of the same model differ by up to
   6 percentage points, which is wider than the spread across the whole leaderboard.
   So every reported number needs an interval, not just a point.

2. Collapsing a rich per-task rubric into one 0-3 score throws away most of the
   information that separates models. A task graded on 8 criteria carries far more
   signal than the same task graded pass/fail — enough to move detection of a real
   15-point gap from ~15% to ~75% at 15 tasks. Prefer the vector; keep 0-3 for display.
"""

import random
from collections import defaultdict

# Deterministic by default — the same scores always produce the same interval.
DEFAULT_SEED = 20260830
DEFAULT_ITERS = 2000


# ──────────────────────────────────────────────
# Bootstrap intervals
# ──────────────────────────────────────────────

def bootstrap_ci(values, iters=DEFAULT_ITERS, alpha=0.05, seed=DEFAULT_SEED):
    """Percentile bootstrap CI for the mean of `values`.

    Returns (lo, hi) in the same units as the input. An empty input gives
    (0.0, 0.0); a single value gives (v, v) — no pretending we know the spread.
    """
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (float(vals[0]), float(vals[0]))

    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        total = 0.0
        for _ in range(n):
            total += vals[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    lo_idx = int(alpha / 2 * iters)
    hi_idx = min(iters - 1, int((1 - alpha / 2) * iters))
    return (means[lo_idx], means[hi_idx])


def score_pct_ci(scores, max_per_item=3, iters=DEFAULT_ITERS, alpha=0.05,
                 seed=DEFAULT_SEED):
    """Bootstrap CI expressed as a percentage of the maximum attainable score.

    `scores` is the list of per-prompt scores actually obtained (0-3 each by
    default). Unscored prompts must be excluded by the caller — an unscored
    prompt is missing data, not a zero.
    """
    if not scores or max_per_item <= 0:
        return (0.0, 0.0)
    lo, hi = bootstrap_ci(scores, iters=iters, alpha=alpha, seed=seed)
    return (100.0 * lo / max_per_item, 100.0 * hi / max_per_item)


def check_pct_ci(check_scores, iters=DEFAULT_ITERS, alpha=0.05, seed=DEFAULT_SEED):
    """Interval for a run scored on its criteria vector.

    `check_score` is already a 0.0-1.0 fraction of the criteria weight earned, so
    this is the same bootstrap as `score_pct_ci` with a maximum of 1 rather than 3.
    Ranking on this instead of the 0-3 collapse is what keeps a well-built eval set
    able to separate models — the collapse throws most of that away.
    """
    return score_pct_ci(check_scores, max_per_item=1, iters=iters, alpha=alpha, seed=seed)


def overlaps(ci_a, ci_b):
    """True when two intervals overlap — i.e. the difference is not resolved."""
    return ci_a[0] <= ci_b[1] and ci_b[0] <= ci_a[1]


# ──────────────────────────────────────────────
# Paired comparison
# ──────────────────────────────────────────────

def paired_bootstrap(a_by_id, b_by_id, iters=DEFAULT_ITERS, alpha=0.05,
                     seed=DEFAULT_SEED):
    """Compare two models on the tasks they both attempted.

    Pairing cancels task difficulty, which is the dominant source of variance in a
    small eval set. This resolves much smaller differences than comparing two
    independent intervals, and it is the right test for "is A actually better than B".

    `a_by_id` / `b_by_id` map prompt id -> score.

    Returns a dict with the mean paired difference (a - b), its CI, the number of
    paired tasks, how many each model won, and whether the CI excludes zero.
    """
    shared = sorted(set(a_by_id) & set(b_by_id))
    diffs = [a_by_id[i] - b_by_id[i] for i in shared]

    if not diffs:
        return {
            "n": 0, "mean_diff": 0.0, "ci": (0.0, 0.0),
            "a_wins": 0, "b_wins": 0, "ties": 0, "significant": False,
        }

    lo, hi = bootstrap_ci(diffs, iters=iters, alpha=alpha, seed=seed)
    return {
        "n": len(diffs),
        "mean_diff": sum(diffs) / len(diffs),
        "ci": (lo, hi),
        "a_wins": sum(1 for d in diffs if d > 0),
        "b_wins": sum(1 for d in diffs if d < 0),
        "ties": sum(1 for d in diffs if d == 0),
        # CI excluding zero is the claim that the ordering is real
        "significant": lo > 0 or hi < 0,
    }


# ──────────────────────────────────────────────
# Criteria-vector scoring
# ──────────────────────────────────────────────

def is_criteria_rubric(rubric):
    """Mirror of eval._is_criteria_rubric, kept here so stats has no import cycle."""
    return isinstance(rubric, dict) and "criteria" in rubric


def criteria_score(criteria_results, rubric):
    """Continuous 0.0-1.0 score from a criteria vector.

    This is the number to rank on. `_aggregate_criteria` in cupel.eval collapses the
    same data to 0-3 for display; that collapse costs most of the discriminating
    power, so it should not be what comparisons are built on.

    Positive criteria earn their weight when met; negative criteria subtract theirs.
    The result is clamped to [0, 1] and normalised by the total positive weight.
    """
    if not is_criteria_rubric(rubric) or not criteria_results:
        return None

    lookup = {c["id"]: (c["type"], c.get("weight", 1)) for c in rubric["criteria"]}
    pos_total = sum(w for t, w in lookup.values() if t == "positive")
    if pos_total <= 0:
        return None

    earned = 0
    for cr in criteria_results:
        entry = lookup.get(cr.get("id"))
        if not entry or not cr.get("met"):
            continue
        ctype, weight = entry
        if ctype == "positive":
            earned += weight
        elif ctype == "negative":
            earned -= weight

    return max(0.0, min(1.0, earned / pos_total))


def criterion_discrimination(runs_criteria):
    """Find which criteria carry the bench and which are dead weight.

    `runs_criteria` is an iterable of {criterion_id: bool} — one dict per scored
    task across all models. A criterion every model passes (or every model fails)
    contributes nothing to separating them and should be dropped or hardened.

    Returns {criterion_id: {"n": int, "met": int, "pass_rate": float,
                            "discrimination": float}} where discrimination peaks
    at 0.5 pass rate and is 0 when every model does the same thing.
    """
    tally = defaultdict(lambda: [0, 0])  # id -> [met, total]
    for row in runs_criteria:
        for cid, met in row.items():
            tally[cid][1] += 1
            if met:
                tally[cid][0] += 1

    out = {}
    for cid, (met, total) in tally.items():
        rate = met / total if total else 0.0
        out[cid] = {
            "n": total,
            "met": met,
            "pass_rate": rate,
            # 4*p*(1-p): 1.0 at an even split, 0.0 when unanimous
            "discrimination": 4 * rate * (1 - rate),
        }
    return out


# ──────────────────────────────────────────────
# Run aggregation
# ──────────────────────────────────────────────

def run_group_key(run):
    """Identity of a comparable configuration.

    Repeat runs sharing this key are samples of the same thing and must be
    aggregated, not ranked against each other. Anything that changes what is being
    measured — the eval set, the judge, the judge's prompt revision, the sampling
    settings — belongs in the key.
    """
    eval_set = run.get("eval_set")
    if isinstance(eval_set, dict):
        es = eval_set.get("hash") or eval_set.get("name", "")
    else:
        es = eval_set or ""

    judges = run.get("judges")
    if isinstance(judges, list) and judges:
        judge_key = ",".join(sorted(str(j.get("model", "")) for j in judges))
        scoring = ",".join(sorted({str(j.get("scoring_version", "")) for j in judges}))
    else:
        judge_key = str(run.get("judge", ""))
        scoring = "legacy"

    return (
        run.get("model", "unknown"),
        es,
        judge_key,
        scoring,
        run.get("thinking_budget"),
        run.get("temperature"),
        run.get("harness_version", ""),
    )


def aggregate_runs(runs, score_of=None, max_per_item=3, seed=DEFAULT_SEED):
    """Group repeat runs and summarise each group with an interval.

    `runs` is an iterable of run-file dicts. `score_of` extracts the per-prompt
    scores from one run; the default reads `score` from each scored result and
    skips the rest — an errored prompt is missing data, never a zero.

    Returns a list of group summaries sorted by pct descending.
    """
    if score_of is None:
        def score_of(run):
            return [r["score"] for r in run.get("results", [])
                    if r.get("score") is not None]

    groups = defaultdict(list)
    for run in runs:
        groups[run_group_key(run)].append(run)

    out = []
    for key, members in groups.items():
        pooled = []
        per_run_pct = []
        for run in members:
            scores = score_of(run)
            pooled.extend(scores)
            if scores:
                per_run_pct.append(100.0 * sum(scores) / (len(scores) * max_per_item))

        if not pooled:
            continue

        lo, hi = score_pct_ci(pooled, max_per_item=max_per_item, seed=seed)
        pct = 100.0 * sum(pooled) / (len(pooled) * max_per_item)

        out.append({
            "model": key[0],
            "eval_set": key[1],
            "judge": key[2],
            "scoring_version": key[3],
            "thinking_budget": key[4],
            "temperature": key[5],
            "harness_version": key[6],
            "n_runs": len(members),
            "n_scored": len(pooled),
            "pct": round(pct, 1),
            "ci_lo": round(lo, 1),
            "ci_hi": round(hi, 1),
            "per_run_pct": [round(p, 1) for p in per_run_pct],
            "spread": round(max(per_run_pct) - min(per_run_pct), 1) if len(per_run_pct) > 1 else 0.0,
            "filenames": [r.get("_filename", "") for r in members],
        })

    out.sort(key=lambda e: e["pct"], reverse=True)
    return out


def measured_noise_floor(per_run_pcts):
    """How far apart the same configuration's repeat runs actually landed.

    `per_run_pcts` is an iterable of per-run percentage lists, one list per
    configuration that was run more than once.

    This replaces modelling the uncertainty. A bootstrap interval has to assume the
    prompts are a random sample of some population, which is false for a
    hand-authored eval set — the prompts are the definition, not a draw from it.
    Repeat runs need no such assumption: the same model, the same prompts, run
    twice, landed this far apart. Gaps smaller than that did not survive a rerun.

    Returns {"floor", "mean", "n_pairs", "n_configs"} or None when nothing repeats.
    """
    gaps = []
    configs = 0
    for pcts in per_run_pcts:
        if len(pcts) < 2:
            continue
        configs += 1
        for i in range(len(pcts)):
            for j in range(i + 1, len(pcts)):
                gaps.append(abs(pcts[i] - pcts[j]))
    if not gaps:
        return None
    return {
        "floor": round(max(gaps), 1),
        "mean": round(sum(gaps) / len(gaps), 1),
        "n_pairs": len(gaps),
        "n_configs": configs,
    }


def rank_within_noise(entries, floor):
    """Rank entries, treating gaps smaller than the measured noise floor as ties.

    Entries must be sorted by pct descending. With no floor (nothing has been run
    twice) every entry gets a distinct rank and none are marked tied — better to
    show no claim than to invent a band from an assumption.
    """
    if floor is None:
        for i, e in enumerate(entries):
            e["rank"] = i + 1
            e["tied"] = False
        return entries

    rank = 0
    leader_pct = None
    for i, e in enumerate(entries):
        if leader_pct is None or (leader_pct - e["pct"]) > floor:
            rank = i + 1
            leader_pct = e["pct"]
        e["rank"] = rank
        e["tied"] = (rank != i + 1)
    return entries


def rank_with_ties(entries):
    """Assign ranks, giving equal rank to entries whose intervals overlap.

    Entries must already be sorted by pct descending. A new rank is only started
    when an entry's interval clears the band leader's — otherwise the ordering
    between them is not something the data supports, and showing distinct ranks
    would be inventing a result.
    """
    rank = 0
    leader_ci = None
    for i, e in enumerate(entries):
        ci = (e["ci_lo"], e["ci_hi"])
        if leader_ci is None or not overlaps(ci, leader_ci):
            rank = i + 1
            leader_ci = ci
        e["rank"] = rank
        e["tied"] = (rank != i + 1)
    return entries
