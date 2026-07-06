#!/usr/bin/env python3
"""Sequential batch runner for hyperparameter sweeps.

Edit ``RUNS`` below — a list of dicts, one per run. Each dict must have an
``"experiment"`` key naming ``experiments/<experiment>.py``; every other key
becomes a CLI flag on that experiment (underscores → dashes, so
``"cosine_decay": 0.0`` → ``--cosine-decay 0.0``). A **bool** value is treated
as a flag: ``True`` adds ``--key``, ``False`` omits it (for ``store_true`` /
``store_false`` args like ``--quiet`` or ``--no-augment``).

Use ``grid(...)`` to build a cartesian-product sweep: list-valued keywords are
swept, scalar ones are fixed (e.g. ``grid("poisson_pinn", optimizer="gnome",
lr=[1e-3, 1e-2], eps=[1e-5, 1e-6])`` → 4 runs). Grids concatenate with ``+``.

The runner executes each run one at a time, streaming its output live, and
prints a pass/fail summary at the end. Each experiment writes its own JSONL
artifact under ``runs/`` as usual, so results are captured automatically.

    uv run python scripts/sweep.py            # run the sweep
    uv run python scripts/sweep.py --dry-run  # just print the commands
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time

STEPS  = 70_000
# ----------------------------------------------------------------------
# Grid helper
# ----------------------------------------------------------------------
def grid(experiment: str, **params) -> list[dict]:
    """Expand a cartesian-product grid into a list of run dicts.

    Any keyword whose value is a ``list``/``tuple`` is **swept**; scalar
    keywords are held **fixed** across every generated run. Returns a plain
    list of run dicts (each with ``"experiment"`` set), so grids can be
    concatenated with ``+`` or mixed with hand-written dicts.

        grid("poisson_pinn", optimizer="gnome", steps=STEPS,
             lr=[1e-4, 1e-3], eps=[1e-5, 1e-6])
        # -> 4 runs (lr × eps), all with optimizer=gnome, steps=STEPS
    """
    fixed = {k: v for k, v in params.items() if not isinstance(v, (list, tuple))}
    swept = {k: list(v) for k, v in params.items() if isinstance(v, (list, tuple))}
    keys = list(swept)
    runs = []
    for combo in itertools.product(*(swept[k] for k in keys)):
        runs.append({"experiment": experiment, **fixed, **dict(zip(keys, combo))})
    return runs


# ----------------------------------------------------------------------
# The sweep. Edit this.
# ----------------------------------------------------------------------
# lr × eps grid for Gnome on Poisson (3 × 3 = 9 runs):

RUNS = grid(
    "ols_regression", optimizer="gnome", epochs=100, seed=0,
    lr=1e-2,
    eps=1e-6,
    beta2=.99,
    dim=128,
)

RUNS += grid(
    "ols_regression", optimizer="soap", epochs=100, seed=0,
    lr=1e-2,
    beta2=.99,
    dim=128,
    cosine=1.0
)

RUNS += grid(
    "ols_regression", optimizer="adamw", epochs=100, seed=0,
    lr=1e-2,
    dim=128,
    cosine_decay=1.0
)


RUNS += grid(
    "poisson_pinn", optimizer="gnome", steps=STEPS, seed=0,
    lr=[1e-2, 3e-2],
    eps=[1e-4, 1e-6],
    beta2=.95,
    depth=8,
)

RUNS += grid(
    "poisson_pinn", optimizer="soap", steps=STEPS, seed=0,
    lr=[7e-4, 1e-3, 3e-3],
    beta2=.95,
    hidden = 64,
    depth = 8,
)

RUNS += grid(
    "poisson_pinn", optimizer="adamw", steps=STEPS, seed=0,
    lr=1e-3,
    hidden=64,
    depth=8,
)

RUNS += grid(
    "burgers_pinn", optimizer="gnome", steps=STEPS, seed=0,
    lr=[1e-2, 3e-2],
    eps=[1e-4, 1e-6],
    beta2=.95,
    hidden=64,
    depth=8,
)

RUNS += grid(
    "burgers_pinn", optimizer="soap", steps=STEPS, seed=0,
    lr=[7e-4, 1e-3, 3e-3],
    beta2=.95,
    hidden=64,
    depth=8,
)


RUNS += grid(
    "burgers_pinn", optimizer="adamw", steps=STEPS, seed=0,
    lr=1e-3,
    hidden=64,
    depth=8,
)



RUNS += grid(
    "kuramoto_sivashinsky_pinn", optimizer="gnome", steps=STEPS, seed=0,
    lr=[1e-2, 3e-2],
    eps=[1e-4, 1e-6],
    beta2=.95,
    hidden=64,
    depth=8,
)

RUNS += grid(
    "kuramoto_sivashinsky_pinn", optimizer="soap", steps=STEPS, seed=0,
    lr=[7e-4, 1e-3, 3e-3],
    beta2=.95,
    hidden=64,
    depth=8,
)

RUNS += grid(
    "kuramoto_sivashinsky_pinn", optimizer="adamw", steps=STEPS, seed=0,
    lr=1e-3,
    hidden=64,
    depth=8,
)


RUNS += grid(
    "navier_stokes_pinn", optimizer="gnome", steps=STEPS, seed=0,
    lr=[3e-2, 5e-2],
    eps=[1e-4, 1e-6],
    beta2=.95,
    hidden=64,
    depth=8,
)

RUNS += grid(
    "navier_stokes_pinn", optimizer="soap", steps=STEPS, seed=0,
    lr=[1e-3, 3e-3, 5e-3],
    beta2=.95,
    hidden=64,
    depth=8,
)

RUNS += grid(
    "navier_stokes_pinn", optimizer="adamw", steps=STEPS, seed=0,
    lr=1e-3,
    hidden=64,
    depth=8,
)

print(f"NUMBER OF RUNS: {len(RUNS)}")





# Concatenate grids to sweep several experiments / optimizers in one batch:
# RUNS = (
#     grid("poisson_pinn", optimizer="gnome", steps=STEPS,
#          lr=[1e-3, 1e-2], eps=[1e-5, 1e-6], beta2=[0.9, 0.95])
#     + grid("poisson_pinn", optimizer="soap", steps=STEPS, cosine_decay=0.0,
#            lr=[5e-4, 1e-3])
#     + [{"experiment": "ols_regression", "optimizer": "gnome", "lr": 0.1}]  # or hand-write
# )

# Stop the whole sweep on the first failing run? Default False: log it and
# keep going (a bad config shouldn't abort the rest of an overnight sweep).
STOP_ON_ERROR = False


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Exit code an experiment uses to signal "training diverged" (see
# experiments/common/divergence.py). An expected sweep outcome — reported
# separately and does NOT fail the batch.
DIVERGED_EXIT = 3


def build_cmd(run: dict) -> list[str]:
    """Turn one run dict into a ``python -m experiments.<exp> --flag val ...``
    argv list."""
    run = dict(run)  # don't mutate the caller's dict
    try:
        experiment = run.pop("experiment")
    except KeyError:
        raise SystemExit(f"run is missing the required 'experiment' key: {run}")
    cmd = [sys.executable, "-u", "-m", f"experiments.{experiment}"]
    for key, val in run.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                cmd.append(flag)     # bare flag; False → omit entirely
        else:
            cmd += [flag, str(val)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="Print the commands that would run, then exit.")
    args = ap.parse_args()

    total = len(RUNS)
    if not total:
        raise SystemExit("RUNS is empty — nothing to do.")

    if args.dry_run:
        for i, run in enumerate(RUNS, 1):
            print(f"[{i}/{total}] {' '.join(build_cmd(run)[2:])}")
        return

    results = []  # (index, run, returncode, seconds)
    sweep_t0 = time.time()
    try:
        for i, run in enumerate(RUNS, 1):
            cmd = build_cmd(run)
            pretty = " ".join(cmd[3:])  # drop python -u -m
            print(f"\n{'='*70}\n[{i}/{total}] {pretty}\n{'='*70}", flush=True)
            t0 = time.time()
            rc = subprocess.call(cmd, cwd=REPO)
            dt = time.time() - t0
            results.append((i, run, rc, dt))
            tag = ("ok" if rc == 0 else
                   "DIVERGED" if rc == DIVERGED_EXIT else f"FAILED (exit {rc})")
            print(f"[{i}/{total}] {tag} in {dt:.0f}s", flush=True)
            if rc not in (0, DIVERGED_EXIT) and STOP_ON_ERROR:
                print("STOP_ON_ERROR set — aborting the rest of the sweep.")
                break
    except KeyboardInterrupt:
        print("\ninterrupted — stopping the sweep.", flush=True)

    # -- summary --
    print(f"\n{'='*70}\nsweep summary ({time.time()-sweep_t0:.0f}s total)\n{'='*70}")
    ran = len(results)
    failed = [r for r in results if r[2] not in (0, DIVERGED_EXIT)]
    div = [r for r in results if r[2] == DIVERGED_EXIT]
    for i, run, rc, dt in results:
        exp = run.get("experiment", "?")
        tag = "ok  " if rc == 0 else ("DIV " if rc == DIVERGED_EXIT else "FAIL")
        print(f"  [{i}/{total}] {tag}  {dt:6.0f}s  {exp}  {_run_summary(run)}")
    if ran < total:
        print(f"  ... {total - ran} run(s) not started (interrupted/aborted)")
    parts = [f"{ran - len(failed) - len(div)} ok"]
    if div:
        parts.append(f"{len(div)} diverged")
    if failed:
        parts.append(f"{len(failed)} FAILED")
    print(f"\n{', '.join(parts)}  (of {ran} run)")
    if failed:   # divergence is an expected outcome; only real crashes fail the batch
        sys.exit(1)


def _run_summary(run: dict) -> str:
    """Compact one-line view of a run's key hyperparameters (sans experiment)."""
    return " ".join(f"{k}={v}" for k, v in run.items() if k != "experiment")


if __name__ == "__main__":
    main()
