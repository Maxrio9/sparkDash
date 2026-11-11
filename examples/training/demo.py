#!/usr/bin/env python3
"""Publish wholly synthetic training/evaluation progress; performs no ML work.

Copyright 2026 Maxrio9. SPDX-License-Identifier: Apache-2.0
"""

import argparse
import os
from pathlib import Path
import time

from reporter import ProgressReporter


def positive(value):
    number = float(value)
    if not 0 < number <= 3600:
        raise argparse.ArgumentTypeError("must be greater than zero and at most 3600")
    return number


def comparison(value, checkpoint=None, *, baseline=False):
    result = {
        "metric": "accuracy",
        "metricLabel": "Synthetic validation accuracy",
        "value": value,
        "unit": "fraction",
        "seriesId": "baseline" if baseline else "demo-tuning",
        "seriesLabel": "Baseline" if baseline else "Example tuning",
        "cohortId": "synthetic-validation-v1",
        "contextId": "synthetic-fixed-settings-v1",
        "contextLabel": "Synthetic examples · fixed settings",
        "goal": "maximize",
    }
    if baseline:
        result["baseline"] = True
    else:
        result["checkpoint"] = checkpoint
        result["checkpointUnit"] = "steps"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / "training-runs" / "synthetic-demo")
    parser.add_argument("--run-id", default="synthetic-demo")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--interval", type=positive, default=2)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000:
        parser.error("--steps must be between 1 and 10000")
    reporter = ProgressReporter(args.output_dir, args.run_id,
                                "Synthetic training demo", model="Example model")
    prepared = [{"id": "prepare", "label": "Prepare synthetic examples", "state": "complete",
                 "completed": 100, "total": 100, "unit": "samples"}]
    results = [{"id": "baseline", "label": "Baseline validation", "state": "complete",
                "completed": 20, "total": 20, "unit": "samples",
                "comparison": comparison(0.6, baseline=True)}]
    training = []
    try:
        affinity = ",".join(map(str, sorted(os.sched_getaffinity(0))))
        resources = [{"kind": "cpu", "id": affinity, "label": "Demo process CPU affinity", "scope": "affinity"}] if len(affinity) <= 160 else []
    except AttributeError:
        resources = []
    print(f"Writing synthetic progress to {reporter.output_dir / 'training-progress.json'}", flush=True)
    print("No model is loaded and no real evaluation is performed.", flush=True)
    try:
        for step in range(args.steps + 1):
            now = round(time.time() * 1000)
            training = [{
                "id": "train", "label": "Example training", "state": "running",
                "completed": step, "total": args.steps, "unit": "steps",
                "loss": round(1 / (1 + step / 10), 4),
                "etaSeconds": (args.steps - step) * args.interval,
                "heartbeatAt": now, "lastProgressAt": now,
                "heartbeatTimeoutSeconds": max(30, args.interval * 3),
                "progressTimeoutSeconds": max(60, args.interval * 5),
                "activity": {"kind": "training", "label": "Synthetic training loop"},
                "resources": resources,
            }]
            if step and (step % 10 == 0 or step == args.steps):
                accuracy = round(0.6 + 0.25 * step / args.steps, 4)
                results.append({
                    "id": f"validation-{step}", "label": f"Validation at step {step}",
                    "state": "complete", "completed": 20, "total": 20, "unit": "samples",
                    "updatedAt": now,
                    "metrics": [{"label": "Synthetic accuracy", "value": accuracy, "unit": "fraction"}],
                    "comparison": comparison(accuracy, step),
                })
            results = [results[0], *results[1:][-46:]]
            pending = [] if step == args.steps else [{
                "id": "validation-live", "label": "Synthetic validation worker", "state": "running",
                "completed": step % 10 * 2, "total": 20, "unit": "samples",
                "heartbeatAt": now, "heartbeatTimeoutSeconds": max(30, args.interval * 3),
                "activity": {"kind": "validating", "label": "Synthetic validation loop"},
                "dependencies": [{"id": "dataset", "label": "Synthetic validation examples", "state": "ready"}],
            }]
            reporter.write(data=prepared, training=training, evaluation=[*results, *pending])
            if step < args.steps:
                time.sleep(args.interval)
        training[0]["state"] = "complete"
        training[0].pop("etaSeconds", None)
        reporter.write(data=prepared, training=training, evaluation=results)
        print("Synthetic run complete. The main training pane will hide on its next refresh.", flush=True)
    except (KeyboardInterrupt, Exception) as error:
        if training:
            training[0]["state"] = "stopped" if isinstance(error, KeyboardInterrupt) else "failed"
            training[0].pop("etaSeconds", None)
            reporter.write(data=prepared, training=training, evaluation=results)
        if not isinstance(error, KeyboardInterrupt):
            raise


if __name__ == "__main__":
    main()
