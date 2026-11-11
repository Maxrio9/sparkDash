# Training reporting examples

These examples use Python's standard library and do not load a model, start a GPU job or contact a service. All example scores are synthetic. See the [module guide](../../server/training/README.md) for configuration, supported native logs and the reporting contract.

## Try the live synthetic demo

Enable `TRAINING_PROGRESS_ENABLED=1` and set `TRAINING_PROGRESS_ROOTS=["~/training-runs"]` in the dashboard's `.env`, then restart it. Register the machine running this example as a local unit, or connect it through the dashboard's usual SSH setup. The file collector needs the same user and Linux process namespace as the demo to verify its PID.

```bash
python3 examples/training/demo.py --output-dir "$HOME/training-runs/synthetic-demo"
```

The demo publishes training counters, loss, completed preparation, a synthetic baseline, checkpoint accuracy and an overlapping validation worker. It takes about two minutes by default; `--steps 20 --interval 1` gives a shorter run. Open Overview while it runs. Select Evaluation to inspect the checkpoint graph, and select Preparation to revisit completed input work. The main training pane hides after completion; the API retains the saved result while its root is configured. Ctrl+C marks the synthetic training stage stopped.

Use a different `--run-id` and output directory for independent experiments. Use the same run ID with distinct reporter IDs to combine workers into one experiment. Files in this directory contain no production results or private device configuration.

## Report a real loop

Copy `reporter.py` into your training project, or add this directory to your Python import path. Construct the reporter inside the worker process. Call `write` after real measurements change, using complete snapshots of the stage groups you want retained:

```python
import time
from reporter import ProgressReporter

progress = ProgressReporter(
    "~/training-runs/experiment-a/trainer",
    run_id="experiment-a",
    campaign="Classification fine-tuning",
    source_id="trainer-0",
    source_label="Trainer rank 0",
)

# In your existing training loop, after obtaining an actual step and loss:
def report_step(step, total_steps, measured_loss):
    now = round(time.time() * 1000)
    progress.write(training=[{
        "id": "fit", "label": "Fine-tuning", "state": "running",
        "completed": step, "total": total_steps, "unit": "steps",
        "loss": measured_loss,
        "heartbeatAt": now,
        # Choose this from your worker's expected reporting cadence.
        "heartbeatTimeoutSeconds": 120,
    }])
```

The helper adds its PID and Linux process-start identity, serializes strict JSON and replaces `training-progress.json` atomically in the output directory. It does not inspect your model, guess a total, calculate a loss or invent a heartbeat. `heartbeatAt` above is valid because the training worker is actively calling the reporting code. For a long training step, arrange a genuine worker heartbeat or omit a timeout that the worker cannot meet.

A call replaces the entire document. Preserve completed preparation and evaluation rows in later snapshots when you want their history available. On success, write the final `state: "complete"`; on a caught failure, write `"failed"`. A process that exits cannot keep establishing live training just because its last saved state said running. Do not share one output file between concurrent writers.

## Separate trainer and evaluator

Construct another reporter in the evaluator process, with the same experiment ID and a different output directory/source ID:

```python
validation = ProgressReporter(
    "~/training-runs/experiment-a/evaluator",
    run_id="experiment-a",
    campaign="Classification fine-tuning",
    source_id="evaluator-0",
    source_label="Validation worker",
)
```

Write actual evaluation rows under `evaluation`, including their own counters, metric units and optional comparison metadata. Sharing a device does not merge reporters; separate configured hosts can join the same run too. The filesystem provider attaches each reporter to the host where it was collected. For a multi-process document, individual stages can override `pid` and `process_start_ticks`.

`manifest.example.json` illustrates a completed preparation stage, training and checkpoint/baseline evaluations. It deliberately omits a live PID, so copying it alone will not claim that a job is running. Use the helper from the actual worker to report live work.
