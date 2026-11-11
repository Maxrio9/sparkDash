# Training and evaluation module

Optional, read-only monitoring in the [community feature fork](https://github.com/Maxrio9/sparkDash) of [MiaAI-Lab/sparkDash](https://github.com/MiaAI-Lab/sparkDash). This is an integrated feature module with trusted code providers. It does not assume a general plugin API in upstream sparkDash. See the [community release installation instructions](../../TRAINING-RELEASE.md).

The module discovers training jobs and converts their progress into one display format. Providers contain the framework and filesystem knowledge. The dashboard renders the reported runs, stages, units and metrics.

The pane appears while a provider confirms any running training stage. While it is open, each run with live preparation, training or evaluation work can appear, including a run whose trainer has finished while another run still trains. Completed work and evaluation queues remain available through the API. Evaluation tiles can continue displaying progress after all training ends. The pane hides after the next status refresh confirms that all training has stopped. Unreachable or expired readings become stale and cease to establish training activity.

## Enable and configure

Set the following in `.env` (or the server environment), then restart sparkDash:

```dotenv
TRAINING_PROGRESS_ENABLED=1
TRAINING_PROGRESS_ROOTS=["~/training-runs"]
```

The module is off unless `TRAINING_PROGRESS_ENABLED=1` is set. Roots default to an empty array; enabling the feature alone does not scan home directories or infer search roots from running command lines. Configure only directories intended for monitoring. The same root list applies to each configured Unix unit. A root is expanded by the collector user on that host, so `~` refers to that user's home.

For different layouts, override roots by the stable unit IDs shown by `GET /api/sparks`:

```dotenv
TRAINING_PROGRESS_ROOTS_BY_UNIT={"training-host":["/srv/experiments"],"validation-host":["~/validation-runs"]}
```

These are JSON values, not shell-separated lists. Unit names above are placeholders; use your actual configured IDs. An override replaces the global list for that unit; an empty array excludes it from artifact discovery. Invalid configuration is rejected rather than silently broadening the search. Keep local paths and credentials in `.env`; do not commit your environment file.

The native artifact provider requires Python 3 and Linux `/proc` access under the same user as the reported process. It uses the configured local/SSH transport and does not open a new network service. The stock Docker deployment runs the collector as root, so `~` means `/root` and it cannot attest a host trainer owned by a different UID simply by sharing `/proc`. For Docker, the simplest setup is to register the training host as a remote SSH unit (`isLocal: false`) using the actual training user, even when it is the machine hosting the container; roots then refer to that user's host paths. A custom local container deployment needs explicit read-only progress-directory mounts and matching process/user identity. This module does not add privileged namespace access. Remote SSH collection runs in the remote host's normal namespace. See the repository's existing [security and deployment guidance](../../README.md#security) for access control. Native live-job detection is supported on Linux. On macOS, the Python reader can read compatible saved metadata, but without Linux `/proc` it cannot confirm a live trainer; its running state is unknown and cannot keep the pane open. Windows units are skipped by this artifact provider. Other platforms and agent-only transports need a trusted provider or bridge with their own real liveness evidence.

## Built-in provider

- `providers/artifacts.js` discovers Transformers/TRL `trainer_state.json`, Lightning `metrics.csv`, Keras CSVLogger epoch logs, and the common `training-progress.json` format. It reads explicit configured roots; live process arguments are used only to identify the owner of discovered artifacts. Set `TRAINING_PROGRESS_ROOTS` or `TRAINING_PROGRESS_ROOTS_BY_UNIT` to choose the locations searched. No roots are searched by default.

Discovery is bounded and skips model weights, recordings, dependency trees and caches. Trainer state becomes active when its output directory matches a live training process. A common progress document supplies its process ID. Saved metrics alone cannot establish activity.

| Native format | Progress and results |
| --- | --- |
| Transformers Trainer / TRL | Steps, loss, epoch, learning rate, numeric objective metrics such as reward accuracy, reward margin and KL, and the latest `eval_*` results. This also covers PEFT/LoRA jobs using Trainer; the adapter reads logger output independently of model architecture. |
| PyTorch Lightning CSVLogger | `metrics.csv` step/epoch counters, sparse training and validation metrics, and scalar metadata from `hparams.yaml` or JSON configuration. Works for language, vision and other Lightning modules. |
| Keras CSVLogger | Epoch progress, loss, accuracy and `val_*` results from `history.csv`, `training.csv`, `training.log`, or epoch-based `metrics.csv`. Comma, semicolon and tab delimiters are accepted. |
| Custom loops, pretraining and distributed jobs | The common progress document supplies arbitrary stages, counters and metrics. The same run ID joins sources across hosts. |

CSV discovery reads a bounded tail rather than loading entire training logs. Its activity check also requires a measurement written after the matched process started. Missing step/epoch limits stay unknown, and a quiet or saved CSV file does not establish that training completed. Native support is verified with representative fixtures; installing the adapter does not launch training jobs.

Configuration can provide `max_steps` or `epochs`/`max_epochs`, model identity and descriptive run names. The collector reads bounded JSON files (`config.json`, `training_args.json`, `run_config.json`, `hparams.json`, `adapter_config.json`) and a scalar subset of `hparams.yaml`/`config.yaml`; it does not execute or deserialize arbitrary training code, checkpoints or YAML objects.

## Reporting from training code

[`examples/training/`](../../examples/training/) includes a small atomic Python writer, a complete manifest and a CPU-only live demo with synthetic counters and scores. The helper is optional; any language can publish this JSON contract.

Any training loop can atomically write `training-progress.json` in its output directory under a discovery root. For example, Python can report a pretraining run with arbitrary counters and metrics.

```python
import json
import os
from pathlib import Path

output = Path.home() / "training-runs" / "pretraining-demo"
output.mkdir(parents=True, exist_ok=True)

def report(step, total_steps, loss, tokens_per_second, state="running"):
    document = {
        "schema_version": 1,
        "id": "pretraining-demo",
        "campaign": "Pretraining experiment",
        "model": "My model",
        "pid": os.getpid(),
        "stages": {
            "data": [],
            "training": [{
                "id": "pretrain",
                "label": "Pretraining",
                "state": state,
                "completed": step,
                "total": total_steps,
                "unit": "steps",
                "loss": loss,
                "metrics": [{
                    "label": "Throughput",
                    "value": tokens_per_second,
                    "unit": "tokens/s",
                }],
            }],
            "evaluation": [],
        },
    }
    temporary = output / "training-progress.json.tmp"
    temporary.write_text(json.dumps(document) + "\n")
    temporary.replace(output / "training-progress.json")
```

Call `report` from the training process as progress changes and with `state="complete"` when it finishes. Process exit is detected even if the last document still says running. Optional `process_start_ticks` can bind the document to Linux's precise process start identity. Shared run IDs join readings from multiple hosts.

## Runs, reporters and devices

These are separate identities. A run is an experiment; a reporter publishes progress for one or more jobs; a device is the physical host. Sharing a device does not merge experiments, and one distributed experiment can span many devices.

| Setup | Reporting |
| --- | --- |
| Train and evaluate in one process | One document can contain both stage groups and use its document PID. |
| Separate trainer and evaluator on one device | Use separate reporter documents with the same `run_id`, different `source_id` values and their own PIDs; both remain visible. |
| Several processes writing one document | Each stage can override `pid` and `process_start_ticks`, so an exited evaluator does not appear alive just because the trainer remains alive. |
| Distributed training, evaluation elsewhere | Reuse a run ID across hosts. Give reporters useful labels such as rank, trainer or evaluator; the host is retained separately. |
| Four Sparks running four independent experiments | Register the devices normally and use distinct run IDs. Each experiment has independent panels, stage selections, metrics and plots. No Spark-number-specific code is needed. |
| Multiple experiments or evaluations on one GPU | Use distinct run/reporter identities. The device tile shows each evaluation's own progress, rate and concurrency, with extra jobs expandable. |
| Streaming preparation, training and evaluation together | Report running stages in all relevant groups. The view can show all three together without adding counters of different units. |

For common progress documents, `id` or `run_id` identifies the experiment. An optional `source_id` (`job_id` alias) identifies a reporter within it; without one, the document path supplies a stable reporter identity. `source_label` (`job_label` alias) gives the reporter a display name. SparkDash combines the reporter identity with the physical host so colocated jobs remain distinct and ranks on separate hosts do not overwrite each other. Saved copies of the same explicit reporter are deduplicated by their newest measurement.

For example, a trainer document can use `run_id: "classification-experiment-a"`, `source_id: "trainer-0"`, `source_label: "Trainer rank 0"` and the trainer PID. A separate evaluator uses that same run ID, `source_id: "evaluator-0"`, `source_label: "Validation worker"` and its own PID. This works whether their folders are on one machine or different registered hosts. A per-stage PID override uses its own start identity rather than inheriting an unrelated document PID's start ticks.

Native Trainer and CSV configurations can also supply `run_id` or `sparkdash.run_id` to associate distributed reporters and companion manifests. Without an explicit shared ID, their host/output-directory identities keep unrelated runs separate even when directory names match. Explicit IDs should be unique to the experiment they describe.

Progress counts stay scoped to their reporter; replicated global steps from multiple ranks are not summed into a larger training total. Evaluation rates and retries are scoped to provider, run, reporter and stage. One reporter going stale does not turn another host's fresh reading into a stale measurement. Plots remain scoped to the run and its comparison metric/unit/corpus.

## Worker activity, resources and health

Sources and individual stages can declare `resources`, for example:

```json
[
  {"kind": "cpu", "id": "0-7", "label": "CPU cores 0–7", "scope": "affinity"},
  {"kind": "gpu", "id": "GPU-example-uuid", "label": "GPU 1", "scope": "assigned"}
]
```

Resource kinds are `cpu`, `gpu` and `accelerator`. Scope distinguishes a declared assignment from process visibility or CPU affinity; none claims an exclusive GPU reservation. Native artifact discovery can read the matched process's CPU affinity and explicit CUDA/HIP/ROCr visibility. It does not infer job ownership from VRAM usage. Custom workers can report other accelerators with stable identities and readable labels. Sources and stages accept up to 32 resources each.

A stage can describe what its worker is currently doing, independently of the broad training/evaluation group:

```json
{
  "activity": {
    "kind": "saving",
    "label": "Saving checkpoint",
    "detail": "Publishing shard 2 of 4",
    "since": 1794400000000
  },
  "dependencies": [
    {"id": "dataset", "label": "Prepared examples", "state": "ready"},
    {"id": "checkpoint-copy", "label": "Checkpoint transfer", "state": "waiting"}
  ],
  "heartbeatAt": 1794400000000,
  "heartbeatTimeoutSeconds": 120,
  "lastProgressAt": 1794400000000,
  "progressTimeoutSeconds": 900
}
```

The reporting code supplies real activity and input states (`ready`, `waiting`, `failed`, `unknown`); SparkDash displays them without executing a dependency graph or managing a scheduler. Set `activity.since` when the activity changes, rather than on every heartbeat. Examples include loading data, waiting for a checkpoint, transferring an artifact, validating and saving. Cached activity is labeled as historical when the source is stale.

Health is derived by the server, not accepted as a publisher's health claim. `heartbeatAt` establishes whether the worker is reporting; changing counters or activities establishes observed progress. `lastProgressAt` can supply the timestamp of a measurement before the dashboard first saw it. Explicit heartbeat/progress timeouts are positive seconds, up to seven days. Without a declared expected interval, the module does not guess a stall threshold. A successful file read or a live PID alone is not a heartbeat.

An overdue heartbeat becomes `unresponsive`; counters that fail to advance within a declared progress interval become `stalled`. Explicit waiting/queued/blocked activities do not get counter-stall warnings, although their declared heartbeat still applies. Unreachable sources and timestamps ahead of the dashboard clock remain unknown. Retries and counter resets start a new progress observation. Stalled or unresponsive stages omit their ETA and finish prediction until reporting/progress recovers.

Discovery is bounded: up to 24 discovered runs per host, 64 runs and 64 reporters per run, with a one-MiB normalized payload limit. Live work receives priority when selecting bounded results. File discovery supports the listed Unix-host formats; other frameworks or external schedulers can use the common document or a code provider rather than relying on guessed progress.

Stages can use steps, epochs, batches, tokens, samples, bytes or other units. `completed` and `total` can be fractional or null when unknown. Custom metrics contain a label, a finite number, text or null value, and an optional unit. Each stage can also report loss, ETA and checkpoint progress. Valid states are `running`, `waiting`, `complete`, `stopped`, `failed` and `unknown`.

Evaluation stages use the same metric format for accuracy, validation loss, reward, acceptance scores, confidence intervals, throughput or other measured results. The newest completed evaluation stays visible; earlier evaluations expand on demand. The metric labels and units come from the provider or reporting code, so a score such as accepted tokens per round remains distinct from evaluation speed in prompts per minute. Unknown measurements should be omitted or reported as null.

The stage navigation shows preparation, training and evaluation status and progress. Its initial view follows current work, including overlapping stages. Each stage button then independently selects or deselects its panel, so preparation, training and evaluation can be combined in any order. Clicking an already selected stage turns it off. Selecting none leaves a compact prompt rather than reopening a stage automatically. Each run keeps its own selection across polling and run reordering. Completed preparation starts out of the main layout and stays accessible through its button. Percentages describe reported counters within a stage, rather than combining unrelated prompts, tokens and steps into one campaign percentage.

Individual checkpoint evaluations can be complete while the evaluation stage remains open. While training is running or waiting, an otherwise finished evaluation queue is shown as awaiting checkpoints with final evaluation pending. The completed-result count describes available results, rather than claiming the whole evaluation stage is finished. Failed or stale readings retain their own status.

Live ETA values show both remaining duration and an estimated finish date/time in the browser's local timezone. They are anchored to the source reading; stale sources do not present a live prediction. Estimates can change as measured training rate changes. Completion timestamps are shown when a provider reports them.

## Checkpoint comparisons

A completed evaluation can provide structured `comparison` metadata to appear on the checkpoint graph. For example:

```json
{
  "metric": "validation-accuracy",
  "metricLabel": "Validation accuracy",
  "value": 0.84,
  "unit": "fraction",
  "seriesId": "lora-r64",
  "seriesLabel": "LoRA rank 64",
  "checkpoint": 2000,
  "checkpointUnit": "steps",
  "cohortId": "validation-set-v1",
  "contextId": "deterministic-evaluation-v1",
  "contextLabel": "Validation split · fixed decoding settings",
  "goal": "maximize",
  "lower": 0.82,
  "upper": 0.86
}
```

Attach this object to a completed evaluation stage alongside its ordinary metrics. A reference evaluation uses `baseline: true` and omits `checkpoint`. Keep metric, unit, cohort and settings identifiers consistent only when scores are comparable. Missing measurements are never synthesized.

A stage can report up to 20 metrics in `comparisons`, using an array of these objects. The array takes precedence over the legacy single `comparison`. Accuracy, loss, reward and throughput remain separate comparisons instead of being averaged into one score. `checkpointUnit` defaults to steps for existing reports; epoch, token or other coordinates remain distinct and are labeled accordingly.

Each metric declares `goal: "maximize"`, `"minimize"`, `"target"` or `"observe"`. Target goals also supply a finite `target`; their outcome compares distance from that target. Missing goals and observe-only metrics make no improvement claim. Best-checkpoint labels and baseline outcomes compare point estimates, not statistical significance. Confidence intervals remain available independently.

| Application | Example measured objectives |
| --- | --- |
| Speculative drafter training | Accepted tokens per round or acceptance rate, higher; decoding latency, lower |
| Classification and task fine-tuning | Accuracy, F1, precision, recall or AUROC, higher |
| Language-model pretraining | Validation cross-entropy/loss or perplexity, lower |
| Regression | MAE, MSE or RMSE, lower |
| Speech recognition | Word/character error rate, lower |
| Vision and retrieval | IoU, mAP, Recall@k, MRR or NDCG, higher |
| Generation and summarization | Task-specific BLEU/ROUGE or reported human scores, with an explicit goal |
| Preference optimization and reinforcement learning | Reward or preference accuracy, higher; KL and entropy observed or given an explicit target |
| Efficiency | Throughput, higher; latency or memory, lower, separate from quality |

These are configurable examples, not a universal judgment of model quality. Report the metric definition and raw unit used by the evaluation code. Fractional accuracy and percentage accuracy are not silently converted, and unfamiliar metrics remain observe-only unless configured. Native Trainer/CSV adapters retain validation history and emit known metric goals; `evaluation_metrics` configuration can override a metric's `goal`, `unit`, `label` and `target`.

For a native log reader, a companion JSON configuration can make objectives and baseline identities explicit:

```json
{
  "evaluation_series_id": "lora-r64",
  "evaluation_series_label": "LoRA rank 64",
  "evaluation_cohort_id": "classification-dev-v2",
  "evaluation_context_id": "greedy-v1",
  "evaluation_context_label": "Classification dev v2 · greedy decoding",
  "evaluation_metrics": {
    "accuracy": {"id": "accuracy", "goal": "maximize", "unit": "fraction"},
    "loss": {"id": "cross-entropy", "goal": "minimize"},
    "kl": {"id": "kl", "goal": "target", "target": 0.05}
  }
}
```

Valid explicit cohort/context IDs are preserved so a separately reported baseline can use the same values. `evaluation_metrics.*.id` similarly provides a common metric ID. Default native metric identities retain their validation/test split; descriptive dataset names or settings objects are hashed rather than being treated as measured settings. Baselines must use the matching goal, unit and checkpoint coordinate unit as well.

`cohortId` identifies the dataset/split, and `contextId` identifies comparable evaluation settings such as decoding options, model precision or scoring protocol. `contextLabel` provides a readable description. The chart separates different metric/unit/corpus/settings/goal/coordinate combinations. Missing settings identity is displayed explicitly; a matching corpus alone does not establish identical settings.

The graph uses the reported checkpoint coordinates, a horizontal baseline, series visibility controls and focusable measured points. Its expandable score table provides exact values and an alternative to reading the graph. Optional 95% confidence bounds are retained in the comparison metadata. This contract also works for custom progress documents and code providers without frontend changes.

## Descriptive names

Training code supplies the visible run name through the progress document's `campaign` or `label`, and the stage name through `stages.*[].label`. Stable run IDs identify a job independently of its display name. Configured unit names identify the physical hosts independently of those run names.

Native artifact readers prefer explicit `display_name`, `run_name` or `experiment_name`, then logger/config names, before falling back to the output directory. Live launch arguments and an accompanying `--config` file can supply these values. Manifest `run_name` and `name` are also accepted after `campaign`/`label`.

The dashboard can read explicit names but cannot infer an experiment's scientific purpose from arbitrary training code. Supply a descriptive name in launch metadata or the progress document to avoid directory-name fallbacks.

## Adding a code provider

Place a JavaScript module in `server/training/providers/` and export a provider as its default export. SparkDash loads these modules at startup when the feature is enabled. A new provider needs no frontend changes.

```javascript
export default {
  id: "my-framework",
  async collect({ sparks, execute, runPython, now }) {
    // Read this framework's live job state and return its current runs.
    // Each run uses the shape below.
    return [];
  },
};
```

A run has `id`, `campaign`, optional `model`, and `sources`. Each source has `id`, `sparkId`, `label`, `status`, `updatedAt` in epoch milliseconds, and `stages` containing `data`, `training` and `evaluation` arrays. Provider IDs namespace run IDs automatically. The stage format is the same as the common progress document above.

`sparks` contains the configured host records. `runPython(spark, sourceText, args)` executes a bundled Python probe locally or through the existing SSH connection, with bounded execution and output. `execute` exposes the existing SSH transport. `now()` returns epoch milliseconds. Provider modules are trusted server code and should read monitoring state.

Provider failures are isolated. The module keeps their previous readings as stale while other providers continue polling. It validates and normalizes output before returning it from `GET /api/training-progress`. The API is covered by the dashboard's existing access controls. This module adds no job-control routes.

## Boundaries and limits

The feature polls at ten-second intervals while enabled. Providers do not overlap their own in-flight polls. Readings expire after thirty seconds and no longer establish active training. Each provider's normalized result allows up to 64 runs, 64 sources per run and 128 stages per group, within a one-MiB payload limit. The bundled filesystem probe uses narrower bounds (including 24 discovered runs per host, depth five, 3,000 entries, 48 metadata files and a bounded CSV tail) and skips weight/dependency/cache directories. Live training has priority when selecting bounded results. These limits are monitoring bounds, not guarantees that an arbitrary framework's output will be discovered.

Only confirmed running training establishes the main pane's visibility. Saved and completed results stay available through the API, and evaluation tiles may continue after the main pane hides. Evaluations completed at intermediate checkpoints do not imply final evaluation is done while training is still active. Stale or failed readings remain distinguishable from complete work.

The module displays declared concurrency and actual counters; it does not modify an evaluator's concurrency. Report `promptConcurrency` and `promptsPerMinute` for prompt-based evaluation when measured. Otherwise the module can derive a rolling rate from observed prompt counters with a sufficiently long observation window, scoped to a single job attempt. It does not add replicated distributed counters or average unrelated objectives. Other evaluation units keep their native counters without invented prompt rates.

Custom provider modules are trusted executable server code loaded at startup. Review and install them like any server extension. This is not a sandbox or a web API for uploading code. Providers should expose only the bounded monitoring data needed for the UI and must avoid forwarding secrets, full command environments, prompts or training examples.

`schema.js`, the Python collector's normalization functions and the frontend API types define the accepted version-1 fields. Unknown measurements should be absent or `null`, never healthy zero placeholders. The module reads metadata and does not deserialize model checkpoints, pickle objects or arbitrary YAML constructors.
