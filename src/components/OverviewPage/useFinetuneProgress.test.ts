import { describe, expect, it } from "vitest";
import type { FinetuneProgress, FinetuneSource, FinetuneStage, TrainingRun } from "../../api/types";
import { hasLiveTrainingOrEvaluation, runningEvaluationsBySpark, trainingRuns } from "./useFinetuneProgress";

function stage(id: string, state: FinetuneStage["state"] = "running", fields: Partial<FinetuneStage> = {}): FinetuneStage {
  return { id, label: "Validation", state, completed: 20, total: 100, unit: "prompts", ...fields };
}

function source(id: string, sparkId: string, stages: FinetuneStage[], fields: Partial<FinetuneSource> = {}): FinetuneSource {
  return { id, sparkId, label: id, kind: "training", status: "live", updatedAt: 1000,
    stages: { data: [], training: [], evaluation: stages }, ...fields };
}

function snapshot(runs: TrainingRun[]): FinetuneProgress {
  return { enabled: true, active: runs.some((run) => run.active), campaign: "Legacy", model: null, sources: [], runs };
}

function run(id: string, sources: FinetuneSource[], active = true): TrainingRun {
  return { id, campaign: `Experiment ${id}`, model: `Model ${id}`, active, sources };
}

describe("training run visibility", () => {
  it.each(["running", "waiting"] as const)("retains %s final evaluation after the trainer completes", (state) => {
    const worker = source("same-host", "gpu-host", [stage("final", state)]);
    worker.stages.training = [stage("train", "complete")];
    expect(hasLiveTrainingOrEvaluation(run("same-device", [worker], false))).toBe(true);
    const legacy: FinetuneProgress = { enabled: true, campaign: "Legacy evaluation", model: null, sources: [worker] };
    expect(trainingRuns(legacy)[0].active).toBe(true);
  });

  it.each(["complete", "failed", "stopped", "unknown"] as const)("does not keep a completed trainer active for %s evaluation", (state) => {
    const trainer = source("trainer", "training-host", []);
    trainer.stages.training = [stage("train", "complete")];
    expect(hasLiveTrainingOrEvaluation(run("distributed", [trainer, source("eval", "evaluation-host", [stage("final", state)])]))).toBe(false);
  });

  it.each(["stale", "unavailable"] as const)("does not treat %s evaluation snapshots as active work", (status) => {
    expect(hasLiveTrainingOrEvaluation(run("cached", [source("eval", "gpu-host", [stage("running"), stage("queued", "waiting")], { status })]))).toBe(false);
  });
});

describe("running evaluations by physical device", () => {
  it("retains independent runs and reporters with the same stage names on one training device", () => {
    const colocated = source("trainer", "spark-a", [stage("validation", "running", { promptsPerMinute: 8, promptConcurrency: 2 })]);
    colocated.stages.training = [stage("train", "running", { unit: "steps" })];
    const result = runningEvaluationsBySpark(snapshot([
      run("sft", [colocated, source("evaluator", "spark-a", [stage("validation", "running", { promptsPerMinute: 4, promptConcurrency: 1 })])]),
      run("dpo", [source("trainer", "spark-a", [stage("validation", "running", { promptsPerMinute: 6, promptConcurrency: 4 })])]),
    ]));
    expect(Object.keys(result)).toEqual(["spark-a"]);
    expect(result["spark-a"]).toHaveLength(3);
    expect(new Set(result["spark-a"].map((job) => job.id)).size).toBe(3);
    expect(result["spark-a"].map((job) => [job.campaign, job.sourceId, job.promptsPerMinute, job.promptConcurrency])).toEqual([
      ["Experiment sft", "trainer", 8, 2], ["Experiment sft", "evaluator", 4, 1], ["Experiment dpo", "trainer", 6, 4],
    ]);
  });

  it("routes distributed evaluation reporters to their own devices and excludes stale sources", () => {
    const result = runningEvaluationsBySpark(snapshot([
      run("distributed", [
        source("rank-0", "spark-a", [stage("eval")]),
        source("rank-1", "spark-b", [stage("eval")]),
        source("old-worker", "spark-c", [stage("eval")], { status: "stale" }),
        source("missing-worker", "spark-d", [stage("eval")], { status: "unavailable" }),
      ]),
      run("other", [source("other", "spark-b", [stage("done", "complete"), stage("queued", "waiting")])]),
    ]));
    expect(Object.keys(result)).toEqual(["spark-a", "spark-b"]);
    expect(result["spark-a"][0].sourceId).toBe("rank-0");
    expect(result["spark-b"][0].sourceId).toBe("rank-1");
  });

  it("keeps final evaluation visible after training stops and preserves arbitrary units and metrics", () => {
    const result = runningEvaluationsBySpark(snapshot([run("vision", [source("vision-evaluator", "gpu-host", [
      stage("test", "running", { completed: 2.5, total: 10, unit: "batches", metrics: [{ label: "Images per second", value: 54, unit: "images/s" }] }),
    ])], false)]));
    expect(result["gpu-host"][0]).toMatchObject({ campaign: "Experiment vision", completed: 2.5, total: 10, unit: "batches",
      metrics: [{ label: "Images per second", value: 54, unit: "images/s" }] });
  });

  it("supports legacy physical source IDs and disabled or missing snapshots", () => {
    const legacySource = source("legacy-host", "unused", [stage("eval")]);
    delete legacySource.sparkId;
    const legacy: FinetuneProgress = { enabled: true, campaign: "Old campaign", model: null, sources: [legacySource] };
    expect(runningEvaluationsBySpark(legacy)["legacy-host"][0].model).toBe("Old campaign");
    expect(runningEvaluationsBySpark({ ...legacy, enabled: false })).toEqual({});
    expect(runningEvaluationsBySpark(null)).toEqual({});
  });
});
