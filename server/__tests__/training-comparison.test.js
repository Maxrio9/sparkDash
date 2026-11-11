// Copyright 2026 Maxrio9. SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { normalizeRuns } from "../training/schema.js";

const score = { metric: "accuracy", metricLabel: "Accuracy", value: 0, unit: "%", seriesId: "lora", seriesLabel: "LoRA", cohortId: "validation-v1" };
function normalize(comparison, stage = {}) {
  return normalizeRuns([{ id: "run", campaign: "Training", sources: [{ id: "worker", label: "Worker", stages: {
    data: [], training: [], evaluation: [{ id: "evaluation", label: "Evaluation", state: "complete", comparison, ...stage }],
  } }] }], "fixture", 1000)[0].sources[0].stages.evaluation[0].comparison;
}

test("evaluation comparison preserves zero baseline scores and strips unknown fields", () => {
  assert.deepEqual(normalize({ ...score, baseline: true, privateReport: "secret" }), { ...score, baseline: true });
});

test("evaluation comparison preserves checkpoint zero, separate cohorts and confidence intervals", () => {
  assert.deepEqual(normalize({ ...score, checkpoint: 0, lower: -1, upper: 1 }), { ...score, checkpoint: 0, lower: -1, upper: 1 });
  assert.equal(normalize({ ...score, checkpoint: 10, cohortId: "different-validation" }).cohortId, "different-validation");
});

test("evaluation comparisons reject invalid scores, ambiguous coordinates and malformed bounds", () => {
  for (const fields of [
    { value: NaN }, { value: true }, { value: Infinity }, { checkpoint: -1 }, { checkpoint: null },
    { baseline: true, checkpoint: 0 }, { baseline: "yes" }, { lower: 0 }, { lower: 1, upper: 0 },
    { lower: NaN, upper: 1 }, { seriesId: "bad id" }, { metricLabel: "x".repeat(81) }, { cohortId: "bad/id" },
  ]) assert.throws(() => normalize({ ...score, checkpoint: 10, ...fields }), /Invalid evaluation comparison/);
  assert.throws(() => normalize(score), /Invalid evaluation comparison/);
});

test("partial evaluation results cannot enter checkpoint comparisons", () => {
  assert.throws(() => normalize({ ...score, checkpoint: 10 }, { state: "running" }), /completed evaluation/);
});

test("multiple objectives retain direction, targets, settings identity and checkpoint coordinate units", () => {
  const accuracy = { ...score, checkpoint: 1, checkpointUnit: "epochs", goal: "maximize", contextId: "dataset-v1-temp0", contextLabel: "Dev split · greedy decoding" };
  const loss = { ...accuracy, metric: "loss", metricLabel: "Loss", goal: "minimize", value: 1.2 };
  const target = { ...accuracy, metric: "kl", metricLabel: "KL", goal: "target", target: 0 };
  const value = normalizeRuns([{ id: "job", campaign: "Job", sources: [{ id: "worker", label: "Worker", stages: {
    data: [], training: [], evaluation: [{ id: "validation", label: "Validation", state: "complete", comparisons: [accuracy, loss, target] }],
  } }] }], "fixture", 1000)[0].sources[0].stages.evaluation[0];
  assert.deepEqual(value.comparisons, [accuracy, loss, target]);
  for (const changes of [{ goal: "higher" }, { goal: "target" }, { goal: "minimize", target: 0 },
    { target: 0 }, { goal: "target", target: NaN }, { contextId: "bad/id" }, { contextLabel: "x".repeat(121) }, { checkpointUnit: "x".repeat(33) }]) {
    assert.throws(() => normalize({ ...score, checkpoint: 10, ...changes }), /Invalid evaluation comparison/);
  }
  assert.throws(() => normalize(score, { comparison: undefined, comparisons: Array(21).fill(accuracy) }), /at most 20/);
  assert.throws(() => normalize(score, { comparison: undefined, comparisons: [accuracy], state: "running" }), /completed evaluation/);
});
