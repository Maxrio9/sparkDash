// Copyright 2026 Maxrio9. SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { registerTrainingProgressRoutes } from "../training/index.js";

const train = () => ({ id: "train", label: "Training", state: "running", completed: 10, total: 100, unit: "steps" });
const evaluate = (completed) => ({ id: "validation", label: "Validation", state: "running", completed, total: 1000,
  unit: "prompts", attemptId: 123, promptConcurrency: 2 });
const source = (id, sparkId, training = [], evaluation = []) => ({ id, sparkId, label: `${sparkId} · ${id}`,
  status: "live", stages: { data: [], training, evaluation } });
const run = (id, sources) => ({ id, campaign: id, model: "Fixture model", sources });

test("same-device, distributed and independent runs retain identities and independent evaluation rates", async () => {
  let time = 1000;
  const local = run("local-loop", [source("trainer", "gpu-a", [train()]), source("evaluator", "gpu-a", [], [evaluate(10)])]);
  const distributed = run("distributed", [source("rank-0", "gpu-a", [train()]), source("rank-1", "gpu-b", [train()]),
    source("evaluator", "gpu-a", [], [evaluate(100)])]);
  const separateA = run("another-method", [source("trainer", "gpu-c", [train()], [evaluate(50)])]);
  const separateB = run("fourth-experiment", [source("trainer", "gpu-d", [train()])]);
  const dashboard = registerTrainingProgressRoutes({ get() {} }, { enabled: true, start: false, now: () => time,
    providers: [{ id: "topology", collect: async () => structuredClone([local, distributed, separateA, separateB]) }] });
  await dashboard.poll();
  let snapshot = dashboard.snapshot();
  assert.equal(snapshot.runs.length, 4);
  assert.equal(snapshot.runs.filter((entry) => entry.active).length, 4);
  assert.equal(snapshot.runs[1].sources.length, 3);
  assert.equal(snapshot.runs[1].sources[0].stages.training[0].completed, 10, "replicated global counters must not be summed");
  assert.equal(snapshot.runs[0].sources[0].sparkId, snapshot.runs[0].sources[1].sparkId);
  time += 20_000;
  local.sources[1].stages.evaluation[0].completed += 2;
  distributed.sources[2].stages.evaluation[0].completed += 7;
  separateA.sources[0].stages.evaluation[0].completed += 4;
  await dashboard.poll();
  snapshot = dashboard.snapshot();
  assert.equal(snapshot.runs[0].sources[1].stages.evaluation[0].promptsPerMinute, 6);
  assert.equal(snapshot.runs[1].sources[2].stages.evaluation[0].promptsPerMinute, 21);
  assert.equal(snapshot.runs[2].sources[0].stages.evaluation[0].promptsPerMinute, 12);
  distributed.sources[1].status = "stale";
  await dashboard.poll();
  snapshot = dashboard.snapshot();
  assert.equal(snapshot.runs[1].active, true, "one unreachable rank cannot invalidate other confirmed workers");
  assert.equal(snapshot.runs[1].sources[1].status, "stale");
  for (const entry of [local, distributed, separateA, separateB]) {
    for (const reporter of entry.sources) for (const stage of reporter.stages.training) stage.state = "complete";
  }
  await dashboard.poll();
  snapshot = dashboard.snapshot();
  assert.equal(snapshot.active, true, "the pane remains visible while final evaluations continue");
  assert.equal(snapshot.runs.filter((entry) => entry.active).length, 3);
  assert.equal(snapshot.runs[0].sources[1].stages.evaluation[0].state, "running");
  for (const entry of [local, distributed, separateA, separateB]) {
    for (const reporter of entry.sources) for (const stage of reporter.stages.evaluation) stage.state = "complete";
  }
  await dashboard.poll();
  assert.equal(dashboard.snapshot().active, false, "the pane closes only after the final evaluation has ended");
});

test("a reporter retry resets only its own rate on a shared device", async () => {
  let time = 1000;
  const jobs = run("shared-device", [source("first", "gpu-a", [train()], [evaluate(10)]),
    source("second", "gpu-a", [], [evaluate(20)])]);
  const dashboard = registerTrainingProgressRoutes({ get() {} }, { enabled: true, start: false, now: () => time,
    providers: [{ id: "topology", collect: async () => structuredClone([jobs]) }] });
  await dashboard.poll();
  time += 20_000;
  jobs.sources[0].stages.evaluation[0].attemptId++;
  jobs.sources[0].stages.evaluation[0].completed++;
  jobs.sources[1].stages.evaluation[0].completed += 5;
  await dashboard.poll();
  const sources = dashboard.snapshot().runs[0].sources;
  assert.equal(sources[0].stages.evaluation[0].promptsPerMinute, undefined);
  assert.equal(sources[1].stages.evaluation[0].promptsPerMinute, 15);
});
