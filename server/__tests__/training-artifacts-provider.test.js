// Copyright 2026 Maxrio9. SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { createArtifactProvider, artifactRootsFromEnv } from "../training/providers/artifacts.js";

const fixtureProvider = () => createArtifactProvider({ getRoots: () => ({ roots: ["~/training-runs"], rootsByUnit: {} }) });

function run(id, host, state, updatedAt) {
  return {
    id, campaign: id, model: null,
    sources: [{ id: host, label: host, sparkId: host, kind: "training", status: "live", updatedAt,
      stages: { data: [], training: [{ id: "training", label: "Training", state, completed: 1, total: 10, unit: "steps", updatedAt }], evaluation: [] } }],
  };
}

test("artifact provider deduplicates copied manifests by newest measurement", async () => {
  const provider = fixtureProvider();
  const runs = await provider.collect({
    sparks: [{ id: "worker", name: "Worker" }], now: () => 5000,
    runPython: async () => JSON.stringify([run("same-job", "worker", "running", 1000), run("same-job", "worker", "complete", 2000)]),
  });
  assert.equal(runs.length, 1);
  assert.equal(runs[0].sources.length, 1);
  assert.equal(runs[0].sources[0].stages.training[0].state, "complete");
});

test("artifact provider respects global run bound and prioritizes live training", async () => {
  const provider = fixtureProvider();
  const sparks = Array.from({ length: 100 }, (_, i) => ({ id: `worker-${i}` }));
  const runs = await provider.collect({
    sparks, now: () => 5000,
    runPython: async (spark) => {
      const i = Number(spark.id.split("-")[1]);
      return JSON.stringify([run(`job-${i}`, spark.id, i >= 90 ? "running" : "complete", i)]);
    },
  });
  assert.equal(runs.length, 64);
  assert.ok(runs.slice(0, 10).every((entry) => entry.sources[0].stages.training[0].state === "running"));
  assert.equal(new Set(runs.map((entry) => entry.id)).size, 64);
});

test("colocated reporters keep separate sources and readable device/job labels", async () => {
  const provider = fixtureProvider();
  const trainer = run("pipeline", "trainer", "running", 1000);
  trainer.sources[0].label = "Trainer";
  const evaluator = run("pipeline", "evaluator", "complete", 1000);
  evaluator.sources[0].label = "Evaluator";
  evaluator.sources[0].stages.evaluation = evaluator.sources[0].stages.training;
  evaluator.sources[0].stages.training = [];
  const runs = await provider.collect({
    sparks: [{ id: "device", name: "One GPU host" }], now: () => 5000,
    runPython: async () => JSON.stringify([trainer, evaluator]),
  });
  assert.equal(runs.length, 1);
  assert.equal(runs[0].sources.length, 2);
  assert.deepEqual(runs[0].sources.map((source) => source.label), ["One GPU host · Trainer", "One GPU host · Evaluator"]);
  assert.deepEqual(runs[0].sources.map((source) => source.sparkId), ["device", "device"]);
});

test("distributed and unrelated runs remain separate while a failed host becomes stale", async () => {
  const provider = fixtureProvider();
  const sparks = Array.from({ length: 4 }, (_, i) => ({ id: `spark-${i}`, name: `Spark ${i}` }));
  let failed = false;
  const collect = (at) => provider.collect({ sparks, now: () => at, runPython: async (spark) => {
    if (failed && spark.id === "spark-0") throw new Error("unreachable");
    return JSON.stringify([run("shared-distributed", `${spark.id}-rank`, "running", at), run(`independent-${spark.id}`, `${spark.id}-job`, "running", at)]);
  } });
  assert.equal((await collect(1000)).length, 5);
  failed = true;
  const runs = await collect(2000);
  assert.equal(runs.length, 5);
  const shared = runs.find((entry) => entry.id === "shared-distributed");
  assert.equal(shared.sources.length, 4);
  const stale = shared.sources.find((source) => source.sparkId === "spark-0");
  assert.equal(stale.status, "stale");
  assert.equal(stale.updatedAt, 1000);
  assert.equal(shared.sources.filter((source) => source.status === "live").length, 3);
});

test("oversized distributed runs preserve active reporters within source and payload bounds", async () => {
  const provider = fixtureProvider();
  const sparks = Array.from({ length: 100 }, (_, i) => ({ id: `spark-${i}` }));
  const runs = await provider.collect({ sparks, now: () => 10000, runPython: async (spark) => {
    const number = Number(spark.id.split("-")[1]);
    return JSON.stringify([run("distributed", `${spark.id}-rank`, number >= 90 ? "running" : "complete", number)]);
  } });
  assert.equal(runs.length, 1);
  assert.equal(runs[0].sources.length, 64);
  assert.ok(runs[0].sources.slice(0, 10).every((source) => source.stages.training[0].state === "running"));
  assert.ok(Buffer.byteLength(JSON.stringify(runs)) < 1024 * 1024);
});

test("final evaluation queues retain priority over completed runs and reporters at discovery bounds", async () => {
  for (const distributed of [true, false]) {
    const provider = fixtureProvider();
    const sparks = Array.from({ length: 100 }, (_, i) => ({ id: `worker-${i}` }));
    const runs = await provider.collect({ sparks, now: () => 10000, runPython: async (spark) => {
      const number = Number(spark.id.split("-")[1]);
      const entry = run(distributed ? "distributed" : `job-${number}`, spark.id, "complete", number);
      entry.sources[0].stages.evaluation = [{ id: "validation", label: "Final validation",
        state: number < 2 ? number === 0 ? "waiting" : "running" : "complete", updatedAt: number }];
      return JSON.stringify([entry]);
    } });
    const selected = distributed ? runs[0].sources : runs.map((entry) => entry.sources[0]);
    assert.equal(selected.length, 64);
    assert.deepEqual(new Set(selected.slice(0, 2).map((source) => source.stages.evaluation[0].state)), new Set(["waiting", "running"]));
    assert.ok(Buffer.byteLength(JSON.stringify(runs)) < 1024 * 1024);
  }
});

test("empty default roots perform no filesystem or host probes", async () => {
  const provider = createArtifactProvider({ getRoots: () => artifactRootsFromEnv({}) });
  assert.deepEqual(await provider.collect({ sparks: [{ id: "unit-a" }], now: Date.now,
    runPython: async () => assert.fail("No roots configured"),
  }), []);
});

test("explicit roots are bounded and per-unit empty overrides disable discovery", async () => {
  const settings = artifactRootsFromEnv({ TRAINING_PROGRESS_ROOTS: '["~/runs","~/runs"]',
    TRAINING_PROGRESS_ROOTS_BY_UNIT: '{"unit-b":[],"unit-c":["/srv/experiment"]}' });
  assert.deepEqual(settings, { roots: ["~/runs"], rootsByUnit: { "unit-b": [], "unit-c": ["/srv/experiment"] } });
  const provider = createArtifactProvider({ getRoots: () => settings });
  const observed = [];
  await provider.collect({ sparks: [{ id: "unit-a" }, { id: "unit-b" }, { id: "unit-c" }, { id: "windows", platform: "windows" }],
    now: Date.now, runPython: async (unit, _source, args) => { observed.push(JSON.parse(args[0])); return "[]"; } });
  assert.deepEqual(observed, [{ host: "unit-a", roots: ["~/runs"] }, { host: "unit-c", roots: ["/srv/experiment"] }]);
  for (const env of [
    { TRAINING_PROGRESS_ROOTS: 'bad' }, { TRAINING_PROGRESS_ROOTS: '["relative/path"]' },
    { TRAINING_PROGRESS_ROOTS: '["/secret\\npath"]' }, { TRAINING_PROGRESS_ROOTS: JSON.stringify(Array(33).fill("/tmp")) },
    { TRAINING_PROGRESS_ROOTS_BY_UNIT: '[]' }, { TRAINING_PROGRESS_ROOTS_BY_UNIT: '{"invalid id":[]}' },
  ]) assert.throws(() => artifactRootsFromEnv(env), /^Error: Invalid training discovery configuration$/);
});
