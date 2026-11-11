// Copyright 2026 Maxrio9. SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { createPythonRunner, trainingProgressEnabled, registerTrainingProgressRoutes } from "../training/index.js";
import { loadProgressProviders } from "../training/registry.js";

const empty = { data: [], training: [], evaluation: [] };
const training = { id: "training", label: "Pretraining", state: "running", completed: 1.5, total: 3, unit: "epochs" };
const evaluation = { id: "evaluation", label: "Validation", state: "running", completed: 100, total: 500, unit: "prompts", attemptId: 10, promptConcurrency: 4 };

function run(stages = { ...empty, training: [training] }, fields = {}) {
  return { id: "experiment", campaign: "Any training job", model: null, sources: [{
    id: "worker", label: "Worker", sparkId: "unit-a", kind: "training", status: "live", stages,
  }], ...fields };
}

function fixture(options) {
  const handlers = new Map();
  const runtime = registerTrainingProgressRoutes({ get: (route, callback) => handlers.set(route, callback) }, {
    enabled: true, start: false, providers: [], ...options,
  });
  return { ...runtime, read(route = "/api/training-progress") {
    let body;
    handlers.get(route)({}, {
      set: (key, value) => assert.deepEqual([key, value], ["Cache-Control", "no-store"]),
      json: (value) => { body = JSON.parse(JSON.stringify(value)); },
    });
    return body;
  } };
}

test("disabled module does not discover providers or read hosts", async () => {
  const dashboard = fixture({ enabled: false, providers: undefined,
    discover: () => assert.fail("Disabled discovery"), getSparks: () => assert.fail("Disabled hosts"),
  });
  await dashboard.poll();
  assert.deepEqual(dashboard.read(), { enabled: false, active: false, runs: [], campaign: "Training", model: null, sources: [] });
  assert.equal(trainingProgressEnabled({}), false);
  assert.equal(trainingProgressEnabled({ TRAINING_PROGRESS_ENABLED: "0" }), false);
  assert.equal(trainingProgressEnabled({ TRAINING_PROGRESS_ENABLED: "1" }), true);
});

test("providers accept fractional progress and custom metrics while activity follows live training", async () => {
  let stages = { ...empty, training: [{ ...training, metrics: [{ label: "Tokens/s", value: 1200, unit: "tokens/s" }] }] };
  const dashboard = fixture({ now: () => 1000, providers: [{ id: "custom", collect: async () => [run(stages)] }] });
  await dashboard.poll();
  const result = dashboard.read();
  assert.equal(result.active, true);
  assert.equal(result.runs[0].id, "custom:experiment");
  assert.deepEqual(result.runs[0].sources[0].stages.training[0].metrics, [{ label: "Tokens/s", value: 1200, unit: "tokens/s" }]);
  assert.equal(result.runs[0].sources[0].stages.training[0].completed, 1.5);
  stages = { ...empty, data: [{ ...training, state: "complete" }], evaluation: [evaluation] };
  await dashboard.poll();
  assert.equal(dashboard.read().active, false);
  assert.equal(dashboard.read().runs[0].sources[0].stages.evaluation[0].promptConcurrency, 4);
});

test("one broken provider preserves stale history and cannot hold other training open", async () => {
  let time = 1000;
  let fail = false;
  let live = true;
  const dashboard = fixture({ now: () => time, providers: [
    { id: "first", collect: async () => { if (fail) throw new Error("offline"); return [run()]; } },
    { id: "second", collect: async () => [run({ ...empty, training: [{ ...training, state: live ? "running" : "complete" }] })] },
  ] });
  await dashboard.poll();
  fail = true;
  await dashboard.poll();
  const result = dashboard.read();
  assert.equal(result.runs[0].active, false);
  assert.equal(result.runs[0].sources[0].status, "stale");
  assert.equal(result.runs[1].active, true);
  live = false;
  await dashboard.poll();
  assert.equal(dashboard.read().active, false);
  time += 31_000;
  assert.equal(dashboard.read().runs[1].sources[0].status, "stale");
});

test("a blocked provider is not polled twice and its previous reading expires", async () => {
  let time = 1000;
  let calls = 0;
  let release;
  const dashboard = fixture({ now: () => time, providers: [{ id: "slow", collect: async () => {
    calls++;
    if (calls === 2) await new Promise((resolve) => { release = resolve; });
    return [run()];
  } }] });
  await dashboard.poll();
  const pending = dashboard.poll();
  await Promise.resolve();
  time += 31_000;
  await dashboard.poll();
  assert.equal(calls, 2);
  assert.equal(dashboard.read().active, false);
  assert.equal(dashboard.read().sources[0].status, "stale");
  release();
  await pending;
  assert.equal(dashboard.read().active, true);
});

test("evaluation rates stay scoped to run and attempt and reset after pause or failure", async () => {
  let time = 1000;
  let row = { ...evaluation };
  let fail = false;
  const dashboard = fixture({ now: () => time, providers: [{ id: "evaluation", collect: async () => {
    if (fail) throw new Error("offline");
    return [run({ ...empty, evaluation: [row] })];
  } }] });
  async function read(advance, fields) {
    time += advance;
    row = { ...row, ...fields };
    await dashboard.poll();
    return dashboard.read().runs[0].sources[0].stages.evaluation[0];
  }
  assert.equal((await read(0)).promptsPerMinute, undefined);
  assert.equal((await read(20_000, { completed: 104 })).promptsPerMinute, 12);
  assert.equal((await read(20_000, { completed: 105, attemptId: 11 })).promptsPerMinute, undefined);
  assert.equal((await read(20_000, { completed: 107 })).promptsPerMinute, 6);
  assert.equal((await read(10_000, { state: "waiting" })).promptConcurrency, undefined);
  assert.equal((await read(10_000, { state: "running" })).promptsPerMinute, undefined);
  fail = true;
  await read(10_000);
  fail = false;
  assert.equal((await read(10_000)).promptsPerMinute, undefined);
  assert.equal((await read(20_000)).promptsPerMinute, 0);
  assert.equal((await read(10_000, { completed: 106 })).promptsPerMinute, undefined);
  assert.equal(dashboard.read().runs[0].sources[0].stages.evaluation[0].attemptId, undefined);
});

test("invalid provider data does not leak unknown fields or replace a valid snapshot", async () => {
  let value = run({ ...empty, training: [{ ...training, privateLog: "secret" }] });
  const dashboard = fixture({ providers: [{ id: "custom", collect: async () => [value] }] });
  await dashboard.poll();
  assert.equal(dashboard.read().sources[0].stages.training[0].privateLog, undefined);
  value = run({ ...empty, training: [{ ...training, completed: 8 }] });
  await dashboard.poll();
  assert.equal(dashboard.read().active, false);
  assert.equal(dashboard.read().sources[0].stages.training[0].completed, 1.5);
});

test("trusted provider modules load independently and duplicate ids are rejected", async (t) => {
  const directory = await mkdtemp(path.join(tmpdir(), "training-providers-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  t.mock.method(console, "warn", () => {});
  await writeFile(path.join(directory, "package.json"), '{"type":"module"}');
  await writeFile(path.join(directory, "a.js"), 'export default { id: "ok", collect: async () => [] };');
  await writeFile(path.join(directory, "b.js"), 'throw new Error("broken provider");');
  await writeFile(path.join(directory, "c.js"), 'export default { id: "ok", collect: async () => [] };');
  assert.deepEqual((await loadProgressProviders(directory)).map((provider) => provider.id), ["ok"]);
});

test("probe arguments travel safely with script input on local and remote hosts", async () => {
  const script = 'import json,sys; print(json.dumps(sys.argv[1:]))';
  const argument = "spaces ' and $(literal)";
  const local = await createPythonRunner()({ isLocal: true }, script, [argument]);
  assert.deepEqual(JSON.parse(local), [argument]);
  let observed;
  const remote = createPythonRunner(async (_spark, command, options) => { observed = { command, options }; return "[]"; });
  await remote({ id: "configured" }, script, [argument]);
  assert.equal(observed.options.input, script);
  assert.equal(observed.options.timeoutMs, 8000);
  assert.equal(observed.command, "python3 - 'spaces '\\'' and $(literal)'");
});
