// Copyright 2026 Maxrio9. SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { registerTrainingProgressRoutes } from "../training/index.js";
import { normalizeRuns } from "../training/schema.js";

function stage(fields = {}) {
  return { id: "train", label: "Training", state: "running", completed: 10, total: 100, unit: "steps", etaSeconds: 90, ...fields };
}

function run(id, stages = [stage()], sourceFields = {}) {
  return { id, campaign: id, sources: [{ id: "worker", sparkId: "gpu-a", label: "GPU A", status: "live",
    stages: { data: [], training: stages, evaluation: [] }, ...sourceFields }] };
}

function fixture(value = [run("job")]) {
  let time = 1000000;
  let failing = false;
  const dashboard = registerTrainingProgressRoutes({ get() {} }, { enabled: true, start: false, now: () => time,
    providers: [{ id: "fixture", collect: async () => {
      if (failing) throw new Error("unreachable");
      return structuredClone(value);
    } }] });
  return { ...dashboard, value, get time() { return time; }, advance(ms) { time += ms; }, fail(value) { failing = value; },
    read(runIndex = 0, stageIndex = 0) { return dashboard.snapshot().runs[runIndex].sources[0].stages.training[stageIndex]; } };
}

test("health is not guessed without a reporting policy", async () => {
  const dashboard = fixture();
  await dashboard.poll();
  assert.equal(dashboard.read().health, undefined);
  dashboard.advance(15000);
  assert.equal(dashboard.read().health, undefined);
  assert.equal(dashboard.read().etaSeconds, 90);
});

test("heartbeat and progress age between reads, suppress ETA on trouble and recover on measured progress", async () => {
  const row = stage({ heartbeatAt: 1000000, heartbeatTimeoutSeconds: 3, progressTimeoutSeconds: 5 });
  const dashboard = fixture([run("job", [row])]);
  await dashboard.poll();
  assert.equal(dashboard.read().health.state, "healthy");
  dashboard.advance(3500);
  assert.equal(dashboard.read().health.state, "unresponsive", "health changes without another provider read");
  assert.equal(dashboard.read().etaSeconds, undefined);
  dashboard.advance(2500);
  row.heartbeatAt = dashboard.time;
  await dashboard.poll();
  assert.equal(dashboard.read().health.state, "stalled", "fresh heartbeat is not evidence that counters progressed");
  assert.equal(dashboard.read().etaSeconds, undefined);
  row.completed++;
  dashboard.advance(500);
  row.heartbeatAt = dashboard.time;
  await dashboard.poll();
  assert.equal(dashboard.read().health.state, "healthy");
  assert.equal(dashboard.read().health.lastProgressAt, dashboard.time);
  assert.equal(dashboard.read().etaSeconds, 90);
});

test("waiting stages and explicit waiting activity do not get false stalled warnings", async () => {
  const dashboard = fixture([run("job", [stage({ state: "waiting", lastProgressAt: 900000, progressTimeoutSeconds: 5 }),
    stage({ id: "checkpoint", activity: { kind: "waiting", label: "Waiting for checkpoint transfer" },
      lastProgressAt: 900000, progressTimeoutSeconds: 5 })])]);
  await dashboard.poll();
  assert.equal(dashboard.read(0, 0).health.state, "waiting");
  assert.equal(dashboard.read(0, 1).health.state, "waiting");
  assert.equal(dashboard.read(0, 1).health.detail, "Waiting for checkpoint transfer");
});

test("observed counter or activity progress overrides a lagging explicit progress timestamp after the first read", async () => {
  const row = stage({ lastProgressAt: 900000, progressTimeoutSeconds: 5 });
  const dashboard = fixture([run("job", [row])]);
  await dashboard.poll();
  assert.equal(dashboard.read().health.state, "stalled", "reading an old timestamp does not freshen it");
  row.completed++;
  dashboard.advance(1000);
  await dashboard.poll();
  assert.equal(dashboard.read().health.state, "healthy");
  assert.equal(dashboard.read().health.lastProgressAt, dashboard.time);
  dashboard.advance(6000);
  await dashboard.poll();
  assert.equal(dashboard.read().health.state, "stalled", "unchanged counters cannot freshen the observed floor");
  row.activity = { kind: "saving", label: "Saving checkpoint" };
  await dashboard.poll();
  assert.equal(dashboard.read().health.state, "healthy");
  assert.equal(dashboard.read().health.lastProgressAt, dashboard.time);
});

test("stale collector readings mean unknown health and do not poison fresh retry observations", async () => {
  const dashboard = fixture([run("job", [stage({ progressTimeoutSeconds: 5 })])]);
  await dashboard.poll();
  dashboard.fail(true);
  dashboard.advance(10000);
  await dashboard.poll();
  assert.equal(dashboard.read().health.state, "unknown");
  assert.match(dashboard.read().health.detail, /Source reading/);
  dashboard.fail(false);
  await dashboard.poll();
  assert.equal(dashboard.read().health.state, "healthy");
  dashboard.advance(31000);
  assert.equal(dashboard.read().health.state, "unknown", "an expired source cannot establish a hung worker");
});

test("counter/activity/attempt resets stay scoped to each run and do not leak private attempt IDs", async () => {
  const first = stage({ attemptId: "attempt-1", progressTimeoutSeconds: 5 });
  const second = stage({ attemptId: "attempt-1", progressTimeoutSeconds: 5 });
  const dashboard = fixture([run("first", [first]), run("second", [second])]);
  await dashboard.poll();
  dashboard.advance(6000);
  second.completed++;
  await dashboard.poll();
  assert.equal(dashboard.read(0).health.state, "stalled");
  assert.equal(dashboard.read(1).health.state, "healthy");
  first.attemptId = "attempt-2";
  await dashboard.poll();
  assert.equal(dashboard.read(0).health.state, "healthy");
  assert.equal(dashboard.read(0).attemptId, undefined);
  dashboard.advance(6000);
  first.completed = 1;
  second.activity = { kind: "saving", label: "Saving checkpoint" };
  await dashboard.poll();
  assert.equal(dashboard.read(0).health.state, "healthy");
  assert.equal(dashboard.read(1).health.state, "healthy");
});

test("missing measurement and future timestamps stay unknown instead of healthy forever", async () => {
  const dashboard = fixture([run("job", [
    stage({ progressTimeoutSeconds: 5, completed: null }),
    stage({ id: "heartbeat", heartbeatTimeoutSeconds: 5 }),
    stage({ id: "clock", heartbeatAt: 9999999, heartbeatTimeoutSeconds: 5 }),
    stage({ id: "reported-only", lastProgressAt: 1000000 }),
  ])]);
  await dashboard.poll();
  for (let index = 0; index < 4; index++) assert.equal(dashboard.read(0, index).health.state, "unknown");
  assert.match(dashboard.read(0, 2).health.detail, /clock/);
});

test("resources, activity, dependencies and timing survive validation without accepting claimed health", () => {
  const row = stage({ resources: [{ kind: "cpu", id: "0-15,32-47", label: "CPU cores", scope: "affinity", secret: "removed" }],
    activity: { kind: "saving", label: "Saving checkpoint", detail: "Writing shard 2", since: 1000, secret: "removed" },
    dependencies: [{ id: "checkpoint-10", label: "Checkpoint 10", state: "ready", secret: "removed" }],
    heartbeatAt: 1000, heartbeatTimeoutSeconds: 30, lastProgressAt: 1000, progressTimeoutSeconds: 600,
    health: { state: "healthy", detail: "Untrusted claim" } });
  const result = normalizeRuns([run("job", [row], { resources: [{ kind: "gpu", id: "GPU-uuid", scope: "assigned" }] })], "fixture", 1000)[0];
  const output = result.sources[0].stages.training[0];
  assert.equal(result.sources[0].resources[0].id, "GPU-uuid");
  assert.deepEqual(output.resources[0], { kind: "cpu", id: "0-15,32-47", label: "CPU cores", scope: "affinity" });
  assert.equal(output.activity.secret, undefined);
  assert.equal(output.dependencies[0].secret, undefined);
  assert.equal(output.health, undefined);
  for (const fields of [
    { resources: Array(33).fill({ kind: "gpu", id: "0" }) }, { resources: [{ kind: "gpu", id: "bad\nid" }] },
    { activity: { kind: "bad id", label: "No" } }, { activity: { kind: "saving", label: "No", since: -1 } },
    { dependencies: [{ id: "x", label: "X", state: "complete" }] },
    { dependencies: [{ id: "x", label: "X", state: "ready" }, { id: "x", label: "X", state: "ready" }] },
    { heartbeatAt: NaN }, { lastProgressAt: -1 }, { heartbeatTimeoutSeconds: 0 }, { progressTimeoutSeconds: 604801 },
  ]) assert.throws(() => normalizeRuns([run("job", [stage(fields)])], "fixture", 1000));
});
