// Copyright 2026 Maxrio9. SPDX-License-Identifier: Apache-2.0
const groups = ["data", "training", "evaluation"];
const clockToleranceMs = 5000;

function keyFor(providerId, run, source, group, stage) {
  return JSON.stringify([providerId, run.id, source.id, group, stage.id]);
}

function activityIdentity(activity) {
  return activity ? JSON.stringify([activity.kind, activity.label, activity.since ?? null]) : null;
}

/** Observe progress, without treating a successful filesystem read as a heartbeat. */
export function createProgressHealthTracker() {
  const observations = new Map();

  function resetProvider(providerId) {
    for (const [key, value] of observations) if (value.providerId === providerId) observations.delete(key);
  }

  function observe(providerId, runs, time) {
    const seen = new Set();
    for (const run of runs) for (const source of run.sources) for (const group of groups) for (const stage of source.stages[group]) {
      if (source.status !== "live" || stage.state !== "running") continue;
      const key = keyFor(providerId, run, source, group, stage);
      seen.add(key);
      const previous = observations.get(key);
      const activity = activityIdentity(stage.activity);
      const restarted = previous && (previous.attemptId !== stage.attemptId || previous.unit !== stage.unit ||
        (previous.completed !== null && stage.completed !== null && stage.completed < previous.completed));
      const changed = !previous || restarted || activity !== previous.activity ||
        (stage.completed !== null && (previous.completed === null || stage.completed > previous.completed));
      // A publisher timestamp can lag its counters. Preserve it on the first
      // read, but do not ignore progress we actually observe after that read.
      const minimumProgressAt = previous && changed ? time : previous?.minimumProgressAt ?? 0;
      const inferredProgressAt = changed ? time : previous.inferredProgressAt;
      const lastProgressAt = stage.lastProgressAt === undefined ? inferredProgressAt : Math.max(stage.lastProgressAt, minimumProgressAt);
      observations.set(key, { providerId, completed: stage.completed, unit: stage.unit, attemptId: stage.attemptId,
        activity, minimumProgressAt, inferredProgressAt, lastProgressAt,
        measurable: stage.completed !== null || stage.activity !== undefined || stage.lastProgressAt !== undefined });
    }
    for (const [key, value] of observations) if (value.providerId === providerId && !seen.has(key)) observations.delete(key);
  }

  function assess(providerId, run, source, group, stage, time) {
    if (!["running", "waiting"].includes(stage.state)) return undefined;
    const configured = stage.heartbeatAt !== undefined || stage.lastProgressAt !== undefined ||
      stage.heartbeatTimeoutSeconds !== undefined || stage.progressTimeoutSeconds !== undefined;
    if (!configured) return undefined;
    const observed = observations.get(keyFor(providerId, run, source, group, stage));
    const lastProgressAt = observed?.lastProgressAt ?? stage.lastProgressAt;
    const times = { ...(lastProgressAt === undefined ? {} : { lastProgressAt }),
      ...(stage.heartbeatAt === undefined ? {} : { heartbeatAt: stage.heartbeatAt }) };
    const result = (state, detail) => ({ state, detail, ...times });
    if (source.status !== "live") return result("unknown", "Source reading is unavailable; worker health is unknown.");
    if ([stage.heartbeatAt, lastProgressAt].some((value) => value !== undefined && value - time > clockToleranceMs)) {
      return result("unknown", "Worker timestamps are ahead of the dashboard clock.");
    }
    if (stage.heartbeatTimeoutSeconds !== undefined) {
      if (stage.heartbeatAt === undefined) return result("unknown", "The configured worker heartbeat has not been reported.");
      if (time - stage.heartbeatAt > stage.heartbeatTimeoutSeconds * 1000) return result("unresponsive", "Worker heartbeat is overdue.");
    }
    if (stage.state === "waiting" || ["waiting", "queued", "blocked"].includes(stage.activity?.kind)) {
      return result("waiting", stage.activity?.label ?? "Waiting for work or dependencies.");
    }
    if (stage.progressTimeoutSeconds !== undefined) {
      if (!observed?.measurable || lastProgressAt === undefined) return result("unknown", "Progress updates have not been reported.");
      if (time - lastProgressAt > stage.progressTimeoutSeconds * 1000) return result("stalled", "No progress within the reported update interval.");
    }
    if (stage.heartbeatTimeoutSeconds === undefined && stage.progressTimeoutSeconds === undefined) {
      return result("unknown", "Update timing is reported without an expected update interval.");
    }
    return result("healthy", "Updates are within the reported intervals.");
  }

  return { observe, assess, resetProvider };
}
