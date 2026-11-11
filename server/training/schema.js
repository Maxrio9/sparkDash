// Copyright 2026 Maxrio9. SPDX-License-Identifier: Apache-2.0
const states = new Set(["complete", "running", "waiting", "failed", "stopped", "unknown"]);
const statuses = new Set(["live", "stale", "unavailable"]);
const controls = /[\u0000-\u001f\u007f-\u009f]/;
const groups = ["data", "training", "evaluation"];

export function validId(value) {
  return typeof value === "string" && /^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,159}$/.test(value);
}

function text(value, length) {
  return typeof value === "string" && value.length <= length && value.trim().length > 0 && !controls.test(value);
}

function count(value) {
  return Number.isFinite(value) && value >= 0 && value <= Number.MAX_SAFE_INTEGER;
}

function timestamp(value) {
  return Number.isSafeInteger(value) && value >= 0 && value <= 8.64e15;
}

function normalizeResources(value) {
  if (value === undefined) return undefined;
  if (!Array.isArray(value) || value.length > 32) throw new Error("Invalid training resources");
  return value.map((resource) => {
    if (!resource || typeof resource !== "object" || Array.isArray(resource)) throw new Error("Invalid training resource");
    const { kind, id, label, scope } = resource;
    if (!["cpu", "gpu", "accelerator"].includes(kind) || !text(id, 160) ||
        (label !== undefined && !text(label, 120)) ||
        (scope !== undefined && !["assigned", "visible", "affinity"].includes(scope))) throw new Error("Invalid training resource");
    return { kind, id, ...(label === undefined ? {} : { label }), ...(scope === undefined ? {} : { scope }) };
  });
}

function normalizeActivity(value) {
  if (value === undefined) return undefined;
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid worker activity");
  const { kind, label, detail, since } = value;
  if (!validId(kind) || !text(label, 120) || (detail !== undefined && !text(detail, 240)) ||
      (since !== undefined && !timestamp(since))) throw new Error("Invalid worker activity");
  return { kind, label, ...(detail === undefined ? {} : { detail }), ...(since === undefined ? {} : { since }) };
}

function normalizeDependencies(value) {
  if (value === undefined) return undefined;
  if (!Array.isArray(value) || value.length > 32) throw new Error("Invalid training dependencies");
  const ids = new Set();
  return value.map((dependency) => {
    if (!dependency || typeof dependency !== "object" || Array.isArray(dependency)) throw new Error("Invalid training dependency");
    const { id, label, state } = dependency;
    if (!validId(id) || ids.has(id) || !text(label, 120) || !["waiting", "ready", "failed", "unknown"].includes(state)) {
      throw new Error("Invalid training dependency");
    }
    ids.add(id);
    return { id, label, state };
  });
}

function normalizeMetrics(metrics) {
  if (metrics === undefined) return undefined;
  if (!Array.isArray(metrics) || metrics.length > 20) throw new Error("Invalid training metrics");
  return metrics.map(({ label, value, unit }) => {
    if (!text(label, 80) || !(value === null || Number.isFinite(value) || text(value, 120)) ||
        (unit !== undefined && !text(unit, 32))) throw new Error("Invalid training metric");
    return { label, value, ...(unit === undefined ? {} : { unit }) };
  });
}

function normalizeComparison(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid evaluation comparison");
  const { metric, metricLabel, seriesId, seriesLabel, checkpoint, baseline, cohortId, lower, upper, unit,
    goal, target, contextId, contextLabel, checkpointUnit } = value;
  if (!validId(metric) || !text(metricLabel, 80) || !validId(seriesId) || !text(seriesLabel, 120) ||
      !Number.isFinite(value.value) || (unit !== undefined && !text(unit, 32)) ||
      (baseline !== undefined && typeof baseline !== "boolean") ||
      (baseline === true ? checkpoint !== undefined : !count(checkpoint)) ||
      (cohortId !== undefined && !validId(cohortId)) ||
      (goal !== undefined && !["maximize", "minimize", "target", "observe"].includes(goal)) ||
      (goal === "target" ? !Number.isFinite(target) : target !== undefined) ||
      (contextId !== undefined && !validId(contextId)) || (contextLabel !== undefined && !text(contextLabel, 120)) ||
      (checkpointUnit !== undefined && !text(checkpointUnit, 32)) ||
      ((lower !== undefined || upper !== undefined) && (!Number.isFinite(lower) || !Number.isFinite(upper) || lower > upper))) {
    throw new Error("Invalid evaluation comparison");
  }
  const comparison = { metric, metricLabel, value: value.value, seriesId, seriesLabel };
  for (const [key, entry] of Object.entries({ checkpoint, baseline, cohortId, lower, upper, unit, goal, target, contextId, contextLabel, checkpointUnit })) {
    if (entry !== undefined) comparison[key] = entry;
  }
  return comparison;
}

function normalizeStages(rows = [], evaluation = false) {
  if (!Array.isArray(rows) || rows.length > 128) throw new Error("Invalid training stages");
  const ids = new Set();
  return rows.map((row) => {
    if (!row || typeof row !== "object" || Array.isArray(row)) throw new Error("Invalid training stage");
    const { id, label, state, detail, updatedAt, loss, etaSeconds, latestCheckpoint, nextCheckpoint } = row;
    const completed = row.completed ?? null;
    const total = row.total ?? null;
    const unit = row.unit ?? null;
    if (!validId(id) || ids.has(id) || !text(label, 120) || !states.has(state) ||
        (completed !== null && !count(completed)) || (total !== null && !count(total)) ||
        (completed !== null && total !== null && completed > total) ||
        (unit !== null && !text(unit, 32)) || (detail !== undefined && !text(detail, 240)) ||
        (updatedAt !== undefined && !timestamp(updatedAt)) || (loss !== undefined && !Number.isFinite(loss)) ||
        (etaSeconds !== undefined && !count(etaSeconds)) ||
        (latestCheckpoint !== undefined && !count(latestCheckpoint)) ||
        (nextCheckpoint !== undefined && !count(nextCheckpoint))) throw new Error("Invalid training stage");
    ids.add(id);
    const stage = { id, label, state, completed, total, unit };
    for (const [key, value] of Object.entries({ detail, updatedAt, loss, etaSeconds, latestCheckpoint, nextCheckpoint })) {
      if (value !== undefined) stage[key] = value;
    }
    const metrics = normalizeMetrics(row.metrics);
    if (metrics !== undefined) stage.metrics = metrics;
    const resources = normalizeResources(row.resources);
    const activity = normalizeActivity(row.activity);
    const dependencies = normalizeDependencies(row.dependencies);
    for (const [key, value] of Object.entries({ resources, activity, dependencies })) {
      if (value !== undefined) stage[key] = value;
    }
    for (const key of ["heartbeatAt", "lastProgressAt"]) {
      if (row[key] !== undefined) {
        if (!timestamp(row[key])) throw new Error("Invalid training health timestamp");
        stage[key] = row[key];
      }
    }
    for (const key of ["heartbeatTimeoutSeconds", "progressTimeoutSeconds"]) {
      if (row[key] !== undefined) {
        if (!Number.isFinite(row[key]) || row[key] <= 0 || row[key] > 604800) throw new Error("Invalid training health timeout");
        stage[key] = row[key];
      }
    }
    if (row.comparison !== undefined) {
      if (!evaluation || state !== "complete") throw new Error("Comparison requires a completed evaluation");
      stage.comparison = normalizeComparison(row.comparison);
    }
    if (row.comparisons !== undefined) {
      if (!evaluation || state !== "complete" || !Array.isArray(row.comparisons) || row.comparisons.length > 20) {
        throw new Error("Comparisons require a completed evaluation and at most 20 metrics");
      }
      stage.comparisons = row.comparisons.map(normalizeComparison);
    }
    if (row.attemptId !== undefined && state === "running") {
      if (!(Number.isSafeInteger(row.attemptId) && row.attemptId > 0) && !text(row.attemptId, 160)) throw new Error("Invalid training attempt");
      stage.attemptId = row.attemptId;
    }
    if (evaluation && state === "running") {
      const { promptConcurrency, promptsPerMinute, attemptId } = row;
      if ((promptConcurrency !== undefined && (!Number.isSafeInteger(promptConcurrency) || promptConcurrency < 1)) ||
          (promptsPerMinute !== undefined && !count(promptsPerMinute)) ||
          (attemptId !== undefined && !(Number.isSafeInteger(attemptId) && attemptId > 0) && !text(attemptId, 160))) {
        throw new Error("Invalid evaluation telemetry");
      }
      for (const [key, value] of Object.entries({ promptConcurrency, promptsPerMinute, attemptId })) {
        if (value !== undefined) stage[key] = value;
      }
    }
    return stage;
  });
}

export function normalizeRuns(value, providerId, now) {
  if (!Array.isArray(value) || value.length > 64 || Buffer.byteLength(JSON.stringify(value)) > 1024 * 1024) {
    throw new Error("Invalid training runs");
  }
  const ids = new Set();
  return value.map((run) => {
    if (!run || typeof run !== "object" || Array.isArray(run) || !validId(run.id) || ids.has(run.id) ||
        !text(run.campaign, 160) || (run.model != null && !text(run.model, 160)) ||
        !Array.isArray(run.sources) || run.sources.length > 64) throw new Error("Invalid training run");
    ids.add(run.id);
    const sourceIds = new Set();
    const sources = run.sources.map((source) => {
      if (!source || typeof source !== "object" || Array.isArray(source)) throw new Error("Invalid training source");
      const { id, label } = source;
      const sparkId = source.sparkId ?? id;
      const kind = source.kind ?? "training";
      const status = source.status ?? "live";
      const updatedAt = source.updatedAt ?? (status === "live" ? now : null);
      if (!validId(id) || sourceIds.has(id) || !text(label, 160) || !validId(sparkId) ||
          !["desktop", "training"].includes(kind) || !statuses.has(status) ||
          (updatedAt !== null && !timestamp(updatedAt)) || !source.stages || typeof source.stages !== "object") {
        throw new Error("Invalid training source");
      }
      sourceIds.add(id);
      const stages = Object.fromEntries(groups.map((group) => [group, normalizeStages(source.stages[group], group === "evaluation")]));
      const resources = normalizeResources(source.resources);
      return { id, label, sparkId, kind, status, updatedAt, stages, ...(resources === undefined ? {} : { resources }) };
    });
    return { id: `${providerId}:${run.id}`, providerId, campaign: run.campaign, model: run.model ?? null, sources };
  });
}
