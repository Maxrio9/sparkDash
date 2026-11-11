// Copyright 2026 Maxrio9. SPDX-License-Identifier: Apache-2.0
import { readFile } from "node:fs/promises";
import { validId } from "../schema.js";

// Discovery is explicitly scoped by the operator. No home/work tree is scanned
// by default, and process output arguments cannot extend these directories.
function validateRoots(value) {
  if (!Array.isArray(value) || value.length > 32 || value.some((entry) =>
    typeof entry !== "string" || !entry.length || entry.length > 1024 ||
    /[\u0000-\u001f\u007f-\u009f]/.test(entry) || !(entry.startsWith("/") || entry.startsWith("~/")))) {
    throw new Error("Invalid training discovery roots");
  }
  return [...new Set(value)];
}

export function artifactRootsFromEnv(env = process.env) {
  try {
    const roots = validateRoots(JSON.parse(env.TRAINING_PROGRESS_ROOTS || "[]"));
    const byUnit = JSON.parse(env.TRAINING_PROGRESS_ROOTS_BY_UNIT || "{}");
    if (!byUnit || typeof byUnit !== "object" || Array.isArray(byUnit) || Object.keys(byUnit).length > 128) throw new Error();
    const rootsByUnit = Object.fromEntries(Object.entries(byUnit).map(([id, paths]) => {
      if (!validId(id)) throw new Error();
      return [id, validateRoots(paths)];
    }));
    return { roots, rootsByUnit };
  } catch {
    // Never include paths or raw environment values in logs or API errors.
    throw new Error("Invalid training discovery configuration");
  }
}

const probe = new URL("../collectors/artifacts.py", import.meta.url);
const MAX_SOURCES = 64;
const MAX_BYTES = 900 * 1024;

function measurementTime(source) {
  const times = Object.values(source.stages ?? {}).flat().map((stage) => stage.updatedAt ?? 0);
  return Math.max(0, ...times);
}

function hasLiveTraining(run) {
  return run.sources.some((source) => source.status === "live" &&
    source.stages.training.some((stage) => stage.state === "running"));
}

function sourcePriority(source) {
  if (source.status !== "live") return 0;
  if (source.stages.training.some((stage) => stage.state === "running")) return 3;
  return Object.values(source.stages).flat().some((stage) => stage.state === "running") ? 2 : 1;
}

export function createArtifactProvider({ getRoots = artifactRootsFromEnv } = {}) {
  const previous = new Map();
  return {
    id: "artifacts",
    async collect({ sparks, runPython, now }) {
      const { roots, rootsByUnit } = getRoots();
      const pathsFor = (spark) => Object.hasOwn(rootsByUnit, spark.id) ? rootsByUnit[spark.id] : roots;
      const hosts = sparks.filter((spark) => spark.platform !== "windows" && pathsFor(spark).length > 0);
      for (const key of previous.keys()) if (!hosts.some((spark) => spark.id === key)) previous.delete(key);
      if (!hosts.length) return [];
      const source = await readFile(probe, "utf8");
      const results = await Promise.allSettled(hosts.map(async (spark) => {
        const output = await runPython(spark, source, [JSON.stringify({ host: spark.id, roots: pathsFor(spark) })]);
        const runs = JSON.parse(output);
        if (!Array.isArray(runs)) throw new Error("Invalid training discovery result");
        for (const run of runs) {
          for (const entry of run.sources ?? []) {
            const device = spark.name || spark.id;
            const reporter = typeof entry.label === "string" ? entry.label.trim() : "";
            entry.label = (reporter && reporter !== spark.id && reporter !== device ? `${device} · ${reporter}` : device).slice(0, 160);
            entry.sparkId = spark.id;
            entry.updatedAt = now();
          }
        }
        previous.set(spark.id, runs);
        return runs;
      }));
      const all = results.flatMap((result, index) => {
        if (result.status === "fulfilled") return result.value;
        return (previous.get(hosts[index].id) ?? []).map((run) => ({
          ...run, sources: run.sources.map((entry) => ({ ...entry, status: "stale" })),
        }));
      });
      const grouped = new Map();
      for (const run of all) {
        const existing = grouped.get(run.id);
        if (!existing) {
          grouped.set(run.id, { ...run, sources: [...run.sources] });
          continue;
        }
        for (const source of run.sources) {
          const index = existing.sources.findIndex((entry) => entry.id === source.id);
          if (index < 0) existing.sources.push(source);
          else if (measurementTime(source) >= measurementTime(existing.sources[index])) {
            existing.sources[index] = source;
            existing.campaign = run.campaign;
            existing.model = run.model;
          }
        }
      }
      const ordered = [...grouped.values()].sort((left, right) =>
        Number(hasLiveTraining(right)) - Number(hasLiveTraining(left)) ||
        Math.max(...right.sources.map(measurementTime)) - Math.max(...left.sources.map(measurementTime))
      );
      const selected = [];
      let bytes = 2;
      for (const run of ordered) {
        const candidates = [...run.sources].sort((left, right) => sourcePriority(right) - sourcePriority(left)
          || measurementTime(right) - measurementTime(left)).slice(0, MAX_SOURCES);
        const bounded = { ...run, sources: [] };
        for (const source of candidates) {
          const next = { ...bounded, sources: [...bounded.sources, source] };
          if (bytes + Buffer.byteLength(JSON.stringify(next)) + 1 <= MAX_BYTES) bounded.sources.push(source);
        }
        if (bounded.sources.length) {
          selected.push(bounded);
          bytes += Buffer.byteLength(JSON.stringify(bounded)) + 1;
        }
        if (selected.length >= 64) break;
      }
      return selected;
    },
  };
}

export default createArtifactProvider();
