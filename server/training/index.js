// Copyright 2026 Maxrio9. SPDX-License-Identifier: Apache-2.0
import { execFile } from "node:child_process";
import { sshExec } from "../collectors/ssh.js";
import { loadProgressProviders } from "./registry.js";
import { normalizeRuns } from "./schema.js";
import { createProgressHealthTracker } from "./health.js";

const intervalMs = 10_000;
const staleAfterMs = 30_000;
const sampleWindowMs = 120_000;
const minimumSampleMs = 20_000;

function shellQuote(value) {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

export function trainingProgressEnabled(env = process.env) {
  return env.TRAINING_PROGRESS_ENABLED === "1";
}

export function createPythonRunner(execute = sshExec) {
  return async (spark, source, args = []) => {
    if (spark.platform === "windows") throw new Error("Training probes require Python on a Unix host");
    if (typeof source !== "string" || !Array.isArray(args) || args.some((arg) => typeof arg !== "string" || arg.includes("\0"))) {
      throw new Error("Invalid training probe");
    }
    if (!spark.isLocal) {
      const output = await execute(spark, `python3 - ${args.map(shellQuote).join(" ")}`, { input: source, timeoutMs: 8_000 });
      if (typeof output !== "string" || Buffer.byteLength(output) > 1024 * 1024) throw new Error("Training probe output too large");
      return output;
    }
    const { HOME, PATH, XDG_RUNTIME_DIR, DBUS_SESSION_BUS_ADDRESS, DOCKER_HOST, DOCKER_CONTEXT } = process.env;
    return new Promise((resolve, reject) => {
      const child = execFile("python3", ["-", ...args], {
        timeout: 8_000, killSignal: "SIGKILL", maxBuffer: 1024 * 1024,
        env: { HOME, PATH, XDG_RUNTIME_DIR, DBUS_SESSION_BUS_ADDRESS, DOCKER_HOST, DOCKER_CONTEXT },
      }, (error, stdout) => error ? reject(error) : resolve(stdout.trim()));
      child.stdin?.on("error", () => {});
      child.stdin?.end(source);
    });
  };
}

export function registerTrainingProgressRoutes(app, {
  enabled = trainingProgressEnabled(), getSparks = () => [], providers,
  discover = loadProgressProviders, execute = sshExec, runPython = createPythonRunner(execute),
  now = Date.now, start = true,
} = {}) {
  const records = new Map();
  const pending = new Set();
  const evaluationSamples = new Map();
  const progressHealth = createProgressHealthTracker();
  let providerList;
  const load = () => providerList ??= Promise.resolve(providers ?? discover());

  function measure(providerId, runs, time) {
    const prefix = `${providerId}/`;
    const measured = new Set();
    for (const run of runs) {
      for (const source of run.sources) {
        source.stages.evaluation = source.stages.evaluation.map(({ attemptId, ...stage }) => {
          if (source.status !== "live" || stage.state !== "running" || stage.unit !== "prompts" || !attemptId || stage.completed === null) return stage;
          const key = `${prefix}${run.id}/${source.id}/${stage.id}`;
          const previous = evaluationSamples.get(key);
          const last = previous?.samples.at(-1);
          const continuous = previous?.attemptId === attemptId && last.time < time && last.completed <= stage.completed;
          const samples = continuous ? previous.samples.filter((sample) => sample.time >= time - sampleWindowMs) : [];
          samples.push({ time, completed: stage.completed });
          measured.add(key);
          evaluationSamples.set(key, { attemptId, samples });
          const first = samples[0];
          const elapsed = time - first.time;
          return elapsed < minimumSampleMs ? stage : { ...stage, promptsPerMinute: 60_000 * (stage.completed - first.completed) / elapsed };
        });
      }
    }
    for (const key of evaluationSamples.keys()) {
      if (key.startsWith(prefix) && !measured.has(key)) evaluationSamples.delete(key);
    }
    return runs;
  }

  async function pollProvider(provider) {
    if (pending.has(provider.id)) return;
    pending.add(provider.id);
    try {
      const value = await provider.collect({ sparks: getSparks(), execute, runPython, now });
      const time = now();
      const normalized = normalizeRuns(value, provider.id, time);
      progressHealth.observe(provider.id, normalized, time);
      const runs = measure(provider.id, normalized, time);
      records.set(provider.id, runs);
    } catch {
      const previous = records.get(provider.id) ?? [];
      records.set(provider.id, previous.map((run) => ({
        ...run, sources: run.sources.map((source) => ({ ...source, status: source.updatedAt === null ? "unavailable" : "stale" })),
      })));
      for (const key of evaluationSamples.keys()) if (key.startsWith(`${provider.id}/`)) evaluationSamples.delete(key);
      progressHealth.resetProvider(provider.id);
    } finally {
      pending.delete(provider.id);
    }
  }

  async function poll() {
    if (!enabled) return;
    try {
      await Promise.all((await load()).map(pollProvider));
    } catch {
      // Discovery failures leave independent hardware monitoring and the API alive.
    }
  }

  function snapshot() {
    const time = now();
    const runs = [...records.values()].flat().map((run) => {
      const sources = run.sources.map((source) => {
        const current = { ...source,
          status: source.status === "live" && (source.updatedAt === null || Math.abs(time - source.updatedAt) > staleAfterMs) ? "stale" : source.status };
        const stages = Object.fromEntries(["data", "training", "evaluation"].map((group) => [group, source.stages[group].map((stage) => {
          const { attemptId, ...publicStage } = stage;
          const health = progressHealth.assess(run.providerId, run, current, group, stage, time);
          if (health) publicStage.health = health;
          if (health && ["stalled", "unresponsive"].includes(health.state)) delete publicStage.etaSeconds;
          return publicStage;
        })]));
        return { ...current, stages };
      });
      const active = sources.some((source) => source.status === "live" && (
        source.stages.training.some((stage) => stage.state === "running") ||
        source.stages.evaluation.some((stage) => stage.state === "running" || stage.state === "waiting")
      ));
      return { ...run, sources, active };
    });
    const first = runs.find((run) => run.active) ?? runs[0];
    return {
      enabled, active: runs.some((run) => run.active), runs,
      campaign: first?.campaign ?? "Training", model: first?.model ?? null, sources: first?.sources ?? [],
    };
  }

  const handler = (_req, res) => {
    res.set("Cache-Control", "no-store");
    res.json(snapshot());
  };
  app.get("/api/training-progress", handler);
  let timer;
  if (enabled && start) {
    void poll();
    timer = setInterval(poll, intervalMs);
    timer.unref();
  }
  return { enabled, poll, snapshot, dispose: () => clearInterval(timer) };
}
