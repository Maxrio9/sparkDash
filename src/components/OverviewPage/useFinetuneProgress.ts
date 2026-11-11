import { useEffect, useMemo, useState } from "react";
import { fetchTrainingProgress } from "../../api/client";
import type { FinetuneProgress, FinetuneStage, TrainingRun } from "../../api/types";

export type HostEvaluation = Pick<FinetuneStage, "label" | "promptConcurrency" | "promptsPerMinute"> & {
  model: string;
  id?: string;
  runId?: string;
  sourceId?: string;
  sourceLabel?: string;
  campaign?: string;
  completed?: number | null;
  total?: number | null;
  unit?: string | null;
  metrics?: FinetuneStage["metrics"];
  resources?: FinetuneStage["resources"];
  activity?: FinetuneStage["activity"];
  dependencies?: FinetuneStage["dependencies"];
  health?: FinetuneStage["health"];
};

export function hasLiveTraining(run: TrainingRun) {
  return run.sources.some((source) => source.status === "live" &&
    source.stages.training.some((stage) => stage.state === "running"));
}

export function trainingRuns(snapshot: FinetuneProgress | null): TrainingRun[] {
  if (!snapshot?.enabled) return [];
  if (snapshot.runs) return snapshot.runs;
  const run = { id: "legacy", campaign: snapshot.campaign, model: snapshot.model, sources: snapshot.sources, active: false };
  return [{ ...run, active: hasLiveTraining(run) }];
}

/** A physical device can host several reporters and independent evaluations. */
export function runningEvaluationsBySpark(snapshot: FinetuneProgress | null): Record<string, HostEvaluation[]> {
  const evaluations: Record<string, HostEvaluation[]> = {};
  const seen = new Set<string>();
  for (const run of trainingRuns(snapshot)) {
    for (const source of run.sources) {
      if (source.status !== "live") continue;
      const sparkId = source.sparkId ?? source.id;
      for (const stage of source.stages.evaluation) {
        if (stage.state !== "running") continue;
        const id = JSON.stringify([run.id, source.id, stage.id]);
        if (seen.has(id)) continue;
        seen.add(id);
        const { label, promptConcurrency, promptsPerMinute, completed, total, unit, metrics, activity, dependencies, health } = stage;
        (evaluations[sparkId] ??= []).push({ id, runId: run.id, sourceId: source.id, sourceLabel: source.label,
          campaign: run.campaign, model: run.model ?? run.campaign,
          label, promptConcurrency, promptsPerMinute, completed, total, unit, metrics, activity, dependencies, health,
          resources: stage.resources ?? source.resources });
      }
    }
  }
  return evaluations;
}

export function useFinetuneProgress(enabled: boolean) {
  const [snapshot, setSnapshot] = useState<FinetuneProgress | null>(null);
  const [requestFailed, setRequestFailed] = useState(false);

  useEffect(() => {
    if (!enabled) {
      setSnapshot(null);
      setRequestFailed(false);
      return;
    }
    let request: AbortController | null = null;
    const refresh = async () => {
      if (request) return;
      const controller = new AbortController();
      request = controller;
      try {
        const next = await fetchTrainingProgress(controller.signal);
        if (controller.signal.aborted) return;
        setSnapshot(next);
        setRequestFailed(false);
      } catch {
        if (!controller.signal.aborted) setRequestFailed(true);
      } finally {
        request = null;
      }
    };
    const poll = () => {
      if (!document.hidden) void refresh();
    };
    void refresh();
    const timer = window.setInterval(poll, 10_000);
    document.addEventListener("visibilitychange", poll);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", poll);
      request?.abort();
    };
  }, [enabled]);

  const evaluationsBySpark = useMemo(() => enabled && !requestFailed ? runningEvaluationsBySpark(snapshot) : {},
    [enabled, requestFailed, snapshot]);

  return { snapshot, requestFailed, evaluationsBySpark };
}
