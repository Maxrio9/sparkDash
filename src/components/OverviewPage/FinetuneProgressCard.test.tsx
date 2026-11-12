import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { fetchTrainingProgress } from "../../api/client";
import type { FinetuneProgress, FinetuneStage, TrainingRun } from "../../api/types";
import { makeSpark } from "../../testing/fixtures";
import { cleanupRenders, flush, render } from "../../testing/render";
import { OverviewPage } from "./OverviewPage";
import { FinetuneProgressCard } from "./FinetuneProgressCard";

vi.mock("../../api/client", async (importOriginal) => ({
  ...await importOriginal<typeof import("../../api/client")>(),
  fetchTrainingProgress: vi.fn(),
}));

function stage(id: string, label: string, state: FinetuneStage["state"], fields: Partial<FinetuneStage> = {}) {
  return { id, label, state, completed: null, total: null, unit: null, ...fields };
}

const snapshot: FinetuneProgress = {
  enabled: true,
  campaign: "Text classification",
  model: "Example-7B",
  sources: [
    {
      id: "desktop", label: "Desktop", kind: "desktop", status: "live", updatedAt: 1794356160000,
      stages: {
        data: [
          stage("corpus", "Training corpus", "complete", { completed: 800, unit: "prompts" }),
          stage("dataset", "Dataset packaged", "complete", { completed: 800, unit: "prompts" }),
        ],
        training: [],
        evaluation: [
          stage("main-1324", "Main checkpoint 1324", "running", {
            completed: 462, total: 810, unit: "prompts", updatedAt: 1794356160000,
            promptsPerMinute: 6.2, promptConcurrency: 1,
          }),
          stage("ce-1986", "PR checkpoint 1986", "waiting"),
          stage("preference-1986", "Preference objective checkpoint 1986", "waiting"),
          stage("baseline", "Baseline evaluation", "complete", { completed: 810, total: 810, unit: "prompts" }),
        ],
      },
    },
    {
      id: "spark1", label: "Spark 1", kind: "training", status: "live", updatedAt: 1794356160000,
      stages: {
        data: [], evaluation: [],
        training: [stage("ce", "Cross entropy", "running", {
          completed: 450, total: 900, unit: "steps", loss: 4.125, etaSeconds: 5450, latestCheckpoint: 300, nextCheckpoint: 600,
        })],
      },
    },
    {
      id: "spark2", label: "Spark 2", kind: "training", status: "stale", updatedAt: 1794356160000,
      stages: {
        data: [], evaluation: [],
        training: [
          stage("preference", "Preference objective", "running", { completed: 420, total: 900, etaSeconds: 6000 }),
          stage("main", "Main objective", "stopped", { completed: 300, total: 900, unit: "steps" }),
        ],
      },
    },
  ],
};

afterEach(() => vi.useRealTimers());

it("shows worker activity, CPU/GPU assignments and declared input dependencies", () => {
  const worker: FinetuneProgress = { ...snapshot, sources: snapshot.sources.map((source) => source.id !== "spark1" ? source : {
    ...source,
    resources: [{ kind: "gpu", id: "GPU-worker-uuid", label: "GPU 1", scope: "assigned" }],
    stages: { ...source.stages, training: [{ ...source.stages.training[0],
      resources: [{ kind: "cpu", id: "0-7", label: "CPU cores 0–7", scope: "affinity" }],
      activity: { kind: "saving", label: "Saving checkpoint", detail: "Publishing checkpoint 600" },
      dependencies: [{ id: "data", label: "Prepared training examples", state: "ready" },
        { id: "copy", label: "Checkpoint transfer", state: "waiting" }],
    }] },
  }) };
  const { container } = render(<FinetuneProgressCard snapshot={worker} requestFailed={false} />);
  expect(container.querySelector('[aria-label="Spark 1 resources"]')?.textContent).toContain("AssignedGPU 1");
  expect(container.querySelector('[aria-label="Cross entropy resources"]')?.textContent).toContain("AffinityCPU cores 0–7");
  expect(container.querySelector('[data-activity="saving"]')?.textContent).toContain("Saving checkpoint");
  expect(container.textContent).toContain("Publishing checkpoint 600");
  const inputs = container.querySelector('[aria-label="Required inputs"]');
  expect(inputs?.textContent).toContain("Prepared training examplesReady");
  expect(inputs?.textContent).toContain("Checkpoint transferWaiting");
  expect(container.textContent).not.toContain("Scheduler");
});

it.each(["stalled", "unresponsive"] as const)("shows %s progress health and suspends the finish prediction", (state) => {
  const worker: FinetuneProgress = { ...snapshot, sources: snapshot.sources.map((source) => source.id !== "spark1" ? source : {
    ...source, stages: { ...source.stages, training: [{ ...source.stages.training[0],
      health: { state, detail: "The declared reporting interval has elapsed." },
    }] },
  }) };
  const { container } = render(<FinetuneProgressCard snapshot={worker} requestFailed={false} />);
  expect(container.querySelector(`[data-health="${state}"]`)?.textContent).toContain(state === "stalled" ? "No recent progress" : "Heartbeat overdue");
  expect(container.textContent).toContain("The declared reporting interval has elapsed.");
  expect(container.querySelector('[aria-label="Cross entropy estimated finish"]')).toBeNull();
  expect(container.textContent).not.toContain("ETA 1h 31m");
  expect(container.querySelector('[aria-label="Cross entropy progress"]')).not.toBeNull();
});

it("keeps a stale worker's activity historical and does not present its cached health as current", () => {
  const worker: FinetuneProgress = { ...snapshot, sources: snapshot.sources.map((source) => source.id === "spark2" ? { ...source, status: "live" }
    : source.id !== "spark1" ? source : { ...source, status: "stale", stages: { ...source.stages, training: [{ ...source.stages.training[0],
      activity: { kind: "saving", label: "Saving checkpoint" }, health: { state: "healthy", detail: "Heartbeat received." },
    }] } }) };
  const { container } = render(<FinetuneProgressCard snapshot={worker} requestFailed={false} />);
  expect(container.querySelector('[data-activity="saving"]')?.textContent).toContain("Last reported activity");
  expect(container.querySelector('[data-health="healthy"]')).toBeNull();
  expect(container.querySelector('[aria-label="Cross entropy estimated finish"]')).toBeNull();
});

it("prioritizes current training and evaluation while completed preparation stays accessible", async () => {
  vi.mocked(fetchTrainingProgress).mockResolvedValue(snapshot);
  const { container } = render(<OverviewPage sparks={[]} showFinetuneProgress />);
  await flush();

  expect(container.textContent).toContain("Text classification");
  expect(container.textContent).not.toContain("Training corpus");
  expect(container.querySelector('[data-group="data"]')).toBeNull();
  expect(container.querySelectorAll(".ov-finetune-group")).toHaveLength(2);
  expect(container.textContent).toContain("462 / 810 prompts");
  expect(container.textContent).toContain("57%");
  const updated = container.querySelector<HTMLTimeElement>("time:not([aria-label])");
  expect(updated?.textContent).toContain("Last update");
  expect(updated?.dateTime).toBe(new Date(1794356160000).toISOString());
  expect(container.textContent).toContain("PR checkpoint 1986");
  expect(container.textContent).toContain("Latest checkpoint 300");
  expect(container.textContent).toContain("Next checkpoint 600");
  expect(container.textContent).toContain("ETA 1h 31m");
  expect(container.textContent).toContain("Loss 4.125");
  expect(container.textContent).toContain("Last seen running");
  expect(container.textContent).toContain("Stopped");
  expect(container.textContent).not.toContain("ETA 1h 40m");
  const evaluation = container.querySelector<HTMLProgressElement>('[aria-label="Main checkpoint 1324 progress"]');
  expect(evaluation?.value).toBe(462);
  expect(evaluation?.max).toBe(810);
  const later = Array.from(container.querySelectorAll("details"))
    .find((details) => details.querySelector("summary")?.textContent === "Later evaluations (1)");
  expect(later?.open).toBe(false);
  expect(later?.textContent).toContain("Later evaluations (1)");
  expect(later?.textContent).toContain("Preference objective checkpoint 1986");
  expect(container.textContent).toContain("Latest result");
  const baseline = Array.from(container.querySelectorAll(".ov-finetune-stage"))
    .find((row) => row.textContent?.includes("Baseline evaluation"));
  expect(baseline).toBeDefined();
  expect(baseline?.closest("details")).toBeNull();

  const preparation = container.querySelector<HTMLButtonElement>('[data-stage="data"]');
  expect(preparation?.textContent).toContain("Complete");
  expect(preparation?.textContent).toContain("2 / 2 stages complete");
  expect(preparation?.getAttribute("aria-pressed")).toBe("false");
  act(() => preparation?.click());
  expect(container.querySelectorAll(".ov-finetune-group")).toHaveLength(3);
  expect(container.querySelector('[data-group="data"]')).not.toBeNull();
  const data = container.querySelector("details");
  expect(data?.open).toBe(false);
  expect(data?.textContent).toContain("Completed data preparation (1)");
  expect(data?.textContent).toContain("Training corpus");
  expect(container.textContent).toContain("Dataset packaged");
  expect(container.textContent).toContain("800 prompts");
  expect(container.textContent).not.toContain("Back to current work");
  act(() => preparation?.click());
  expect(container.querySelector('[data-group="data"]')).toBeNull();
  expect(container.querySelectorAll(".ov-finetune-group")).toHaveLength(2);
});

it("toggles each stage independently and keeps an intentionally empty selection across refreshes", () => {
  const { container, root } = render(<FinetuneProgressCard snapshot={snapshot} requestFailed={false} />);
  const button = (id: string) => container.querySelector<HTMLButtonElement>(`[data-stage="${id}"]`)!;
  expect(button("training").getAttribute("aria-pressed")).toBe("true");
  expect(button("evaluation").getAttribute("aria-pressed")).toBe("true");

  act(() => button("training").click());
  expect(button("training").getAttribute("aria-pressed")).toBe("false");
  expect(container.querySelector('[data-group="training"]')).toBeNull();
  expect(container.querySelector('[data-group="evaluation"]')).not.toBeNull();
  act(() => button("evaluation").click());
  expect(container.querySelectorAll(".ov-finetune-group")).toHaveLength(0);
  expect(container.textContent).toContain("No stages selected. Select a stage above.");
  act(() => root.render(<FinetuneProgressCard snapshot={{ ...snapshot, sources: [...snapshot.sources] }} requestFailed={false} />));
  expect(container.querySelectorAll(".ov-finetune-group")).toHaveLength(0);

  act(() => button("training").click());
  expect(container.querySelectorAll(".ov-finetune-group")).toHaveLength(1);
  expect(container.querySelector('[data-group="training"]')).not.toBeNull();
  act(() => button("evaluation").click());
  expect(container.querySelectorAll(".ov-finetune-group")).toHaveLength(2);
  expect(button("training").getAttribute("aria-pressed")).toBe("true");
  expect(button("evaluation").getAttribute("aria-pressed")).toBe("true");
});

it("follows current work until stages are manually selected, then retains the chosen combination", () => {
  const preparing: FinetuneProgress = { ...snapshot, sources: [{
    ...snapshot.sources[0], stages: { ...snapshot.sources[0].stages,
      data: [stage("preparing", "Prepare next shard", "running", { completed: 1, total: 5, unit: "shards" })],
    },
  }, ...snapshot.sources.slice(1)] };
  const { container, root } = render(<FinetuneProgressCard snapshot={preparing} requestFailed={false} />);
  expect(container.querySelector('[data-group="data"]')).not.toBeNull();
  act(() => root.render(<FinetuneProgressCard snapshot={snapshot} requestFailed={false} />));
  expect(container.querySelector('[data-group="data"]')).toBeNull();
  act(() => container.querySelector<HTMLButtonElement>('[data-stage="training"]')!.click());
  act(() => root.render(<FinetuneProgressCard snapshot={preparing} requestFailed={false} />));
  expect(container.querySelectorAll(".ov-finetune-group")).toHaveLength(1);
  expect(container.querySelector('[data-group="evaluation"]')).not.toBeNull();
  expect(container.querySelector('[data-group="data"]')).toBeNull();
  expect(container.querySelector('[data-group="training"]')).toBeNull();
});

it("keeps stage choices separate for independent runs sharing a host and stable when runs reorder", () => {
  const runs: TrainingRun[] = ["First experiment", "Second experiment"].map((campaign, index) => ({
    id: `run-${index}`, campaign, model: null, active: true, sources: snapshot.sources,
  }));
  const { container, root } = render(<FinetuneProgressCard snapshot={{ ...snapshot, runs }} requestFailed={false} />);
  const article = (name: string) => Array.from(container.querySelectorAll<HTMLElement>(".ov-finetune-run"))
    .find((run) => run.querySelector("h3")?.textContent === name)!;
  act(() => article("First experiment").querySelector<HTMLButtonElement>('[data-stage="training"]')!.click());
  expect(article("First experiment").querySelectorAll(".ov-finetune-group")).toHaveLength(1);
  expect(article("Second experiment").querySelectorAll(".ov-finetune-group")).toHaveLength(2);
  act(() => root.render(<FinetuneProgressCard snapshot={{ ...snapshot, runs: [...runs].reverse() }} requestFailed={false} />));
  expect(article("First experiment").querySelector('[data-stage="training"]')?.getAttribute("aria-pressed")).toBe("false");
  expect(article("First experiment").querySelector('[data-stage="evaluation"]')?.getAttribute("aria-pressed")).toBe("true");
  expect(article("Second experiment").querySelector('[data-stage="training"]')?.getAttribute("aria-pressed")).toBe("true");
  expect(article("Second experiment").querySelectorAll(".ov-finetune-group")).toHaveLength(2);
});

it("keeps a run's evaluation visible after every trainer finishes and hides after evaluation finishes", () => {
  const evaluating: TrainingRun = {
    id: "evaluating", campaign: "Finished training, evaluating", model: null, active: false,
    sources: [{ ...snapshot.sources[0], sparkId: "gpu-lab", stages: {
      data: [], training: [stage("done", "Training finished", "complete")],
      evaluation: [stage("eval", "Final validation", "running", { completed: 25, total: 100, unit: "samples" })],
    } }],
  };
  const training: TrainingRun = {
    id: "training", campaign: "Another training run", model: null, active: true, sources: [snapshot.sources[1]],
  };
  const stopped: TrainingRun = {
    id: "stopped", campaign: "Stopped experiment", model: null, active: false,
    sources: [{ ...snapshot.sources[1], stages: { data: [], evaluation: [], training: [stage("stopped", "Stopped", "stopped")] } }],
  };
  const preparing: TrainingRun = {
    id: "preparing", campaign: "Preparing the next experiment", model: null, active: false,
    sources: [{ ...snapshot.sources[0], stages: { training: [], evaluation: [], data: [stage("prepare", "Preparing shards", "running")] } }],
  };
  const reported = { ...snapshot, active: true, runs: [evaluating, training, stopped, preparing] };
  const { container, root } = render(<FinetuneProgressCard snapshot={reported} requestFailed={false} />);
  expect(container.querySelectorAll(".ov-finetune-run")).toHaveLength(3);
  expect(container.textContent).toContain("Finished training, evaluating");
  expect(container.textContent).toContain("Final validation");
  expect(container.textContent).toContain("Preparing shards");
  expect(container.textContent).not.toContain("Stopped experiment");
  const finished: TrainingRun = { ...training, active: false, sources: [{
    ...snapshot.sources[1], stages: { data: [], evaluation: [], training: [stage("done", "Training finished", "complete")] },
  }] };
  // Reporter stages keep the pane open even while an older server still reports active=false.
  act(() => root.render(<FinetuneProgressCard snapshot={{ ...reported, runs: [evaluating, finished, preparing] }} requestFailed={false} />));
  expect(container.querySelector(".ov-finetune")).not.toBeNull();
  expect(container.querySelectorAll(".ov-finetune-run")).toHaveLength(2);
  expect(container.querySelector('.ov-finetune-run [data-group="evaluation"]')).not.toBeNull();
  expect(container.querySelector('.ov-finetune-run [data-group="training"]')).toBeNull();
  const evaluated = { ...evaluating, sources: evaluating.sources.map((source) => ({ ...source, stages: {
    ...source.stages, evaluation: source.stages.evaluation.map((row) => ({ ...row, state: "complete" as const })),
  } })) };
  act(() => root.render(<FinetuneProgressCard snapshot={{ ...reported, runs: [evaluated, finished, preparing] }} requestFailed={false} />));
  expect(container.querySelector(".ov-finetune")).toBeNull();
});

it.each(["running", "waiting"] as const)("opens the current evaluation view for a %s final evaluator with completed training accessible", (state) => {
  const evaluating: FinetuneProgress = { ...snapshot, sources: [{
    ...snapshot.sources[0], stages: {
      data: [stage("data", "Preparation finished", "complete")],
      training: [stage("train", "Training finished", "complete")],
      evaluation: [stage("final", "Final validation", state)],
    },
  }] };
  const { container } = render(<FinetuneProgressCard snapshot={evaluating} requestFailed={false} />);
  expect(container.querySelector(".ov-finetune")).not.toBeNull();
  expect(container.querySelector('[data-group="evaluation"]')?.textContent).toContain("Final validation");
  expect(container.querySelector('[data-group="training"]')).toBeNull();
  expect(container.querySelector('[data-group="data"]')).toBeNull();
  act(() => container.querySelector<HTMLButtonElement>('[data-stage="training"]')!.click());
  expect(container.querySelector('[data-group="training"]')?.textContent).toContain("Training finished");
});

it.each(["stale", "unavailable"] as const)("does not retain the pane for cached running or queued evaluation from a %s source", (status) => {
  const evaluating: TrainingRun = { id: "cached", campaign: "Cached final evaluation", model: null, active: true,
    sources: [{ ...snapshot.sources[0], status, stages: {
      data: [], training: [stage("done", "Training finished", "complete")],
      evaluation: [stage("eval", "Final validation", "running"), stage("queued", "Queued validation", "waiting")],
    } }],
  };
  const { container } = render(<FinetuneProgressCard snapshot={{ ...snapshot, active: true, runs: [evaluating] }} requestFailed={false} />);
  expect(container.querySelector(".ov-finetune")).toBeNull();
});

it("keeps the evaluation view current when another checkpoint failed after training finished", () => {
  const evaluating: FinetuneProgress = { ...snapshot, sources: [{ ...snapshot.sources[0], stages: {
    data: [stage("data", "Preparation finished", "complete")],
    training: [stage("train", "Training finished", "complete")],
    evaluation: [stage("failed", "Earlier validation failed", "failed"), stage("final", "Final validation", "running")],
  } }] };
  const { container } = render(<FinetuneProgressCard snapshot={evaluating} requestFailed={false} />);
  expect(container.querySelector('[data-group="evaluation"]')?.textContent).toContain("Earlier validation failed");
  expect(container.querySelector('[data-group="evaluation"]')?.textContent).toContain("Final validation");
  expect(container.querySelector('[data-group="training"]')).toBeNull();
  expect(container.querySelector('[data-group="data"]')).toBeNull();
});

it("retains a manual history selection as training finishes and evaluation continues", () => {
  const { container, root } = render(<FinetuneProgressCard snapshot={snapshot} requestFailed={false} />);
  act(() => container.querySelector<HTMLButtonElement>('[data-stage="evaluation"]')!.click());
  const trained = { ...snapshot, sources: snapshot.sources.map((source) => ({ ...source, stages: {
    ...source.stages, training: source.stages.training.map((row) => ({ ...row, state: "complete" as const })),
  } })) };
  act(() => root.render(<FinetuneProgressCard snapshot={trained} requestFailed={false} />));
  expect(container.querySelector(".ov-finetune")).not.toBeNull();
  expect(container.querySelector('[data-group="training"]')).not.toBeNull();
  expect(container.querySelector('[data-group="evaluation"]')).toBeNull();
  expect(container.querySelector('[data-stage="training"]')?.getAttribute("aria-pressed")).toBe("true");
  expect(container.querySelector('[data-stage="evaluation"]')?.getAttribute("aria-pressed")).toBe("false");
  act(() => container.querySelector<HTMLButtonElement>('[data-stage="training"]')!.click());
  expect(container.querySelector(".ov-finetune")).not.toBeNull();
  expect(container.querySelectorAll(".ov-finetune-group")).toHaveLength(0);
  expect(container.textContent).toContain("No stages selected. Select a stage above.");
});

it("counts physical devices once when several reporters share one device", () => {
  const run: TrainingRun = {
    id: "one-host", campaign: "Multiple jobs on one GPU host", model: null, active: true,
    sources: Array.from({ length: 4 }, (_, index) => ({ ...snapshot.sources[1], id: `reporter-${index}`, sparkId: "shared-host" })),
  };
  const { container } = render(<FinetuneProgressCard snapshot={{ ...snapshot, runs: [run] }} requestFailed={false} />);
  expect(container.querySelector(".ov-finetune-devices")?.textContent).toBe("1 device");
});

it("shows concurrent stage progress without combining counters from different stages", () => {
  const activePreparation: FinetuneProgress = {
    ...snapshot,
    sources: [{
      ...snapshot.sources[0],
      stages: { ...snapshot.sources[0].stages, data: [
        stage("download", "Download source", "complete", { completed: 5000, total: 5000, unit: "bytes" }),
        stage("prepare", "Prepare examples", "running", { completed: 200, total: 800, unit: "prompts" }),
      ] },
    }, snapshot.sources[1]],
  };
  const { container } = render(<FinetuneProgressCard snapshot={activePreparation} requestFailed={false} />);
  expect(container.querySelectorAll(".ov-finetune-group")).toHaveLength(3);
  const preparation = container.querySelector('[data-stage="data"]');
  expect(preparation?.textContent).toContain("In progress");
  expect(preparation?.textContent).toContain("200 / 800 prompts · 25%");
  expect(container.querySelector('[data-stage="training"]')?.textContent).toContain("In progress");
  expect(container.querySelector('[data-stage="evaluation"]')?.textContent).toContain("In progress");
  expect(container.querySelector('[data-group="data"]')?.textContent).toContain("5,000 / 5,000 bytes");
});

it("keeps the evaluation stage open for future checkpoints while individual results stay complete", () => {
  const knownResults: FinetuneProgress = {
    ...snapshot,
    sources: [{
      ...snapshot.sources[0],
      stages: { ...snapshot.sources[0].stages, evaluation: [
        stage("ce-662", "PR checkpoint 662", "complete", { updatedAt: 100 }),
        stage("ce-1324", "PR checkpoint 1324", "complete", { updatedAt: 200 }),
      ] },
    }, snapshot.sources[1]],
  };
  const { container } = render(<FinetuneProgressCard snapshot={knownResults} requestFailed={false} />);
  const evaluation = container.querySelector('[data-stage="evaluation"]');
  expect(evaluation?.getAttribute("data-state")).toBe("waiting");
  expect(evaluation?.textContent).toContain("Awaiting checkpoints");
  expect(evaluation?.textContent).toContain("2 checkpoint evaluations complete · Final evaluation pending");
  expect(evaluation?.querySelector(".ov-finetune-step__number")?.textContent).toBe("3");
  const latest = container.querySelector('[data-group="evaluation"] .ov-finetune-stage');
  expect(latest?.getAttribute("data-state")).toBe("complete");
  expect(latest?.textContent).toContain("PR checkpoint 1324");
  expect(latest?.querySelector(".tag")?.textContent).toBe("Complete");
  expect(container.querySelector('[data-group="evaluation"]')).not.toBeNull();
});

it.each(["stale", "unavailable"] as const)("does not claim all evaluations are up to date from a %s source", (status) => {
  const knownResults: FinetuneProgress = {
    ...snapshot,
    sources: [{ ...snapshot.sources[0], status, stages: {
      ...snapshot.sources[0].stages, evaluation: [stage("done", "Saved checkpoint", "complete")],
    } }, snapshot.sources[1]],
  };
  const { container } = render(<FinetuneProgressCard snapshot={knownResults} requestFailed={false} />);
  const evaluation = container.querySelector('[data-stage="evaluation"]');
  expect(evaluation?.getAttribute("data-state")).toBe("unknown");
  expect(evaluation?.textContent).toContain("Status unavailable");
  expect(evaluation?.textContent).toContain("Last read: 1 checkpoint evaluation complete");
  expect(evaluation?.textContent).not.toContain("Awaiting checkpoints");
});

it.each([
  { state: "running", expectedState: "running", status: "In progress" },
  { state: "waiting", expectedState: "waiting", status: "Waiting" },
  { state: "failed", expectedState: "failed", status: "Needs attention" },
] as const)("keeps a $state evaluation distinct from waiting for future checkpoints", ({ state, expectedState, status }) => {
  const { container } = render(<FinetuneProgressCard snapshot={{
    ...snapshot,
    sources: [{ ...snapshot.sources[0], stages: { ...snapshot.sources[0].stages, evaluation: [
      stage("done", "Finished checkpoint", "complete"),
      stage("pending", "Next checkpoint", state, { completed: 5, total: 10, unit: "prompts" }),
    ] } }, snapshot.sources[1]],
  }} requestFailed={false} />);
  const evaluation = container.querySelector('[data-stage="evaluation"]');
  expect(evaluation?.getAttribute("data-state")).toBe(expectedState);
  expect(evaluation?.textContent).toContain(status);
  expect(evaluation?.textContent).not.toContain("Awaiting checkpoints");
});

it("does not hide a failed evaluation behind another checkpoint being scored", () => {
  const { container } = render(<FinetuneProgressCard snapshot={{
    ...snapshot,
    sources: [{ ...snapshot.sources[0], stages: { ...snapshot.sources[0].stages, evaluation: [
      stage("failed", "Failed checkpoint", "failed"),
      stage("running", "Scoring checkpoint", "running"),
    ] } }, snapshot.sources[1]],
  }} requestFailed={false} />);
  expect(container.querySelector('[data-stage="evaluation"]')?.textContent).toContain("Needs attention");
  expect(container.querySelector('[aria-label="Scoring checkpoint progress"]')).not.toBeNull();
});

it("anchors estimated finish to the successful reading and omits predictions for stale sources", () => {
  const { container, root } = render(<FinetuneProgressCard snapshot={snapshot} requestFailed={false} />);
  const finish = container.querySelector<HTMLTimeElement>('[aria-label="Cross entropy estimated finish"]');
  const expected = new Date(snapshot.sources[1].updatedAt! + 5450 * 1000).toISOString();
  expect(finish?.dateTime).toBe(expected);
  expect(finish?.textContent).toMatch(/\d/);
  expect(container.textContent).toContain("ETA 1h 31m");
  expect(container.textContent).toContain("Est. finish");
  expect(container.querySelector('[aria-label="Preference objective estimated finish"]')).toBeNull();
  vi.spyOn(Date, "now").mockReturnValue(snapshot.sources[1].updatedAt! + 30 * 60 * 1000);
  act(() => root.render(<FinetuneProgressCard snapshot={snapshot} requestFailed={false} />));
  expect(container.querySelector<HTMLTimeElement>('[aria-label="Cross entropy estimated finish"]')?.dateTime).toBe(expected);
});

it("shows the newest measured evaluation statistics and keeps earlier results collapsed", async () => {
  vi.mocked(fetchTrainingProgress).mockResolvedValue({
    ...snapshot,
    sources: [{
      ...snapshot.sources[0],
      stages: { ...snapshot.sources[0].stages, evaluation: [
        stage("old", "Old checkpoint", "complete", { updatedAt: 100 }),
        stage("new", "New checkpoint", "complete", {
          completed: 810, total: 810, unit: "prompts", updatedAt: 300,
          metrics: [
            { label: "Tree score", value: 4.125, unit: "tokens/round" },
            { label: "Paired improvement", value: 0 },
            { label: "95% interval", value: "3.98–4.27" },
            { label: "Validation accuracy", value: null },
          ],
        }),
        stage("middle", "Middle checkpoint", "complete", { updatedAt: 200 }),
      ] },
    }, snapshot.sources[1]],
  });
  const { container } = render(<OverviewPage sparks={[]} showFinetuneProgress />);
  await flush();

  const statistics = container.querySelector('[aria-label="New checkpoint statistics"]');
  expect(statistics?.closest("details")).toBeNull();
  expect(statistics?.querySelectorAll("dt")).toHaveLength(4);
  expect(statistics?.textContent).toContain("Tree score 4.125 tokens/round");
  expect(statistics?.textContent).toContain("Paired improvement 0");
  expect(statistics?.textContent).toContain("95% interval 3.98–4.27");
  expect(statistics?.textContent).toContain("Validation accuracy —");
  const history = Array.from(container.querySelectorAll("details"))
    .find((details) => details.querySelector("summary")?.textContent === "Earlier evaluations (2)");
  expect(history?.open).toBe(false);
  expect(history?.textContent).toContain("Old checkpoint");
  expect(history?.textContent).toContain("Middle checkpoint");
  expect(history?.textContent).not.toContain("New checkpoint");
});

it("does not fetch fine-tuning progress when the overview feature is disabled", () => {
  render(<OverviewPage sparks={[]} />);
  expect(fetchTrainingProgress).not.toHaveBeenCalled();
});

it("marks retained values stale after a failed refresh and cancels an in-flight read on unmount", async () => {
  vi.useFakeTimers();
  vi.spyOn(document, "hidden", "get").mockReturnValue(false);
  vi.mocked(fetchTrainingProgress)
    .mockResolvedValueOnce(snapshot)
    .mockRejectedValueOnce(new Error("Connection lost"))
    .mockImplementationOnce(() => new Promise(() => {}));

  const { container } = render(<OverviewPage sparks={[makeSpark("desktop"), makeSpark("spark1")]} showFinetuneProgress />);
  await flush();
  expect(fetchTrainingProgress).toHaveBeenCalledTimes(1);
  expect(container.querySelectorAll('[aria-label="Evaluation"]')).toHaveLength(1);
  await act(async () => vi.advanceTimersByTimeAsync(10_000));
  expect(fetchTrainingProgress).toHaveBeenCalledTimes(2);
  expect(container.querySelector('[aria-label="Evaluation"]')).toBeNull();
  expect(container.querySelector(".ov-finetune")).toBeNull();

  await act(async () => vi.advanceTimersByTimeAsync(10_000));
  const signal = vi.mocked(fetchTrainingProgress).mock.calls.at(-1)?.[0];
  expect(signal?.aborted).toBe(false);
  cleanupRenders();
  expect(signal?.aborted).toBe(true);
});

it.each(["stale", "unavailable"] as const)("omits GPU evaluation for a %s source", async (status) => {
  vi.mocked(fetchTrainingProgress).mockResolvedValue({
    ...snapshot, sources: [{ ...snapshot.sources[0], status }],
  });

  const { container } = render(<OverviewPage sparks={[makeSpark("desktop")]} showFinetuneProgress />);
  await flush();
  expect(container.querySelector('[aria-label="Evaluation"]')).toBeNull();
  expect(container.querySelector(".ov-finetune")).toBeNull();
});

it("renders multiple providers with arbitrary metrics, units and stage groups on any source", async () => {
  const pretraining: TrainingRun = {
    id: "pretraining", campaign: "Corpus pretraining", model: "Research model", active: true,
    sources: [{
      ...snapshot.sources[0], id: "gpu-process", sparkId: "gpu-lab", label: "GPU lab",
      stages: {
        data: [], evaluation: [],
        training: [stage("epoch", "Token training", "running", {
          completed: 25000, total: 100000, unit: "tokens",
          metrics: [{ label: "Epoch", value: 2.5 }, { label: "Throughput", value: 321, unit: "tokens/s" },
            { label: "Schedule", value: "cosine" }, { label: "Validation loss", value: null }],
        })],
      },
    }],
  };
  const finetuning: TrainingRun = {
    id: "finetuning", campaign: "Adapter training", model: null, active: true,
    sources: [{
      ...snapshot.sources[1], sparkId: "spark1",
      stages: { ...snapshot.sources[1].stages, evaluation: [stage("quality", "Perplexity evaluation", "complete", {
        metrics: [{ label: "Perplexity", value: 7.8 }],
      })] },
    }],
  };
  vi.mocked(fetchTrainingProgress).mockResolvedValue({ ...snapshot, active: true, runs: [pretraining, finetuning] });
  const { container } = render(<OverviewPage sparks={[]} showFinetuneProgress />);
  await flush();

  expect(container.querySelectorAll(".ov-finetune-run")).toHaveLength(2);
  expect(container.textContent).toContain("Training progress");
  expect(container.textContent).toContain("Corpus pretraining");
  expect(container.textContent).toContain("Adapter training");
  expect(container.textContent).toContain("25,000 / 100,000 tokens");
  expect(container.textContent).toContain("Epoch 2.5");
  expect(container.textContent).toContain("Throughput 321 tokens/s");
  expect(container.textContent).toContain("Schedule cosine");
  expect(container.textContent).toContain("Validation loss —");
  expect(container.textContent).toContain("Perplexity 7.8");
  expect(container.textContent).not.toContain("No source available");
  const firstGroups = container.querySelector(".ov-finetune-run")?.querySelectorAll(".ov-finetune-group");
  expect(firstGroups).toHaveLength(1);
});

it("keeps polling through completed training and queued final evaluation, then hides only after the last result", async () => {
  vi.useFakeTimers();
  vi.spyOn(document, "hidden", "get").mockReturnValue(false);
  const evaluationOnly: TrainingRun = {
    id: "experiment", campaign: "Evaluation experiment", model: snapshot.model, active: false,
    sources: [{ ...snapshot.sources[0], sparkId: "desktop" }],
  };
  const active: TrainingRun = { ...evaluationOnly, active: true, sources: [...evaluationOnly.sources, snapshot.sources[1]] };
  const trained: TrainingRun = {
    ...active, active: false, sources: [active.sources[0], {
      ...active.sources[1], stages: { ...active.sources[1].stages, training: [stage("done", "Training complete", "complete")] },
    }],
  };
  const queued: TrainingRun = { ...trained, sources: trained.sources.map((source) => ({ ...source, stages: {
    ...source.stages, evaluation: source.stages.evaluation.map((row, index) => ({
      ...row, state: index === 1 ? "waiting" as const : "complete" as const,
    })),
  } })) };
  const finished: TrainingRun = { ...queued, sources: queued.sources.map((source) => ({ ...source, stages: {
    ...source.stages, evaluation: source.stages.evaluation.map((row) => ({ ...row, state: "complete" as const })),
  } })) };
  vi.mocked(fetchTrainingProgress)
    .mockResolvedValueOnce({ ...snapshot, active: true, runs: [active] })
    .mockResolvedValueOnce({ ...snapshot, active: false, runs: [trained] })
    .mockResolvedValueOnce({ ...snapshot, active: true, runs: [queued] })
    .mockResolvedValueOnce({ ...snapshot, active: false, runs: [finished] });
  const { container } = render(<OverviewPage sparks={[makeSpark("desktop")]} showFinetuneProgress />);
  await flush();
  expect(container.querySelector(".ov-finetune")).not.toBeNull();
  expect(container.querySelector('[data-group="training"]')).not.toBeNull();
  expect(container.querySelector('[aria-label="Evaluation"]')).not.toBeNull();
  expect(fetchTrainingProgress).toHaveBeenCalledTimes(1);

  await act(async () => vi.advanceTimersByTimeAsync(10_000));
  expect(container.querySelector(".ov-finetune")).not.toBeNull();
  expect(container.querySelector('[data-group="training"]')).toBeNull();
  expect(container.querySelector('[data-group="evaluation"]')).not.toBeNull();
  expect(container.querySelector('[data-stage="training"]')).not.toBeNull();
  expect(fetchTrainingProgress).toHaveBeenCalledTimes(2);
  await act(async () => vi.advanceTimersByTimeAsync(10_000));
  expect(container.querySelector(".ov-finetune")).not.toBeNull();
  expect(container.querySelector('[data-group="training"]')).toBeNull();
  expect(container.querySelector('[data-group="evaluation"]')).not.toBeNull();
  expect(container.querySelector('[aria-label="Evaluation"]')).toBeNull();
  expect(fetchTrainingProgress).toHaveBeenCalledTimes(3);
  await act(async () => vi.advanceTimersByTimeAsync(10_000));
  expect(container.querySelector(".ov-finetune")).toBeNull();
  expect(container.querySelector('[aria-label="Evaluation"]')).toBeNull();
  expect(fetchTrainingProgress).toHaveBeenCalledTimes(4);
});

it("routes evaluation metrics by host identity independently of the provider source name", async () => {
  vi.mocked(fetchTrainingProgress).mockResolvedValue({
    ...snapshot, active: false, runs: [{
      id: "custom", campaign: "Custom evaluation", model: null, active: false,
      sources: [{ ...snapshot.sources[0], id: "python-job", sparkId: "gpu-lab", kind: "training" }],
    }],
  });
  const host = makeSpark("gpu-lab");
  host.metrics.llm = [];
  const { container } = render(<OverviewPage sparks={[host]} showFinetuneProgress />);
  await flush();
  expect(container.querySelector(".ov-finetune")).not.toBeNull();
  expect(container.querySelector('[data-group="evaluation"]')).not.toBeNull();
  expect(container.querySelector('[aria-label="Evaluation"]')?.textContent).toContain("6.2 prompts/min");
  expect(container.querySelector(".ov-modelrow")?.textContent).toContain("Custom evaluation");
});
