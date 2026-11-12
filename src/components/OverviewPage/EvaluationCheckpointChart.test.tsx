import { act } from "react";
import { expect, it } from "vitest";
import type { EvaluationComparison, FinetuneSource, FinetuneStage } from "../../api/types";
import { render } from "../../testing/render";
import { EvaluationCheckpointChart } from "./EvaluationCheckpointChart";

function result(id: string, changes: Partial<EvaluationComparison> = {}, state: FinetuneStage["state"] = "complete"): FinetuneStage {
  return { id, label: id, state, completed: 10, total: 10, unit: "prompts", updatedAt: 100,
    comparison: { metric: "tree", metricLabel: "Tree score", unit: "tokens/round", value: 4,
      seriesId: "ce", seriesLabel: "Cross entropy", checkpoint: 100, cohortId: "same-set", ...changes } };
}

function source(evaluation: FinetuneStage[], status: FinetuneSource["status"] = "live"): FinetuneSource {
  return { id: "desktop", label: "Desktop", kind: "desktop", status, updatedAt: 100,
    stages: { data: [], training: [], evaluation } };
}

function fixture() {
  return [source([
    result("baseline", { seriesId: "baseline", seriesLabel: "Released baseline", value: 3.8, baseline: true, checkpoint: undefined }),
    result("ce100", { value: 3.9, lower: 3.85, upper: 3.95 }),
    result("ce1000", { checkpoint: 1000, value: 4.1 }),
    result("preference1000", { seriesId: "preference", seriesLabel: "Preference objective", checkpoint: 1000, value: 4.2 }),
    result("pending", { checkpoint: 2000, value: 99 }, "running"),
  ])];
}

it("plots measured checkpoint positions and a dashed baseline without plotting unfinished evaluations", () => {
  const { container } = render(<EvaluationCheckpointChart sources={fixture()} />);
  expect(container.querySelectorAll("circle")).toHaveLength(4);
  expect(container.querySelectorAll(".ov-evaluation-chart__baseline")).toHaveLength(1);
  expect(container.querySelectorAll(".ov-evaluation-chart__interval")).toHaveLength(0);
  const line = container.querySelector('[data-series="ce"] polyline');
  const points = line!.getAttribute("points")!.split(" ").map((point) => point.split(",").map(Number));
  const xStart = 48;
  expect((points[0][0] - xStart) / (points[1][0] - xStart)).toBeCloseTo(.1);
  expect(container.textContent).toContain("Checkpoint step");
  const details = container.querySelector<HTMLDetailsElement>("details")!;
  expect(container.querySelectorAll("details")).toHaveLength(1);
  expect(details.open).toBe(false);
  expect(details.querySelector("summary")?.textContent).toBe("Details");
  expect(details.querySelector(".ov-evaluation-chart__note")?.textContent).toContain("Same evaluation set.");
  expect(container.querySelector('[role="status"]')).toBeNull();
  const table = container.querySelector("table");
  expect(table?.querySelectorAll("tbody tr")).toHaveLength(4);
  expect(table?.textContent).not.toContain("pending");
  expect(table?.textContent).toContain("+0.4");
});

it("keeps score trends prominent by default and expands the axis only when intervals are shown", () => {
  const sources = fixture();
  sources[0].stages.evaluation[1].comparison!.lower = 2;
  sources[0].stages.evaluation[1].comparison!.upper = 6;
  const { container } = render(<EvaluationCheckpointChart sources={sources} />);
  const checkbox = container.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
  const point = () => container.querySelector('[aria-label^="Cross entropy · Step 100:"]')!;
  const initialY = point().getAttribute("cy");
  expect(checkbox.checked).toBe(false);
  expect(point().getAttribute("aria-label")).toContain("95% CI 2–6");
  expect(container.querySelector(".ov-evaluation-chart__note")?.textContent).toContain("95% intervals are hidden");
  act(() => checkbox.click());
  expect(checkbox.checked).toBe(true);
  expect(container.querySelectorAll(".ov-evaluation-chart__interval")).toHaveLength(1);
  expect(point().getAttribute("cy")).not.toBe(initialY);
  expect(container.querySelector(".ov-evaluation-chart__note")?.textContent).toContain("Whiskers show reported 95% intervals");
  act(() => checkbox.click());
  expect(point().getAttribute("cy")).toBe(initialY);
});

it("exposes exact scores, confidence intervals and baseline changes on keyboard focus", () => {
  const { container } = render(<EvaluationCheckpointChart sources={fixture()} />);
  const point = container.querySelector('[aria-label^="Cross entropy · Step 100:"]')!;
  expect(point.getAttribute("aria-label")).toContain("3.9 tokens/round; +0.1 vs Released baseline; 95% CI 3.85–3.95");
  act(() => point.dispatchEvent(new FocusEvent("focusin", { bubbles: true })));
  expect(container.querySelector('[role="status"]')?.textContent).toContain("Cross entropy · Step 100: 3.9 tokens/round");
  act(() => point.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true })));
  expect(container.querySelector('[role="status"]')?.textContent).toContain("+0.1 vs Released baseline");
});

it("keeps a keyboard-focused score visible when the pointer leaves the chart", () => {
  const { container } = render(<EvaluationCheckpointChart sources={fixture()} />);
  const point = container.querySelector('[aria-label^="Cross entropy · Step 100:"]')!;
  const plot = container.querySelector("svg")!;
  act(() => point.dispatchEvent(new FocusEvent("focusin", { bubbles: true })));
  act(() => plot.dispatchEvent(new MouseEvent("mouseout", { bubbles: true, relatedTarget: document.body })));
  expect(container.querySelector('[role="status"]')?.textContent).toContain("Cross entropy · Step 100: 3.9 tokens/round");
  expect(container.querySelector('[role="status"]')?.textContent).toContain("+0.1 vs Released baseline");
  act(() => point.dispatchEvent(new FocusEvent("focusout", { bubbles: true })));
  expect(container.querySelector('[role="status"]')).toBeNull();
});

it("lets readers toggle series without losing the exact score table", () => {
  const { container } = render(<EvaluationCheckpointChart sources={fixture()} />);
  const button = [...container.querySelectorAll<HTMLButtonElement>("button[aria-pressed]")].find((item) => item.textContent === "Cross entropy")!;
  act(() => button.click());
  expect(button.getAttribute("aria-pressed")).toBe("false");
  expect(container.querySelector('[data-series="ce"]')?.querySelector("circle")).toBeNull();
  expect(container.querySelectorAll("tbody tr")).toHaveLength(4);
  act(() => button.click());
  expect(container.querySelectorAll('[data-series="ce"] circle')).toHaveLength(2);
});

it("keeps the all-hidden state actionable", () => {
  const { container } = render(<EvaluationCheckpointChart sources={fixture()} />);
  act(() => { container.querySelectorAll<HTMLButtonElement>("button[aria-pressed]").forEach((button) => button.click()); });
  expect(container.querySelectorAll("circle")).toHaveLength(0);
  expect(container.querySelector('[role="status"]')?.textContent).toContain("All series are hidden");
  expect(container.querySelectorAll("tbody tr")).toHaveLength(4);
});

it("separates distinct evaluation sets and score units before comparing to a baseline", () => {
  const sources = fixture();
  sources[0].stages.evaluation.push(result("other-cohort", { cohortId: "different-set", value: 8 }),
    result("other-metric", { metric: "accuracy", metricLabel: "Accuracy", unit: "%", value: 80 }));
  const { container } = render(<EvaluationCheckpointChart sources={sources} />);
  const select = container.querySelector("select")!;
  expect(select.options).toHaveLength(3);
  expect(container.querySelectorAll("tbody tr")).toHaveLength(4);
  const other = [...select.options].find((option) => JSON.parse(option.value)[2] === "different-set")!;
  act(() => { select.value = other.value; select.dispatchEvent(new Event("change", { bubbles: true })); });
  expect(container.querySelectorAll("tbody tr")).toHaveLength(1);
  expect(container.querySelector("tbody")?.textContent).toContain("8");
  expect(container.querySelector(".ov-evaluation-chart__baseline")).toBeNull();
  expect(container.querySelector("tbody")?.textContent).not.toContain("+");
  expect(container.querySelector('[role="status"]')?.textContent).toBe("No matching baseline");
});

it("preserves a zero score and checkpoint, handles a baseline alone and omits unreported comparisons", () => {
  const { container, root } = render(<EvaluationCheckpointChart sources={[source([result("zero", { checkpoint: 0, value: 0 })])]} />);
  expect(container.querySelector("circle")?.getAttribute("aria-label")).toContain("Step 0: 0 tokens/round");
  expect(container.innerHTML).not.toContain("NaN");
  act(() => root.render(<EvaluationCheckpointChart sources={[source([result("baseline", { baseline: true, checkpoint: undefined, value: 0 })])]} />));
  expect(container.textContent).toContain("Baseline recorded. Checkpoint scores will appear");
  expect(container.innerHTML).not.toContain("NaN");
  act(() => root.render(<EvaluationCheckpointChart sources={[source([{ ...result("missing"), comparison: undefined }])]} />));
  expect(container.querySelector("figure")).toBeNull();
});

it("chooses the newest duplicate result and marks results retained from stale sources", () => {
  const older = result("old", { value: 3 });
  const newer = { ...result("new", { value: 4 }), updatedAt: 200 };
  const { container } = render(<EvaluationCheckpointChart sources={[source([newer, older], "stale")]} />);
  expect(container.querySelectorAll("circle")).toHaveLength(1);
  expect(container.querySelector("circle")?.getAttribute("aria-label")).toContain("4 tokens/round");
  expect(container.querySelector("circle")?.getAttribute("aria-label")).toContain("stale source");
});

it.each([
  { metric: "accuracy", label: "Accuracy", goal: "maximize" as const, baseline: .7, better: .8, worse: .6, direction: "Higher is better" },
  { metric: "loss", label: "Validation loss", goal: "minimize" as const, baseline: .7, better: .6, worse: .8, direction: "Lower is better" },
])("interprets $metric changes using its declared objective", ({ metric, label, goal, baseline, better, worse, direction }) => {
  const common = { metric, metricLabel: label, unit: undefined, goal, contextId: "same-settings", contextLabel: "Greedy · dataset v2" };
  const { container } = render(<EvaluationCheckpointChart sources={[source([
    result("baseline", { ...common, seriesId: "baseline", seriesLabel: "Baseline", baseline: true, checkpoint: undefined, value: baseline }),
    result("better", { ...common, value: better }),
    result("worse", { ...common, checkpoint: 200, value: worse }),
  ])]} />);
  expect(container.querySelector(".ov-evaluation-chart__objective")?.textContent).toContain(direction);
  expect(container.querySelector(".ov-evaluation-chart__best")?.textContent).toContain(`Best shown: ${better} · Cross entropy · Step 100`);
  expect(container.querySelector('[aria-label^="Cross entropy · Step 100:"]')?.getAttribute("aria-label")).toContain("Improved");
  expect(container.querySelector('[aria-label^="Cross entropy · Step 200:"]')?.getAttribute("aria-label")).toContain("Worse");
  expect(container.querySelector(".ov-evaluation-chart__note")?.textContent).toContain("Settings: Greedy · dataset v2. Same reported evaluation settings.");
  const rows = [...container.querySelectorAll("tbody tr")].map((row) => row.textContent);
  expect(rows.some((row) => row?.includes("Improved"))).toBe(true);
  expect(rows.some((row) => row?.includes("Worse"))).toBe(true);
});

it("compares distance to a declared target and preserves a zero-valued best checkpoint", () => {
  const common = { metric: "calibration-error", metricLabel: "Calibration error", unit: undefined, goal: "target" as const, target: 0 };
  const { container } = render(<EvaluationCheckpointChart sources={[source([
    result("baseline", { ...common, seriesId: "baseline", seriesLabel: "Baseline", baseline: true, checkpoint: undefined, value: .2 }),
    result("zero", { ...common, checkpoint: 0, value: 0 }),
    result("farther", { ...common, checkpoint: 200, value: -.3 }),
  ])]} />);
  expect(container.querySelector(".ov-evaluation-chart__objective")?.textContent).toContain("Target 0");
  expect(container.querySelector(".ov-evaluation-chart__best")?.textContent).toContain("Best shown: 0 · Cross entropy · Step 0");
  expect(container.querySelector('[aria-label^="Cross entropy · Step 0:"]')?.getAttribute("aria-label")).toContain("Improved");
  expect(container.querySelector('[aria-label^="Cross entropy · Step 200:"]')?.getAttribute("aria-label")).toContain("Worse");
});

it.each([undefined, "observe"] as const)("leaves direction %s unranked while preserving numeric baseline changes", (goal) => {
  const sources = fixture();
  for (const stage of sources[0].stages.evaluation) stage.comparison!.goal = goal;
  const { container } = render(<EvaluationCheckpointChart sources={sources} />);
  expect(container.querySelector(".ov-evaluation-chart__best")).toBeNull();
  expect(container.textContent).not.toContain("Higher is better");
  expect(container.textContent).not.toContain("Improved");
  expect(container.textContent).not.toContain("Worse");
  expect(container.querySelector('[aria-label^="Cross entropy · Step 100:"]')?.getAttribute("aria-label")).toContain("+0.1 vs Released baseline");
  expect(container.querySelector(".ov-evaluation-chart__note")?.textContent).toContain("Evaluation settings identity was not reported.");
});

it("offers several metrics from one evaluation without duplicating a legacy primary score", () => {
  const stage = result("checkpoint", { value: 999 });
  stage.comparisons = [
    { ...stage.comparison!, metric: "accuracy", metricLabel: "Accuracy", value: .85, goal: "maximize", unit: "fraction" },
    { ...stage.comparison!, metric: "loss", metricLabel: "Validation loss", value: .42, goal: "minimize", unit: undefined },
  ];
  const { container } = render(<EvaluationCheckpointChart sources={[source([stage])]} />);
  const select = container.querySelector("select")!;
  expect(select.options).toHaveLength(2);
  expect([...select.options].some((option) => option.text.includes("Accuracy (fraction)"))).toBe(true);
  expect(container.querySelectorAll("circle")).toHaveLength(1);
  expect(container.querySelector("circle")?.getAttribute("aria-label")).toContain("0.85 fraction");
  const loss = [...select.options].find((option) => JSON.parse(option.value)[0] === "loss")!;
  act(() => { select.value = loss.value; select.dispatchEvent(new Event("change", { bubbles: true })); });
  expect(container.querySelector("circle")?.getAttribute("aria-label")).toContain("0.42");
  expect(container.textContent).toContain("Lower is better");
  expect(container.querySelector("table")?.textContent).not.toContain("999");
});

it("separates settings, objective directions and target values before finding a baseline", () => {
  const common = { metric: "accuracy", metricLabel: "Accuracy", unit: "fraction", contextId: "settings-a", contextLabel: "Settings A", goal: "maximize" as const };
  const sources = [source([
    result("baseline", { ...common, seriesId: "baseline", baseline: true, checkpoint: undefined, value: .5 }),
    result("same", { ...common, value: .6 }),
    result("other-settings", { ...common, contextId: "settings-b", contextLabel: "Settings B", value: .7 }),
    result("other-goal", { ...common, goal: "minimize", value: .7 }),
    result("target-one", { ...common, goal: "target", target: 1, value: .7 }),
    result("target-zero", { ...common, goal: "target", target: 0, value: .7 }),
  ])];
  const { container } = render(<EvaluationCheckpointChart sources={sources} />);
  const select = container.querySelector("select")!;
  expect(select.options).toHaveLength(5);
  const other = [...select.options].find((option) => JSON.parse(option.value)[3] === "settings-b")!;
  act(() => { select.value = other.value; select.dispatchEvent(new Event("change", { bubbles: true })); });
  expect(container.querySelector(".ov-evaluation-chart__baseline")).toBeNull();
  expect(container.querySelector("tbody")?.textContent).not.toContain("Improved");
  expect(container.querySelector(".ov-evaluation-chart__note")?.textContent).toContain("Settings: Settings B.");
});

it("keeps epoch coordinates separate from step coordinates and labels them correctly", () => {
  const sources = fixture();
  sources[0].stages.evaluation.push(result("epoch", { checkpoint: 2.5, checkpointUnit: "epochs", value: .8 }));
  const { container } = render(<EvaluationCheckpointChart sources={sources} />);
  const select = container.querySelector("select")!;
  const epochs = [...select.options].find((option) => JSON.parse(option.value)[6] === "epochs")!;
  expect(epochs.text).toContain("epochs");
  act(() => { select.value = epochs.value; select.dispatchEvent(new Event("change", { bubbles: true })); });
  expect(container.querySelector("svg")?.textContent).toContain("Checkpoint epoch");
  expect(container.querySelector("circle")?.getAttribute("aria-label")).toContain("Epoch 2.5");
  expect(container.querySelector(".ov-evaluation-chart__baseline")).toBeNull();
  expect(container.querySelector("thead")?.textContent).toContain("Checkpoint epoch");
});
