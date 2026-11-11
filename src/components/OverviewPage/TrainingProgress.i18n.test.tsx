import { act } from "react";
import { afterEach, expect, it } from "vitest";
import { setLocale } from "../../i18n";
import type { FinetuneProgress } from "../../api/types";
import { render } from "../../testing/render";
import { FinetuneProgressCard } from "./FinetuneProgressCard";

const snapshot: FinetuneProgress = {
  enabled: true, campaign: "Example classification", model: "Example-7B", sources: [{
    id: "worker", label: "Worker A", kind: "training", status: "live", updatedAt: 1000,
    stages: {
      data: [{ id: "data", label: "Training examples", state: "complete", completed: 20, total: 20, unit: "samples" }],
      training: [{ id: "fit", label: "Fit model", state: "running", completed: 10, total: 100, unit: "steps", etaSeconds: 300 }],
      evaluation: [0, 1].map((index) => ({
        id: `result-${index}`, label: `Checkpoint ${index}`, state: "complete" as const, completed: 20, total: 20, unit: "samples",
        comparison: { metric: "accuracy", metricLabel: "Accuracy", value: .7 + index * .1, unit: "fraction",
          seriesId: index ? "candidate" : "baseline", seriesLabel: index ? "Candidate" : "Baseline", checkpoint: index ? 100 : undefined,
          baseline: !index, cohortId: "examples-v1", contextId: "greedy-v1", goal: "maximize" as const, lower: .6, upper: .9 },
      })),
    },
  }],
};

afterEach(async () => { await act(async () => { await setLocale("en"); }); });

it("translates training controls and keeps stage selection, hidden series and intervals across language switches", async () => {
  const { container } = render(<FinetuneProgressCard snapshot={snapshot} requestFailed={false} />);
  const training = container.querySelector<HTMLButtonElement>('[data-stage="training"]')!;
  const data = container.querySelector<HTMLButtonElement>('[data-stage="data"]')!;
  act(() => { training.click(); data.click(); });
  const candidate = [...container.querySelectorAll<HTMLButtonElement>(".ov-evaluation-chart__series")]
    .find((button) => button.textContent === "Candidate")!;
  const intervals = container.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
  act(() => { candidate.click(); intervals.click(); });
  await act(async () => { await setLocale("zh-CN"); });
  expect(container.querySelector("h2")?.textContent).toBe("训练进度");
  expect(container.querySelector('[data-stage="training"]')?.getAttribute("aria-pressed")).toBe("false");
  expect(container.querySelector('[data-stage="data"]')?.getAttribute("aria-pressed")).toBe("true");
  expect(container.textContent).toContain("检查点评估对比");
  expect(container.textContent).toContain("越高越好");
  expect(container.querySelector('[aria-label="得分序列"]')).not.toBeNull();
  expect(candidate.getAttribute("aria-pressed")).toBe("false");
  expect(intervals.checked).toBe(true);
  expect(container.querySelector('[data-series="candidate"] circle')).toBeNull();
  expect(container.textContent).toContain("Candidate");
  expect(container.textContent).toContain("Worker A");
  await act(async () => { await setLocale("en"); });
  expect(container.querySelector("h2")?.textContent).toBe("Training progress");
  expect(container.querySelector('[data-stage="training"]')?.getAttribute("aria-pressed")).toBe("false");
  expect(candidate.getAttribute("aria-pressed")).toBe("false");
  expect(intervals.checked).toBe(true);
});
