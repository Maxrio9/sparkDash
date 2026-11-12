// Modified by Maxrio9 in 2026: per-device training evaluation regressions.
import type { HostEvaluation } from "./useFinetuneProgress";
import { act } from "react";
import { describe, expect, it, vi } from "vitest";

vi.mock("../../api/client", () => ({
  fetchLaunchers: () => new Promise(() => {}),
  runLauncher: vi.fn(),
  wakeSpark: vi.fn(),
}));

import { SparkCard } from "./SparkCard";
import { makeSpark } from "../../testing/fixtures";
import { render } from "../../testing/render";
import type { SparkSnapshot } from "../../api/types";

function worker(): SparkSnapshot {
  return { ...makeSpark("w"), role: "worker", workerNode: true, workerHeadId: "h", workerLabel: "cfg-label" } as SparkSnapshot;
}
function head(online: boolean): SparkSnapshot {
  const h = makeSpark("h", online);
  return {
    ...h,
    role: "head",
    metrics: { ...h.metrics, llm: [{ available: true, backend: "vllm", modelId: "big-model", generationTps: 12, prefillTps: 100, contextLength: 4096 }] },
  } as unknown as SparkSnapshot;
}

describe("SparkCard", () => {
  it("shows one bar per card on a multi-GPU host, with load, temperature and power under it", () => {
    const spark = makeSpark("rtx");
    const card = (index: number, used: number, temperature: number, draw: number) => ({
      index, name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", uuid: `GPU-${index}`, temperature, usage: 100,
      power: { draw, limit: 300 }, vram: { used, total: 97887, percentage: Math.round((used / 97887) * 100), available: 97887 - used },
    });
    spark.metrics.gpu = {
      ...spark.metrics.gpu!, usage: 100, temperature: 81, power: { draw: 600.08, limit: 600 },
      vram: { used: 59426, total: 195774, percentage: 30, available: 136348 },
      gpus: [card(0, 52646, 81, 300.04), card(1, 6780, 63, 299.97)],
    };
    const { container } = render(<SparkCard spark={spark} temperatureUnit="celsius" />);
    const bars = [...container.querySelectorAll(".ov-lbl")].map((row) => row.textContent);
    const stats = [...container.querySelectorAll(".ov-gpu-stats")].map((row) => row.textContent);
    expect(bars.filter((b) => b?.startsWith("GPU"))).toEqual(["GPU 0 VRAM51.4 / 95.6 GB", "GPU 1 VRAM6.6 / 95.6 GB"]);
    expect(stats).toEqual(["Load 100%Temp 81°CPower 300 / 300 W", "Load 100%Temp 63°CPower 300 / 300 W"]);
    // Only the hot card's temperature is marked.
    expect([...container.querySelectorAll(".ov-gpu-stats__hot")].map((el) => el.textContent)).toEqual(["81°C"]);
    // No aggregate VRAM, usage, temperature or power rows next to the per-card bars.
    expect(bars.some((b) => /^(VRAM|GPU temp|GPU power)/.test(b ?? ""))).toBe(false);
  });

  it("keeps the single GPU rows on a one-GPU unit", () => {
    const spark = makeSpark("one");
    spark.metrics.gpu = { ...spark.metrics.gpu!, gpus: [{ index: 0, name: "NVIDIA GB10", uuid: null, temperature: 55, usage: 42, power: { draw: 50, limit: 100 }, vram: { used: 1024, total: 4096, percentage: 25, available: 3072 } }] };
    const { container } = render(<SparkCard spark={spark} temperatureUnit="celsius" />);
    const bars = [...container.querySelectorAll(".ov-lbl")].map((row) => row.textContent);
    expect(bars).toContain("GPU42%");
    expect(bars).toContain("GPU temp55°C");
    expect(bars).toContain("GPU power50 / 100 W");
    expect(bars.some((b) => b?.startsWith("GPU 0"))).toBe(false);
  });

  it.each([null, undefined, 0, 2])("renders queue %s without inventing a zero", (waiting) => {
    const spark = makeSpark("fixture");
    Object.assign(spark.metrics.llm[0], {
      backend: "tensorfold", slotsActive: 8, slotsTotal: 8,
      requestsRunning: 8, requestsWaiting: waiting,
    });
    const { container } = render(<SparkCard spark={spark} temperatureUnit="celsius" />);
    expect(container.querySelector(".ov-chips")?.textContent).toContain(
      waiting == null ? "8 running · queue unknown" : `8 running · ${waiting} queued`
    );
  });

  it("renders unknown slot occupancy without presenting it as zero", () => {
    const spark = makeSpark("fixture");
    Object.assign(spark.metrics.llm[0], {
      backend: "tensorfold", slotsActive: null, slotsTotal: 8, requestsRunning: null,
    });
    const { container } = render(<SparkCard spark={spark} temperatureUnit="celsius" />);
    expect(container.querySelector(".ov-chips")?.textContent).toContain("?/8 slots");
  });

  it("names the parallel capacity next to running and queued, so --parallel 4 and 8 read differently", () => {
    for (const total of [4, 8]) {
      const spark = makeSpark("fixture");
      Object.assign(spark.metrics.llm[0], {
        backend: "tensorfold", slotsActive: 1, slotsTotal: total, requestsRunning: 1, requestsWaiting: 0,
      });
      const { container } = render(<SparkCard spark={spark} temperatureUnit="celsius" />);
      const chip = container.querySelector(".ov-chips")!;
      expect(chip.textContent).toContain(`1 running · 0 queued · ${total} slots`);
      expect(chip.querySelector(`[title="${total} requests decode in parallel on this engine"]`)).not.toBeNull();
    }
  });

  it("adds no slot count when the engine reports no capacity", () => {
    const spark = makeSpark("fixture");
    Object.assign(spark.metrics.llm[0], {
      backend: "vllm", slotsActive: 0, slotsTotal: 0, requestsRunning: 2, requestsWaiting: 1,
    });
    const { container } = render(<SparkCard spark={spark} temperatureUnit="celsius" />);
    const text = container.querySelector(".ov-chips")!.textContent!;
    expect(text).toContain("2 running · 1 queued");
    expect(text).not.toContain("slots");
  });
  it("lists private loaded models alongside the monitored model without duplicates", () => {
    const spark = head(true);
    spark.metrics.residentModels = [
      { modelId: "big-model", container: "primary-vllm" },
      { modelId: "datalab-to/surya-ocr-2", container: "surya-vllm" },
    ];
    const { container } = render(<SparkCard spark={spark} temperatureUnit="celsius" />);
    const rows = [...container.querySelectorAll(".ov-modelrow code")].map((row) => row.textContent);
    expect(rows).toEqual(["big-model", "datalab-to/surya-ocr-2"]);
    expect(container.querySelectorAll(".ov-modelrow")[1].textContent).toContain("Loaded");
  });

  it("does not present an offline unit's discovered models as loaded", () => {
    const spark = head(false);
    spark.metrics.residentModels = [{ modelId: "surya-private", container: "surya-vllm" }];
    const { container } = render(<SparkCard spark={spark} temperatureUnit="celsius" />);
    expect(container.textContent).not.toContain("surya-private");
  });
  it("is a plain container whose name is the only link; no interactive element nests another", () => {
    const onSelect = vi.fn();
    const { container } = render(<SparkCard spark={makeSpark("a")} temperatureUnit="celsius" onSelect={onSelect} />);
    const card = container.querySelector(".ov-sc")!;
    expect(card.getAttribute("role")).toBeNull();
    expect(card.getAttribute("tabindex")).toBeNull();
    const link = card.querySelector<HTMLAnchorElement>("a.ov-sc__link")!;
    expect(link.textContent).toBe("Spark a");
    for (const el of card.querySelectorAll("button, a, [role=button]")) {
      expect(el.querySelector("button, a, [role=button]")).toBeNull();
    }
    act(() => link.click());
    expect(onSelect).toHaveBeenCalledWith("a");
  });

  it("does not select when clicking the card body (only the stretched name does)", () => {
    const onSelect = vi.fn();
    const { container } = render(<SparkCard spark={makeSpark("a")} temperatureUnit="celsius" onSelect={onSelect} />);
    act(() => container.querySelector<HTMLElement>(".ov-sc__body")?.click());
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("a worker shows its online head's model", () => {
    const { container } = render(<SparkCard spark={worker()} headSpark={head(true)} temperatureUnit="celsius" />);
    expect(container.textContent).toContain("big-model");
  });

  it("a worker does not show an offline head's stale model", () => {
    const { container } = render(<SparkCard spark={worker()} headSpark={head(false)} temperatureUnit="celsius" />);
    expect(container.textContent).not.toContain("big-model");
    expect(container.textContent).toContain("cfg-label");
    expect(container.textContent).toContain("Head is offline");
  });

  it("an online head with no available probe is unready, not known to have no model", () => {
    // A head that is cold-starting or switching model produces the same empty probe
    // state as an idle one, so residency is unknown and must not be asserted.
    const unready = { ...head(true) } as SparkSnapshot;
    unready.metrics = {
      ...unready.metrics,
      llm: [{ available: false, backend: "vllm", modelId: null, generationTps: 0, prefillTps: 0, contextLength: null }],
    } as unknown as SparkSnapshot["metrics"];
    const bare = { ...worker(), workerLabel: "", workerDerivedLabel: "" } as SparkSnapshot;
    const { container } = render(<SparkCard spark={bare} headSpark={unready} temperatureUnit="celsius" />);
    expect(container.textContent).not.toContain("No model");
    expect(container.textContent).toContain("not ready");
  });

  it("says when no LLM port is watched on the head instead of claiming no model", () => {
    const unwatched = { ...head(true) } as SparkSnapshot;
    unwatched.metrics = { ...unwatched.metrics, llm: [] } as unknown as SparkSnapshot["metrics"];
    const bare = { ...worker(), workerLabel: "", workerDerivedLabel: "" } as SparkSnapshot;
    const { container } = render(<SparkCard spark={bare} headSpark={unwatched} temperatureUnit="celsius" />);
    expect(container.textContent).not.toContain("No model");
    expect(container.textContent).toContain("No LLM port is watched on the head");
  });
});

describe("SparkCard per-stream decode rate", () => {
  it("shows the per-stream rate when the engine reports one", () => {
    const spark = head(true);
    Object.assign(spark.metrics.llm[0], { generationTps: 40, perStreamTps: 18.5 });
    const { container } = render(<SparkCard spark={spark} temperatureUnit="celsius" />);
    expect(container.textContent).toContain("18.5 tok/s per stream");
  });
  it("omits the per-stream line when the engine has no histogram", () => {
    const spark = head(true);
    Object.assign(spark.metrics.llm[0], { generationTps: 40, perStreamTps: null });
    const { container } = render(<SparkCard spark={spark} temperatureUnit="celsius" />);
    expect(container.textContent).not.toContain("per stream");
  });
});

describe("SparkCard effective prefill rate", () => {
  it("does not present an unavailable live TensorFold prefill as measured zero", () => {
    const spark = head(true);
    Object.assign(spark.metrics.llm[0], { backend: "tensorfold", prefillTps: 0, perStreamTps: 40, requestsRunning: 3 });
    const { container } = render(<SparkCard spark={spark} temperatureUnit="celsius" />);
    expect(container.textContent).toContain("no live sample");
    expect(container.textContent).toContain("Decode · total");
    expect(container.textContent).toContain("40.0 tok/s · completed");
    expect(container.textContent).not.toContain("tok/s per stream");
  });
  it("labels the completed prefill as including cached tokens", () => {
    const spark = head(true);
    Object.assign(spark.metrics.llm[0], { backend: "tensorfold", prefillTps: 2813, perStreamPrefillTps: 2813, prefillTpsSource: "completed-effective" });
    const { container } = render(<SparkCard spark={spark} temperatureUnit="celsius" />);
    expect(container.textContent).toContain("Prefill · completed");
    expect(container.textContent).toContain("incl. cached tokens");
  });
  it("omits the prefill line when no request has finished", () => {
    const spark = head(true);
    Object.assign(spark.metrics.llm[0], { prefillTps: 0, perStreamPrefillTps: null });
    const { container } = render(<SparkCard spark={spark} temperatureUnit="celsius" />);
    expect(container.textContent).not.toContain("incl. cached tokens");
  });
});
describe("per-device evaluation", () => {
  const desktopEvaluation = {
    model: "Example-7B", label: "Main objective · Step 1,324",
    promptsPerMinute: 6.2, promptConcurrency: 1,
  };

  it("keeps meaningful worker activity and blockers visible with warning details collapsed", () => {
    const evaluationJobs: HostEvaluation[] = [{ ...desktopEvaluation, id: "worker-eval",
      activity: { kind: "transferring", label: "Transferring checkpoint" },
      resources: [{ kind: "gpu", id: "GPU-a", label: "GPU 2", scope: "assigned" }],
      dependencies: [{ id: "checkpoint", label: "Checkpoint 4000", state: "waiting" }],
      health: { state: "unresponsive", detail: "Worker heartbeat is overdue." },
    }];
    const { container } = render(<SparkCard spark={makeSpark("gpu-lab")} evaluationJobs={evaluationJobs} temperatureUnit="celsius" />);
    const evaluation = container.querySelector('[aria-label="Evaluation"]');
    expect(evaluation?.textContent).toContain("Transferring checkpoint");
    expect(evaluation?.textContent).not.toContain("Assigned GPU 2");
    expect(evaluation?.textContent).toContain("Waiting for: Checkpoint 4000");
    expect(evaluation?.textContent).toContain("Heartbeat overdue");
    const health = evaluation?.querySelector<HTMLDetailsElement>(".ov-device-evaluation__health");
    expect(health?.open).toBe(false);
    expect(health?.querySelector("summary")?.textContent).toBe("Heartbeat overdue");
    act(() => { health!.open = true; });
    expect(health?.querySelector("p")?.textContent).toBe("Worker heartbeat is overdue.");
  });

  it("keeps scoring counters, speed and concurrency without repeating routine worker or hardware details", () => {
    const evaluation: HostEvaluation = { ...desktopEvaluation, id: "scoring", campaign: "Acceptance run", sourceLabel: "Desktop evaluator",
      completed: 25, total: 100, unit: "prompts",
      activity: { kind: "evaluating", label: "Scoring prompts", detail: "Scoring prompts with the current model." },
      resources: [{ kind: "cpu", id: "0-7", scope: "affinity" }, { kind: "gpu", id: "GPU-private", scope: "visible" }],
      health: { state: "healthy", detail: "Updates are within the reported interval." },
    };
    const { container } = render(<SparkCard spark={makeSpark("desktop")} evaluationJobs={[evaluation]} temperatureUnit="celsius" />);
    const tile = container.querySelector('[aria-label="Evaluation"]');
    expect(tile?.textContent).toContain("Acceptance run");
    expect(tile?.textContent).toContain("25 / 100 prompts");
    expect(tile?.textContent).toContain("6.2 prompts/min");
    expect(tile?.textContent).toContain("1 prompt at a time");
    expect(tile?.textContent).not.toContain("Scoring prompts");
    expect(tile?.textContent).not.toContain("Desktop evaluator");
    expect(tile?.textContent).not.toContain("GPU-private");
    expect(tile?.textContent).not.toContain("0-7");
    expect(tile?.textContent).not.toContain("Updates are within");
    expect(tile?.querySelector(".ov-device-evaluation__heading")?.getAttribute("title")).toContain("Desktop evaluator");
  });

  it("shows evaluation metrics and model without treating scoring as inference", async () => {
    const spark = makeSpark("desktop");
    spark.metrics.llm = [];
    spark.metrics.gpu!.processes = [{ pid: 1, name: "evaluate.py", vramMB: 3000 }];
    const { container } = render(<SparkCard spark={spark} evaluationJobs={[desktopEvaluation]} temperatureUnit="celsius" />);
    await act(async () => {});

    const metrics = container.querySelector('[aria-label="Evaluation"]');
    expect(metrics?.textContent).toContain("6.2 prompts/min");
    expect(metrics?.textContent).toContain("1 prompt at a time");
    expect(metrics?.textContent).toContain(desktopEvaluation.label);
    expect(container.querySelector(".ov-modelrow")?.textContent).toContain(desktopEvaluation.model);
    expect(container.querySelector(".ov-launch")).toBeNull();
    expect(container.textContent).not.toContain("Decode");
    expect(container.textContent).not.toContain("Prefill");
  });

  it("keeps the monitored model visible while evaluation speed is still being measured", async () => {
    const spark = makeSpark("desktop");
    const evaluation = { ...desktopEvaluation, promptsPerMinute: undefined };
    const { container } = render(<SparkCard spark={spark} evaluationJobs={[evaluation]} temperatureUnit="celsius" />);
    await act(async () => {});

    const metrics = container.querySelector('[aria-label="Evaluation"]');
    expect(metrics?.textContent).toContain("— prompts/min");
    expect(container.textContent).toContain("Decode");
    const models = [...container.querySelectorAll(".ov-modelrow code")].map((row) => row.textContent);
    expect(models).toEqual([spark.metrics.llm[0].modelId, evaluation.model]);
  });

  it("shows independent evaluation jobs alongside a monitored model on the same device", () => {
    const spark = makeSpark("training-and-evaluation");
    const evaluationJobs: HostEvaluation[] = [
      { id: "sft:reporter:validation", model: "SFT model", campaign: "Medical SFT", label: "Validation", completed: 20, total: 100,
        unit: "prompts", promptsPerMinute: 5, promptConcurrency: 2 },
      { id: "dpo:reporter:validation", model: "DPO model", campaign: "Preference alignment", label: "Validation", completed: 40, total: 200,
        unit: "prompts", promptsPerMinute: 7, promptConcurrency: 3 },
    ];
    const { container } = render(<SparkCard spark={spark} evaluationJobs={evaluationJobs} temperatureUnit="celsius" />);
    const evaluation = container.querySelector('[aria-label="Evaluation"]');
    expect(evaluation?.textContent).toContain("2 evaluations running");
    expect(evaluation?.textContent).toContain("Medical SFT");
    expect(evaluation?.textContent).not.toContain("SFT model");
    expect(evaluation?.textContent).toContain("Preference alignment");
    expect(evaluation?.textContent).not.toContain("DPO model");
    expect(evaluation?.textContent).toContain("20 / 100 prompts");
    expect(evaluation?.textContent).toContain("40 / 200 prompts");
    expect(evaluation?.textContent).toContain("5.0 prompts/min");
    expect(evaluation?.textContent).toContain("7.0 prompts/min");
    expect(evaluation?.textContent).not.toContain("12.0 prompts/min");
    expect(container.textContent).toContain("Decode");
    expect(container.querySelector(".ov-modelrow")?.textContent).toContain(spark.metrics.llm[0].modelId);
    expect(container.querySelector(".ov-launch")).toBeNull();
  });

  it("disambiguates colocated jobs that share a campaign and evaluation label", () => {
    const evaluationJobs: HostEvaluation[] = [
      { ...desktopEvaluation, id: "one", campaign: "Shared campaign", label: "Validation", model: "First model", sourceLabel: "Worker one" },
      { ...desktopEvaluation, id: "two", campaign: "Shared campaign", label: "Validation", model: "Second model", sourceLabel: "Worker two" },
    ];
    const { container } = render(<SparkCard spark={makeSpark("desktop")} evaluationJobs={evaluationJobs} temperatureUnit="celsius" />);
    expect(container.querySelector('[data-evaluation-id="one"] .ov-device-evaluation__identity')?.textContent).toBe("Shared campaign · First model · Worker one");
    expect(container.querySelector('[data-evaluation-id="two"] .ov-device-evaluation__identity')?.textContent).toBe("Shared campaign · Second model · Worker two");
  });

  it.each(["batches", null])("bounds the compact job list and preserves evaluation unit %s in the expanded list", (unit) => {
    const spark = makeSpark("gpu-lab");
    const evaluationJobs: HostEvaluation[] = [
      { id: "one", model: "Model 1", label: "Scoring one", promptsPerMinute: 1 },
      { id: "two", model: "Model 2", label: "Scoring two", promptsPerMinute: 2 },
      { id: "vision", model: "Vision model", campaign: "Image classification", label: "Validation images", completed: 25, total: 100,
        unit, metrics: [{ label: "Throughput", value: 54, unit: "images/s" }] },
    ];
    const { container } = render(<SparkCard spark={spark} evaluationJobs={evaluationJobs} temperatureUnit="celsius" />);
    const evaluation = container.querySelector('[aria-label="Evaluation"]');
    expect(evaluation?.querySelectorAll(":scope > .ov-device-evaluation")).toHaveLength(2);
    const more = evaluation?.querySelector("details");
    expect(more?.open).toBe(false);
    expect(more?.querySelector("summary")?.textContent).toBe("More evaluations (1)");
    act(() => { more!.open = true; });
    expect(more?.textContent).toContain("Image classification");
    expect(more?.textContent).not.toContain("Vision model");
    expect(more?.textContent).toContain(`25 / 100${unit ? ` ${unit}` : ""}`);
    expect(more?.textContent).toContain("Throughput 54 images/s");
    expect(more?.textContent).not.toContain("prompts/min");
    const progress = more?.querySelector<HTMLProgressElement>("progress");
    expect([progress?.value, progress?.max]).toEqual([25, 100]);
  });

  it("keeps evaluation available on worker devices and hides all jobs on an offline device", () => {
    const evaluationJobs = [{ ...desktopEvaluation, id: "worker-eval" }];
    const { container, root } = render(<SparkCard spark={worker()} evaluationJobs={evaluationJobs} temperatureUnit="celsius" />);
    expect(container.querySelector('[aria-label="Evaluation"]')?.textContent).toContain(desktopEvaluation.label);
    act(() => root.render(<SparkCard spark={{ ...worker(), online: false }} evaluationJobs={evaluationJobs} temperatureUnit="celsius" />));
    expect(container.querySelector('[aria-label="Evaluation"]')).toBeNull();
  });
});
