import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import "../../styles/training-progress.css";
import { useId, useState } from "react";
import type { FinetuneProgress, FinetuneSource, FinetuneStage, TrainingRun } from "../../api/types";
import { EvaluationCheckpointChart } from "./EvaluationCheckpointChart";
import { hasLiveTraining, trainingRuns } from "./useFinetuneProgress";

const count = new Intl.NumberFormat();
const loss = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 6 });
function statusLabels(t: TFunction) { return {
  complete: [t("status.complete"), "tag--good"],
  running: [t("status.running"), "tag--acc"],
  waiting: [t("status.waiting"), ""],
  stopped: [t("status.stopped"), ""],
  failed: [t("status.failed"), "tag--bad"],
  unknown: [t("status.unknown"), "tag--warn"],
} satisfies Record<FinetuneStage["state"], [string, string]>; }

function formatEta(seconds: number, t: TFunction) {
  const minutes = Math.round(seconds / 60);
  if (minutes < 1) return t("eta.short");
  if (minutes < 60) return t("eta.minutes", { minutes });
  return t("eta.hours", { hours: Math.floor(minutes / 60), minutes: minutes % 60 });
}

function stageCount(stage: FinetuneStage) {
  if (stage.completed === null && stage.total === null) return null;
  const amount = stage.completed === null ? "—" : count.format(stage.completed);
  const fraction = stage.total === null ? amount : `${amount} / ${count.format(stage.total)}`;
  return stage.unit ? `${fraction} ${stage.unit}` : fraction;
}

function metricValue(value: number | string | null) {
  return value === null ? "—" : typeof value === "number" ? loss.format(value) : value;
}

function ResourceList({ resources, label }: { resources: NonNullable<FinetuneStage["resources"]>; label: string }) {
  const { t } = useTranslation("training");
  const names = { cpu: "CPU", gpu: "GPU", accelerator: t("resource.accelerator") };
  const rows = (items: typeof resources) => (
    <ul className="ov-finetune-resources" aria-label={label}>
      {items.map((resource, index) => (
        <li key={`${resource.kind}:${resource.id}:${index}`} title={`${names[resource.kind]} ${resource.id}`}>
          <span>{resource.scope === "affinity" ? t("resource.affinity") : resource.scope === "visible" ? t("resource.visible") : resource.scope === "assigned" ? t("resource.assigned") : t("resource.reported")}</span>
          {resource.label ?? `${names[resource.kind]} ${resource.id}`}
        </li>
      ))}
    </ul>
  );
  return resources.length ? (
    <div className="ov-finetune-resource-list">
      {rows(resources.slice(0, 3))}
      {resources.length > 3 ? <details className="ov-finetune-details">
        <summary>{t("resource.more", { count: resources.length - 3 })}</summary>
        {rows(resources.slice(3))}
      </details> : null}
    </div>
  ) : null;
}

function StageDependencies({ dependencies }: { dependencies: NonNullable<FinetuneStage["dependencies"]> }) {
  const { t } = useTranslation("training");
  const labels = { waiting: t("status.waiting"), ready: t("status.ready"), failed: t("status.failed"), unknown: t("status.unknown") };
  const tones = { waiting: "tag--warn", ready: "tag--good", failed: "tag--bad", unknown: "" };
  const rows = (items: typeof dependencies) => <ul className="ov-finetune-dependencies" aria-label={t("input.required")}>
    {items.map((dependency) => <li key={dependency.id} title={dependency.id}>
      <span>{dependency.label}</span><span className={`tag ${tones[dependency.state]}`}>{labels[dependency.state]}</span>
    </li>)}
  </ul>;
  return dependencies.length ? (
    <div className="ov-finetune-inputs">
      <span className="ov-finetune-stage__detail">{t("input.title")}</span>
      {rows(dependencies.slice(0, 3))}
      {dependencies.length > 3 ? <details className="ov-finetune-details">
        <summary>{t("input.more", { count: dependencies.length - 3 })}</summary>
        {rows(dependencies.slice(3))}
      </details> : null}
    </div>
  ) : null;
}

function StageRow({ stage, stale, resultMetrics, observedAt }: {
  stage: FinetuneStage; stale: boolean; resultMetrics: boolean; observedAt: number | null;
}) {
  const { t, i18n } = useTranslation("training");
  const time = new Intl.DateTimeFormat(i18n.language, { hour: "2-digit", minute: "2-digit" });
  const finishTime = new Intl.DateTimeFormat(i18n.language, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", timeZoneName: "short" });
  const [label, tone] = statusLabels(t)[stage.state];
  const quantity = stageCount(stage);
  const measured = stage.completed !== null && stage.total !== null && stage.total > 0
    ? { value: stage.completed, max: stage.total }
    : null;
  const showProgress = stage.state !== "complete" && (measured || (stage.state === "running" && !stale));
  const showTimestamp = stage.state === "running" || stage.state === "complete" || (stage.state === "waiting" && stage.completed != null);
  const updated = showTimestamp && stage.updatedAt != null ? new Date(stage.updatedAt) : null;
  const health = stale ? undefined : stage.health;
  const healthProblem = health?.state === "stalled" || health?.state === "unresponsive";
  const eta = stage.state === "running" && !stale && !healthProblem && stage.etaSeconds != null ? stage.etaSeconds : null;
  const finish = eta !== null && observedAt !== null ? new Date(observedAt + eta * 1000) : null;
  const metadata = [
    stage.loss != null && t("stage.loss", { value: loss.format(stage.loss) }),
    stage.latestCheckpoint != null && t("stage.latest", { value: count.format(stage.latestCheckpoint) }),
    stage.nextCheckpoint != null && t("stage.next", { value: count.format(stage.nextCheckpoint) }),
    ...(!resultMetrics ? stage.metrics ?? [] : []).map(({ label, value, unit }) =>
      `${label} ${metricValue(value)}${unit ? ` ${unit}` : ""}`),
  ].filter(Boolean).join(" · ");

  return (
    <li className="ov-finetune-stage" data-state={stage.state}>
      <div className="ov-finetune-stage__line">
        <span className="ov-finetune-stage__label">{stage.label}</span>
        <span className={`tag ${stale && stage.state === "running" ? "tag--warn" : tone}`}>
          {stale && stage.state === "running" ? t("stage.lastRunning") : label}
        </span>
      </div>
      {stage.activity ? <div className="ov-finetune-activity" data-activity={stage.activity.kind}>
        <span>{stale ? t("stage.lastActivity") : t("stage.activity")}</span>
        <b>{stage.activity.label}</b>
        {stage.activity.detail ? <span>{stage.activity.detail}</span> : null}
      </div> : null}
      {stage.resources?.length ? <ResourceList resources={stage.resources} label={t("resource.label", { label: stage.label })} /> : null}
      {quantity ? (
        <div className="ov-finetune-stage__line ov-finetune-stage__quantity mono">
          <span>{quantity}</span>
          {stage.state !== "complete" && measured ? <span>{Math.floor(100 * measured.value / measured.max)}%</span> : null}
        </div>
      ) : null}
      {health ? <div className="ov-finetune-health" data-health={health.state}>
        <span className={healthProblem ? "tag tag--warn" : "ov-finetune-stage__detail"}>
          {({ healthy: t("health.healthy"), waiting: t("health.waiting"), stalled: t("health.stalled"), unresponsive: t("health.unresponsive"), unknown: t("health.unknown") })[health.state]}
        </span>
        <span className="ov-finetune-stage__detail">{health.detail}</span>
      </div> : null}
      {showProgress ? (
        <progress
          className="ov-finetune-progress"
          aria-label={t("stage.progress", { label: stage.label })}
          value={measured?.value}
          max={measured?.max}
        />
      ) : null}
      {eta !== null ? (
        <div className="ov-finetune-estimate">
          <span>{t("stage.eta", { duration: formatEta(eta, t) })}</span>
          {finish && Number.isFinite(finish.valueOf()) ? (
            <span>{t("stage.finish")} <time aria-label={t("stage.finishLabel", { label: stage.label })} dateTime={finish.toISOString()} title={finish.toLocaleString()}>
              {finishTime.format(finish)}
            </time></span>
          ) : null}
        </div>
      ) : null}
      {metadata ? <span className="ov-finetune-stage__detail">{metadata}</span> : null}
      {resultMetrics && stage.metrics && stage.metrics.length > 0 ? (
        <dl className="ov-finetune-statistics" aria-label={t("stage.statistics", { label: stage.label })}>
          {stage.metrics.map(({ label, value, unit }, index) => (
            <div key={`${label}-${index}`}>
              <dt>{label}</dt>{" "}
              <dd className="mono">{metricValue(value)}{unit ? ` ${unit}` : ""}</dd>
            </div>
          ))}
        </dl>
      ) : null}
      {stage.detail ? <span className="ov-finetune-stage__detail">{stage.detail}</span> : null}
      {stage.dependencies?.length ? <StageDependencies dependencies={stage.dependencies} /> : null}
      {updated && Number.isFinite(updated.valueOf()) ? (
        <time className="ov-finetune-stage__detail" dateTime={updated.toISOString()} title={updated.toLocaleString()}>
          {stage.state === "complete" ? t("stage.completedAt", { time: finishTime.format(updated) }) : t("stage.updatedAt", { time: time.format(updated) })}
        </time>
      ) : null}
    </li>
  );
}

function StageList({ stages, stale, resultMetrics = false, observedAt = null }: {
  stages: FinetuneStage[]; stale: boolean; resultMetrics?: boolean; observedAt?: number | null;
}) {
  return stages.length > 0 ? (
    <ul className="ov-finetune-list">
      {stages.map((stage) => <StageRow key={stage.id} stage={stage} stale={stale} resultMetrics={resultMetrics} observedAt={observedAt} />)}
    </ul>
  ) : null;
}

function DataStages({ stages, stale, observedAt }: { stages: FinetuneStage[]; stale: boolean; observedAt: number | null }) {
  const { t } = useTranslation("training");
  const summaryStage = stages.at(-1);
  if (!summaryStage || stages.length < 2 || stages.some((stage) => stage.state !== "complete")) {
    return <StageList stages={stages} stale={stale} observedAt={observedAt} />;
  }
  const preparation = stages.filter((stage) => stage !== summaryStage);
  return (
    <>
      <StageList stages={[summaryStage]} stale={stale} observedAt={observedAt} />
      <details className="ov-finetune-details">
        <summary>{t("history.data", { count: preparation.length })}</summary>
        <StageList stages={preparation} stale={stale} />
      </details>
    </>
  );
}

function EvaluationStages({ stages, stale, observedAt }: { stages: FinetuneStage[]; stale: boolean; observedAt: number | null }) {
  const { t } = useTranslation("training");
  const current = stages.filter(({ state }) => state !== "complete" && state !== "waiting");
  const [next, ...later] = stages.filter(({ state }) => state === "waiting");
  const completed = stages.filter(({ state }) => state === "complete");
  const latest = completed.reduce<FinetuneStage | undefined>((previous, stage) =>
    !previous || (stage.updatedAt ?? -Infinity) >= (previous.updatedAt ?? -Infinity) ? stage : previous, undefined);
  const older = completed.filter((stage) => stage !== latest);
  return (
    <>
      <StageList stages={next ? [...current, next] : current} stale={stale} resultMetrics observedAt={observedAt} />
      {later.length > 0 ? (
        <details className="ov-finetune-details">
          <summary>{t("history.later", { count: later.length })}</summary>
          <StageList stages={later} stale={stale} resultMetrics />
        </details>
      ) : null}
      {latest ? (
        <>
          <p className="ov-finetune-result-label">{t("history.latest")}</p>
          <StageList stages={[{ ...latest, metrics: latest.metrics?.slice(0, 4) }]} stale={stale} resultMetrics />
          {latest.metrics && latest.metrics.length > 4 ? (
            <details className="ov-finetune-details">
              <summary>{t("history.full", { count: latest.metrics.length })}</summary>
              <StageList stages={[{ ...latest, metrics: latest.metrics.slice(4), detail: undefined }]} stale={stale} resultMetrics />
            </details>
          ) : null}
        </>
      ) : null}
      {older.length > 0 ? (
        <details className="ov-finetune-details">
          <summary>{t("history.earlier", { count: older.length })}</summary>
          <StageList stages={older} stale={stale} resultMetrics />
        </details>
      ) : null}
    </>
  );
}

function TrainingStages({ stages, stale, observedAt }: { stages: FinetuneStage[]; stale: boolean; observedAt: number | null }) {
  const { t } = useTranslation("training");
  const history = stages.filter(({ state }) => state === "complete" || state === "stopped");
  const current = stages.filter((stage) => !history.includes(stage));
  return (
    <>
      <StageList stages={current} stale={stale} observedAt={observedAt} />
      {history.length > 0 ? (
        <details className="ov-finetune-details">
          <summary>{t("history.training", { count: history.length })}</summary>
          <StageList stages={history} stale={stale} />
        </details>
      ) : null}
    </>
  );
}

function SourceStages({ source, group, requestFailed }: {
  source: FinetuneSource;
  group: keyof FinetuneSource["stages"];
  requestFailed: boolean;
}) {
  const { t, i18n } = useTranslation("training");
  const stages = source.stages[group];
  const stale = requestFailed || source.status !== "live";
  const updated = source.updatedAt === null ? null : new Date(source.updatedAt);
  const status = source.status === "unavailable" ? t("source.unavailable") : stale ? t("source.stale") : null;

  return (
    <div className="ov-finetune-source">
      <div className="ov-finetune-source__line">
        <span>{source.label}</span>
        {status ? (
          <span className="tag tag--warn" title={updated ? t("source.lastRead", { time: updated.toLocaleString(i18n.language) }) : undefined}>
            {status}
          </span>
        ) : null}
      </div>
      {source.resources?.length ? <ResourceList resources={source.resources} label={t("resource.label", { label: source.label })} /> : null}
      {stages.length === 0 ? (
        <p className="ov-finetune-empty">
          {source.status === "live" && !requestFailed ? t("source.noProgress") : t("source.waiting")}
        </p>
      ) : group === "evaluation" ? (
        <EvaluationStages stages={stages} stale={stale} observedAt={source.updatedAt} />
      ) : group === "data" ? (
        <DataStages stages={stages} stale={stale} observedAt={source.updatedAt} />
      ) : (
        <TrainingStages stages={stages} stale={stale} observedAt={source.updatedAt} />
      )}
    </div>
  );
}

const groups = [ { id: "data" }, { id: "training" }, { id: "evaluation" } ] as const;
function groupTitles(t: TFunction) { return { data: t("group.data"), training: t("group.training"), evaluation: t("group.evaluation") }; }
type GroupId = typeof groups[number]["id"];

function groupSummary(sources: FinetuneSource[], group: GroupId, t: TFunction) {
  const rows = sources.flatMap((source) => source.stages[group].map((stage) => ({ stage, live: source.status === "live" })));
  const complete = rows.filter(({ stage }) => stage.state === "complete").length;
  const running = rows.filter(({ stage, live }) => stage.state === "running" && live);
  const failed = rows.some(({ stage }) => stage.state === "failed");
  const trainingContinues = sources.some((source) => source.status === "live" && source.stages.training
    .some((stage) => stage.state === "running" || stage.state === "waiting"));
  if (group === "evaluation" && complete === rows.length && trainingContinues) {
    const stale = rows.some(({ live }) => !live);
    return {
      state: stale ? "unknown" : "waiting",
      status: stale ? t("group.unavailable") : t("group.awaiting"),
      detail: stale ? t("group.lastRead", { detail: t("group.pending", { count: complete }) }) : t("group.pending", { count: complete }),
    };
  }
  const state = group === "evaluation" && failed ? "failed" : running.length > 0 ? "running" : rows.length === complete ? "complete"
    : failed ? "failed"
      : rows.some(({ stage }) => stage.state === "running") ? "unknown" : "waiting";
  const status = state === "running" ? t("group.inProgress") : state === "complete" ? t("status.complete")
    : state === "failed" ? t("group.attention") : state === "unknown" ? t("stage.lastRunning") : t("status.waiting");
  const current = running.length === 1 ? running[0].stage : null;
  const quantity = current ? stageCount(current) : null;
  const progress = current && current.completed !== null && current.total !== null && current.total > 0
    ? Math.floor(100 * current.completed / current.total) : null;
  const detail = quantity ? `${quantity}${progress === null ? "" : ` · ${progress}%`}`
    : running.length > 0 ? t("group.runningCount", { running: running.length, complete, total: rows.length })
      : t("group.completedCount", { complete, total: rows.length });
  return { state, status, detail };
}

function RunProgress({ run, runId }: { run: TrainingRun; runId: string }) {
  const { t } = useTranslation("training");
  const titles = groupTitles(t);
  const [selection, setSelection] = useState<Set<GroupId> | null>(null);
  const devices = new Set(run.sources.map((source) => source.sparkId ?? source.id)).size;
  const populated = groups.filter(({ id }) => run.sources.some((source) => source.stages[id].length > 0));
  const running = populated.filter(({ id }) => groupSummary(run.sources, id, t).state === "running");
  const current = populated.filter(({ id }) => running.some((group) => group.id === id)
    || (id === "evaluation" && running.some((group) => group.id === "training")));
  const automatic = current.length ? current : populated;
  const visible = selection === null ? automatic : populated.filter(({ id }) => selection.has(id));

  function toggleStage(id: GroupId) {
    setSelection((previous) => {
      const next = new Set(previous ?? automatic.map((group) => group.id));
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  return (
    <article className="ov-finetune-run" aria-labelledby={runId}>
      <header className="ov-finetune-head">
        <h3 className="ov-card__sub" id={runId}>{run.campaign}</h3>
        <div className="ov-finetune-context">
          <span className="ov-finetune-devices">{t("run.devices", { count: devices })}</span>
          {run.model ? <span className="ov-finetune-model mono">{run.model}</span> : null}
        </div>
      </header>
      <nav className="ov-finetune-pipeline" aria-label={t("run.stages", { campaign: run.campaign })}>
        {populated.map(({ id }, index) => {
          const summary = groupSummary(run.sources, id, t);
          const showing = visible.some((group) => group.id === id);
          return (
            <button
              type="button"
              className="ov-finetune-step"
              data-state={summary.state}
              data-stage={id}
              aria-pressed={showing}
              aria-controls={`${runId}-work`}
              key={id}
              onClick={() => toggleStage(id)}
            >
              <span className="ov-finetune-step__number" aria-hidden="true">{summary.state === "complete" ? "✓" : index + 1}</span>
              <span className="ov-finetune-step__content">
                <span className="ov-finetune-step__title">{titles[id]}<span>{summary.status}</span></span>
                <span className="ov-finetune-step__detail">{summary.detail}</span>
              </span>
            </button>
          );
        })}
      </nav>
      <div className="ov-finetune-view">
        <span>{selection === null ? t("run.current") : t("run.selected")}</span>
      </div>
      <div className="ov-finetune-grid" data-count={visible.length} data-evaluation={visible.some(({ id }) => id === "evaluation")} id={`${runId}-work`}>
        {visible.length === 0 ? <p className="ov-finetune-empty ov-finetune-empty-selection" role="status">{t("run.none")}</p> : null}
        {visible.map(({ id }) => (
          <section className="ov-finetune-group" data-group={id} key={id} aria-labelledby={`${runId}-${id}`}>
            <h4 id={`${runId}-${id}`}>{titles[id]}</h4>
            {id === "evaluation" ? <EvaluationCheckpointChart sources={run.sources} /> : null}
            {run.sources.filter((source) => source.stages[id].length > 0).map((source) => (
              <SourceStages key={source.id} source={source} group={id} requestFailed={false} />
            ))}
          </section>
        ))}
      </div>
    </article>
  );
}

export function FinetuneProgressCard({ snapshot, requestFailed }: {
  snapshot: FinetuneProgress | null;
  requestFailed: boolean;
}) {
  const { t } = useTranslation("training");
  const headingId = useId();
  const reported = trainingRuns(snapshot);
  const hasTraining = reported.some(hasLiveTraining);
  const runs = reported.filter((run) => run.sources.some((source) => source.status === "live" &&
    groups.some(({ id }) => source.stages[id].some((stage) => stage.state === "running"))));

  if (requestFailed || !hasTraining || runs.length === 0) return null;
  return (
    <section className="panel ov-card ov-finetune" aria-labelledby={headingId}>
      <header className="ov-finetune-head">
        <h2 className="ov-card__title" id={headingId}>{t("title")}</h2>
      </header>
      {runs.map((run, index) => <RunProgress run={run} key={run.id} runId={`${headingId}-${index}`} />)}
    </section>
  );
}
