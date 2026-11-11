import { useTranslation } from "react-i18next";
import type { HostEvaluation } from "./useFinetuneProgress";

const evaluationCount = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 });

/** A single reported job; rates are never aggregated across independent evaluations. */
export function EvaluationSummary({ evaluation }: { evaluation: HostEvaluation }) {
  const { t } = useTranslation("training");
  const completed = evaluation.completed ?? null;
  const total = evaluation.total ?? null;
  const quantity = completed === null && total === null ? null :
    `${completed === null ? "—" : evaluationCount.format(completed)}${total === null ? "" : ` / ${evaluationCount.format(total)}`}${evaluation.unit ? ` ${evaluation.unit}` : ""}`;
  const identity = [...new Set([evaluation.campaign, evaluation.model, evaluation.sourceLabel].filter(Boolean))].join(" · ");
  const promptTelemetry = evaluation.promptsPerMinute !== undefined || evaluation.promptConcurrency !== undefined
    || evaluation.unit === undefined || evaluation.unit === "prompts";
  const resourceScopes = { visible: t("resource.visible"), affinity: t("resource.affinity"), assigned: t("resource.assigned") };
  return (
    <div className="ov-device-evaluation" data-evaluation-id={evaluation.id}>
      <div className="ov-device-evaluation__heading">
        <b>{evaluation.label}</b>
        {quantity ? <span className="mono">{quantity}</span> : null}
      </div>
      <p className="ov-device-evaluation__identity">{identity}</p>
      {evaluation.activity ? <p className="ov-device-evaluation__identity" data-worker-activity={evaluation.activity.kind}>
        {evaluation.activity.label}{evaluation.activity.detail ? ` · ${evaluation.activity.detail}` : ""}
      </p> : null}
      {evaluation.resources?.length ? <p className="ov-device-evaluation__identity">
        {evaluation.resources.slice(0, 2).map((resource) => `${resource.scope ? `${resourceScopes[resource.scope]} ` : ""}${resource.label ?? `${resource.kind === "accelerator" ? t("resource.accelerator") : resource.kind.toUpperCase()} ${resource.id}`}`).join(" · ")}
      </p> : null}
      {evaluation.health && ["stalled", "unresponsive"].includes(evaluation.health.state) ? <p className="ov-device-evaluation__health" title={evaluation.health.detail}>
        <span className="tag tag--warn">{evaluation.health.state === "stalled" ? t("health.stalled") : t("health.unresponsive")}</span>
        {evaluation.health.detail}
      </p> : null}
      {evaluation.dependencies?.some((input) => input.state === "waiting" || input.state === "failed") ? <p className="ov-device-evaluation__identity">
        {evaluation.dependencies.filter((input) => input.state === "waiting" || input.state === "failed").slice(0, 2)
          .map((input) => input.state === "failed" ? t("device.inputFailed", { label: input.label }) : t("device.waitingFor", { label: input.label })).join(" · ")}
      </p> : null}
      {promptTelemetry ? <div className="ov-tps">
        <div title={t("device.speedHelp")}>
          <div className="eyebrow">{t("device.speed")}</div>
          <div className="big-num ov-tps__num">
            {evaluation.promptsPerMinute?.toFixed(1) ?? "—"}{" "}<small>{t("device.promptsPerMinute")}</small>
          </div>
        </div>
        <div>
          <div className="eyebrow">{t("device.concurrency")}</div>
          <div className="big-num ov-tps__num">
            {evaluation.promptConcurrency ?? "—"}{" "}
            <small>{t("device.atOnce", { count: evaluation.promptConcurrency ?? 0 })}</small>
          </div>
        </div>
      </div> : total !== null && total > 0 && completed !== null ? (
        <progress className="ov-finetune-progress" value={completed} max={total} aria-label={t("stage.progress", { label: evaluation.label })} />
      ) : null}
      {evaluation.metrics?.length ? <p className="ov-device-evaluation__identity">{evaluation.metrics.slice(0, 2)
        .map(({ label, value, unit }) => `${label} ${value === null ? "—" : typeof value === "number" ? evaluationCount.format(value) : value}${unit ? ` ${unit}` : ""}`).join(" · ")}</p> : null}
    </div>
  );
}
