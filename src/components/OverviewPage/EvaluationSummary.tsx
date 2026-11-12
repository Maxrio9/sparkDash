import { useTranslation } from "react-i18next";
import type { HostEvaluation } from "./useFinetuneProgress";

const evaluationCount = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 });

/** A single reported job; rates are never aggregated across independent evaluations. */
export function EvaluationSummary({ evaluation, showIdentity = false }: { evaluation: HostEvaluation; showIdentity?: boolean }) {
  const { t } = useTranslation("training");
  const completed = evaluation.completed ?? null;
  const total = evaluation.total ?? null;
  const quantity = completed === null && total === null ? null :
    `${completed === null ? "—" : evaluationCount.format(completed)}${total === null ? "" : ` / ${evaluationCount.format(total)}`}${evaluation.unit ? ` ${evaluation.unit}` : ""}`;
  const identity = [...new Set([evaluation.campaign, evaluation.model, evaluation.sourceLabel].filter(Boolean))].join(" · ");
  const campaign = evaluation.campaign !== evaluation.model && evaluation.campaign !== evaluation.label ? evaluation.campaign : undefined;
  const visibleIdentity = showIdentity ? identity : campaign;
  const activity = evaluation.activity && !["running", "training", "evaluating", "scoring", "validating"].includes(evaluation.activity.kind)
    && evaluation.activity.label !== evaluation.label ? evaluation.activity : undefined;
  const promptTelemetry = evaluation.promptsPerMinute !== undefined || evaluation.promptConcurrency !== undefined
    || evaluation.unit === undefined || evaluation.unit === "prompts";
  return (
    <div className="ov-device-evaluation" data-evaluation-id={evaluation.id}>
      <div className="ov-device-evaluation__heading" title={identity}>
        <b>{evaluation.label}</b>
        {quantity ? <span className="mono">{quantity}</span> : null}
      </div>
      {visibleIdentity ? <p className="ov-device-evaluation__identity">{visibleIdentity}</p> : null}
      {activity ? <p className="ov-device-evaluation__identity" data-worker-activity={activity.kind} title={activity.detail}>
        {activity.label}
      </p> : null}
      {evaluation.health && ["stalled", "unresponsive"].includes(evaluation.health.state) ? evaluation.health.detail ? (
        <details className="ov-device-evaluation__health">
          <summary><span className="tag tag--warn">{evaluation.health.state === "stalled" ? t("health.stalled") : t("health.unresponsive")}</span></summary>
          <p>{evaluation.health.detail}</p>
        </details>
      ) : <p className="ov-device-evaluation__health"><span className="tag tag--warn">{evaluation.health.state === "stalled" ? t("health.stalled") : t("health.unresponsive")}</span></p> : null}
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
