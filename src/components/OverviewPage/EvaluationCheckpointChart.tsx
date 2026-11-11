import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import { useId, useMemo, useState } from "react";
import type { FinetuneSource, FinetuneStage } from "../../api/types";
import { useElementWidth } from "../ui/useElementWidth";
import "../../styles/training-chart.css";

type Comparison = NonNullable<FinetuneStage["comparison"]>;
type Point = Comparison & { key: string; label: string; updatedAt: number; stale: boolean };
type ComparisonGroup = Pick<Comparison, "metric" | "metricLabel" | "unit" | "cohortId" | "contextId" | "contextLabel" | "goal" | "target" | "checkpointUnit"> & {
  key: string; points: Point[];
};
const score = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 6 });
const axisScore = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 });
const step = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 6 });
const delta = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 5, signDisplay: "always" });
const height = 250;
const pad = { left: 48, right: 16, top: 14, bottom: 42 };

function collectGroups(sources: FinetuneSource[]): ComparisonGroup[] {
  const groups = new Map<string, ComparisonGroup>();
  for (const source of sources) {
    for (const stage of source.stages.evaluation) {
      if (stage.state !== "complete") continue;
      const comparisons = stage.comparisons ?? (stage.comparison ? [stage.comparison] : []);
      for (const comparison of comparisons) {
        if (!Number.isFinite(comparison.value)) continue;
        if (!comparison.baseline && (comparison.checkpoint === undefined || !Number.isFinite(comparison.checkpoint))) continue;
        const key = JSON.stringify([comparison.metric, comparison.unit ?? "", comparison.cohortId ?? "",
          comparison.contextId ?? "", comparison.goal ?? "observe", comparison.target ?? null, comparison.checkpointUnit ?? "steps"]);
        let group = groups.get(key);
        if (!group) {
          const { metric, metricLabel, unit, cohortId, contextId, contextLabel, goal, target, checkpointUnit } = comparison;
          group = { key, points: [], metric, metricLabel, unit, cohortId, contextId, contextLabel, goal, target, checkpointUnit };
          groups.set(key, group);
        }
        const point: Point = { ...comparison, key: `${source.id}:${stage.id}:${comparison.metric}`, label: stage.label,
          updatedAt: stage.updatedAt ?? 0, stale: source.status !== "live" };
        const index = group.points.findIndex((item) => item.seriesId === point.seriesId &&
          item.baseline === point.baseline && item.checkpoint === point.checkpoint);
        if (index < 0) group.points.push(point);
        else if (point.updatedAt >= group.points[index].updatedAt) group.points[index] = point;
      }
    }
  }
  return [...groups.values()].sort((a, b) => b.points.length - a.points.length || a.key.localeCompare(b.key));
}

function seriesNames(points: Point[]) {
  const names = new Map(points.map((point) => [point.seriesId, point.seriesLabel]));
  const counts = new Map<string, number>();
  for (const label of names.values()) counts.set(label, (counts.get(label) ?? 0) + 1);
  for (const [id, label] of names) {
    if (counts.get(label)! > 1) names.set(id, `${label} (${id})`);
  }
  return names;
}

function checkpointDescription(point: Point, t: TFunction) {
  const unit = point.checkpointUnit ?? "steps";
  if (unit === "steps") return t("chart.step", { value: step.format(point.checkpoint!) });
  if (unit === "epochs") return t("chart.epoch", { value: step.format(point.checkpoint!) });
  return t("chart.checkpoint", { value: step.format(point.checkpoint!), unit });
}

function goalLabel(t: TFunction, goal: Comparison["goal"], target?: number, unit?: string) {
  if (goal === "maximize") return t("chart.higher");
  if (goal === "minimize") return t("chart.lower");
  if (goal === "target" && Number.isFinite(target)) return t("chart.target", { value: score.format(target!), unit: unit ? ` ${unit}` : "" });
  return goal === "observe" ? t("chart.observe") : t("chart.unspecified");
}

function objectiveDistance(point: Comparison) {
  if (point.goal === "maximize") return -point.value;
  if (point.goal === "minimize") return point.value;
  if (point.goal === "target" && Number.isFinite(point.target)) return Math.abs(point.value - point.target!);
  return null;
}

function outcome(point: Point, baseline: Point | undefined, t: TFunction) {
  if (point.baseline || !baseline) return null;
  const current = objectiveDistance(point);
  const reference = objectiveDistance(baseline);
  if (current === null || reference === null) return null;
  return current < reference ? t("chart.improved") : current > reference ? t("chart.worse") : t("chart.unchanged");
}

function pointDescription(point: Point, seriesLabel: string, baseline: Point | undefined, t: TFunction) {
  const location = point.baseline ? t("chart.baseline") : checkpointDescription(point, t);
  const value = `${score.format(point.value)}${point.unit ? ` ${point.unit}` : ""}`;
  const change = !point.baseline && baseline ? t("chart.change", { value: delta.format(point.value - baseline.value), baseline: baseline.seriesLabel }) : "";
  const interval = point.lower !== undefined && point.upper !== undefined
    ? t("chart.interval", { lower: score.format(point.lower), upper: score.format(point.upper) }) : "";
  const verdict = outcome(point, baseline, t);
  return `${seriesLabel} · ${location}: ${value}${change}${interval}${verdict ? `; ${verdict}` : ""}${point.stale ? t("chart.stale") : ""}`;
}

/** Compare explicit score metadata only; display labels never establish comparability. */
export function EvaluationCheckpointChart({ sources }: { sources: FinetuneSource[] }) {
  const { t } = useTranslation("training");
  const id = useId();
  const groups = useMemo(() => collectGroups(sources), [sources]);
  const [selected, setSelected] = useState<string | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(() => new Set());
  const [active, setActive] = useState<string | null>(null);
  const [focused, setFocused] = useState<string | null>(null);
  const [showIntervals, setShowIntervals] = useState(false);
  const [ref, width] = useElementWidth();
  const group = groups.find((item) => item.key === selected) ?? groups[0];
  if (!group) return null;

  const names = seriesNames(group.points);
  const series = [...names.entries()].sort(([a], [b]) => a.localeCompare(b));
  const visible = group.points.filter((point) => !hidden.has(`${group.key}:${point.seriesId}`));
  const baselines = group.points.filter((point) => point.baseline);
  const baseline = baselines.length === 1 ? baselines[0] : undefined;
  const checkpoints = visible.filter((point) => !point.baseline);
  const allCheckpoints = group.points.filter((point) => !point.baseline);
  const ranked = checkpoints.filter((point) => objectiveDistance(point) !== null)
    .sort((a, b) => objectiveDistance(a)! - objectiveDistance(b)! || a.checkpoint! - b.checkpoint! || a.seriesId.localeCompare(b.seriesId));
  const best = ranked[0];
  const tied = best ? ranked.filter((point) => objectiveDistance(point) === objectiveDistance(best)).length : 0;
  const checkpointUnit = group.checkpointUnit ?? "steps";
  const coordinateLabel = checkpointUnit === "steps" ? t("chart.coordinateStep") : checkpointUnit === "epochs" ? t("chart.coordinateEpoch") : t("chart.coordinate", { unit: checkpointUnit });
  const maximumStep = Math.max(1, ...allCheckpoints.map((point) => point.checkpoint!));
  const values = visible.flatMap((point) => [point.value, ...(!showIntervals || point.lower === undefined ? [] : [point.lower]),
    ...(!showIntervals || point.upper === undefined ? [] : [point.upper])]);
  const minimum = Math.min(...(values.length ? values : [0]));
  const maximum = Math.max(...(values.length ? values : [1]));
  const span = maximum - minimum || Math.max(Math.abs(maximum) * .1, .1);
  const lower = minimum - span * .12;
  const upper = maximum + span * .12;
  const plotWidth = Math.max(1, width - pad.left - pad.right);
  const plotHeight = height - pad.top - pad.bottom;
  const x = (value: number) => pad.left + value / maximumStep * plotWidth;
  const y = (value: number) => pad.top + (upper - value) / (upper - lower) * plotHeight;
  const yTicks = Array.from({ length: 5 }, (_, index) => lower + index / 4 * (upper - lower));
  const tickCount = width < 420 ? 3 : 5;
  const xTicks = Array.from({ length: tickCount }, (_, index) => index / (tickCount - 1) * maximumStep);
  const selectedPoint = focused ?? active;
  const hovered = group.points.find((point) => point.key === selectedPoint && !hidden.has(`${group.key}:${point.seriesId}`));
  const selectedDescription = hovered ? pointDescription(hovered, names.get(hovered.seriesId)!, baseline, t) : null;
  const choiceLabels = groups.map((item) => `${item.metricLabel}${item.unit ? ` (${item.unit})` : ""} · ${goalLabel(t, item.goal, item.target, item.unit)} · ${item.checkpointUnit ?? "steps"}${item.contextLabel ? ` · ${item.contextLabel}` : ""}`);
  const groupLabel = (_item: ComparisonGroup, index: number) => {
    const label = choiceLabels[index];
    return `${label}${choiceLabels.filter((candidate) => candidate === label).length > 1 ? t("chart.comparisonNumber", { number: index + 1 }) : ""}`;
  };

  return (
    <figure className="ov-evaluation-chart" aria-labelledby={`${id}-title`}>
      <div className="ov-evaluation-chart__head">
        <h5 id={`${id}-title`}>{t("chart.title")}</h5>
        <span>{group.metricLabel}{group.unit ? ` · ${group.unit}` : ""}</span>
      </div>
      <div className="ov-evaluation-chart__objective">
        <span>{goalLabel(t, group.goal, group.target, group.unit)}</span>
        {best ? <span className="ov-evaluation-chart__best">{t("chart.best", { value: score.format(best.value), unit: group.unit ? ` ${group.unit}` : "", series: names.get(best.seriesId), checkpoint: checkpointDescription(best, t) })}{tied > 1 ? t("chart.tied", { count: tied }) : ""}</span> : null}
      </div>
      {groups.length > 1 ? (
        <label className="ov-evaluation-chart__select">
          {t("chart.comparison")}
          <select value={group.key} onChange={(event) => { setSelected(event.target.value); setActive(null); setFocused(null); }}>
            {groups.map((item, index) => <option key={item.key} value={item.key}>{groupLabel(item, index)}</option>)}
          </select>
        </label>
      ) : null}
      <div className="ov-evaluation-chart__legend" aria-label={t("chart.series")}>
        {series.map(([seriesId, seriesLabel], index) => {
          const key = `${group.key}:${seriesId}`;
          const isBaseline = group.points.some((point) => point.seriesId === seriesId && point.baseline);
          return (
            <button key={seriesId} type="button" aria-pressed={!hidden.has(key)}
              className={`ov-evaluation-chart__series ov-evaluation-chart__series--${index % 6}${isBaseline ? " is-baseline" : ""}`}
              onClick={() => setHidden((previous) => {
                const next = new Set(previous);
                if (next.has(key)) next.delete(key); else next.add(key);
                return next;
              })}>
              <span aria-hidden="true" />{seriesLabel}
            </button>
          );
        })}
      </div>
      {group.points.some((point) => point.lower !== undefined && point.upper !== undefined) ? (
        <label className="ov-evaluation-chart__interval-toggle">
          <input type="checkbox" checked={showIntervals} onChange={(event) => setShowIntervals(event.target.checked)} />
          {t("chart.showIntervals")}
        </label>
      ) : null}
      <div ref={ref} className="ov-evaluation-chart__viewport">
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img"
        aria-labelledby={`${id}-title ${id}-description`} className="ov-evaluation-chart__plot"
        onMouseLeave={() => setActive(null)}>
        <desc id={`${id}-description`}>{t("chart.description", { metric: group.metricLabel, unit: checkpointUnit })}</desc>
        {yTicks.map((value, index) => (
          <g key={index}>
            <line x1={pad.left} x2={width - pad.right} y1={y(value)} y2={y(value)} className="ov-evaluation-chart__grid" />
            <text x={pad.left - 7} y={y(value) + 3} textAnchor="end" className="ov-evaluation-chart__tick">{axisScore.format(value)}</text>
          </g>
        ))}
        {xTicks.map((value, index) => <text key={index} x={x(value)} y={height - 23} textAnchor="middle" className="ov-evaluation-chart__tick">{step.format(value)}</text>)}
        <text x={pad.left + plotWidth / 2} y={height - 5} textAnchor="middle" className="ov-evaluation-chart__tick">{coordinateLabel}</text>
        {series.map(([seriesId, seriesLabel], index) => {
          const points = visible.filter((point) => point.seriesId === seriesId).sort((a, b) => (a.checkpoint ?? 0) - (b.checkpoint ?? 0));
          const measured = points.filter((point) => !point.baseline);
          const className = `ov-evaluation-chart__series--${index % 6}`;
          return (
            <g key={seriesId} className={className} data-series={seriesId}>
              {measured.length > 1 ? <polyline points={measured.map((point) => `${x(point.checkpoint!)},${y(point.value)}`).join(" ")}
                className="ov-evaluation-chart__line" /> : null}
              {points.map((point) => {
                const description = pointDescription(point, seriesLabel, baseline, t);
                const pointX = point.baseline ? width - pad.right : x(point.checkpoint!);
                return (
                  <g key={point.key}>
                    {point.baseline ? <line x1={pad.left} x2={width - pad.right} y1={y(point.value)} y2={y(point.value)}
                      className="ov-evaluation-chart__baseline" /> : null}
                    {showIntervals && point.lower !== undefined && point.upper !== undefined ? (
                      <path d={`M${pointX},${y(point.lower)} V${y(point.upper)} M${pointX - 3},${y(point.lower)} H${pointX + 3} M${pointX - 3},${y(point.upper)} H${pointX + 3}`}
                        className="ov-evaluation-chart__interval" />
                    ) : null}
                    <circle cx={pointX} cy={y(point.value)} r={selectedPoint === point.key ? 5 : 3.5}
                      className="ov-evaluation-chart__point" tabIndex={0} role="button" aria-label={description}
                      onMouseEnter={() => setActive(point.key)} onFocus={() => setFocused(point.key)}
                      onBlur={() => setFocused(null)} onClick={() => setActive(point.key)}
                      onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setFocused(point.key); } }}>
                      <title>{description}</title>
                    </circle>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      </div>
      <div className="ov-evaluation-chart__readout" role="status">
        {selectedDescription ?? (visible.length === 0 ? t("chart.hidden") : checkpoints.length === 0
          ? t("chart.onlyBaseline") : baseline
            ? t("chart.inspectBaseline")
            : t("chart.inspect"))}
      </div>
      <figcaption className="ov-evaluation-chart__note">
        {group.cohortId ? t("chart.sameSet") : t("chart.unknownSet")}{" "}
        {group.contextLabel ? `${t("chart.settings", { label: group.contextLabel })} ` : ""}
        {group.contextId ? `${t("chart.sameSettings")} ` : `${t("chart.unknownSettings")} `}
        {t("chart.lines")}
        {group.goal && group.goal !== "observe" ? ` ${t("chart.estimates")}` : ""}
        {showIntervals ? ` ${t("chart.whiskers")}` : ` ${t("chart.intervalsHidden")}`}
      </figcaption>
      <details className="ov-evaluation-chart__table">
        <summary>{t("chart.table", { count: group.points.length })}</summary>
        <div className="ov-evaluation-chart__table-scroll">
          <table>
            <caption>{group.metricLabel}{group.unit ? ` (${group.unit})` : ""}</caption>
            <thead><tr><th scope="col">{t("chart.run")}</th><th scope="col">{coordinateLabel}</th><th scope="col">{group.metricLabel}</th><th scope="col">{t("chart.baselineChange")}</th><th scope="col">{t("chart.outcome")}</th></tr></thead>
            <tbody>{[...group.points].sort((a, b) => Number(!!b.baseline) - Number(!!a.baseline) ||
              a.seriesId.localeCompare(b.seriesId) || (a.checkpoint ?? 0) - (b.checkpoint ?? 0)).map((point) => (
              <tr key={point.key}><th scope="row">{names.get(point.seriesId)}</th>
                <td>{point.baseline ? t("chart.baseline") : step.format(point.checkpoint!)}</td>
                <td>{score.format(point.value)}</td>
                <td>{!point.baseline && baseline ? delta.format(point.value - baseline.value) : "—"}</td>
                <td>{outcome(point, baseline, t) ?? "—"}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
