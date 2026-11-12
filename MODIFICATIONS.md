# Community fork modifications

Modified by Maxrio9, 2026. This independently maintained distribution is based on
MiaAI-Lab/sparkDash main at `163d109738d7c3cb8ae06b62f05840083a879a5f`
(upstream version 2.77.0), under Apache-2.0. LICENSE, LICENSE-MIT, upstream NOTICE,
third-party notices, credits and upstream history are retained.

The additions implement optional read-only training/evaluation monitoring. They
do not include private deployment adapters, controllers, addresses, credentials,
training examples from real datasets, model weights or real evaluation results.
The preview images and reporting examples use wholly synthetic data.

Modified upstream files carry change notices. JSON package metadata carries a
fork notice. New module sources and fixtures are contributed under Apache-2.0.
The lockfile refresh resolves the inherited npm advisories within the existing
compatible ranges; no new direct dependency is introduced. Server tests use four
concurrent files to reduce ephemeral-port fixture races during full validation.
The default feature state is off, with no discovery roots. Preview 2 also copies the root license/notice files into the runtime image; CI runs in a reaping Debian container under a normal user to exercise shell and permission fixtures.

Preview 3 keeps the pane visible through live running or queued final evaluation after
training stops, follows evaluation in its automatic view and retains manual stage
choices. Terminal or stale results do not keep it open. This changes monitoring
visibility and discovery priority, not training/evaluation control or metric values.

## Modified upstream files

- `.dockerignore`
- `.env.example`
- `.gitignore`
- `CHANGELOG.md`
- `CODEBASE.md`
- `Dockerfile`
- `NOTICE`
- `README.md`
- `docker-compose.image.yml`
- `docker-compose.yml`
- `docs/I18N-COVERAGE.md`
- `package-lock.json`
- `package.json`
- `server/__tests__/isolatedEnv.js`
- `server/__tests__/isolatedEnv.test.js`
- `server/index.js`
- `src/App.tsx`
- `src/api/client.ts`
- `src/api/types.ts`
- `src/components/OverviewPage/OverviewPage.tsx`
- `src/components/OverviewPage/SparkCard.test.tsx`
- `src/components/OverviewPage/SparkCard.tsx`
- `src/i18n/index.ts`

## Added module, examples, checks and release files

- `.github/training-preview-mobile.png`
- `.github/training-preview.png`
- `.github/workflows/validate-training.yml`
- `MODIFICATIONS.md`
- `TRAINING-RELEASE.md`
- `examples/training/README.md`
- `examples/training/demo.py`
- `examples/training/manifest.example.json`
- `examples/training/reporter.py`
- `scripts/__tests__/training-artifacts.test.py`
- `scripts/package-training-release.mjs`
- `server/__tests__/training-artifacts-provider.test.js`
- `server/__tests__/training-comparison.test.js`
- `server/__tests__/training-health.test.js`
- `server/__tests__/training-progress-route.test.js`
- `server/__tests__/training-topologies.test.js`
- `server/training/README.md`
- `server/training/collectors/artifacts.py`
- `server/training/health.js`
- `server/training/index.js`
- `server/training/providers/artifacts.js`
- `server/training/registry.js`
- `server/training/schema.js`
- `src/components/OverviewPage/EvaluationCheckpointChart.test.tsx`
- `src/components/OverviewPage/EvaluationCheckpointChart.tsx`
- `src/components/OverviewPage/EvaluationSummary.tsx`
- `src/components/OverviewPage/FinetuneProgressCard.test.tsx`
- `src/components/OverviewPage/FinetuneProgressCard.tsx`
- `src/components/OverviewPage/TrainingProgress.i18n.test.tsx`
- `src/components/OverviewPage/useFinetuneProgress.test.ts`
- `src/components/OverviewPage/useFinetuneProgress.ts`
- `src/locales/en/training.json`
- `src/locales/zh-CN/training.json`
- `src/styles/device-evaluations.css`
- `src/styles/training-chart.css`
- `src/styles/training-progress.css`
