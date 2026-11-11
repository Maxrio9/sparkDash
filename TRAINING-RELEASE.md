# Installing the community training release

This is the independently maintained [Maxrio9 training-module fork](https://github.com/Maxrio9/sparkDash) of [MiaAI-Lab/sparkDash](https://github.com/MiaAI-Lab/sparkDash), based on upstream 2.77.0. The optional read-only module is a preview release. Upstream licenses, attribution and existing features are retained. [MODIFICATIONS.md](MODIFICATIONS.md) records the changes.

## Prebuilt frontend archive (recommended)

Requires a supported Node.js release (validated with **22.23.2**), npm, Linux/POSIX tools, Git, and Python **3.11+** on hosts whose training artifacts are collected. SSH collection uses the dashboard's existing unit configuration. No GPU or training framework is required to install the dashboard.

Download `sparkdash-training-2.77.0-training.2.tar.gz` and `SHA256SUMS` from the [release](https://github.com/Maxrio9/sparkDash/releases/tag/v2.77.0-training.2), then:

```sh
sha256sum -c SHA256SUMS
tar -xzf sparkdash-training-2.77.0-training.2.tar.gz
cd sparkdash-training-2.77.0-training.2
npm ci --omit=dev
npm start
```

Open <http://127.0.0.1:5555>. The archive includes the built `dist/` frontend; no frontend build or Git checkout is needed. Dependencies are installed from the lockfile. Add devices using the ordinary dashboard UI. The included settings file contains upstream defaults; private runtime configuration and credentials are never bundled.

Enable monitoring and declare only the folders you intend to expose, on each configured unit:

```sh
TRAINING_PROGRESS_ENABLED=1 \
TRAINING_PROGRESS_ROOTS='["~/training-runs"]' \
npm start
```

Defaults are **off**, with **no search roots**. Per-unit paths, supported native logs, reporting examples, liveness requirements and custom collector interfaces are documented in [the module guide](server/training/README.md). The pane follows active training and hides after all training has stopped; evaluation jobs can still appear in device tiles. Use the synthetic example to explore the UI without running a model.

Keep the default loopback bind. For remote access, follow the upstream README's authentication and allowed-host instructions. Training output can itself contain sensitive names or metrics; select folders and dashboard access accordingly.

## Source installation

```sh
git clone --branch v2.77.0-training.2 https://github.com/Maxrio9/sparkDash.git
cd sparkDash
npm ci
npm run build
npm start
```

## Containers

The fork retains the upstream Dockerfile and compose deployment model. Build from this tagged source with `docker compose build` after `npm ci && npm run build`. Optional training environment variables are passed through both compose files. Python is included in the runtime image. Configure explicit paths as seen **inside the collection process**. The training probe runs as the dashboard process's user; it does not enter the host mount namespace. A root container cannot attest a different user's host trainer, and its `~` points to the container home. For host training, configure an SSH unit as the training user, or deliberately provide matching user IDs, PID visibility and read-only artifact mounts. Remote collection uses the remote user's paths. Linux process evidence is required for native live detection; Windows artifact collection is skipped, and macOS readings cannot establish live training without a custom provider. A trusted provider can normalize another source into the same schema.

Tagged image builds target `ghcr.io/maxrio9/sparkdash:2.77.0-training.2` for amd64 and arm64. Only use that image after its publish workflow succeeds and the package is public; the release archive above is independently installable. Container privileges, host mounts and NVIDIA library paths retain upstream's Linux/Spark defaults; adapt them for your host as described in README. Do not run a second dashboard against an existing installation's configuration files.

## Release verification

```sh
npm ci
npm test
npm run typecheck
npm run build
npm run release:pack
```

`release:pack` archives **committed files only** plus the newly built frontend. It excludes untracked/runtime files and emits a SHA-256 checksum. Keep release outputs in the ignored `release/` directory. The release notes record checks actually completed for that release; architecture targets in an image workflow are not a claim that every physical host has been exercised.
