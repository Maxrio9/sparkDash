// Copyright 2026 Maxrio9. SPDX-License-Identifier: Apache-2.0
// Package committed public source and the production frontend, never runtime data.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const git = (...args) => execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
if (git("status", "--porcelain", "--untracked-files=no")) {
  throw new Error("Commit the public release sources before packaging.");
}
const { version } = JSON.parse(await readFile(path.join(root, "package.json"), "utf8"));
if (!/^[0-9A-Za-z.-]+$/.test(version)) throw new Error("Invalid release version");
await readFile(path.join(root, "dist/index.html"));
const name = `sparkdash-training-${version}`;
const output = path.join(root, "release");
await mkdir(output, { recursive: true });
const stage = await mkdtemp(path.join(tmpdir(), "sparkdash-release-"));
try {
  const sourceTar = path.join(stage, "source.tar");
  execFileSync("git", ["archive", "--format=tar", `--prefix=${name}/`, "-o", sourceTar, "HEAD"], { cwd: root });
  execFileSync("tar", ["-xf", sourceTar, "-C", stage]);
  const install = path.join(stage, name);
  await cp(path.join(root, "dist"), path.join(install, "dist"), { recursive: true });
  await writeFile(path.join(install, "RELEASE.json"), JSON.stringify({
    version, commit: git("rev-parse", "HEAD"),
    upstream: "MiaAI-Lab/sparkDash", upstreamBase: "163d109738d7c3cb8ae06b62f05840083a879a5f",
    distribution: "Maxrio9 community training-module fork", frontendBuilt: true,
  }, null, 2) + "\n");
  const archive = path.join(output, `${name}.tar.gz`);
  execFileSync("tar", ["-czf", archive, "-C", stage, name]);
  const checksum = createHash("sha256").update(await readFile(archive)).digest("hex");
  await writeFile(path.join(output, "SHA256SUMS"), `${checksum}  ${path.basename(archive)}\n`);
  console.log(archive);
} finally {
  await rm(stage, { recursive: true, force: true });
}
