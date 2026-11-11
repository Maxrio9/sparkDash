// Copyright 2026 Maxrio9. SPDX-License-Identifier: Apache-2.0
import { readdir } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import path from "node:path";
import { validId } from "./schema.js";

export async function loadProgressProviders(directory = new URL("./providers/", import.meta.url)) {
  let files;
  try {
    files = await readdir(directory, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
  const providers = [];
  const ids = new Set();
  for (const file of files.sort((a, b) => a.name.localeCompare(b.name))) {
    if (!file.isFile() || !file.name.endsWith(".js")) continue;
    try {
      const url = directory instanceof URL ? new URL(file.name, directory) : pathToFileURL(path.join(directory, file.name));
      const provider = (await import(url.href)).default;
      if (!provider || !validId(provider.id) || typeof provider.collect !== "function" || ids.has(provider.id)) {
        throw new Error("Invalid provider export");
      }
      providers.push(provider);
      ids.add(provider.id);
    } catch {
      console.warn(`[training] Could not load progress provider ${file.name}`);
    }
  }
  return providers;
}
