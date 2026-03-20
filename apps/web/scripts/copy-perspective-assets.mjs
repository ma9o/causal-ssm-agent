/**
 * Copies Perspective WASM and CSS assets from node_modules into public/
 * so they can be fetched at runtime without bundler WASM support.
 *
 * Run automatically via postinstall.
 */
import { cpSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const dest = resolve(__dirname, "../public/perspective");

mkdirSync(resolve(dest, "wasm"), { recursive: true });

const assets = [
  {
    src: "@finos/perspective/dist/wasm/perspective-server.wasm",
    dest: "wasm/perspective-server.wasm",
  },
  {
    src: "@finos/perspective-viewer/dist/wasm/perspective-viewer.wasm",
    dest: "wasm/perspective-viewer.wasm",
  },
  {
    src: "@finos/perspective-viewer/dist/css/pro-dark.css",
    dest: "pro-dark.css",
  },
];

function findModule(pkg) {
  try {
    const resolved = import.meta.resolve(pkg);
    return dirname(fileURLToPath(resolved));
  } catch {
    return null;
  }
}

for (const { src, dest: rel } of assets) {
  // Resolve from monorepo root or local node_modules
  const [scope, pkg, ...rest] = src.split("/");
  const pkgRoot = findModule(`${scope}/${pkg}/package.json`);
  if (!pkgRoot) {
    console.warn(`⚠ Could not resolve ${scope}/${pkg} — may not be installed yet`);
    continue;
  }
  const from = resolve(pkgRoot, ...rest);
  const to = resolve(dest, rel);
  try {
    cpSync(from, to);
  } catch (e) {
    console.warn(`⚠ Could not copy ${src}: ${e.message}`);
  }
}
