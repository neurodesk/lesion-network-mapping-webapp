#!/usr/bin/env node --no-warnings
// Contract test for web/js/lnm-app.js: the orchestrator class structure and
// import surface. Written before lnm-app.js is created per the project's TDD
// policy.
//
// We inspect the source rather than executing it because the module pulls in
// browser-only globals (fetch, document, NiiVue) that aren't trivial to stub
// in Node. Acorn parses the file to confirm it's syntactically valid; the
// regex checks pin the shape we rely on at runtime.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'acorn';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const APP_PATH = path.join(ROOT, 'web/js/lnm-app.js');

assert.ok(fs.existsSync(APP_PATH), 'web/js/lnm-app.js must exist');
assert.ok(
  !fs.existsSync(path.join(ROOT, 'web/js/spinalcordtoolbox-app.js')),
  'web/js/spinalcordtoolbox-app.js must be deleted (renamed)'
);

const src = fs.readFileSync(APP_PATH, 'utf8');

// Acorn parse — catches stray syntax errors before they ship.
parse(src, { ecmaVersion: 'latest', sourceType: 'module' });

// Class + key methods must exist. Phase 1 stubs runYeoOverlap and exportCsv;
// later phases extend them with chart rendering and full CSV serialization.
assert.match(src, /export\s+class\s+LesionNetworkMappingApp\b/,
  'must export class LesionNetworkMappingApp');
for (const method of ['init', 'setStructural', 'setLesion', 'runYeoOverlap', 'exportCsv']) {
  const re = new RegExp(`\\b${method}\\s*\\(`);
  assert.match(src, re, `LesionNetworkMappingApp must define method ${method}`);
}

// Imports: must pull in the pieces we expect, and must NOT pull in any of the
// SCT modules we deleted in Phase 0.
const requiredImports = [
  /from\s+['"]\.\/controllers\/FileIOController\.js['"]/,
  /from\s+['"]\.\/controllers\/ViewerController\.js['"]/,
  /from\s+['"]\.\/app\/lnm-tasks\.js['"]/,
  /from\s+['"]\.\/app\/lnm-labels\.js['"]/,
  /from\s+['"]\.\/modules\/parcel-overlap\.js['"]/,
  /from\s+['"]\.\/modules\/atlas-loader\.js['"]/
];
for (const re of requiredImports) {
  assert.match(src, re, `lnm-app.js must import ${re}`);
}
const forbiddenImports = [
  /sct-tasks/i,
  /sct-processing/i,
  /\bvertebrae\b/i,
  /\bSpinalCordToolbox\b/,
  /['"]\.\/app\/labels\.js['"]/   // old labels.js (without lnm- prefix)
];
for (const re of forbiddenImports) {
  assert.doesNotMatch(src, re, `lnm-app.js must not reference ${re}`);
}

// runYeoOverlap is the linchpin of the Phase 1 flow: it must call both
// computeParcelOverlap and summarizeNetworkOverlap from parcel-overlap.js so
// the UI gets per-network aggregates. The acorn parse above guarantees the
// file is parseable; here we pin behaviour.
assert.match(src, /computeParcelOverlap\s*\(/,
  'runYeoOverlap must call computeParcelOverlap');
assert.match(src, /summarizeNetworkOverlap\s*\(/,
  'runYeoOverlap must call summarizeNetworkOverlap');

// Yeo asset ID matches the manifest entry; if these drift the loader silently
// fails. Pin the string here so the test catches typos.
assert.match(src, /['"]yeo7-2mm['"]/,
  'lnm-app.js must reference the yeo7-2mm asset ID literal');

// Phase 1c.2: the orchestrator must surface 'X voxels of lesion fall outside
// the atlas' as soon as voxelsOutsideAtlas > 0. Source-grep the wiring so we
// catch typos that would silently drop the warning.
assert.match(src, /voxelsOutsideAtlas/,
  'lnm-app.js must reference voxelsOutsideAtlas (outside-atlas warning wiring)');
assert.match(src, /outsideAtlasWarning/,
  'lnm-app.js must reference the #outsideAtlasWarning element');

// runYeoOverlap must call the atlas loader rather than the Phase 1c.1 stub.
assert.match(src, /loadAtlasFromManifest|fetchAndDecodeAtlas|loadAtlas/,
  'runYeoOverlap must invoke the atlas-loader (no longer a stub)');

console.log('LNM app skeleton OK: class + 5 methods + import surface + atlas wiring validated.');
