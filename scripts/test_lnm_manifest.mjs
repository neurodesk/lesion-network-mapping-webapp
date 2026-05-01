#!/usr/bin/env node --no-warnings
// Contract test for web/models/manifest.json under the LNM schema. Written
// before the manifest is rewritten per the project's TDD policy.
//
// The LNM manifest extends the SCT shape with:
//  - top-level 'pipelines' (renamed from 'tasks')
//  - 'atlasAssets' and 'connectomeAssets' alongside 'modelAssets'
//  - every stage in every pipeline references an asset that exists in one of
//    the asset registries (no silent fallbacks).

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifestPath = path.join(ROOT, 'web/models/manifest.json');
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

assert.equal(typeof manifest.schemaVersion, 'string');
assert.ok(Array.isArray(manifest.pipelines),
  "manifest must use 'pipelines' (renamed from 'tasks')");
assert.ok(!('tasks' in manifest),
  "manifest must not retain SCT 'tasks' key after LNM migration");

// Asset registries must exist (may be empty but must be arrays).
for (const key of ['modelAssets', 'atlasAssets', 'connectomeAssets']) {
  assert.ok(Array.isArray(manifest[key]),
    `manifest.${key} must be an array (may be empty)`);
}

// Every asset entry must have id + filename + sizeBytes + checksum + cacheKey.
for (const key of ['modelAssets', 'atlasAssets', 'connectomeAssets']) {
  for (const asset of manifest[key]) {
    for (const field of ['id', 'filename', 'sizeBytes', 'checksum', 'cacheKey']) {
      assert.ok(asset[field] !== undefined,
        `${key} asset '${asset.id || '?'}' missing field '${field}'`);
    }
  }
}

// Asset IDs must be unique across the union of all three registries (so a
// stage referencing assetId 'foo' has an unambiguous source).
const allIds = [
  ...manifest.modelAssets.map(a => a.id),
  ...manifest.atlasAssets.map(a => a.id),
  ...manifest.connectomeAssets.map(a => a.id)
];
assert.equal(new Set(allIds).size, allIds.length,
  'asset IDs must be globally unique across modelAssets/atlasAssets/connectomeAssets');

// Phase 1: the Yeo7 atlas must exist and have a sensible parcel count
// (network labels 1..7 plus 0 background -> at least 7 nonzero networks).
const yeoAtlas = manifest.atlasAssets.find(a => /yeo/i.test(a.id));
assert.ok(yeoAtlas, "Phase 1 manifest must register a Yeo atlas in atlasAssets");
assert.equal(typeof yeoAtlas.parcelCount, 'number');
assert.ok(yeoAtlas.parcelCount >= 7, 'Yeo atlas must declare >= 7 networks');
assert.ok(typeof yeoAtlas.networkLabels === 'object' && yeoAtlas.networkLabels !== null,
  'Yeo atlas must declare networkLabels (label -> network name) so the UI can render an overlap chart');
// Yeo7 network names must be present.
const yeoLabelValues = new Set(Object.values(yeoAtlas.networkLabels));
for (const expected of ['Visual', 'Default']) {
  assert.ok(yeoLabelValues.has(expected),
    `Yeo7 networkLabels must include '${expected}'`);
}

// Pipelines must reference atlas/model/connectome assets that exist.
const knownIds = new Set(allIds);
for (const pipeline of manifest.pipelines) {
  assert.match(pipeline.id, /^lnm-/);
  assert.ok(Array.isArray(pipeline.stages));
  for (const stage of pipeline.stages) {
    for (const refKey of ['modelAssetId', 'atlasAssetId', 'connectomeAssetId']) {
      if (stage[refKey] !== undefined) {
        assert.ok(knownIds.has(stage[refKey]),
          `pipeline ${pipeline.id} stage ${stage.id} references unknown ${refKey} '${stage[refKey]}'`);
      }
    }
  }
}

// Cross-check with lnm-tasks.js: every pipeline declared in code must exist
// in the manifest (so the UI's pipeline dropdown can't list a pipeline whose
// assets aren't fetchable).
const tasks = await import(pathToFileURL(path.join(ROOT, 'web/js/app/lnm-tasks.js')));
const codePipelineIds = new Set(tasks.LNM_PIPELINES.map(p => p.id));
const manifestPipelineIds = new Set(manifest.pipelines.map(p => p.id));
for (const id of codePipelineIds) {
  assert.ok(manifestPipelineIds.has(id),
    `code pipeline '${id}' missing from manifest.pipelines`);
}

console.log(`LNM manifest OK: ${manifest.pipelines.length} pipeline(s); ${allIds.length} unique asset(s).`);
