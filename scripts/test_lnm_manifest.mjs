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

// Phase 2a.1: SynthStrip brain-extraction model must be registered as a
// supported modelAsset so the worker can fetch it. Pin the asset id literal
// here so a typo in either the manifest or the orchestrator surfaces fast.
const synthstrip = manifest.modelAssets.find(a => a.id === 'lnm-synthstrip');
assert.ok(synthstrip, "Phase 2a.1 manifest must register 'lnm-synthstrip' under modelAssets");
assert.equal(synthstrip.supportStatus, 'supported',
  'lnm-synthstrip must be marked supported once the model is uploaded');
assert.match(synthstrip.checksum, /^sha256:[0-9a-f]{64}$/i,
  'lnm-synthstrip must declare a real sha256 checksum');
assert.ok(typeof synthstrip.sizeBytes === 'number' && synthstrip.sizeBytes > 0,
  'lnm-synthstrip must declare a non-zero sizeBytes');
assert.match(synthstrip.sourceUrl, /huggingface\.co.+\.onnx$/,
  'lnm-synthstrip sourceUrl must point at an ONNX file on Hugging Face');

// Phase 3: SynthMorph MNI registration model (SVF sub-model).
const synmorph = manifest.modelAssets.find(a => a.id === 'lnm-synthmorph-mni');
assert.ok(synmorph, "Phase 3 manifest must register 'lnm-synthmorph-mni' under modelAssets");
assert.equal(synmorph.supportStatus, 'supported',
  'lnm-synthmorph-mni must be supported once the SVF ONNX is uploaded');
assert.match(synmorph.checksum, /^sha256:[0-9a-f]{64}$/i,
  'lnm-synthmorph-mni must declare a real sha256 checksum');
assert.ok(typeof synmorph.sizeBytes === 'number' && synmorph.sizeBytes > 0,
  'lnm-synthmorph-mni must declare a non-zero sizeBytes');
assert.match(synmorph.sourceUrl, /huggingface\.co.+\.onnx$/,
  'lnm-synthmorph-mni sourceUrl must point at an ONNX file on Hugging Face');
assert.ok(Array.isArray(synmorph.inputShape) && synmorph.inputShape.length === 5,
  'lnm-synthmorph-mni must declare a 5D inputShape (1, X, Y, Z, 1)');
assert.ok(Array.isArray(synmorph.svfShape) && synmorph.svfShape.length === 5,
  'lnm-synthmorph-mni must declare a 5D svfShape (1, X/2, Y/2, Z/2, 3)');

// Phase 2a.2: lesion-segmentation model (SynthStroke baseline) registered.
const stroke = manifest.modelAssets.find(a => a.id === 'lnm-stroke-lesion');
assert.ok(stroke, "Phase 2a.2 manifest must register 'lnm-stroke-lesion' under modelAssets");
assert.equal(stroke.supportStatus, 'supported',
  'lnm-stroke-lesion must be supported once the ONNX is exported and uploaded');
assert.match(stroke.checksum, /^sha256:[0-9a-f]{64}$/i,
  'lnm-stroke-lesion must declare a real sha256 checksum');
assert.ok(typeof stroke.sizeBytes === 'number' && stroke.sizeBytes > 0,
  'lnm-stroke-lesion must declare a non-zero sizeBytes');
assert.match(stroke.sourceUrl, /huggingface\.co.+\.onnx$/,
  'lnm-stroke-lesion sourceUrl must point at an ONNX file on Hugging Face');
assert.ok(Array.isArray(stroke.patchSize) && stroke.patchSize.length === 3,
  'lnm-stroke-lesion must declare a 3-tuple patchSize');
assert.ok(typeof stroke.probabilityThreshold === 'number',
  'lnm-stroke-lesion must declare a probabilityThreshold');
assert.ok(typeof stroke.minComponentSize === 'number',
  'lnm-stroke-lesion must declare a minComponentSize');

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
