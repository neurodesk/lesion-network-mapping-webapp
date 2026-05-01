#!/usr/bin/env node --no-warnings
// Contract test for web/js/app/lnm-tasks.js: the LNM_PIPELINES manifest must
// expose well-formed pipelines and the helpers used by the app to validate
// stage definitions. Written before lnm-tasks.js per the project's TDD policy.

import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const tasksUrl = pathToFileURL(path.join(ROOT, 'web/js/app/lnm-tasks.js'));

const {
  LNM_PIPELINES,
  getPipelineById,
  getRequiredAssetIds,
  isStageRunnable
} = await import(tasksUrl);

assert.ok(Array.isArray(LNM_PIPELINES) && LNM_PIPELINES.length >= 1,
  'LNM_PIPELINES must export a non-empty array');

// Phase 1 ships the 'lnm-yeo-only' pipeline (manual mask -> Yeo7 overlap).
// The 'lnm-default' Schaefer/GSP1000 pipeline is declared but stages may be
// marked unsupported until later phases.
const yeo = getPipelineById('lnm-yeo-only');
assert.ok(yeo, "Pipeline 'lnm-yeo-only' must exist for Phase 1");
assert.ok(typeof yeo.displayName === 'string' && yeo.displayName.length > 0);
assert.ok(Array.isArray(yeo.stages) && yeo.stages.length >= 1,
  "lnm-yeo-only must declare at least one stage");

// The Yeo-only pipeline must include a parcel-overlap stage that references
// the Yeo7 atlas.
const overlapStage = yeo.stages.find(s => s.module === 'parcel-overlap');
assert.ok(overlapStage, "lnm-yeo-only must contain a parcel-overlap stage");
assert.ok(overlapStage.atlasAssetId, 'parcel-overlap stage must declare atlasAssetId');
assert.match(overlapStage.atlasAssetId, /yeo/i, 'Phase 1 atlas must be Yeo-based');

// Stage IDs must be unique within a pipeline.
for (const pipeline of LNM_PIPELINES) {
  const ids = pipeline.stages.map(s => s.id);
  assert.equal(new Set(ids).size, ids.length,
    `pipeline ${pipeline.id} has duplicate stage IDs`);
  // Pipeline IDs must follow lnm-* convention so the routing layer can pick
  // them up consistently (see test_lnm_manifest).
  assert.match(pipeline.id, /^lnm-/,
    `pipeline ID must start with 'lnm-': got ${pipeline.id}`);
}

// getRequiredAssetIds returns the union of modelAssetId + atlasAssetId +
// connectomeAssetId across all stages in a pipeline. Used by the loader to
// fetch the right entries from manifest.json before running.
const yeoAssets = getRequiredAssetIds(yeo);
assert.ok(Array.isArray(yeoAssets) && yeoAssets.includes(overlapStage.atlasAssetId),
  'getRequiredAssetIds must surface the Yeo atlas asset ID');

// isStageRunnable returns true only for stages whose module is implemented and
// (if required) whose asset ID is provided. For the Phase 1 Yeo overlap stage
// this must be true.
assert.equal(isStageRunnable(overlapStage), true,
  'Phase 1 Yeo overlap stage must be runnable');

// A stage with required:true and no module/assets must NOT be runnable —
// guards against the SCT regression where missing assets fell back silently.
assert.equal(
  isStageRunnable({ id: 'broken', module: 'fc-weighted-sum', required: true }),
  false,
  'a stage missing its required assets must be flagged not-runnable'
);

console.log(`LNM tasks OK: ${LNM_PIPELINES.length} pipeline(s), Yeo overlap stage runnable.`);
