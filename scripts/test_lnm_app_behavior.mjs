#!/usr/bin/env node --no-warnings
// Phase 33 audit follow-up: behavior tests for the orchestrator's
// pipeline-driven runFullPipeline + _runStage dispatch + auto-promote.
//
// scripts/test_lnm_app.mjs is 100% source-grep — it asserts that the
// for-loop and case statements EXIST but never that they actually run
// the right stages in the right order. A regression where _runStage
// returns early after the first case, or runFullPipeline `return`s
// inside the loop, passes the source-grep but breaks the chain.
//
// This test stubs the browser globals (niivue, document, fetch, etc.)
// just enough to import LesionNetworkMappingApp, then:
//   - Replaces _runStage with a recording spy.
//   - Calls runFullPipeline with a synthetic pipeline.
//   - Asserts every stage was dispatched in declared order.
//   - Verifies an exception in one stage halts the chain (the for-loop
//     `try/catch ... return` path).
//   - Verifies _autoPromotePipeline switches selectedPipeline only
//     when the user hasn't manually picked.
//   - Verifies loading a structural image does not start processing;
//     the user must click Run analysis or an explicit stage button.

import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// ---- Stub browser globals before importing the orchestrator ----
// Niivue constructor is called in LesionNetworkMappingApp's constructor.
// We never touch the resulting nv instance in these tests.
globalThis.niivue = {
  Niivue: class {
    constructor() {}
    async attachTo() {}
    setMultiplanarPadPixels() {}
    setSliceType() {}
    sliceTypeMultiplanar = 0;
  }
};

// Minimal document stub — bindEvents() walks the DOM looking for IDs
// during init(). We never call init() in these tests so most of
// document is unused; but the constructor instantiates ConsoleOutput,
// ProgressManager, ModalManager which read DOM IDs lazily on .log()
// etc. Provide a no-op querySelector chain.
globalThis.document = {
  getElementById: () => null,
  querySelector: () => null,
  querySelectorAll: () => [],
  createElement: () => ({
    appendChild: () => {},
    setAttribute: () => {},
    addEventListener: () => {},
    style: {},
    classList: { add: () => {}, remove: () => {}, contains: () => false }
  }),
  addEventListener: () => {},
  body: { appendChild: () => {} }
};
globalThis.window = globalThis;
globalThis.URL = { createObjectURL: () => 'blob:fake', revokeObjectURL: () => {} };
globalThis.Blob = class { constructor() {} };
globalThis.File = class { constructor() {} };

const { LesionNetworkMappingApp, formatVersionLabel } =
  await import(path.join(ROOT, 'web/js/lnm-app.js'));

function makeApp() {
  const app = new LesionNetworkMappingApp();
  // Capture updateOutput so we can inspect status messages without
  // dragging in ConsoleOutput's DOM dependency.
  app._messages = [];
  app.updateOutput = (m) => { app._messages.push(m); };
  return app;
}

// ---- Test 1: runFullPipeline dispatches every stage in declared order ----
{
  const app = makeApp();
  const calls = [];
  app._runStage = async (stage) => { calls.push(stage.id); };
  // Skip the precondition gate for parcel-overlap-first pipelines.
  app._lesionFileMatchesYeoGrid = async () => true;
  app.lesionFile = {};   // satisfy precondition
  app.selectedPipeline = {
    id: 'test-pipeline',
    displayName: 'Test',
    stages: [
      { id: 'overlap', module: 'parcel-overlap', required: true },
      { id: 'fc',      module: 'fc-weighted-sum', required: true },
      { id: 'thresh',  module: 'threshold', required: false }
    ]
  };
  await app.runFullPipeline();
  assert.deepEqual(calls, ['overlap', 'fc', 'thresh'],
    `expected stages in declared order; got ${JSON.stringify(calls)}`);
  // _perfStats accumulated one entry per stage.
  assert.equal(app._perfStats.length, 3,
    `expected 3 perf entries; got ${app._perfStats.length}`);
  assert.deepEqual(
    app._perfStats.map(s => s.id),
    ['overlap', 'fc', 'thresh'],
    'perf entries must match stage order'
  );
}

// ---- Test 2: a stage exception halts the chain (no further stages run) ----
{
  const app = makeApp();
  const calls = [];
  app._runStage = async (stage) => {
    calls.push(stage.id);
    if (stage.id === 'fc') throw new Error('synthetic FC failure');
  };
  app._lesionFileMatchesYeoGrid = async () => true;
  app.lesionFile = {};
  app.selectedPipeline = {
    id: 'test-pipeline',
    displayName: 'Test',
    stages: [
      { id: 'overlap', module: 'parcel-overlap', required: true },
      { id: 'fc',      module: 'fc-weighted-sum', required: true },
      { id: 'thresh',  module: 'threshold', required: false }
    ]
  };
  await app.runFullPipeline();
  assert.deepEqual(calls, ['overlap', 'fc'],
    `expected chain to halt after the failing stage; got ${JSON.stringify(calls)}`);
  // The failure message must surface to the user.
  assert.ok(
    app._messages.some(m => /Stage 'fc'.*failed.*synthetic FC failure/.test(m)),
    `expected a 'Stage fc failed' message; got ${JSON.stringify(app._messages)}`
  );
}

// ---- Test 3: precondition gate — parcel-overlap-first pipeline + missing
//      lesion file -> no stages run ----
{
  const app = makeApp();
  const calls = [];
  app._runStage = async (stage) => { calls.push(stage.id); };
  app.lesionFile = null;   // missing!
  app.selectedPipeline = {
    id: 'test-pipeline', displayName: 'Test',
    stages: [{ id: 'overlap', module: 'parcel-overlap', required: true }]
  };
  await app.runFullPipeline();
  assert.deepEqual(calls, [],
    `parcel-overlap-first pipeline must NOT run without a lesion file; got ${JSON.stringify(calls)}`);
}

// ---- Test 4: precondition gate — brain-extraction-first pipeline +
//      missing structural -> no stages run ----
{
  const app = makeApp();
  const calls = [];
  app._runStage = async (stage) => { calls.push(stage.id); };
  app.structuralFile = null;
  app.selectedPipeline = {
    id: 'test-pipeline', displayName: 'Test',
    stages: [
      { id: 'brain', module: 'brain-extraction', required: true },
      { id: 'overlap', module: 'parcel-overlap', required: true }
    ]
  };
  await app.runFullPipeline();
  assert.deepEqual(calls, [],
    `brain-extraction-first pipeline must NOT run without structural; got ${JSON.stringify(calls)}`);
}

// ---- Test 5: _autoPromotePipeline only fires before user manual pick ----
{
  const app = makeApp();
  // Initial state: no user override.
  app._userPickedPipeline = false;
  // Auto-promote on file drop.
  app._autoPromotePipeline('lnm-yeo-auto');
  assert.equal(app.selectedPipeline?.id, 'lnm-yeo-auto',
    'auto-promote should switch to lnm-yeo-auto on first call');

  // Now simulate the user manually picking via the dropdown.
  app._userPickedPipeline = true;
  // Subsequent auto-promote calls must be no-ops.
  app._autoPromotePipeline('lnm-network-map');
  assert.equal(app.selectedPipeline?.id, 'lnm-yeo-auto',
    'auto-promote must NOT override a user-picked pipeline');
}

// ---- Test 6: _runStage rejects unknown modules (catches manifest typos) ----
{
  const app = makeApp();
  await assert.rejects(
    () => app._runStage({ id: 'x', module: 'made-up-module' }),
    /unknown module/i,
    '_runStage must throw on an unrecognised module name'
  );
  await assert.rejects(
    () => app._runStage(null),
    /must declare a module/i,
    '_runStage must throw on a null stage'
  );
}

// ---- Test 7: setStructural only loads/selects; it must not run SynthStrip ----
{
  const app = makeApp();
  const loaded = [];
  let synthStripCalls = 0;
  app.viewerController = {
    loadBaseVolume: async (file, opts) => { loaded.push({ file, opts }); }
  };
  app.runBrainExtraction = async () => { synthStripCalls += 1; };
  const file = { name: 'subject-t1.nii.gz' };

  await app.setStructural(file);

  assert.equal(app.structuralFile, file, 'setStructural must store the structural file');
  assert.equal(loaded.length, 1, 'setStructural must still load the structural into the viewer');
  assert.equal(app.selectedPipeline?.id, 'lnm-yeo-auto',
    'setStructural must still select the auto T1 pipeline for the later Run analysis click');
  assert.equal(synthStripCalls, 0,
    'setStructural must not call runBrainExtraction; users explicitly start processing');
}

// ---- Test 8: version label de-duplicates the staging short SHA ----
{
  assert.equal(
    formatVersionLabel('0.17.1-staging+31ff9d1', {
      sha: '31ff9d1',
      branch: 'main',
      dirty: false
    }),
    'v0.17.1-staging+31ff9d1',
    'staging version must not append the same git SHA twice'
  );
  assert.equal(
    formatVersionLabel('0.17.1', {
      sha: '31ff9d1',
      branch: 'main',
      dirty: false
    }),
    'v0.17.1 (31ff9d1)',
    'plain versions should still surface build-info SHA'
  );
}

console.log('lnm-app behavior OK: 8 dispatch + precondition + explicit-start + auto-promote + version-label cases.');
