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
globalThis.location = { href: 'http://localhost:8080/' };
globalThis.URL.createObjectURL = () => 'blob:fake';
globalThis.URL.revokeObjectURL = () => {};
globalThis.Blob = class { constructor() {} };
globalThis.File = class {
  constructor(parts, name) { this.parts = parts; this.name = name; }
  async arrayBuffer() { return this.parts?.[0] || new ArrayBuffer(0); }
};

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

function useMockElements(elementsById) {
  const originalGetElementById = globalThis.document.getElementById;
  globalThis.document.getElementById = (id) => elementsById[id] || null;
  return () => { globalThis.document.getElementById = originalGetElementById; };
}

function waitMs(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForMicrotaskCondition(predicate, message, attempts = 20) {
  for (let i = 0; i < attempts; i++) {
    if (predicate()) return;
    await Promise.resolve();
  }
  assert.fail(message);
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

// ---- Test 8: worker-backed stages resolve only after their output arrives ----
{
  const app = makeApp();
  app.manifest = {
    modelAssets: [{
      id: 'lnm-synthstrip',
      sourceUrl: 'https://example.test/models/synthstrip.onnx',
      cacheKey: 'synthstrip-v1',
      supportStatus: 'supported'
    }]
  };
  app.structuralFile = { name: 't1.nii.gz', arrayBuffer: async () => new ArrayBuffer(8) };
  app.viewerController = { loadOverlay: async () => {} };
  const calls = [];
  app.executor = {
    loadVolume: async () => { calls.push('load'); },
    runSynthStrip: async () => { calls.push('run-synthstrip'); }
  };

  let settled = false;
  const promise = app.runBrainExtraction().then(() => { settled = true; });
  await waitForMicrotaskCondition(
    () => calls.includes('run-synthstrip'),
    'runBrainExtraction must dispatch SynthStrip before waiting for stageData'
  );
  assert.equal(settled, false,
    'runBrainExtraction must not resolve before brainmask stageData arrives');
  assert.deepEqual(calls, ['load', 'run-synthstrip']);

  app.handleStageData({
    stage: 'brainmask',
    niftiData: new ArrayBuffer(8),
    description: 'synthetic brainmask'
  });
  await Promise.resolve();
  assert.equal(settled, false,
    'runBrainExtraction must also wait for brainmask step-complete after stageData');
  app.handleStepComplete('brainmask');
  await promise;
  assert.equal(settled, true,
    'runBrainExtraction must resolve after brainmask stageData and step-complete arrive');
  assert.ok(app.brainmaskFile, 'brainmaskFile must be populated before the stage resolves');
}

// ---- Test 9: registration resolves only after the worker register step completes ----
{
  const app = makeApp();
  app.manifest = {
    modelAssets: [{
      id: 'lnm-synthmorph-mni',
      sourceUrl: 'https://example.test/models/lnm-synthmorph-mni.onnx',
      filename: 'models/lnm-synthmorph-mni.onnx',
      cacheKey: 'synthmorph-v1',
      supportStatus: 'supported',
      inputShape: [1, 48, 48, 64, 1],
      svfShape: [1, 24, 24, 32, 3],
      browserRuntime: {
        inputDims: [48, 48, 64],
        svfDims: [24, 24, 32],
        executionProviders: ['wasm']
      }
    }],
    atlasAssets: [{
      id: 'lnm-mni160',
      sourceUrl: 'https://example.test/templates/lnm-mni160.nii.gz',
      cacheKey: 'mni160-v1',
      supportStatus: 'supported'
    }]
  };
  app.structuralFile = { name: 'lnm-prealign-t1.nii', arrayBuffer: async () => new ArrayBuffer(8) };
  let registrationSettings = null;
  app.executor = {
    loadVolume: async () => {},
    runRegistration: async (settings) => { registrationSettings = settings; }
  };

  let settled = false;
  const promise = app.runRegistration().then(() => { settled = true; });
  await waitForMicrotaskCondition(
    () => registrationSettings !== null,
    'runRegistration must dispatch worker registration before waiting for step-complete'
  );
  assert.equal(settled, false,
    'runRegistration must not resolve before register step-complete arrives');
  assert.deepEqual(registrationSettings.executionProviders, ['wasm']);

  app.handleStepComplete('register');
  await promise;
  assert.equal(settled, true,
    'runRegistration must resolve after register step-complete arrives');
}

// ---- Test 10: lesion segmentation waits for output and inference completion ----
{
  const app = makeApp();
  app.manifest = {
    modelAssets: [{
      id: 'lnm-stroke-lesion',
      sourceUrl: 'https://example.test/models/lnm-stroke-lesion.onnx',
      cacheKey: 'stroke-v1',
      supportStatus: 'supported',
      patchSize: [128, 128, 128],
      probabilityThreshold: 0.4,
      minComponentSize: 30,
      preprocessing: {}
    }]
  };
  app.structuralFile = { name: 'lnm-prealign-t1.nii', arrayBuffer: async () => new ArrayBuffer(8) };
  app.viewerController = { loadOverlay: async () => {} };
  const calls = [];
  app.executor = {
    loadVolume: async () => { calls.push('load'); },
    runInference: async () => { calls.push('run-inference'); }
  };

  let settled = false;
  const promise = app.runLesionSegmentation().then(() => { settled = true; });
  await waitForMicrotaskCondition(
    () => calls.includes('run-inference'),
    'runLesionSegmentation must dispatch inference before waiting for stageData'
  );
  assert.equal(settled, false,
    'runLesionSegmentation must not resolve before segmentation output arrives');
  app.handleStageData({
    stage: 'segmentation',
    niftiData: new ArrayBuffer(8),
    description: 'synthetic segmentation'
  });
  await Promise.resolve();
  assert.equal(settled, false,
    'runLesionSegmentation must also wait for inference step-complete after stageData');
  app.handleStepComplete('inference');
  await promise;
  assert.equal(settled, true,
    'runLesionSegmentation must resolve after segmentation stageData and step-complete arrive');
  assert.ok(app.lesionMaskFile, 'lesionMaskFile must be populated before the stage resolves');
}

// ---- Test 11: thresholding schedules a live viewer preview overlay ----
{
  const app = makeApp();
  const summaryEl = { textContent: '' };
  const restoreDocument = useMockElements({
    networkThresholdValue: { value: '0.5' },
    networkThresholdMode: { value: 'absolute' },
    networkThresholdSymmetric: { checked: true },
    networkThresholdMinCluster: { value: '4' },
    networkThresholdSummary: summaryEl,
    downloadThresholdedNetworkMapButton: { disabled: true }
  });
  const previewCalls = [];
  app.viewerController = {
    replaceOverlayForStage: async (...args) => { previewCalls.push(args); },
    removeVolumeForStage: () => {}
  };
  app.networkMapData = new Float32Array([-1.0, 0.25, 0.75, 2.0]);
  app.networkMapDims = [2, 2, 1];
  app.networkMapSpacing = [1, 1, 1];
  app.networkMapAffine = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0
  ];

  try {
    const mask = app.applyNetworkThreshold();
    assert.equal(mask.reduce((sum, value) => sum + value, 0), 0,
      'symmetric absolute threshold should apply min-cluster cleanup to the binary mask');
    assert.match(summaryEl.textContent, /^0 voxels survive absolute/,
      'threshold summary should update immediately');
    assert.match(summaryEl.textContent, /cluster≥4 removed 3 voxels/,
      'threshold summary should report how many voxels cluster cleanup removed');

    await waitMs(90);
    await app._thresholdPreviewRenderPromise;

    assert.equal(previewCalls.length, 1,
      'thresholding must schedule exactly one live preview replacement');
    assert.equal(previewCalls[0][0], 'threshold-preview');
    assert.equal(previewCalls[0][1], app.thresholdedMaskFile,
      'preview must render the freshly generated thresholded mask file');
    assert.equal(previewCalls[0][2], 'red');
    assert.equal(previewCalls[0][3], 0.65);
  } finally {
    restoreDocument();
    app.cancelThresholdPreviewOverlay();
  }
}

// ---- Test 12: min-cluster input recomputes as the user types ----
{
  const app = makeApp();
  const listeners = {};
  const noOpAddEventListener = () => {};
  const minClusterEl = {
    addEventListener: (eventName, handler) => { listeners[eventName] = handler; }
  };
  const restoreDocument = useMockElements({
    networkThresholdMinCluster: minClusterEl,
    networkThresholdValueLabel: { textContent: '' },
    networkThresholdValue: { value: '0.5', addEventListener: noOpAddEventListener },
    networkThresholdMode: { value: 'absolute', addEventListener: noOpAddEventListener },
    networkThresholdSymmetric: { checked: true, addEventListener: noOpAddEventListener }
  });
  let thresholdCalls = 0;
  app.networkMapData = new Float32Array([0, 1]);
  app.applyNetworkThreshold = () => { thresholdCalls += 1; };

  try {
    app.bindEvents();
    assert.equal(typeof listeners.input, 'function',
      'min-cluster input must recompute on input, not only after blur/change');
    listeners.input();
    assert.equal(thresholdCalls, 1,
      'min-cluster input handler should trigger threshold recompute immediately');
  } finally {
    restoreDocument();
  }
}

// ---- Test 13: percentile UI uses top-percent semantics ----
{
  const app = makeApp();
  const valueEl = { value: '0' };
  const elements = {
    networkThresholdValue: valueEl,
    networkThresholdMode: { value: 'percentile' },
    networkThresholdSymmetric: { checked: true },
    networkThresholdMinCluster: { value: '0' },
    networkThresholdSummary: { textContent: '' },
    downloadThresholdedNetworkMapButton: { disabled: true }
  };
  const restoreDocument = useMockElements(elements);
  app.viewerController = {
    replaceOverlayForStage: async () => {},
    removeVolumeForStage: () => {}
  };
  app.networkMapData = new Float32Array([1, 2, 3, 4]);
  app.networkMapDims = [2, 2, 1];
  app.networkMapSpacing = [1, 1, 1];
  app.networkMapAffine = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0
  ];

  try {
    app.configureThresholdSliderForMode('percentile', { resetValue: true });
    assert.equal(valueEl.min, '0', 'top-percent slider min must be 0');
    assert.equal(valueEl.max, '10', 'top-percent slider max must be 10 for easier adjustment');
    assert.equal(valueEl.step, '0.1', 'top-percent slider step must allow fine 0.1% adjustment');
    assert.equal(valueEl.value, '5', 'top-percent slider reset should land on the 5% default');

    valueEl.value = '0';
    const mask0 = app.applyNetworkThreshold();
    assert.equal(mask0.reduce((sum, value) => sum + value, 0), 0,
      'top 0% must keep no voxels, not the whole map');
    assert.match(elements.networkThresholdSummary.textContent, /top 0%/,
      'summary must report user-facing top-percent semantics');

    valueEl.value = '10';
    const mask10 = app.applyNetworkThreshold();
    assert.equal(mask10.reduce((sum, value) => sum + value, 0), 1,
      'top 10% of four ranked voxels should keep only the strongest voxel');
    assert.match(elements.networkThresholdSummary.textContent, /top 10%/,
      'summary must track the top-percent slider value');
  } finally {
    restoreDocument();
    app.cancelThresholdPreviewOverlay();
  }
}

// ---- Test 14: version label de-duplicates the staging short SHA ----
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

console.log('lnm-app behavior OK: 14 dispatch + precondition + explicit-start + worker-wait + threshold-preview + min-cluster-input + top-percent + auto-promote + version-label cases.');
