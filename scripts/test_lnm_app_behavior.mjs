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
import * as niftiModule from 'nifti-reader-js';
import { writeNifti1 } from '../web/js/modules/nifti-writer.js';

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
globalThis.Blob = class {
  constructor(parts) { this.parts = parts || []; }
  async arrayBuffer() {
    const first = this.parts[0];
    if (first instanceof ArrayBuffer) return first;
    if (ArrayBuffer.isView(first)) {
      return first.buffer.slice(first.byteOffset, first.byteOffset + first.byteLength);
    }
    if (first && typeof first.arrayBuffer === 'function') return first.arrayBuffer();
    return new ArrayBuffer(0);
  }
};
globalThis.File = class {
  constructor(parts, name) { this.parts = parts; this.name = name; }
  async arrayBuffer() {
    const first = this.parts?.[0];
    if (first instanceof ArrayBuffer) return first;
    if (ArrayBuffer.isView(first)) {
      return first.buffer.slice(first.byteOffset, first.byteOffset + first.byteLength);
    }
    if (first && typeof first.arrayBuffer === 'function') return first.arrayBuffer();
    return new ArrayBuffer(0);
  }
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

function makeNiftiFile(name, buffer) {
  return {
    name,
    async arrayBuffer() { return buffer; }
  };
}

function decodeNiftiForTest(buffer) {
  const nifti = niftiModule.default || niftiModule;
  let buf = buffer;
  const bytes = new Uint8Array(buf);
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) buf = nifti.decompress(buf);
  const header = nifti.readHeader(buf);
  return {
    header,
    dims: [Number(header.dims[1]), Number(header.dims[2]), Number(header.dims[3])]
  };
}

function assertAffineClose(actual, expected, message, tolerance = 1e-5) {
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 4; c++) {
      assert.ok(
        Math.abs(actual[r][c] - expected[r][c]) < tolerance,
        `${message}: affine[${r}][${c}] got ${actual[r][c]}, expected ${expected[r][c]}`
      );
    }
  }
}

function makeClassList(initial = []) {
  const classes = new Set(initial);
  return {
    add: (...names) => names.forEach(name => classes.add(name)),
    remove: (...names) => names.forEach(name => classes.delete(name)),
    contains: (name) => classes.has(name)
  };
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

// ---- Test 5: _autoPromotePipeline follows the currently loaded input ----
{
  const app = makeApp();
  assert.equal(app.selectedPipeline?.id, 'lnm-yeo-auto',
    'default Run analysis pipeline should be the structural-T1 auto chain');

  app._autoPromotePipeline('lnm-yeo-auto');
  assert.equal(app.selectedPipeline?.id, 'lnm-yeo-auto',
    'structural T1 input should select the full auto pipeline');

  app._autoPromotePipeline('lnm-network-map');
  assert.equal(app.selectedPipeline?.id, 'lnm-network-map',
    'researcher-mode Yeo mask input should select the manual network-map pipeline');
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

// ---- Test 12: thresholding labels the final affected map by Yeo network ----
{
  const app = makeApp();
  const summaryEl = { textContent: '' };
  const affectedResultsEl = { classList: makeClassList(['hidden']) };
  const affectedTableEl = { innerHTML: '', appendChild: () => {} };
  const elements = {
    networkThresholdValue: { value: '1' },
    networkThresholdMode: { value: 'absolute' },
    networkThresholdSymmetric: { checked: false },
    networkThresholdMinCluster: { value: '0' },
    networkThresholdSummary: summaryEl,
    affectedNetworkResults: affectedResultsEl,
    affectedNetworkTable: affectedTableEl,
    downloadThresholdedNetworkMapButton: { disabled: true }
  };
  const restoreDocument = useMockElements(elements);
  const originalCreateElement = globalThis.document.createElement;
  const renderedText = [];
  globalThis.document.createElement = (tagName) => {
    let text = '';
    return {
      tagName,
      children: [],
      style: {},
      classList: makeClassList(),
      appendChild(child) { this.children.push(child); },
      setAttribute: () => {},
      set textContent(value) {
        text = value;
        renderedText.push(value);
      },
      get textContent() { return text; }
    };
  };
  app.viewerController = {
    replaceOverlayForStage: async () => {},
    removeVolumeForStage: () => {}
  };
  app.networkMapData = new Float32Array([0.1, 2.0, 3.0, 4.0]);
  app.networkMapDims = [2, 2, 1];
  app.networkMapSpacing = [1, 1, 1];
  app.networkMapAffine = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0
  ];
  app.overlapResult = {
    atlas: {
      data: new Int16Array([1, 2, 7, 2]),
      dims: [2, 2, 1],
      networkLabels: {
        1: 'Visual',
        2: 'Somatomotor',
        7: 'Default'
      }
    }
  };

  try {
    const mask = app.applyNetworkThreshold();
    assert.equal(mask.reduce((sum, value) => sum + value, 0), 3,
      'absolute threshold should keep the three high-valued voxels');
    assert.ok(app.affectedNetworkResult,
      'thresholding must store a final affected-network summary');
    assert.deepEqual(
      app.affectedNetworkResult.summary.networks.map(row => [row.network, row.voxelsInLesion]),
      [['Somatomotor', 2], ['Default', 1]],
      'affected-network summary must aggregate thresholded map voxels by Yeo label'
    );
    assert.equal(affectedResultsEl.classList.contains('hidden'), false,
      'affected-network table must become visible after thresholding');
    assert.ok(renderedText.includes('Somatomotor'),
      'affected-network table must render the dominant Yeo network name');
    assert.ok(renderedText.includes('Default'),
      'affected-network table must render secondary Yeo network names');
    assert.ok(renderedText.includes('% of map'),
      'affected-network table must use map-specific percent copy');
  } finally {
    globalThis.document.createElement = originalCreateElement;
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

// ---- Test 15: Yeo label coverage note is neutral, not a brain-mask warning ----
{
  const app = makeApp();
  const classOps = [];
  const coverageEl = {
    textContent: '',
    classList: {
      add: (cls) => { classOps.push(['add', cls]); },
      remove: (cls) => { classOps.push(['remove', cls]); }
    }
  };
  const restoreDocument = useMockElements({ outsideAtlasWarning: coverageEl });

  try {
    app.showYeoLabelCoverageNote(2, 7);
    assert.equal(
      coverageEl.textContent,
      '5 of 7 lesion voxels are assigned to Yeo cortical network labels; 2 are unlabeled by this cortical atlas.',
      'coverage note must report assigned and unlabeled Yeo cortical-label voxels'
    );
    assert.deepEqual(classOps.at(-1), ['remove', 'hidden'],
      'coverage note should be visible when any lesion voxels are unlabeled');

    app.showYeoLabelCoverageNote(0, 7);
    assert.equal(coverageEl.textContent, '',
      'coverage note should clear when all lesion voxels are labelled');
    assert.deepEqual(classOps.at(-1), ['add', 'hidden'],
      'coverage note should hide when there are no unlabeled voxels');
  } finally {
    restoreDocument();
  }
}

// ---- Test 16: viewer layer toggles bind to stage visibility ----
{
  const app = makeApp();
  app.structuralFile = { name: 't1.nii' };
  app.brainmaskFile = { name: 'brainmask.nii' };
  app.lesionMaskFile = { name: 'lesion.nii' };
  app.thresholdedMaskFile = { name: 'threshold.nii' };
  app.patientAtlasFile = { name: 'atlas-patient.nii' };

  const listeners = {};
  const makeToggle = id => ({
    checked: true,
    disabled: true,
    addEventListener: (eventName, handler) => { listeners[`${id}:${eventName}`] = handler; }
  });
  const toggles = {
    layerToggleT1: makeToggle('layerToggleT1'),
    layerToggleBrainMask: makeToggle('layerToggleBrainMask'),
    layerToggleLesionMask: makeToggle('layerToggleLesionMask'),
    layerToggleThresholdMap: makeToggle('layerToggleThresholdMap'),
    layerToggleAtlasQc: makeToggle('layerToggleAtlasQc')
  };
  const restoreDocument = useMockElements(toggles);
  const stageVisibilityCalls = [];
  app.viewerController = {
    setStageVisible: (stage, visible) => {
      stageVisibilityCalls.push([stage, visible]);
      return true;
    }
  };

  try {
    app.bindEvents();
    for (const el of Object.values(toggles)) {
      assert.equal(el.disabled, false, 'available layer toggles must be enabled');
      assert.equal(el.checked, true, 'available layer toggles default visible');
    }

    listeners['layerToggleBrainMask:change']({ target: { checked: false } });
    assert.equal(app.viewerLayerVisibility.brainmask, false,
      'brain-mask toggle must persist app-level visibility state');
    assert.deepEqual(stageVisibilityCalls.at(-1), ['brainmask', false],
      'brain-mask toggle must target the brainmask viewer stage');

    listeners['layerToggleLesionMask:change']({ target: { checked: false } });
    assert.ok(stageVisibilityCalls.some(call => call[0] === 'segmentation' && call[1] === false),
      'lesion toggle must hide the native segmentation stage');
    assert.ok(stageVisibilityCalls.some(call => call[0] === 'lesion' && call[1] === false),
      'lesion toggle must hide the Yeo/manual lesion stage');

    listeners['layerToggleAtlasQc:change']({ target: { checked: false } });
    assert.deepEqual(stageVisibilityCalls.at(-1), ['atlas-qc', false],
      'atlas QC toggle must target the subject-space atlas viewer stage');
  } finally {
    restoreDocument();
  }
}

// ---- Test 17: threshold preview chooses patient-space projection when available ----
{
  const app = makeApp();
  app.structuralFile = { name: 't1.nii' };
  app.thresholdedMaskFile = { name: 'threshold.nii' };
  app.networkMapAffine = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
  app.hasRegistrationDisplacement = true;
  app.viewerController = {};
  const projected = [];
  let patientStackRenders = 0;
  let atlasRenders = 0;
  app.projectThresholdToPatientSpace = async (version) => {
    projected.push(version);
    app.patientThresholdedMaskFile = { name: 'patient-threshold.nii' };
  };
  app.renderPatientLayerStack = async () => { patientStackRenders += 1; };
  app.renderAtlasThresholdPreviewOverlay = async () => { atlasRenders += 1; };

  await app.renderThresholdPreviewOverlay(app._thresholdPreviewVersion);

  assert.equal(projected.length, 1,
    'registered structural runs must project threshold masks to patient space');
  assert.equal(patientStackRenders, 1,
    'patient-space projection must render the patient layer stack');
  assert.equal(atlasRenders, 0,
    'patient-space projection must not fall back to the atlas preview');
}

// ---- Test 18: patient projection resamples Yeo threshold onto lnm-mni160,
//      not the structural/prealign affine ----
{
  globalThis.nifti = niftiModule.default || niftiModule;
  const app = makeApp();
  const thresholdData = new Uint8Array(5 * 5 * 5);
  thresholdData[0] = 1;
  const thresholdAffine = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0
  ];
  const mni160Data = new Uint8Array(2 * 2 * 2);
  const mni160Affine = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0
  ];
  const structuralData = new Uint8Array(2 * 2 * 2);
  const structuralAffine = [
    1, 0, 0, 2,
    0, 1, 0, 0,
    0, 0, 1, 0
  ];
  app.thresholdedMaskFile = makeNiftiFile('threshold.nii', writeNifti1(thresholdData, {
    dims: [5, 5, 5],
    spacing: [1, 1, 1],
    affine: thresholdAffine,
    description: 'synthetic threshold'
  }));
  app.structuralFile = makeNiftiFile('structural-prealign.nii', writeNifti1(structuralData, {
    dims: [2, 2, 2],
    spacing: [1, 1, 1],
    affine: structuralAffine,
    description: 'synthetic structural with shifted affine'
  }));
  const mni160Buffer = writeNifti1(mni160Data, {
    dims: [2, 2, 2],
    spacing: [1, 1, 1],
    affine: mni160Affine,
    description: 'synthetic lnm-mni160'
  });

  const originalFetch = globalThis.fetch;
  const originalCaches = globalThis.caches;
  globalThis.caches = undefined;
  globalThis.fetch = async (url) => {
    const href = String(url);
    if (href.includes('manifest.json')) {
      return {
        ok: true,
        status: 200,
        async json() {
          return {
            atlasAssets: [{
              id: 'lnm-mni160',
              sourceUrl: 'https://example.test/lnm-mni160.nii',
              cacheKey: 'lnm-mni160-test',
              dims: [2, 2, 2],
              supportStatus: 'supported'
            }]
          };
        }
      };
    }
    if (href === 'https://example.test/lnm-mni160.nii') {
      return {
        ok: true,
        status: 200,
        async arrayBuffer() { return mni160Buffer; }
      };
    }
    throw new Error(`unexpected fetch in test: ${href}`);
  };

  try {
    const { mask, dims } = await app.resampleThresholdMaskToStructuralGrid();
    assert.deepEqual(dims, [2, 2, 2],
      'projection input to inverse-warp must use lnm-mni160 dims');
    assert.equal(mask[0], 1,
      'Yeo threshold must be sampled through the lnm-mni160 affine');
    assert.equal(mask[1], 0,
      'synthetic threshold should not be shifted through the structural affine');
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.caches = originalCaches;
  }
}

// ---- Test 19: prealign writes fixed lnm-mni160 headers, not the sampling affine ----
{
  globalThis.nifti = niftiModule.default || niftiModule;
  const app = makeApp();
  app.viewerController = {
    loadBaseVolume: async () => {},
    loadOverlay: async () => {},
    removeVolumeForStage: () => {}
  };
  const dims = [4, 4, 4];
  const t1Data = new Float32Array(64);
  for (let i = 0; i < t1Data.length; i++) t1Data[i] = i + 1;
  const maskData = new Uint8Array(64);
  maskData.fill(1);
  const sourceAffine = [
    1, 0.15, 0, 4,
    0, 1, 0.2, -3,
    0, 0, 1, 2
  ];
  const fixedMniAffine = [
    1, 0, 0, -2,
    0, 1, 0, -2,
    0, 0, 1, -2
  ];
  app.structuralFile = makeNiftiFile('source-oblique.nii', writeNifti1(t1Data, {
    dims,
    spacing: [1, 1, 1],
    affine: sourceAffine,
    description: 'synthetic oblique source'
  }));
  app.brainmaskFile = makeNiftiFile('source-mask.nii', writeNifti1(maskData, {
    dims,
    spacing: [1, 1, 1],
    affine: sourceAffine,
    description: 'synthetic source mask'
  }));
  const mni160Buffer = writeNifti1(new Uint8Array(64), {
    dims,
    spacing: [1, 1, 1],
    affine: fixedMniAffine,
    description: 'synthetic fixed lnm-mni160'
  });

  const originalFetch = globalThis.fetch;
  const originalCaches = globalThis.caches;
  globalThis.caches = undefined;
  globalThis.fetch = async (url) => {
    const href = String(url);
    if (href.includes('manifest.json')) {
      return {
        ok: true,
        status: 200,
        async json() {
          return {
            atlasAssets: [{
              id: 'lnm-mni160',
              sourceUrl: 'https://example.test/prealign-mni160.nii',
              cacheKey: 'lnm-mni160-prealign-test',
              dims,
              supportStatus: 'supported'
            }]
          };
        }
      };
    }
    if (href === 'https://example.test/prealign-mni160.nii') {
      return {
        ok: true,
        status: 200,
        async arrayBuffer() { return mni160Buffer; }
      };
    }
    throw new Error(`unexpected fetch in prealign test: ${href}`);
  };

  try {
    await app.prealignToMni160();
    const structural = decodeNiftiForTest(await app.structuralFile.arrayBuffer());
    const brainmask = decodeNiftiForTest(await app.brainmaskFile.arrayBuffer());
    assert.deepEqual(structural.dims, dims,
      'prealigned structural must use the fixed lnm-mni160 dims');
    assert.deepEqual(brainmask.dims, dims,
      'prealigned brainmask must use the fixed lnm-mni160 dims');
    assertAffineClose(structural.header.affine, [
      [1, 0, 0, -2],
      [0, 1, 0, -2],
      [0, 0, 1, -2],
      [0, 0, 0, 1]
    ], 'prealigned structural header must be fixed lnm-mni160');
    assertAffineClose(brainmask.header.affine, [
      [1, 0, 0, -2],
      [0, 1, 0, -2],
      [0, 0, 1, -2],
      [0, 0, 0, 1]
    ], 'prealigned brainmask header must be fixed lnm-mni160');
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.caches = originalCaches;
  }
}

// ---- Test 20: subject-space Yeo atlas QC uses the fixed MNI160 grid and
//      requests a label-preserving inverse warp ----
{
  globalThis.nifti = niftiModule.default || niftiModule;
  const app = makeApp();
  const mni160Data = new Uint8Array(2 * 2 * 2);
  const yeoData = new Uint8Array(5 * 5 * 5);
  yeoData[0] = 7;
  const affine = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0
  ];
  const mni160Buffer = writeNifti1(mni160Data, {
    dims: [2, 2, 2],
    spacing: [1, 1, 1],
    affine,
    description: 'synthetic lnm-mni160'
  });
  const yeoBuffer = writeNifti1(yeoData, {
    dims: [5, 5, 5],
    spacing: [1, 1, 1],
    affine,
    description: 'synthetic Yeo atlas'
  });
  const projectedAtlasBuffer = writeNifti1(new Uint8Array([7, 0, 0, 0, 0, 0, 0, 0]), {
    dims: [2, 2, 2],
    spacing: [1, 1, 1],
    affine,
    description: 'synthetic subject-space atlas'
  });

  app.structuralFile = makeNiftiFile('lnm-prealign-t1.nii', mni160Buffer);
  app.hasRegistrationDisplacement = true;
  let inverseSettings = null;
  app.executor = {
    runInverseWarpMask: async (settings) => {
      inverseSettings = settings;
      queueMicrotask(() => {
        app.handleStageData({ stage: 'atlas-patient', niftiData: projectedAtlasBuffer });
        app.handleStepComplete('inverse-warp-mask');
      });
    }
  };
  const renderedStacks = [];
  app.viewerController = {
    loadVolumeStack: async (entries) => { renderedStacks.push(entries); },
    setStageVisible: () => true
  };

  const originalFetch = globalThis.fetch;
  const originalCaches = globalThis.caches;
  globalThis.caches = undefined;
  globalThis.fetch = async (url) => {
    const href = String(url);
    if (href.includes('manifest.json')) {
      return {
        ok: true,
        status: 200,
        async json() {
          return {
            atlasAssets: [
              {
                id: 'lnm-mni160',
                sourceUrl: 'https://example.test/qc-mni160.nii',
                cacheKey: 'lnm-mni160-qc-test',
                dims: [2, 2, 2],
                supportStatus: 'supported'
              },
              {
                id: 'yeo7-2mm',
                sourceUrl: 'https://example.test/qc-yeo7.nii',
                cacheKey: 'yeo7-qc-test',
                dims: [5, 5, 5],
                supportStatus: 'supported'
              }
            ]
          };
        }
      };
    }
    if (href === 'https://example.test/qc-mni160.nii') {
      return {
        ok: true,
        status: 200,
        async arrayBuffer() { return mni160Buffer; }
      };
    }
    if (href === 'https://example.test/qc-yeo7.nii') {
      return {
        ok: true,
        status: 200,
        async arrayBuffer() { return yeoBuffer; }
      };
    }
    throw new Error(`unexpected fetch in atlas QC test: ${href}`);
  };

  try {
    await app.showSubjectSpaceAtlas();
    assert.ok(app.patientAtlasFile,
      'subject-space atlas QC must store the projected atlas file');
    assert.equal(inverseSettings.stage, 'atlas-patient',
      'subject-space atlas QC must use the atlas-patient stage');
    assert.equal(inverseSettings.labelMap, true,
      'subject-space atlas QC must request label-preserving inverse warp');
    assert.deepEqual(inverseSettings.maskDims, [2, 2, 2],
      'subject-space atlas QC must inverse-warp from the fixed lnm-mni160 grid');
    assert.equal(new Uint8Array(inverseSettings.maskBuffer)[0], 7,
      'Yeo labels must survive the Yeo-grid to MNI160 resample before inverse warp');
    const atlasEntry = renderedStacks.at(-1).find(entry => entry.stage === 'atlas-qc');
    assert.ok(atlasEntry, 'patient viewer stack must include the atlas QC layer');
    assert.equal(atlasEntry.colormap, 'lnm-yeo7',
      'atlas QC layer must use the Yeo label colormap');
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.caches = originalCaches;
  }
}

console.log('lnm-app behavior OK: 21 dispatch + precondition + explicit-start + worker-wait + threshold-preview/projection + subject-atlas QC + affected-network labels + layer-toggle + min-cluster-input + top-percent + auto-promote + coverage-note + version-label + MNI160 threshold-resample/header cases.');
