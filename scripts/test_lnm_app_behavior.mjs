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
import { LESION_MASK_COLORMAP_ID } from '../web/js/app/lnm-labels.js';

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
  const imageBuffer = nifti.readImage(header, buf);
  const byteOffset = imageBuffer.byteOffset || 0;
  let data;
  switch (header.datatypeCode) {
    case 2:
      data = new Uint8Array(imageBuffer, byteOffset);
      break;
    case 4:
      data = new Int16Array(imageBuffer, byteOffset);
      break;
    case 16:
      data = new Float32Array(imageBuffer, byteOffset);
      break;
    default:
      data = new Uint8Array(imageBuffer, byteOffset);
  }
  return {
    header,
    dims: [Number(header.dims[1]), Number(header.dims[2]), Number(header.dims[3])],
    data
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
    contains: (name) => classes.has(name),
    toggle: (name, force) => {
      const shouldAdd = force === undefined ? !classes.has(name) : !!force;
      if (shouldAdd) classes.add(name);
      else classes.delete(name);
      return shouldAdd;
    }
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

// ---- Test 7a: lesion masks render with the dedicated blue colormap ----
{
  const app = makeApp();
  const overlayCalls = [];
  const renderedStacks = [];
  app.structuralFile = { name: 'subject-t1.nii' };
  app.networkMapFile = { name: 'network-map.nii' };
  app.networkMapSpacing = [1, 1, 1];
  app.viewerController = {
    loadOverlay: async (...args) => { overlayCalls.push(args); },
    loadVolumeStack: async (entries) => { renderedStacks.push(entries); }
  };
  const lesion = { name: 'manual-lesion.nii' };

  await app.setLesion(lesion);
  assert.equal(overlayCalls.at(-1)[1], LESION_MASK_COLORMAP_ID,
    'manual lesion overlays must render with the blue lesion colormap');

  await app.displayNetworkMapOnYeoTemplate({
    data: new Uint8Array([0, 1, 1, 0, 0, 1, 0, 0]),
    dims: [2, 2, 2]
  }, [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0
  ]);
  const lesionEntry = renderedStacks.at(-1).find(entry => entry.stage === 'lesion');
  assert.equal(lesionEntry.colormap, LESION_MASK_COLORMAP_ID,
    'Yeo-grid lesion overlays must render with the blue lesion colormap');
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
  globalThis.nifti = niftiModule.default || niftiModule;
  app.structuralFile = { name: 'lnm-prealign-t1.nii', arrayBuffer: async () => new ArrayBuffer(8) };
  app.brainmaskFile = makeNiftiFile('lnm-prealign-brainmask.nii', writeNifti1(
    new Uint8Array([1, 0, 1, 0, 0, 1, 0, 1]),
    {
      dims: [2, 2, 2],
      spacing: [1, 1, 1],
      affine: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
      description: 'registration brain mask'
    }
  ));
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
  assert.deepEqual(registrationSettings.brainMaskDims, [2, 2, 2],
    'runRegistration must pass the prealigned brain-mask dimensions to the worker');
  assert.equal(registrationSettings.brainMaskBuffer.byteLength, 8,
    'runRegistration must pass a compact binary brain mask to the worker');

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
  assert.ok(app.autoLesionSeedFile, 'autoLesionSeedFile must be populated before the stage resolves');
  assert.equal(app.lesionMaskFile, null,
    'automatic segmentation is only a seed; confirmed manual review populates lesionMaskFile');
}

// ---- Test 10b: auto structural pipeline pauses at manual mask review ----
{
  const app = makeApp();
  const calls = [];
  app.structuralFile = { name: 'subject-t1.nii' };
  app.runBrainExtraction = async () => { calls.push('brain'); };
  app.prealignToMni160 = async () => { calls.push('prealign'); };
  app.runLesionSegmentation = async () => {
    calls.push('segment');
    app.autoLesionSeedFile = { name: 'auto-seed.nii' };
  };
  app.startLesionMaskReview = async () => {
    calls.push('review');
    app.maskReviewActive = true;
  };
  app.runRegistration = async () => { calls.push('register'); };
  app.applyRegistrationToLesion = async () => { calls.push('warp'); };
  app.selectedPipeline = {
    id: 'auto-with-review',
    displayName: 'Auto With Review',
    stages: [
      { id: 'brain', module: 'brain-extraction', required: true },
      { id: 'prealign', module: 'prealign', required: true },
      { id: 'seed', module: 'inference-pipeline', required: true },
      { id: 'register', module: 'registration', required: true }
    ]
  };

  await app.runFullPipeline();

  assert.deepEqual(calls, ['brain', 'prealign', 'segment', 'review'],
    'auto pipeline must stop after loading the editable lesion seed');
  assert.equal(app._pendingMaskResume?.nextStageIndex, 3,
    'paused pipeline must remember the stage after editable-seed review');
  assert.equal(app.maskReviewActive, true,
    'mask review must be active after the auto seed is created');
  assert.ok(app._messages.includes('Pipeline paused for manual lesion-mask review.'),
    'pause must be visible to the user');
}

// ---- Test 10c: blank review path starts an editable native-space mask ----
{
  const app = makeApp();
  const loaded = [];
  let blankStarts = 0;
  let seedLoads = 0;
  const toolCalls = [];
  app.nativeStructuralFile = { name: 'native-t1.nii' };
  app.nativeStructuralInfo = {
    dims: [2, 2, 2],
    affine: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ]
  };
  app.viewerController = {
    loadBaseVolume: async (file, opts) => { loaded.push({ file, opts }); }
  };
  app.maskDrawingController = {
    hasDrawing: true,
    startBlank: () => { blankStarts += 1; },
    loadSeedFile: async () => { seedLoads += 1; },
    ensureDrawing: () => {},
    setTool: (tool) => { toolCalls.push(tool); }
  };

  const seed = await app.startLesionMaskReview({ blank: true });

  assert.equal(seed, null,
    'blank review should not create a projected seed file');
  assert.equal(blankStarts, 1,
    'blank review must create a new editable drawing');
  assert.equal(seedLoads, 0,
    'blank review must not load a seed drawing');
  assert.equal(loaded[0].file.name, 'native-t1.nii',
    'blank review must render the native T1 as the drawing base');
  assert.equal(app.maskReviewActive, true,
    'blank review must activate mask-review mode');
  assert.equal(app.lesionMaskConfirmed, false,
    'blank review must require explicit confirmation');
  assert.deepEqual(toolCalls, ['paint'],
    'blank review must leave the drawing controller in paint mode');
}

// ---- Test 10c.1: advanced Manual mask choice starts blank review ----
{
  const app = makeApp();
  const listeners = {};
  const restoreDocument = useMockElements({
    startManualMaskButton: {
      addEventListener: (eventName, handler) => { listeners[`startManualMaskButton:${eventName}`] = handler; }
    }
  });
  let requestedOptions = null;
  app.startLesionMaskReview = async (options) => { requestedOptions = options; };

  try {
    app.bindEvents();
    assert.equal(typeof listeners['startManualMaskButton:click'], 'function',
      'advanced Manual mask button must bind a click handler');
    listeners['startManualMaskButton:click']();
    await waitForMicrotaskCondition(
      () => requestedOptions !== null,
      'advanced Manual mask button must call startLesionMaskReview'
    );
    assert.deepEqual(requestedOptions, { blank: true },
      'advanced Manual mask button must start a blank editable lesion mask');
  } finally {
    restoreDocument();
  }
}

// ---- Test 10d: confirming a native drawing writes fixed-header MNI160 mask ----
{
  globalThis.nifti = niftiModule.default || niftiModule;
  const app = makeApp();
  const nativeDims = [4, 4, 4];
  const nativeAffine = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ];
  const prealignSamplingAffine = [
    [1, 0, 0, -1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ];
  const fixedMniAffine = [
    [1, 0, 0, -4],
    [0, 1, 0, -4],
    [0, 0, 1, -4],
    [0, 0, 0, 1]
  ];
  const flatten = (affine) => [
    affine[0][0], affine[0][1], affine[0][2], affine[0][3],
    affine[1][0], affine[1][1], affine[1][2], affine[1][3],
    affine[2][0], affine[2][1], affine[2][2], affine[2][3]
  ];
  const nativeMask = new Uint8Array(64);
  nativeMask[2 + 1 * 4 + 1 * 16] = 1;
  const nativeFile = makeNiftiFile('lnm-lesion-edited-native.nii', writeNifti1(nativeMask, {
    dims: nativeDims,
    spacing: [1, 1, 1],
    affine: flatten(nativeAffine),
    description: 'synthetic native drawing'
  }));
  let closeOptions = null;
  let resumed = false;
  const toolbarClassList = makeClassList();
  const toolbar = { classList: toolbarClassList };
  const confirmButton = { disabled: false };
  const downloadEditedButton = { disabled: false };
  const status = { textContent: '' };
  const downloadButton = { disabled: true };
  const restoreDocument = useMockElements({
    maskDrawingToolbar: toolbar,
    confirmLesionMaskButton: confirmButton,
    downloadEditedLesionMaskButton: downloadEditedButton,
    maskReviewStatus: status,
    downloadLesionMaskButton: downloadButton
  });
  app.maskDrawingController = {
    hasDrawing: true,
    exportDrawingFile: async () => nativeFile,
    close: (options) => { closeOptions = options; }
  };
  app.prealignSamplingAffine = prealignSamplingAffine;
  app.fixedMni160Info = {
    dims: nativeDims,
    affine: fixedMniAffine,
    spacing: [1, 1, 1]
  };
  app.resumePipelineAfterMaskConfirmation = async () => { resumed = true; };

  let confirmed;
  let decoded;
  try {
    confirmed = await app.confirmLesionDrawing({ resumePipeline: true });
    decoded = decodeNiftiForTest(await confirmed.arrayBuffer());
  } finally {
    restoreDocument();
  }

  assert.equal(app.confirmedNativeLesionFile, nativeFile,
    'confirmed native drawing must be kept for download/edit provenance');
  assert.equal(app.lesionMaskFile, confirmed,
    'confirmed MNI160 mask must become the downstream lesionMaskFile');
  assert.equal(app.lesionMaskConfirmed, true,
    'confirming must mark the lesion mask as user-approved');
  assert.equal(app.maskReviewActive, false,
    'confirming must leave mask-review mode');
  assert.deepEqual(closeOptions, { clearDrawing: true },
    'confirming must clear the accepted NiiVue drawing overlay');
  assert.equal(toolbarClassList.contains('hidden'), true,
    'confirming must hide the mask-review toolbar');
  assert.equal(resumed, true,
    'confirming from the paused pipeline must resume analysis');
  assert.deepEqual(decoded.dims, nativeDims,
    'confirmed mask must use the fixed MNI160 dimensions');
  assertAffineClose(decoded.header.affine, fixedMniAffine,
    'confirmed mask header must use the fixed MNI160 affine');
  assert.equal(decoded.data[3 + 1 * 4 + 1 * 16], 1,
    'native drawing must be resampled through the saved prealign sampling affine');
  assert.equal(decoded.data[2 + 1 * 4 + 1 * 16], 0,
    'native drawing must not be copied directly onto the MNI160 grid');
}

// ---- Test 10e: confirming the mask resumes the remaining pipeline stages ----
{
  const app = makeApp();
  const calls = [];
  app._perfRunStart = app._now();
  app._runStage = async (stage) => { calls.push(stage.id); };
  app._pendingMaskResume = {
    pipeline: {
      id: 'resume-test',
      displayName: 'Resume Test',
      stages: [
        { id: 'brain', module: 'brain-extraction' },
        { id: 'seed', module: 'inference-pipeline' },
        { id: 'register', module: 'registration' },
        { id: 'overlap', module: 'parcel-overlap' }
      ]
    },
    nextStageIndex: 2
  };

  await app.resumePipelineAfterMaskConfirmation();

  assert.deepEqual(calls, ['register', 'overlap'],
    'mask confirmation must resume at the stage after editable-seed review');
  assert.equal(app._pendingMaskResume, null,
    'pending resume state must be cleared after resuming');
  assert.ok(app._messages.some(message => /Pipeline complete/.test(message)),
    'resumed pipeline must report completion when remaining stages finish');
}

// ---- Test 11: thresholding schedules a live viewer preview overlay ----
{
  const app = makeApp();
  const summaryEl = { textContent: '' };
  const restoreDocument = useMockElements({
    networkThresholdValue: { value: '10' },
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
      'top-percent threshold should apply min-cluster cleanup to the binary mask');
    assert.match(summaryEl.textContent, /^0 voxels survive top 10%/,
      'threshold summary should update immediately');
    assert.match(summaryEl.textContent, /cluster≥4 removed 1 voxels/,
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
  const mapFunctionResultsEl = { classList: makeClassList(['hidden']) };
  const mapFunctionTableEl = { innerHTML: '', appendChild: () => {} };
  const elements = {
    networkThresholdValue: { value: '10' },
    networkThresholdSymmetric: { checked: false },
    networkThresholdMinCluster: { value: '0' },
    networkThresholdSummary: summaryEl,
    affectedNetworkResults: affectedResultsEl,
    affectedNetworkTable: affectedTableEl,
    mapFunctionProfileResults: mapFunctionResultsEl,
    mapFunctionProfileTable: mapFunctionTableEl,
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
  app.functionProfiles = {
    sourceLabel: 'Neurosynth v7 via NiMARE',
    networkProfiles: {
      Somatomotor: [{ term: 'motor', score: 0.9 }],
      Default: [{ term: 'memory', score: 0.8 }]
    }
  };

  try {
    const mask = app.applyNetworkThreshold();
    await app._functionalProfileRenderPromise;
    assert.equal(mask.reduce((sum, value) => sum + value, 0), 1,
      'top-percent threshold should keep only the strongest high-valued voxel in this toy map');
    assert.ok(app.affectedNetworkResult,
      'thresholding must store a final affected-network summary');
    assert.deepEqual(
      app.affectedNetworkResult.summary.networks.map(row => [row.network, row.voxelsInLesion]),
      [['Somatomotor', 1]],
      'affected-network summary must aggregate thresholded map voxels by Yeo label'
    );
    assert.equal(affectedResultsEl.classList.contains('hidden'), false,
      'affected-network table must become visible after thresholding');
    assert.ok(renderedText.includes('Somatomotor'),
      'affected-network table must render the dominant Yeo network name');
    assert.ok(renderedText.includes('% of map'),
      'affected-network table must use map-specific percent copy');
    assert.equal(mapFunctionResultsEl.classList.contains('hidden'), false,
      'connectivity-map functional profile table must become visible after thresholding');
    assert.ok(renderedText.includes('motor'),
      'connectivity-map functional profile must rank terms from affected Yeo networks');
    assert.ok(renderedText.includes('Neurosynth v7 via NiMARE'),
      'connectivity-map functional profile must show its source label');
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
    app.configureTopPercentThresholdSlider({ resetValue: true });
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

// ---- Test 16: unavailable viewer layers are disabled and unchecked ----
{
  const app = makeApp();
  app.structuralFile = { name: 't1.nii' };
  app.viewerController = {
    getVolumeIndexForStage: stage => stage === 'structural' ? 0 : null
  };

  const makeToggle = () => ({
    checked: true,
    disabled: false,
    addEventListener: () => {}
  });
  const toggles = {
    layerToggleT1: makeToggle(),
    layerToggleBrainMask: makeToggle(),
    layerToggleLesionMask: makeToggle(),
    layerToggleThresholdMap: makeToggle(),
    layerToggleAtlasQc: makeToggle()
  };
  const restoreDocument = useMockElements(toggles);

  try {
    app.refreshViewerLayerControls();
    assert.equal(toggles.layerToggleT1.disabled, false,
      'available structural layer must remain enabled');
    assert.equal(toggles.layerToggleT1.checked, true,
      'available structural layer must mirror the visible preference');
    assert.equal(toggles.layerToggleAtlasQc.disabled, true,
      'Yeo atlas toggle must be disabled before an atlas-QC volume is available');
    assert.equal(toggles.layerToggleAtlasQc.checked, false,
      'unavailable Yeo atlas toggle must not appear checked');
  } finally {
    restoreDocument();
  }
}

// ---- Test 16: direct lesion overlap renders exploratory functional terms ----
{
  const app = makeApp();
  const directResultsEl = { classList: makeClassList(['hidden']) };
  const directTableEl = { innerHTML: '', appendChild: () => {} };
  const restoreDocument = useMockElements({
    directFunctionProfileResults: directResultsEl,
    directFunctionProfileTable: directTableEl
  });
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
  app.overlapResult = {
    summary: {
      networks: [
        { network: 'Visual', fractionOfLesion: 0.6 },
        { network: 'Default', fractionOfLesion: 0.4 }
      ]
    }
  };
  app.functionProfiles = {
    sourceLabel: 'Neurosynth v7 via NiMARE',
    networkProfiles: {
      Visual: [{ term: 'visual', score: 0.8 }],
      Default: [{ term: 'memory', score: 0.7 }]
    }
  };

  try {
    const ranked = await app.updateDirectFunctionProfile();
    assert.equal(directResultsEl.classList.contains('hidden'), false,
      'direct functional profile table must become visible after overlap');
    assert.equal(ranked[0].term, 'visual',
      'direct functional profile must weight terms by direct lesion-overlap fractions');
    assert.ok(renderedText.includes('Neurosynth v7 via NiMARE'),
      'direct functional profile must show its source label');
  } finally {
    globalThis.document.createElement = originalCreateElement;
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
  app.maskReviewActive = true;

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
  const drawingVisibilityCalls = [];
  const activeStages = new Set(['structural', 'brainmask', 'segmentation', 'lesion', 'threshold-preview', 'atlas-qc']);
  app.viewerController = {
    getVolumeIndexForStage: stage => activeStages.has(stage) ? 1 : null,
    setStageVisible: (stage, visible) => {
      stageVisibilityCalls.push([stage, visible]);
      return true;
    }
  };
  app.maskDrawingController = {
    hasDrawing: true,
    setVisible: (visible) => { drawingVisibilityCalls.push(visible); }
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
    assert.equal(drawingVisibilityCalls.at(-1), false,
      'lesion toggle must hide the active editable drawing overlay');

    listeners['layerToggleLesionMask:change']({ target: { checked: true } });
    await waitForMicrotaskCondition(() => drawingVisibilityCalls.at(-1) === true,
      'lesion toggle must restore the active editable drawing overlay');
    assert.equal(drawingVisibilityCalls.at(-1), true,
      'lesion toggle must restore the active editable drawing overlay');

    listeners['layerToggleAtlasQc:change']({ target: { checked: false } });
    assert.deepEqual(stageVisibilityCalls.at(-1), ['atlas-qc', false],
      'atlas QC toggle must target the subject-space atlas viewer stage');
  } finally {
    restoreDocument();
  }
}

// ---- Test 16a: layer controls reflect the active viewer stack, not stale files ----
{
  const app = makeApp();
  app.structuralFile = { name: 't1.nii' };
  app.brainmaskFile = { name: 'brainmask.nii' };
  app.lesionMaskFile = { name: 'lesion-mask.nii' };
  app.lesionFile = { name: 'lesion.nii' };
  app.patientAtlasFile = { name: 'atlas-patient.nii' };
  app.patientThresholdedMaskFile = { name: 'threshold-patient.nii' };
  const activeStages = new Set(['structural', 'threshold-preview']);
  app.viewerController = {
    getVolumeIndexForStage: stage => activeStages.has(stage) ? 1 : null
  };
  const makeToggle = () => ({ checked: true, disabled: false });
  const toggles = {
    layerToggleT1: makeToggle(),
    layerToggleBrainMask: makeToggle(),
    layerToggleLesionMask: makeToggle(),
    layerToggleThresholdMap: makeToggle(),
    layerToggleAtlasQc: makeToggle()
  };
  const restoreDocument = useMockElements(toggles);

  try {
    app.refreshViewerLayerControls();
    assert.deepEqual(
      Object.fromEntries(Object.entries(toggles).map(([id, el]) => [id, { checked: el.checked, disabled: el.disabled }])),
      {
        layerToggleT1: { checked: true, disabled: false },
        layerToggleBrainMask: { checked: false, disabled: false },
        layerToggleLesionMask: { checked: false, disabled: false },
        layerToggleThresholdMap: { checked: true, disabled: false },
        layerToggleAtlasQc: { checked: false, disabled: false }
      },
      'final threshold view controls must be unchecked for inactive layers but enabled when data exists'
    );
  } finally {
    restoreDocument();
  }
}

// ---- Test 16b: checking an available inactive layer adds it to the viewer stack ----
{
  const app = makeApp();
  app.structuralFile = { name: 't1.nii' };
  app.brainmaskFile = { name: 'brainmask.nii' };
  app.lesionMaskFile = { name: 'lesion-mask.nii' };
  app.patientAtlasFile = { name: 'atlas-patient.nii' };
  app.patientThresholdedMaskFile = { name: 'threshold-patient.nii' };
  const activeStages = new Set(['structural', 'threshold-preview']);
  const listeners = {};
  const overlayCalls = [];
  const visibilityCalls = [];
  const makeToggle = id => ({
    checked: true,
    disabled: false,
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
  app.viewerController = {
    getVolumeIndexForStage: stage => activeStages.has(stage) ? 1 : null,
    loadOverlay: async (file, colormap, opacity, options = {}) => {
      overlayCalls.push([file, colormap, opacity, options]);
      if (options.stage) activeStages.add(options.stage);
    },
    setStageVisible: (stage, visible) => {
      visibilityCalls.push([stage, visible]);
      return activeStages.has(stage);
    }
  };

  try {
    app.bindViewerLayerToggles();
    assert.equal(toggles.layerToggleBrainMask.checked, false,
      'inactive-but-available brain mask starts unchecked');
    assert.equal(toggles.layerToggleBrainMask.disabled, false,
      'inactive-but-available brain mask can be activated');

    listeners['layerToggleBrainMask:change']({ target: { checked: true } });
    await waitForMicrotaskCondition(() => activeStages.has('brainmask') && toggles.layerToggleBrainMask.checked,
      'checking brain mask must load the brainmask overlay and refresh the checkbox');
    assert.equal(overlayCalls.at(-1)[0], app.brainmaskFile,
      'brain-mask activation must load the stored brain mask file');
    assert.equal(overlayCalls.at(-1)[3].stage, 'brainmask');
    assert.equal(toggles.layerToggleBrainMask.checked, true,
      'brain-mask checkbox must become checked after the overlay is loaded');

    listeners['layerToggleLesionMask:change']({ target: { checked: true } });
    await waitForMicrotaskCondition(() => activeStages.has('segmentation') && toggles.layerToggleLesionMask.checked,
      'checking lesion mask must load the lesion overlay and refresh the checkbox');
    assert.equal(overlayCalls.at(-1)[0], app.lesionMaskFile,
      'lesion activation must prefer the confirmed MNI160 lesion mask');
    assert.equal(overlayCalls.at(-1)[3].stage, 'segmentation');

    listeners['layerToggleAtlasQc:change']({ target: { checked: true } });
    await waitForMicrotaskCondition(() => activeStages.has('atlas-qc') && toggles.layerToggleAtlasQc.checked,
      'checking Yeo atlas must load the atlas-QC overlay and refresh the checkbox');
    assert.equal(overlayCalls.at(-1)[0], app.patientAtlasFile,
      'Yeo atlas activation must load the projected patient-space atlas');
    assert.equal(overlayCalls.at(-1)[3].stage, 'atlas-qc');
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

// ---- Test 17a: final patient-space threshold view only shows T1 + threshold ----
{
  const app = makeApp();
  app.structuralFile = { name: 't1.nii' };
  app.brainmaskFile = { name: 'brainmask.nii' };
  app.lesionMaskFile = { name: 'lesion-mask.nii' };
  app.lesionFile = { name: 'lesion.nii' };
  app.patientAtlasFile = { name: 'atlas-patient.nii' };
  app.patientThresholdedMaskFile = { name: 'threshold-patient.nii' };
  const renderedStacks = [];
  app.viewerController = {
    loadVolumeStack: async (entries) => { renderedStacks.push(entries); },
    setStageVisible: () => true
  };

  await app.renderPatientLayerStack();

  const stages = renderedStacks.at(-1).map(entry => entry.stage);
  assert.deepEqual(stages, ['structural', 'threshold-preview'],
    'final patient-space threshold view must load only the structural T1 and threshold map');
}

// ---- Test 17b: advanced atlas-alignment QC button mirrors subject-atlas
//      enablement and invokes the existing visual QC overlay path ----
{
  const app = makeApp();
  const listeners = {};
  const makeButton = id => ({
    disabled: false,
    addEventListener: (eventName, handler) => { listeners[`${id}:${eventName}`] = handler; }
  });
  const controls = {
    checkAtlasAlignmentButton: makeButton('checkAtlasAlignmentButton'),
    registrationQcMode: {
      value: 'mni',
      addEventListener: (eventName, handler) => { listeners[`registrationQcMode:${eventName}`] = handler; }
    },
    showSubjectAtlasButton: makeButton('showSubjectAtlasButton'),
    downloadSubjectAtlasButton: makeButton('downloadSubjectAtlasButton')
  };
  const restoreDocument = useMockElements(controls);
  let qcCalls = 0;
  app.showRegistrationQc = async () => { qcCalls += 1; };

  try {
    app.bindEvents();
    assert.equal(controls.checkAtlasAlignmentButton.disabled, true,
      'advanced atlas QC button starts disabled before registration');
    assert.equal(controls.showSubjectAtlasButton.disabled, true,
      'results atlas QC button starts disabled before registration');

    app.structuralFile = { name: 't1.nii' };
    app.hasRegistrationDisplacement = false;
    app.refreshSubjectAtlasControls();
    assert.equal(controls.checkAtlasAlignmentButton.disabled, true,
      'advanced atlas QC button stays disabled until registration displacement exists');
    assert.equal(controls.showSubjectAtlasButton.disabled, true,
      'results atlas QC button stays disabled until registration displacement exists');

    app.hasRegistrationDisplacement = true;
    app.refreshSubjectAtlasControls();
    assert.equal(controls.checkAtlasAlignmentButton.disabled, false,
      'advanced atlas QC button enables after structural T1 + registration');
    assert.equal(controls.showSubjectAtlasButton.disabled, false,
      'results atlas QC button enables after structural T1 + registration');

    listeners['checkAtlasAlignmentButton:click']();
    await waitForMicrotaskCondition(() => qcCalls === 1,
      'advanced atlas QC button must invoke showRegistrationQc');
  } finally {
    restoreDocument();
  }
}

// ---- Test 17c: Patient/MNI registration blend slider drives the
//      registered-patient overlay opacity in QC views ----
{
  const app = makeApp();
  const listeners = {};
  const controls = {
    registrationBlendValue: {
      value: '0.75',
      addEventListener: (eventName, handler) => { listeners[`registrationBlendValue:${eventName}`] = handler; }
    },
    registrationBlendLabel: { textContent: '' }
  };
  const restoreDocument = useMockElements(controls);
  const stageOpacityCalls = [];
  const appliedStages = [];
  let glUpdates = 0;

  app.viewerController = {
    setStageOpacity: (stage, opacity, options = {}) => {
      stageOpacityCalls.push([stage, opacity, options]);
      if (options.apply) appliedStages.push(stage);
      if (options.redraw) glUpdates += 1;
      return true;
    },
    nv: {
      updateGLVolume: () => { glUpdates += 1; },
      drawScene: () => {}
    }
  };

  try {
    app.bindEvents();
    assert.equal(app.registrationBlendValue, 0.75,
      'registration blend slider value must initialize app state');
    assert.equal(controls.registrationBlendLabel.textContent, '75% patient',
      'registration blend label must describe the registered-patient opacity');

    controls.registrationBlendValue.value = '0';
    listeners['registrationBlendValue:input']();
    assert.deepEqual(stageOpacityCalls.at(-1).slice(0, 2), ['registered-t1-mni160', 0],
      'MNI-only blend must set registered-patient overlay opacity to zero');
    assert.equal(controls.registrationBlendLabel.textContent, 'MNI template',
      'zero blend label must identify the fixed MNI template');

    controls.registrationBlendValue.value = '1';
    listeners['registrationBlendValue:change']();
    assert.deepEqual(stageOpacityCalls.at(-1).slice(0, 2), ['registered-t1-mni160', 1],
      'patient-only blend must set registered-patient overlay opacity to one');
    assert.equal(controls.registrationBlendLabel.textContent, 'Registered patient',
      'full blend label must identify the registered patient scan');
    assert.ok(appliedStages.includes('registered-t1-mni160'),
      'blend changes must be applied to the active registered-patient stage');
    assert.ok(glUpdates >= 2,
      'viewer must redraw after active registration blend changes');
  } finally {
    restoreDocument();
  }
}

// ---- Test 17d: moving the Patient/MNI blend slider opens MNI QC if
//      registration artifacts exist but the active viewer is still patient-space ----
{
  const app = makeApp();
  const listeners = {};
  const controls = {
    registrationQcMode: {
      value: 'patient',
      addEventListener: (eventName, handler) => { listeners[`registrationQcMode:${eventName}`] = handler; }
    },
    registrationBlendValue: {
      value: '0.6',
      addEventListener: (eventName, handler) => { listeners[`registrationBlendValue:${eventName}`] = handler; }
    },
    registrationBlendLabel: { textContent: '' }
  };
  const restoreDocument = useMockElements(controls);
  let mniRenders = 0;
  app.hasRegistrationDisplacement = true;
  app.registeredT1MniFile = { name: 'registered.nii' };
  app.viewerController = {
    setStageOpacity: () => false
  };
  app.renderMniRegistrationQc = async () => { mniRenders += 1; };

  try {
    app.bindEvents();
    listeners['registrationBlendValue:input']();
    await waitForMicrotaskCondition(() => mniRenders === 1,
      'blend slider must render MNI QC when the blend target is not active');
    assert.equal(controls.registrationQcMode.value, 'mni',
      'blend slider must switch the QC selector to MNI space before rendering');
    assert.equal(app.registrationQcMode, 'mni',
      'blend slider must update app QC mode to MNI space');
  } finally {
    restoreDocument();
  }
}

// ---- Test 17e: registration QC render modes use MNI-space artifacts ----
{
  globalThis.nifti = niftiModule.default || niftiModule;
  const app = makeApp();
  const affine = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0
  ];
  const templateBuffer = writeNifti1(new Float32Array([0, 1]), {
    dims: [2, 1, 1],
    spacing: [1, 1, 1],
    affine,
    description: 'qc template'
  });
  const registeredBuffer = writeNifti1(new Float32Array([1, 0]), {
    dims: [2, 1, 1],
    spacing: [1, 1, 1],
    affine,
    description: 'qc registered'
  });
  const displacementBuffer = writeNifti1(new Float32Array([0, 3]), {
    dims: [2, 1, 1],
    spacing: [1, 1, 1],
    affine,
    description: 'qc displacement'
  });
  const atlasBuffer = writeNifti1(new Uint8Array([1, 2]), {
    dims: [2, 1, 1],
    spacing: [1, 1, 1],
    affine,
    description: 'qc yeo atlas'
  });

  app.registrationTemplateFile = makeNiftiFile('template.nii', templateBuffer);
  app.yeoAtlasMni160File = makeNiftiFile('yeo-mni.nii', atlasBuffer);
  app.handleStageData({ stage: 'registered-t1-mni160', niftiData: registeredBuffer });
  app.handleStageData({ stage: 'registration-displacement-mag', niftiData: displacementBuffer });
  assert.equal(app.registeredT1MniFile.name, 'lnm-registered-t1-mni160.nii',
    'registered T1 QC stage must be cached as a File');
  assert.equal(app.displacementMagnitudeFile.name, 'lnm-registration-displacement-mag.nii',
    'displacement magnitude QC stage must be cached as a File');
  app.registrationBlendValue = 0.8;

  const renderedStacks = [];
  app.viewerController = {
    loadVolumeStack: async (entries) => { renderedStacks.push(entries); },
    setStageVisible: () => true,
    setStageOpacity: () => false
  };

  await app.renderMniRegistrationQc();
  assert.deepEqual(renderedStacks.at(-1).map(entry => entry.stage), [
    'registration-template',
    'registered-t1-mni160'
  ], 'MNI QC view must show only the fixed template and registered T1');
  assert.equal(renderedStacks.at(-1).find(entry => entry.stage === 'registered-t1-mni160').opacity, 0.8,
    'MNI QC view must use the Patient/MNI blend slider for registered T1 opacity');

  await app.renderCheckerboardRegistrationQc();
  assert.deepEqual(renderedStacks.at(-1).map(entry => entry.stage), [
    'registration-checkerboard'
  ], 'checkerboard QC view must show only the fixed-template/registered-T1 checkerboard');
  assert.ok(app.registrationCheckerboardFile,
    'checkerboard QC view must cache the generated checkerboard NIfTI');

  await app.renderDisplacementRegistrationQc();
  assert.ok(renderedStacks.at(-1).some(entry => entry.stage === 'registration-displacement' && entry.scalar),
    'displacement QC view must render the displacement magnitude map as a scalar overlay');
  assert.equal(renderedStacks.at(-1).find(entry => entry.stage === 'registered-t1-mni160').opacity, 0.8,
    'displacement QC view must retain the same registered T1 blend for patient/template comparison');
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

console.log('lnm-app behavior OK: dispatch + precondition + explicit-start + worker-wait + manual mask review/confirm/resume + threshold-preview/projection + subject-atlas QC + advanced atlas-QC button + registration QC modes/blend + affected-network labels + functional profiles + layer-toggle + min-cluster-input + top-percent + auto-promote + coverage-note + version-label + MNI160 threshold-resample/header cases.');
