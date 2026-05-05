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
for (const method of [
  'init', 'setStructural', 'setLesion', 'runYeoOverlap', 'exportCsv',
  // Phase 2a.1.4b additions:
  'runBrainExtraction', 'downloadBrainMask',
  // Phase 2a.2.3 additions:
  'runLesionSegmentation', 'downloadLesionMask',
  // Phase 3.4 additions:
  'runRegistration',
  // Phase 4.4 additions:
  'runFcNetworkMap', 'downloadNetworkMap',
  // Phase 5 additions:
  'applyNetworkThreshold', 'scheduleThresholdPreviewOverlay',
  'renderThresholdPreviewOverlay', 'downloadThresholdedNetworkMap',
  // Phase 6 additions: warp+resample bridge + one-click full chain.
  'applyRegistrationToLesion', 'runFullPipeline',
  // Phase 13 additions: about-modal version wiring.
  'populateVersionLabel',
  // Phase 15 additions: stage dispatch + threshold-default helper.
  '_runStage', '_applyThresholdDefaults',
  // Phase 19 additions: per-stage perf instrumentation helpers.
  '_now', '_formatMs',
  // Phase 16 additions: in-browser affine pre-registration to MNI160 1mm.
  'prealignToMni160',
  // Phase 21 additions: clear-results control.
  'clearResults',
  // Phase 31: auto-promote the pipeline selection on file drop.
  '_autoPromotePipeline'
]) {
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

// Phase 1c.2 + coverage-note follow-up: the reducer still reports
// voxelsOutsideAtlas, but the UI must present it as unlabeled Yeo cortical
// label coverage rather than a brain-mask warning.
assert.match(src, /voxelsOutsideAtlas/,
  'lnm-app.js must reference voxelsOutsideAtlas (Yeo label coverage-note wiring)');
assert.match(src, /outsideAtlasWarning/,
  'lnm-app.js must keep the #outsideAtlasWarning element for compatibility');
assert.match(src, /Yeo cortical network labels/,
  'coverage note must describe labelled Yeo cortical network voxels');
assert.match(src, /unlabeled by this cortical atlas/,
  'coverage note must describe label-0 voxels as unlabeled by the cortical atlas');

// runYeoOverlap must call the atlas loader rather than the Phase 1c.1 stub.
assert.match(src, /loadAtlasFromManifest|fetchAndDecodeAtlas|loadAtlas/,
  'runYeoOverlap must invoke the atlas-loader (no longer a stub)');

// Phase 2a.1.4b: brain-extraction wiring. The orchestrator must spin up an
// InferenceExecutor, kick a 'run-synthstrip' message via runBrainExtraction,
// listen for 'brainmask' stageData, render it as an overlay (or store it
// for download), and offer a NIfTI download via downloadBrainMask.
assert.match(src, /from\s+['"]\.\/controllers\/InferenceExecutor\.js['"]/,
  'lnm-app.js must import InferenceExecutor');
assert.match(src, /new\s+InferenceExecutor\s*\(/,
  'orchestrator must instantiate InferenceExecutor');
assert.match(src, /\brunSynthStrip\s*\(/,
  'runBrainExtraction must call executor.runSynthStrip(...)');
assert.match(src, /['"]lnm-synthstrip['"]/,
  'orchestrator must reference the lnm-synthstrip asset id literal');
assert.match(src, /['"]brainmask['"]/,
  'orchestrator must wire the brainmask stage');

// Phase 3.4: SynthMorph MNI registration wiring. runRegistration reads the
// lnm-synthmorph-mni manifest entry + the lnm-mni160 reference, calls
// executor.runRegistration(...).
assert.match(src, /\brunRegistration\s*\(/,
  'orchestrator must call executor.runRegistration(...)');
assert.match(src, /['"]lnm-synthmorph-mni['"]/,
  'orchestrator must reference the lnm-synthmorph-mni asset id literal');
assert.match(src, /['"]lnm-mni160['"]/,
  'orchestrator must reference the lnm-mni160 atlas asset id literal');
assert.match(src, /executionProviders:\s*model\.browserRuntime\?\.executionProviders/,
  'orchestrator must pass SynthMorph manifest executionProviders into the worker');

// Phase 4.4: FC weighted-sum wiring. runFcNetworkMap loads the
// yeo7-fc-pack via loadConnectomeFromManifest, calls fcWeightedSum,
// wraps as NIfTI, enables #downloadNetworkMapButton.
assert.match(src, /from\s+['"]\.\/modules\/fc-weighted-sum\.js['"]/,
  'lnm-app.js must import fc-weighted-sum.js');
assert.match(src, /\bfcWeightedSum\s*\(/,
  'orchestrator must invoke fcWeightedSum(...)');
assert.match(src, /\bdecodeFcPack\s*\(/,
  'orchestrator must decode the FC pack via decodeFcPack');
assert.match(src, /\bsummaryToNetworkWeights\s*\(/,
  'orchestrator must convert overlap summary to network weights');
assert.match(src, /['"]yeo7-fc-pack['"]/,
  'orchestrator must reference the yeo7-fc-pack connectome asset id literal');
assert.match(src, /\bloadConnectomeFromManifest\s*\(/,
  'orchestrator must load the FC pack via loadConnectomeFromManifest');
assert.match(src, /this\.networkMapAffine\s*=\s*flatAffine/,
  'runFcNetworkMap must retain the atlas affine for network-map NIfTI outputs');
assert.match(src, /affine:\s*this\.networkMapAffine/,
  'network-map NIfTI writers must use the Yeo atlas affine, not a default centered grid');
assert.match(src, /scalar:\s*true[\s\S]*?symmetricCal:\s*true/,
  'network-map overlay must render as a scalar t-map with symmetric calibration');
assert.match(src, /\bdisplayNetworkMapOnYeoTemplate\s*\(/,
  'runFcNetworkMap must display FC maps on a Yeo-grid display base');
assert.match(src, /\bbuildYeoBrainMaskBaseFile\s*\(/,
  'network-map display must use a Yeo atlas brain-mask base with matching FOV');
assert.match(src, /\bloadVolumeStack\s*\(/,
  'network-map display must replace the patient-space viewer stack with an atlas-space stack');
assert.match(src, /stage:\s*['"]yeo-brain-mask['"]/,
  'network-map display base must be stage-tracked as yeo-brain-mask');
assert.match(src, /downloadNetworkMapButton[\s\S]*?disabled\s*=\s*false|disabled\s*=\s*false[\s\S]*?downloadNetworkMapButton/,
  'lnm-app.js must enable #downloadNetworkMapButton after a successful run');

assert.doesNotMatch(src, /pipelineSelect/,
  'lnm-app.js must not bind a visible pipeline selector; Run analysis is input-driven');
assert.match(src, /getPipelineById\(['"]lnm-yeo-auto['"]\)/,
  'default Run analysis pipeline must be the structural-T1 auto chain');
assert.match(src, /populateVersionLabel\s*\(/,
  'populateVersionLabel must be defined');
assert.match(src, /aboutAppVersion/,
  'populateVersionLabel must reference the #aboutAppVersion DOM id');

// Phase 16: in-browser affine pre-registration. The orchestrator must
// import the centroid + affine helpers from prealign.js and surface a
// 'Pre-align to MNI' button.
assert.match(src, /from\s+['"]\.\/modules\/prealign\.js['"]/,
  'lnm-app.js must import prealign.js');
assert.match(src, /\bcentroidOfMask\s*\(/,
  'prealignToMni160 must call centroidOfMask');
// Phase 26: prealignToMni160 must use the PCA principal-axis aligner.
assert.match(src, /\bprincipalAxisAlign\s*\(/,
  'prealignToMni160 must call principalAxisAlign (PCA-based, Phase 26)');
assert.match(src, /['"]prealignToMniButton['"]/,
  '#prealignToMniButton must be referenced for the click binding');

// Phase 19: per-stage perf instrumentation. runFullPipeline must collect
// stage timings into _perfStats and log a [perf] line per stage. Source-
// grep guards so a future refactor that drops the timing loses the test.
assert.match(src, /this\._perfStats\s*=\s*\[\s*\]/,
  'runFullPipeline must reset _perfStats at start');
assert.match(src, /\[perf\]/,
  'each stage must emit a [perf] line into the console');
assert.match(src, /performance\.now/,
  '_now must call performance.now() when available');

// Phase 15 + Phase 34: _runStage must dispatch on stage.module and cover
// every implemented module, including the Phase 34 'prealign' module.
// Source-grep each module's case-clause (a typo or missing branch
// would silently fall through to the throw).
for (const m of ['brain-extraction', 'prealign', 'inference-pipeline', 'registration',
                 'parcel-overlap', 'fc-weighted-sum', 'threshold']) {
  const re = new RegExp(`case\\s+['"]${m}['"]`);
  assert.match(src, re, `_runStage must handle module '${m}'`);
}
assert.match(src, /this\.selectedPipeline/,
  'runFullPipeline must read from this.selectedPipeline');
assert.match(src, /for\s*\(\s*const\s+stage\s+of\s+pipeline\.stages\s*\)/,
  'runFullPipeline must iterate pipeline.stages');

// Phase 14: cancel button must be wired to executor.cancel(). Source-grep
// for the cancel-button id + an executor.cancel call so a regression that
// re-disables the button at boot but never invokes cancel surfaces here.
assert.match(src, /['"]cancelButton['"]/,
  'cancel-button DOM id must be referenced');
assert.match(src, /this\.executor\.cancel\s*\(/,
  'cancel button must invoke this.executor.cancel(...)');

// Phase 6: bridge module + warp+resample wiring. applyRegistrationToLesion
// must invoke executor.runWarpMask, decode the 'mni-lesion' stage data, and
// resample onto the Yeo grid via the new resample module.
assert.match(src, /from\s+['"]\.\/modules\/resample\.js['"]/,
  'lnm-app.js must import resample.js');
assert.match(src, /\bresampleAffine\s*\(/,
  'applyRegistrationToLesion must call resampleAffine(...)');
assert.match(src, /\baffineFromHeader\s*\(/,
  'applyRegistrationToLesion must read affines via affineFromHeader');
assert.match(src, /\brunWarpMask\s*\(/,
  'applyRegistrationToLesion must dispatch runWarpMask');
assert.match(src, /['"]mni-lesion['"]/,
  'orchestrator must wire the mni-lesion stage');

// Phase 5: threshold UI wiring. applyNetworkThreshold reads the slider /
// mode / symmetric / min-cluster controls and updates either the
// thresholded mask state or the live overlay; downloadThresholdedNetworkMap
// emits a Blob NIfTI with the thresholded binary mask.
assert.match(src, /from\s+['"]\.\/modules\/threshold\.js['"]/,
  'lnm-app.js must import threshold.js');
assert.match(src, /\bapplyThresholdDetailed\s*\(/,
  'orchestrator must call applyThresholdDetailed(...) to get cluster cleanup stats');
assert.match(src, /thresholdResult\.threshold/,
  'orchestrator must report the actual percentile cutoff used for top-percent thresholding');
assert.match(src, /1\s*-\s*\(topPercent\s*\/\s*100\)/,
  'percentile UI must convert top-percent slider values to quantile cutoffs');
assert.match(src, /NETWORK_TOP_PERCENT_MAX\s*=\s*10/,
  'percentile slider must be scoped to the useful 0..10% range');
assert.match(src, /NETWORK_TOP_PERCENT_STEP\s*=\s*0\.1/,
  'percentile slider must allow fine 0.1% adjustment');
assert.match(src, /['"]thresholdValue['"]|getElementById\(['"]thresholdValue['"]\)|networkThresholdValue/,
  'orchestrator must read the threshold slider value');
assert.match(src, /thresholdMode|networkThresholdMode/,
  'orchestrator must read the threshold mode (absolute / percentile)');
assert.match(src, /thresholdMinCluster[\s\S]*?addEventListener\(['"]input['"]/,
  'min-cluster changes must recompute while the user types');
assert.match(src, /removedByCluster/,
  'threshold summary must report whether cluster cleanup removed voxels');
assert.match(src, /\bscheduleThresholdPreviewOverlay\s*\(/,
  'applyNetworkThreshold must schedule a live threshold preview overlay');
assert.match(src, /replaceOverlayForStage\s*\(\s*['"]threshold-preview['"]/,
  'threshold preview must replace the existing threshold-preview overlay stage');
assert.match(src, /\bprojectThresholdToPatientSpace\s*\(/,
  'threshold preview must project final threshold masks back to patient T1 space when registration is available');
assert.match(src, /\brunInverseWarpMask\s*\(/,
  'patient-space threshold projection must dispatch the inverse-warp worker path');
assert.match(src, /stage:\s*['"]threshold-patient['"]/,
  'patient-space threshold projection must wait for the threshold-patient stage output');
assert.match(src, /\brunInverseWarpStage\s*\(/,
  'patient-space inverse-warp projections must be serialized through runInverseWarpStage');
assert.match(src, /\brenderPatientLayerStack\s*\(/,
  'patient-space threshold projection must render the structural/brainmask/lesion/threshold viewer stack');
assert.match(src, /\bprojectAtlasToPatientSpace\s*\(/,
  'orchestrator must expose a subject-space Yeo atlas QC projection');
assert.match(src, /stage:\s*['"]atlas-patient['"]/,
  'subject-space Yeo atlas QC must use the atlas-patient stage output');
assert.match(src, /labelMap:\s*true/,
  'subject-space Yeo atlas QC must request label-preserving inverse warp');
assert.match(src, /layerToggleT1[\s\S]*?layerToggleThresholdMap[\s\S]*?layerToggleAtlasQc/,
  'lnm-app.js must bind viewer layer toggles for T1, brain mask, lesion mask, threshold map, and Yeo atlas QC');
assert.match(src, /showSubjectAtlasButton/,
  'lnm-app.js must bind the subject-space atlas QC button');

// Phase 2a.2.3: lesion-segmentation wiring. runLesionSegmentation reads
// the lnm-stroke-lesion manifest entry, calls executor.runInference(...),
// and listens for 'segmentation' stageData. downloadLesionMask emits a
// .nii Blob just like downloadBrainMask did for the brain mask.
assert.match(src, /\brunInference\s*\(/,
  'runLesionSegmentation must call executor.runInference(...)');
assert.match(src, /['"]lnm-stroke-lesion['"]/,
  'orchestrator must reference the lnm-stroke-lesion asset id literal');
assert.match(src, /['"]segmentation['"]/,
  'orchestrator must wire the segmentation stage');
assert.match(src, /downloadLesionMaskButton[\s\S]*?disabled\s*=\s*false|disabled\s*=\s*false[\s\S]*?downloadLesionMaskButton|downloadLesionMaskButton[\s\S]*?removeAttribute\(['"]disabled/,
  'lnm-app.js must enable #downloadLesionMaskButton after a successful run');

// Phase 1c.3: runYeoOverlap must populate #networkOverlapTable via the new
// renderer, and exportCsv must serialise via overlap-export and trigger a
// real Blob download (no more 'not implemented' stub).
assert.match(src, /from\s+['"]\.\/modules\/overlap-export\.js['"]/,
  'lnm-app.js must import from ./modules/overlap-export.js');
assert.match(src, /from\s+['"]\.\/modules\/overlap-render\.js['"]/,
  'lnm-app.js must import from ./modules/overlap-render.js');
assert.match(src, /renderOverlapTable\s*\(/,
  'runYeoOverlap must call renderOverlapTable(...)');
assert.match(src, /serializeOverlapCsv\s*\(/,
  'exportCsv must call serializeOverlapCsv(...)');

// exportCsv must trigger a real download — Blob + createObjectURL + a .csv
// filename. Source-grep is brittle but matches the SCT-style guardrails:
// catches the "stub left behind" regression at lint time.
assert.match(src, /\bnew\s+Blob\b/,
  'exportCsv must construct a Blob for download');
assert.match(src, /URL\.createObjectURL\s*\(/,
  'exportCsv must create an object URL for the Blob');
assert.match(src, /\.csv['"]/,
  'exportCsv must reference a .csv filename');

// renderOverlapTable receives the Yeo7 colormap so the bars match the
// canonical Yeo palette across the app + the NiiVue overlay.
assert.match(src, /YEO7_COLORMAP/,
  'overlap rendering must reference YEO7_COLORMAP for bar colors');

// Once an overlap result exists, the CSV download button must become
// interactive. Source-grep the toggle so a regression that leaves the button
// disabled forever surfaces here, not in user reports.
assert.match(src, /downloadOverlapCsv[\s\S]*?disabled\s*=\s*false|disabled\s*=\s*false[\s\S]*?downloadOverlapCsv|downloadOverlapCsv[\s\S]*?removeAttribute\(['"]disabled/,
  'lnm-app.js must enable #downloadOverlapCsv after a successful run');

console.log('LNM app skeleton OK: class + 5 methods + import surface + atlas + render + CSV wiring validated.');
