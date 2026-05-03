#!/usr/bin/env node --no-warnings
// Phase 35: behavior tests for web/js/controllers/ViewerController.js.
//
// The Phase 4 silent-regression bug was: NiiVue 0.68.x's
// `loadVolumes([base, overlay1, ...])` adds the overlay volumes but
// doesn't initialise their cal_min / cal_max / colormap LUT — so binary
// label-mask overlays render invisible. Fix: load the base via
// `loadVolumes([single])`, add overlays via `addVolumeFromUrl()`. This
// test replicates that contract against a fake NiiVue and asserts the
// call shape so a future ViewerController refactor that "simplifies"
// back to the broken pattern fails immediately.

import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// Minimal browser stubs ViewerController doesn't reach into much, but
// loadBaseVolume calls URL.createObjectURL on the input file.
globalThis.URL = {
  createObjectURL: (file) => `blob:fake-${file?.name || 'noname'}`,
  revokeObjectURL: () => {}
};

const { ViewerController } = await import(path.join(ROOT, 'web/js/controllers/ViewerController.js'));

// Fake NiiVue that records every call. Mirrors the surface ViewerController
// touches: addColormap, loadVolumes, addVolumeFromUrl, setOpacity,
// setColormap, setSliceType, updateGLVolume, drawScene, removeVolumeByIndex,
// volumes (mutable array), sliceType* constants.
function makeNv() {
  const calls = {
    addColormap: [],
    loadVolumes: [],
    addVolumeFromUrl: [],
    setOpacity: [],
    setColormap: [],
    setSliceType: [],
    updateGLVolume: 0,
    drawScene: 0,
    removeVolumeByIndex: []
  };
  const nv = {
    volumes: [],
    sliceTypeMultiplanar: 0,
    sliceTypeAxial: 1,
    sliceTypeCoronal: 2,
    sliceTypeSagittal: 3,
    sliceTypeRender: 4,
    addColormap(id, data) { calls.addColormap.push({ id, data }); },
    async loadVolumes(entries) {
      calls.loadVolumes.push(entries);
      // Mirror NiiVue: append a fake volume per entry.
      entries.forEach(e => nv.volumes.push({
        id: `vol-${nv.volumes.length}`,
        url: e.url, name: e.name,
        cal_min: 0, cal_max: 1, colormap: 'gray', interpolation: true,
        img: new Float32Array([0.5])
      }));
    },
    async addVolumeFromUrl(opts) {
      calls.addVolumeFromUrl.push(opts);
      nv.volumes.push({
        id: `vol-${nv.volumes.length}`,
        url: opts.url, name: opts.name,
        cal_min: 0, cal_max: 1, colormap: opts.colormap || 'gray',
        interpolation: true, img: new Float32Array([1])
      });
    },
    setOpacity(idx, op) { calls.setOpacity.push([idx, op]); },
    setColormap(volId, cm) { calls.setColormap.push([volId, cm]); },
    setSliceType(t) { calls.setSliceType.push(t); },
    updateGLVolume() { calls.updateGLVolume++; },
    drawScene() { calls.drawScene++; },
    removeVolumeByIndex(idx) {
      calls.removeVolumeByIndex.push(idx);
      nv.volumes.splice(idx, 1);
    }
  };
  return { nv, calls };
}

function fakeFile(name) {
  return { name, type: 'application/octet-stream' };
}

// ---- Test 1: loadBaseVolume calls loadVolumes with a single entry ----
// THE Phase 4 regression target: never call loadVolumes([base, overlay, ...]).
{
  const { nv, calls } = makeNv();
  const vc = new ViewerController({ nv });
  await vc.loadBaseVolume(fakeFile('t1.nii'), { stage: 'structural' });
  assert.equal(calls.loadVolumes.length, 1, 'loadVolumes called once');
  assert.equal(calls.loadVolumes[0].length, 1,
    'Phase 4 regression: loadVolumes MUST be called with a single-entry array');
  assert.equal(calls.loadVolumes[0][0].name, 't1.nii');
  assert.equal(vc.currentBaseFile.name, 't1.nii');
  // Stage tracking maps base to volume index 0.
  assert.equal(vc.volumeStageIndices.get('structural'), 0);
}

// ---- Test 2: loadOverlay uses addVolumeFromUrl, NOT loadVolumes ----
// The other half of the Phase 4 fix.
{
  const { nv, calls } = makeNv();
  const vc = new ViewerController({ nv });
  await vc.loadBaseVolume(fakeFile('t1.nii'));
  calls.loadVolumes.length = 0;   // reset to verify overlay path doesn't reuse it

  await vc.loadOverlay(fakeFile('lesion.nii'), 'red', 0.5, { stage: 'lesion' });
  assert.equal(calls.loadVolumes.length, 0,
    'overlay load must NOT call loadVolumes (would reset base + lose overlay LUT)');
  assert.equal(calls.addVolumeFromUrl.length, 1,
    'overlay must use addVolumeFromUrl');
  assert.equal(calls.addVolumeFromUrl[0].colormap, 'red');
  assert.equal(calls.addVolumeFromUrl[0].opacity, 0.5);
  // Overlay is now volume index 1; configureSegmentationVolume kicked in.
  assert.equal(nv.volumes.length, 2);
  assert.equal(nv.volumes[1].interpolation, false,
    'binary overlay must have interpolation disabled');
  assert.equal(vc.volumeStageIndices.get('lesion'), 1);
}

// ---- Test 3: loadVolumeStack delegates to base + overlay path ----
{
  const { nv, calls } = makeNv();
  const vc = new ViewerController({ nv });
  const entries = [
    { file: fakeFile('t1.nii'), stage: 'structural' },
    { file: fakeFile('mask1.nii'), colormap: 'red', opacity: 0.5, stage: 'lesion' },
    { file: fakeFile('mask2.nii'), colormap: 'green', opacity: 0.4, stage: 'brainmask' }
  ];
  await vc.loadVolumeStack(entries);
  assert.equal(calls.loadVolumes.length, 1, 'one loadVolumes call (base)');
  assert.equal(calls.loadVolumes[0].length, 1, 'base load is single-entry');
  assert.equal(calls.addVolumeFromUrl.length, 2, 'two overlays via addVolumeFromUrl');
  assert.equal(nv.volumes.length, 3);
}

// ---- Test 4: empty entries -> clearVolumes ----
{
  const { nv, calls } = makeNv();
  const vc = new ViewerController({ nv });
  await vc.loadBaseVolume(fakeFile('t1.nii'));
  assert.equal(nv.volumes.length, 1);
  await vc.loadVolumeStack([]);
  assert.equal(nv.volumes.length, 0,
    'loadVolumeStack([]) must clear volumes');
  assert.equal(vc.currentBaseFile, null);
}

// ---- Test 5: clearOverlay removes only the overlay ----
{
  const { nv } = makeNv();
  const vc = new ViewerController({ nv });
  await vc.loadBaseVolume(fakeFile('t1.nii'));
  await vc.loadOverlay(fakeFile('lesion.nii'), 'red', 0.5, { stage: 'lesion' });
  assert.equal(nv.volumes.length, 2);
  vc.clearOverlay();
  assert.equal(nv.volumes.length, 1, 'base remains after clearOverlay');
  assert.equal(vc.currentOverlayFile, null);
}

// ---- Test 6: setViewType maps strings to NiiVue slice constants ----
{
  const { nv, calls } = makeNv();
  const vc = new ViewerController({ nv });
  vc.setViewType('axial');
  vc.setViewType('coronal');
  vc.setViewType('render');
  assert.deepEqual(calls.setSliceType, [1, 2, 4]);
  // Unknown type: silent no-op (keeps app robust to typos).
  vc.setViewType('madeup');
  assert.equal(calls.setSliceType.length, 3);
}

// ---- Test 7: registerSctColormap installs once + survives second call ----
{
  const { nv, calls } = makeNv();
  const vc = new ViewerController({ nv });
  vc.registerSctColormap({ R: [0], G: [0], B: [0], A: [0] }, 'lnm-yeo7');
  vc.registerSctColormap({ R: [1], G: [1], B: [1], A: [1] }, 'lnm-yeo7');
  assert.equal(calls.addColormap.length, 2,
    'registerSctColormap must always call addColormap (NiiVue overwrites)');
  assert.equal(vc.sctColormapsRegistered.has('lnm-yeo7'), true);
}

// ---- Test 8: getVolumeIndexForStage returns null for stale/missing ----
{
  const { nv } = makeNv();
  const vc = new ViewerController({ nv });
  await vc.loadBaseVolume(fakeFile('t1.nii'), { stage: 'structural' });
  await vc.loadOverlay(fakeFile('lesion.nii'), 'red', 0.5, { stage: 'lesion' });
  assert.equal(vc.getVolumeIndexForStage('structural'), 0);
  assert.equal(vc.getVolumeIndexForStage('lesion'), 1);
  assert.equal(vc.getVolumeIndexForStage('madeup'), null,
    'unknown stage -> null');
  // Stale: overlay was at index 1, but if volumes shrink, the cached
  // index should be invalidated by the bounds check.
  nv.volumes.pop();
  assert.equal(vc.getVolumeIndexForStage('lesion'), null,
    'stage index pointing past volumes.length -> null');
}

// ---- Test 9: getVolumeDataMax handles empty / missing ----
{
  const { nv } = makeNv();
  const vc = new ViewerController({ nv });
  assert.equal(vc.getVolumeDataMax(undefined), 1, 'undefined -> 1 default');
  assert.equal(vc.getVolumeDataMax({}), 1, 'no .img -> 1 default');
  // Use exact Float32-representable values to avoid precision noise.
  assert.equal(vc.getVolumeDataMax({ img: new Float32Array([0.5, 2.0, 0.25, 1.5]) }), 2.0);
  // Non-finite values are skipped.
  // Use exact Float32-representable values; NaN + Infinity must be filtered.
  assert.equal(
    vc.getVolumeDataMax({ img: new Float32Array([NaN, 0.5, Infinity, 0.75]) }),
    0.75,
    'NaN + Infinity must be filtered out'
  );
}

console.log('ViewerController OK: 9 cases (Phase 4 call-shape, overlay path, view + stage + colormap).');
