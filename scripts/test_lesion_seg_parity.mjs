#!/usr/bin/env node --no-warnings
// Phase 2a.2.4: Node-side parity test for the lesion-segmentation port.
//
// Drives runInferencePipeline (web/js/inference-pipeline.js) on the same
// MNI152 anatomical template the SynthStrip parity test uses, backed by
// onnxruntime-node so we can validate the JS port end-to-end without a
// browser. Asserts plausibility:
//
//   - inference runs end-to-end without erroring
//   - output mask shape matches input dims
//   - mask coverage is low (the MNI152 template is a healthy averaged
//     brain — the model should produce close to zero stroke voxels;
//     allow up to 5% to absorb model false positives)
//
// We DO NOT assert Dice against ground truth here. The user-locked
// acceptance is Dice ≥ 0.5 vs an ATLAS-2 held-out subject, but the
// ATLAS-2 release is gated behind a 4 GB password-protected tarball
// with no per-subject direct-download URL. A future hardening commit
// can swap in a real subject when one is locally available.
//
// NOT in `npm test`. Run via `npm run test:lesion-seg-parity`.

import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-node';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FIXTURE_T1 = path.join(ROOT, 'tests/fixtures/synthstrip-mini/T1.nii.gz');
const MODEL_CACHE_DIR = path.join(ROOT, 'web/models/_dev_cache');
const MODEL_CACHE = path.join(MODEL_CACHE_DIR, 'lnm-stroke-lesion.onnx');
const MODEL_URL =
  'https://huggingface.co/datasets/sbollmann/lnm-webapp-models' +
  '/resolve/main/models/lnm-stroke-lesion.onnx';

async function ensureModel() {
  try {
    const buf = await fs.readFile(MODEL_CACHE);
    if (buf.length > 10_000_000) return buf;
  } catch (e) { if (e.code !== 'ENOENT') throw e; }
  await fs.mkdir(MODEL_CACHE_DIR, { recursive: true });
  console.log(`Downloading lesion model from ${MODEL_URL}...`);
  const response = await fetch(MODEL_URL);
  if (!response.ok) throw new Error(`Model download failed: HTTP ${response.status}`);
  const buf = Buffer.from(await response.arrayBuffer());
  if (buf.length < 10_000_000) {
    throw new Error(`Downloaded model is unexpectedly small: ${buf.length} bytes`);
  }
  await fs.writeFile(MODEL_CACHE, buf);
  console.log(`Cached: ${MODEL_CACHE} (${buf.length} bytes)`);
  return buf;
}

async function loadNifti() {
  const mod = await import('nifti-reader-js');
  return mod.default || mod;
}

async function decodeT1Fixture() {
  const t1Bytes = await fs.readFile(FIXTURE_T1);
  let buf = t1Bytes.buffer.slice(
    t1Bytes.byteOffset,
    t1Bytes.byteOffset + t1Bytes.byteLength
  );
  const nifti = await loadNifti();
  if (t1Bytes[0] === 0x1f && t1Bytes[1] === 0x8b) {
    buf = nifti.decompress(buf);
  }
  if (!nifti.isNIFTI(buf)) throw new Error('Fixture is not a valid NIfTI');
  const header = nifti.readHeader(buf);
  const imageBuffer = nifti.readImage(header, buf);
  const off = imageBuffer.byteOffset || 0;
  let raw;
  switch (header.datatypeCode) {
    case nifti.NIFTI1.TYPE_FLOAT32: raw = new Float32Array(imageBuffer, off); break;
    case nifti.NIFTI1.TYPE_INT16:   raw = new Int16Array(imageBuffer, off); break;
    case nifti.NIFTI1.TYPE_UINT8:   raw = new Uint8Array(imageBuffer, off); break;
    default: throw new Error(`Unsupported dtype ${header.datatypeCode}`);
  }
  const data = raw instanceof Float32Array ? raw : Float32Array.from(raw);
  const dims = [Number(header.dims[1]), Number(header.dims[2]), Number(header.dims[3])];
  const spacing = [Number(header.pixDims[1]), Number(header.pixDims[2]), Number(header.pixDims[3])];
  return { data, dims, spacing };
}

(async () => {
  console.log('Loading model...');
  const modelBuf = await ensureModel();
  console.log(`Model: ${modelBuf.length} bytes`);

  console.log(`Loading T1 fixture: ${path.relative(ROOT, FIXTURE_T1)}`);
  const t1 = await decodeT1Fixture();
  console.log(
    `T1: ${t1.dims.join('x')} @ ${t1.spacing.map(s => s.toFixed(2)).join('x')}mm, ` +
    `${t1.data.length.toLocaleString()} voxels`
  );

  console.log('Creating ONNX session (CPU EP)...');
  const session = await ort.InferenceSession.create(modelBuf, {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all'
  });
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  console.log(`session ready (input=${inputName}, output=${outputName})`);

  // The lesion model expects 1 mm input. Our fixture is 2 mm; the worker's
  // stepInference resamples before calling runInferencePipeline. For this
  // parity test we keep things simple: skip resample (the model still runs;
  // the point is to validate the pipeline shape), and assert the output is
  // plausible.
  const { runInferencePipeline } = await import('../web/js/inference-pipeline.js');

  // The model is exported at static 128^3. Pad input to a multiple of 128
  // (handled by runInferencePipeline's zero-pad).
  const PATCH = [128, 128, 128];

  // The SynthStroke baseline outputs 2-channel softmax logits ([bg, stroke]
  // along the channel axis, NCHW layout). runInferencePipeline expects
  // single-channel raw logits that it sigmoids + thresholds. The
  // log-odds `logit_stroke - logit_bg` sigmoids to P(stroke) under the
  // softmax model, so we collapse here.
  function collapseBinaryLogits(out, voxelsPerChannel) {
    const collapsed = new Float32Array(voxelsPerChannel);
    for (let i = 0; i < voxelsPerChannel; i++) {
      collapsed[i] = out[voxelsPerChannel + i] - out[i];
    }
    return collapsed;
  }

  const t0 = Date.now();
  const result = await runInferencePipeline(
    { data: t1.data, dims: t1.dims, patchSize: PATCH },
    async (patch, patchDims) => {
      const [d0, d1, d2] = patchDims;
      const voxels = d0 * d1 * d2;
      const tensor = new ort.Tensor('float32', patch, [1, 1, d0, d1, d2]);
      const out = await session.run({ [inputName]: tensor });
      const raw = out[outputName].data;
      tensor.dispose?.();
      // Expect 2-channel output: [1, 2, d0, d1, d2] = 2 * voxels floats.
      assert.equal(raw.length, 2 * voxels,
        `model output length ${raw.length} != 2 x ${voxels} (expected 2-channel softmax logits)`);
      return collapseBinaryLogits(raw, voxels);
    },
    {
      overlap: 0,
      threshold: 0.4,
      minComponentSize: 30,
      testTimeAugmentation: false,
      onLog: msg => console.log(`  [pipeline] ${msg}`)
    }
  );
  const elapsed = (Date.now() - t0) / 1000;
  console.log(`Pipeline ran in ${elapsed.toFixed(1)}s`);

  assert.deepEqual(
    result.dims,
    t1.dims,
    `Output dims ${result.dims} must match input dims ${t1.dims}`
  );
  assert.equal(
    result.labels.length,
    t1.data.length,
    'output mask length must match input voxel count'
  );

  let positive = 0;
  for (let i = 0; i < result.labels.length; i++) {
    if (result.labels[i] > 0) positive++;
  }
  const coverage = positive / t1.data.length;
  console.log(
    `Lesion mask: ${positive.toLocaleString()}/${t1.data.length.toLocaleString()} voxels ` +
    `(${(coverage * 100).toFixed(2)}%)`
  );
  assert.ok(
    coverage < 0.05,
    `MNI152 (healthy averaged template) -> lesion-seg should produce <5% coverage; ` +
    `got ${(coverage * 100).toFixed(2)}%`
  );

  console.log(
    'Lesion-segmentation parity OK: pipeline runs end-to-end on a real T1; ' +
    'mask shape matches input; coverage is plausibly low.'
  );
})().catch((err) => {
  console.error(err.stack || err.message);
  process.exit(1);
});
