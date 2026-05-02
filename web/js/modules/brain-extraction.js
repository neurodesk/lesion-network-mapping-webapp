// SynthStrip brain extraction. Ported 1:1 from
// neurodesk/vesselboost-webapp@/web/js/inference-worker.js (stepSynthStrip,
// lines 1562..1880). Behavior preserved, structure normalised: pure helpers
// are isolated and exported for unit testing; runSynthStrip takes its
// dependencies (model bytes, ort namespace, progress callbacks) as
// arguments rather than reaching into a worker-global state object.
//
// Pipeline (RAS-oriented input -> binary brain mask in RAS):
//   1. RAS -> LIA reorientation (model trained on LIA brains)
//   2. Resample to 1mm isotropic ("fast" -> adaptive 1..2mm to keep min
//      resampled axis >= 48 voxels on high-res inputs)
//   3. Crop to bounding box of nonzero voxels
//   4. Center-pad to per-axis multiple of 64 within [192, 320]
//   5. Per-volume P99 normalisation (subtract min, divide by 99th pct of
//      nonzero voxels, clamp to [0,1])
//   6. Fortran-order -> C-order transpose into ONNX tensor
//   7. ONNX inference (WASM execution provider only — WebGPU lacks 3D MaxPool)
//   8. C-order -> Fortran-order transpose back
//   9. SDT < 1 -> binary mask
//   10. Reverse center-pad, reverse crop
//   11. Largest CC + interior fill (FreeSurfer SynthStrip default)
//   12. Resample mask back to LIA original dims (nearest-neighbour)
//   13. LIA -> RAS reorientation
//   14. (Optional) 1-voxel 6-conn dilation — vesselboost behaviour, off by
//       default for LNM where we want a tighter mask.

import {
  resampleVolume,
  resampleLabelsNearest,
  keepLargestComponentAndFill
} from './volume-utils.js';

// ---------------- Pure helpers (exported for unit testing) ----------------

export function computeFreeSurferTargetDims(cropDims) {
  return cropDims.map(s => Math.min(320, Math.max(192, Math.ceil(s / 64) * 64)));
}

export function centerPadConform(croppedData, cropDims, targetDims) {
  const [cnx, cny, cnz] = cropDims;
  const [tnx, tny, tnz] = targetDims;
  const offsets = [
    Math.floor((tnx - cnx) / 2),
    Math.floor((tny - cny) / 2),
    Math.floor((tnz - cnz) / 2)
  ];
  const data = new Float32Array(tnx * tny * tnz);
  for (let z = 0; z < cnz; z++) {
    for (let y = 0; y < cny; y++) {
      for (let x = 0; x < cnx; x++) {
        const dx = x + offsets[0];
        const dy = y + offsets[1];
        const dz = z + offsets[2];
        data[dx + dy * tnx + dz * tnx * tny] =
          croppedData[x + y * cnx + z * cnx * cny];
      }
    }
  }
  return { data, offsets };
}

export function uncenterUnpadMask(paddedMask, targetDims, cropDims, offsets) {
  const [cnx, cny, cnz] = cropDims;
  const [tnx, tny] = targetDims;
  const result = new Uint8Array(cnx * cny * cnz);
  for (let z = 0; z < cnz; z++) {
    for (let y = 0; y < cny; y++) {
      for (let x = 0; x < cnx; x++) {
        const sx = x + offsets[0];
        const sy = y + offsets[1];
        const sz = z + offsets[2];
        result[x + y * cnx + z * cnx * cny] =
          paddedMask[sx + sy * tnx + sz * tnx * tny];
      }
    }
  }
  return result;
}

// FreeSurfer P99 normalisation: subtract min, divide by p99 of nonzero,
// clamp to [0,1]. Returns a NEW array — input is not mutated. p99 is
// computed by sorting the nonzero values (matches vesselboost's exact
// floor-index convention).
export function p99Normalize(data) {
  const n = data.length;
  let vMin = Infinity;
  for (let i = 0; i < n; i++) {
    if (data[i] < vMin) vMin = data[i];
  }
  if (!isFinite(vMin)) vMin = 0;

  const shifted = new Float32Array(n);
  for (let i = 0; i < n; i++) shifted[i] = data[i] - vMin;

  let nonZeroCount = 0;
  for (let i = 0; i < n; i++) {
    if (shifted[i] > 0) nonZeroCount++;
  }
  let p99 = 0;
  if (nonZeroCount > 0) {
    const nonZero = new Float32Array(nonZeroCount);
    let idx = 0;
    for (let i = 0; i < n; i++) {
      if (shifted[i] > 0) nonZero[idx++] = shifted[i];
    }
    nonZero.sort();
    p99 = nonZero[Math.floor(nonZeroCount * 0.99)];
  }
  const denom = p99 || 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.min(1, Math.max(0, shifted[i] / denom));
  }
  return { data: out, vMin, p99 };
}

// Fortran-order (x + y*nx + z*nx*ny) -> C-order (x*ny*nz + y*nz + z).
// SynthStrip's ONNX wrapper expects C-order with dim layout [nx, ny, nz].
export function fortranToCOrder(data, dims) {
  const [nx, ny, nz] = dims;
  const out = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        out[x * ny * nz + y * nz + z] = data[x + y * nx + z * nx * ny];
      }
    }
  }
  return out;
}

export function cOrderToFortran(data, dims) {
  const [nx, ny, nz] = dims;
  const out = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        out[x + y * nx + z * nx * ny] = data[x * ny * nz + y * nz + z];
      }
    }
  }
  return out;
}

// 6-connectivity binary dilation by `radius` voxels. Iterative (one
// 6-neighbour expansion per radius step). Used to expand the SynthStrip
// brain mask outward — vesselboost defaults to radius=1 to catch boundary
// vessels. LNM default is radius=0 (off).
export function dilate3D(mask, dims, radius = 1) {
  if (!radius || radius < 1) {
    const copy = new Uint8Array(mask.length);
    copy.set(mask);
    return copy;
  }
  const [nx, ny, nz] = dims;
  let cur = new Uint8Array(mask.length);
  cur.set(mask);
  let next = new Uint8Array(mask.length);
  for (let r = 0; r < radius; r++) {
    next.fill(0);
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const idx = x + y * nx + z * nx * ny;
          if (cur[idx]) {
            next[idx] = 1;
            if (x > 0) next[idx - 1] = 1;
            if (x < nx - 1) next[idx + 1] = 1;
            if (y > 0) next[idx - nx] = 1;
            if (y < ny - 1) next[idx + nx] = 1;
            if (z > 0) next[idx - nx * ny] = 1;
            if (z < nz - 1) next[idx + nx * ny] = 1;
          }
        }
      }
    }
    const tmp = cur;
    cur = next;
    next = tmp;
  }
  return cur;
}

// ---------------- Orchestration ----------------

function rasToLia(rasData, rasDims) {
  // perm = [0, 2, 1], flip = [true, true, false]
  // (matches vesselboost stepSynthStrip lines 1591..1610.)
  const liaDims = [rasDims[0], rasDims[2], rasDims[1]];
  const [dx, dy, dz] = liaDims;
  const out = new Float32Array(rasData.length);
  for (let oz = 0; oz < dz; oz++) {
    for (let oy = 0; oy < dy; oy++) {
      for (let ox = 0; ox < dx; ox++) {
        // src axis 0 = flip, axis 1 (perm 2 -> axis 1) = flip, axis 2 (perm 1 -> axis 2) = no flip
        const sx = liaDims[0] - 1 - ox;          // flip[0]=true, perm[0]=0
        const sz = liaDims[1] - 1 - oy;          // flip[1]=true, perm[1]=2 -> RAS axis 2
        const sy = oz;                           // flip[2]=false, perm[2]=1 -> RAS axis 1
        const srcIdx = sx + sy * rasDims[0] + sz * rasDims[0] * rasDims[1];
        out[ox + oy * dx + oz * dx * dy] = rasData[srcIdx];
      }
    }
  }
  return { data: out, dims: liaDims };
}

function liaMaskToRas(liaMask, liaDims, rasDims) {
  // Inverse of rasToLia for a binary mask.
  const [dx, dy, dz] = liaDims;
  const out = new Uint8Array(rasDims[0] * rasDims[1] * rasDims[2]);
  for (let oz = 0; oz < dz; oz++) {
    for (let oy = 0; oy < dy; oy++) {
      for (let ox = 0; ox < dx; ox++) {
        if (!liaMask[ox + oy * dx + oz * dx * dy]) continue;
        const dxRas = liaDims[0] - 1 - ox;       // flip[0]=true, perm[0]=0
        const dzRas = liaDims[1] - 1 - oy;       // flip[1]=true, perm[1]=2
        const dyRas = oz;                        // flip[2]=false, perm[2]=1
        out[dxRas + dyRas * rasDims[0] + dzRas * rasDims[0] * rasDims[1]] = 1;
      }
    }
  }
  return out;
}

function computeBoundingBox(data, dims) {
  const [nx, ny, nz] = dims;
  let minX = nx, minY = ny, minZ = nz;
  let maxX = -1, maxY = -1, maxZ = -1;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        if (data[x + y * nx + z * nx * ny] > 0) {
          if (x < minX) minX = x; if (x > maxX) maxX = x;
          if (y < minY) minY = y; if (y > maxY) maxY = y;
          if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        }
      }
    }
  }
  if (maxX < minX) {
    return { min: [0, 0, 0], max: dims, isEmpty: true };
  }
  return { min: [minX, minY, minZ], max: [maxX + 1, maxY + 1, maxZ + 1], isEmpty: false };
}

function cropToBBox(data, dims, bbox) {
  const [nx, ny] = dims;
  const [oxMin, oyMin, ozMin] = bbox.min;
  const [oxMax, oyMax, ozMax] = bbox.max;
  const cdims = [oxMax - oxMin, oyMax - oyMin, ozMax - ozMin];
  const cropped = new Float32Array(cdims[0] * cdims[1] * cdims[2]);
  for (let z = 0; z < cdims[2]; z++) {
    for (let y = 0; y < cdims[1]; y++) {
      for (let x = 0; x < cdims[0]; x++) {
        const srcIdx = (oxMin + x) + (oyMin + y) * nx + (ozMin + z) * nx * ny;
        cropped[x + y * cdims[0] + z * cdims[0] * cdims[1]] = data[srcIdx];
      }
    }
  }
  return { data: cropped, dims: cdims };
}

function placeBackInResampledFrame(croppedMask, cropDims, fullDims, bbox) {
  const out = new Uint8Array(fullDims[0] * fullDims[1] * fullDims[2]);
  const [oxMin, oyMin, ozMin] = bbox.min;
  const [fnx, fny] = fullDims;
  const [cnx, cny, cnz] = cropDims;
  for (let z = 0; z < cnz; z++) {
    for (let y = 0; y < cny; y++) {
      for (let x = 0; x < cnx; x++) {
        const dst = (oxMin + x) + (oyMin + y) * fnx + (ozMin + z) * fnx * fny;
        out[dst] = croppedMask[x + y * cnx + z * cnx * cny];
      }
    }
  }
  return out;
}

// Public entry point. Caller hands in already-decoded, RAS-oriented voxel
// data + the SynthStrip ONNX model bytes + an `ort` namespace (the bundled
// onnxruntime-web). Returns { mask: Uint8Array (RAS, same dims as input),
// voxelCount, coveragePct, vMin, p99 }.
export async function runSynthStrip({
  rasData,
  rasDims,
  rasSpacing,
  modelArrayBuffer,
  ort,
  fast = false,
  dilate = false,
  onProgress = () => {},
  onLog = () => {}
}) {
  if (!rasData || !rasDims || !rasSpacing || !modelArrayBuffer || !ort) {
    throw new Error('runSynthStrip: missing required argument');
  }

  const targetSpacing = (() => {
    if (!fast) return [1.0, 1.0, 1.0];
    const MIN_RESAMPLED_DIM = 48;
    const physicalExtents = rasDims.map((d, i) => d * rasSpacing[i]);
    const minExtent = Math.min(...physicalExtents);
    const maxSpacing = Math.min(2.0, minExtent / MIN_RESAMPLED_DIM);
    const sp = Math.max(1.0, maxSpacing);
    return [sp, sp, sp];
  })();
  const modeLabel = fast ? 'SynthStrip Fast' : 'SynthStrip';
  onProgress(0.02, `${modeLabel}: reorienting RAS->LIA...`);

  // 1. RAS -> LIA
  const lia = rasToLia(rasData, rasDims);
  const liaSpacing = [rasSpacing[0], rasSpacing[2], rasSpacing[1]];
  onLog(`Reoriented RAS->LIA: ${rasDims.join('x')} -> ${lia.dims.join('x')}`);

  // 2. Resample to target spacing
  const needsResample =
    liaSpacing[0] !== targetSpacing[0] ||
    liaSpacing[1] !== targetSpacing[1] ||
    liaSpacing[2] !== targetSpacing[2];
  let workData, workDims;
  if (needsResample) {
    onProgress(0.04, `${modeLabel}: resampling to ${targetSpacing[0].toFixed(2)}mm...`);
    const r = resampleVolume(lia.data, lia.dims, liaSpacing, targetSpacing);
    workData = r.data; workDims = r.dims;
    onLog(`Resampled: ${lia.dims.join('x')} -> ${workDims.join('x')} (${targetSpacing[0]}mm)`);
  } else {
    workData = lia.data; workDims = [...lia.dims];
  }
  const resampledDims = [...workDims];

  // 3. Crop to bbox of nonzero
  onProgress(0.05, `${modeLabel}: cropping to brain bbox...`);
  const bbox = computeBoundingBox(workData, workDims);
  if (bbox.isEmpty) {
    onLog('Empty volume; returning empty mask.');
    return {
      mask: new Uint8Array(rasData.length),
      voxelCount: 0,
      coveragePct: 0,
      vMin: 0,
      p99: 0
    };
  }
  const cropped = cropToBBox(workData, workDims, bbox);
  onLog(`Cropped to bbox: ${workDims.join('x')} -> ${cropped.dims.join('x')}`);

  // 4. Center-pad to FreeSurfer target shape
  const targetDims = computeFreeSurferTargetDims(cropped.dims);
  const { data: conformedData, offsets: centerOffsets } =
    centerPadConform(cropped.data, cropped.dims, targetDims);
  onLog(`Conformed: ${cropped.dims.join('x')} -> ${targetDims.join('x')} (offsets: ${centerOffsets.join(',')})`);

  // 5. Normalise to [0, 1]
  onProgress(0.07, `${modeLabel}: normalising...`);
  const { data: normalized, vMin, p99 } = p99Normalize(conformedData);
  onLog(`Normalized: vMin=${vMin.toFixed(2)}, p99=${p99.toFixed(2)}`);

  // 6. Create ONNX session (WASM only — WebGPU lacks 3D MaxPool)
  onProgress(0.10, `${modeLabel}: loading model...`);
  const session = await ort.InferenceSession.create(modelArrayBuffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
  });
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  onLog(`ONNX session ready. Input=${inputName} Output=${outputName}`);

  // 7. F-order -> C-order, infer, C-order -> F-order
  onProgress(0.30, `${modeLabel}: inference on ${targetDims.join('x')}...`);
  const cInput = fortranToCOrder(normalized, targetDims);
  const inputTensor = new ort.Tensor('float32', cInput, [1, 1, ...targetDims]);
  let sdtData;
  try {
    const results = await session.run({ [inputName]: inputTensor });
    sdtData = cOrderToFortran(results[outputName].data, targetDims);
  } finally {
    if (typeof inputTensor.dispose === 'function') inputTensor.dispose();
    if (typeof session.release === 'function') session.release();
  }

  // 8. SDT < 1 -> binary mask
  onProgress(0.85, `${modeLabel}: thresholding SDT...`);
  const SDT_BORDER = 1;
  const totalConformed = targetDims[0] * targetDims[1] * targetDims[2];
  const conformedMask = new Uint8Array(totalConformed);
  let sdtMin = Infinity, sdtMax = -Infinity;
  for (let i = 0; i < totalConformed; i++) {
    const v = sdtData[i];
    if (v < sdtMin) sdtMin = v;
    if (v > sdtMax) sdtMax = v;
    if (v < SDT_BORDER) conformedMask[i] = 1;
  }
  onLog(`SDT range [${sdtMin.toFixed(2)}, ${sdtMax.toFixed(2)}]; threshold < ${SDT_BORDER}`);

  // 9. Reverse center+pad
  const croppedMask =
    uncenterUnpadMask(conformedMask, targetDims, cropped.dims, centerOffsets);

  // 10. Reverse crop
  let resampledMask = placeBackInResampledFrame(
    croppedMask, cropped.dims, resampledDims, bbox
  );

  // 11. Largest CC + interior fill (FreeSurfer SynthStrip default)
  onProgress(0.90, `${modeLabel}: cleaning mask...`);
  resampledMask = keepLargestComponentAndFill(resampledMask, resampledDims);

  // 12. Resample mask back to LIA original dims (nearest-neighbour)
  let liaMask;
  if (needsResample) {
    liaMask = resampleLabelsNearest(resampledMask, resampledDims, lia.dims);
  } else {
    liaMask = resampledMask;
  }

  // 13. LIA -> RAS
  let finalMask = liaMaskToRas(liaMask, lia.dims, rasDims);

  // 14. Optional dilation (off for LNM by default)
  if (dilate) {
    onProgress(0.95, `${modeLabel}: dilating mask by 1 voxel...`);
    finalMask = dilate3D(finalMask, rasDims, 1);
  }

  let voxelCount = 0;
  for (let i = 0; i < finalMask.length; i++) {
    if (finalMask[i]) voxelCount++;
  }
  const coveragePct = (100 * voxelCount) / Math.max(1, finalMask.length);
  onLog(`${modeLabel} complete: ${voxelCount} brain voxels (${coveragePct.toFixed(1)}% coverage).`);
  onProgress(1.0, `${modeLabel} complete`);

  return { mask: finalMask, voxelCount, coveragePct, vMin, p99 };
}
