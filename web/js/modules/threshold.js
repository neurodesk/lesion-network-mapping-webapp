// Phase 5: threshold + cluster cleanup for the Yeo7 group-FC weighted-sum
// network map (and any future scalar volumes the orchestrator wants to
// threshold). Pure JS, reuses connectedComponents3D + removeSmallComponents
// from web/js/modules/volume-utils.js.
//
// Modes:
//   - absolute: voxel passes if (value > threshold) when one-sided, or
//     (|value| > threshold) when symmetric.
//   - percentile: threshold = quantileAbsValue(data, value), where value
//     is in [0, 1]. Use 0.95 for "top 5 % of |voxel intensity|".
//
// Output is always a Uint8Array of the same length as the input volume.

import { connectedComponents3D, removeSmallComponents } from './volume-utils.js';

const MODES = new Set(['absolute', 'percentile']);

export function applyThreshold(data, dims, options = {}) {
  if (!Array.isArray(dims) || dims.length !== 3) {
    throw new Error('applyThreshold: dims must be [X, Y, Z]');
  }
  const expected = dims[0] * dims[1] * dims[2];
  if (data.length !== expected) {
    throw new Error(`applyThreshold: data length ${data.length} != ${expected}`);
  }
  const {
    mode = 'absolute',
    value = 0,
    symmetric = false,
    minClusterVoxels = 0
  } = options;
  if (!MODES.has(mode)) {
    throw new Error(`applyThreshold: unknown mode '${mode}'; expected one of ${[...MODES].join(', ')}`);
  }

  const threshold = (mode === 'percentile')
    ? quantileAbsValue(data, value)
    : value;

  const mask = new Uint8Array(expected);
  if (symmetric) {
    for (let i = 0; i < expected; i++) {
      mask[i] = Math.abs(data[i]) > threshold ? 1 : 0;
    }
  } else {
    for (let i = 0; i < expected; i++) {
      mask[i] = data[i] > threshold ? 1 : 0;
    }
  }

  if (minClusterVoxels && minClusterVoxels > 1) {
    return removeSmallComponents(mask, dims, minClusterVoxels);
  }
  return mask;
}

// |value|-quantile: returns the threshold T such that fraction `q` of
// |data| values are ≤ T. q = 0.95 -> the 95th percentile of |voxels|.
//
// Linear interpolation between sorted indices (matches numpy default
// quantile method, "linear"). Builds a sorted Float32Array of |x|; for
// large volumes this is the dominant cost.
export function quantileAbsValue(data, q) {
  if (data.length === 0) return 0;
  if (q <= 0) return 0;
  const abs = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) abs[i] = Math.abs(data[i]);
  abs.sort();
  if (q >= 1) return abs[abs.length - 1];
  const idx = q * (abs.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const w = idx - lo;
  return abs[lo] * (1 - w) + abs[hi] * w;
}
