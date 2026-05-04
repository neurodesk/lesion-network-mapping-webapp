// Yeo7 group-FC weighted sum — the algorithmic core of the Phase 4 lesion
// network map.
//
// Inputs (all on the same MNI grid):
//   - networkWeights: Float32Array(7), per-network share of the lesion
//     (output of summaryToNetworkWeights below). Order matches
//     `networkOrder` from the FC pack's index.json (Visual, Somatomotor,
//     DorsalAttention, VentralAttention, Limbic, Frontoparietal, Default).
//   - tMaps: array of 7 Float32Arrays, each length X*Y*Z in NIfTI voxel
//     order (x + y*X + z*X*Y), one per network. Per-network group
//     t-statistic against zero of the seed-to-voxel Fisher-z connectivity.
//     Decoded from the packed .bin via decodeFcPack.
//   - dims: [X, Y, Z] — the atlas grid (Yeo7 = 99x117x95 2mm).
//
// Output:
//   Float32Array length X*Y*Z, the per-voxel weighted-sum t-stat. Lesions
//   that lie wholly in one network reproduce that network's t-map; lesions
//   spanning multiple networks linearly combine. The result is *not*
//   thresholded here — Phase 5's threshold UI lives separately.

export function fcWeightedSum(networkWeights, tMaps, dims) {
  if (!Array.isArray(dims) || dims.length !== 3) {
    throw new Error('fcWeightedSum: dims must be [X, Y, Z]');
  }
  const expected = dims[0] * dims[1] * dims[2];
  if (networkWeights.length !== 7) {
    throw new Error(`fcWeightedSum: weights length must be 7, got ${networkWeights.length}`);
  }
  if (tMaps.length !== 7) {
    throw new Error(`fcWeightedSum: tMaps array must have 7 maps, got ${tMaps.length}`);
  }
  for (let k = 0; k < 7; k++) {
    if (tMaps[k].length !== expected) {
      throw new Error(
        `fcWeightedSum: tMap[${k}] length ${tMaps[k].length} != ${expected} (dim mismatch)`
      );
    }
  }

  const out = new Float32Array(expected);
  for (let k = 0; k < 7; k++) {
    const w = networkWeights[k];
    if (w === 0) continue;
    const t = tMaps[k];
    for (let v = 0; v < expected; v++) out[v] += w * t[v];
  }
  return out;
}

export function rowMajorToNiftiOrder(data, dims) {
  if (!Array.isArray(dims) || dims.length !== 3) {
    throw new Error('rowMajorToNiftiOrder: dims must be [X, Y, Z]');
  }
  const [X, Y, Z] = dims;
  const expected = X * Y * Z;
  if (data.length !== expected) {
    throw new Error(`rowMajorToNiftiOrder: data length ${data.length} != ${expected}`);
  }

  const out = new Float32Array(expected);
  for (let x = 0; x < X; x++) {
    const srcX = x * Y * Z;
    for (let y = 0; y < Y; y++) {
      const srcXY = srcX + y * Z;
      for (let z = 0; z < Z; z++) {
        out[x + y * X + z * X * Y] = data[srcXY + z];
      }
    }
  }
  return out;
}

function fcPackVoxelOrder(index) {
  const order = index.voxelOrder || index.storageOrder || 'row-major';
  if (order === 'row-major' || order === 'c-order') return 'row-major';
  if (order === 'nifti' || order === 'f-order' || order === 'fortran') return 'nifti';
  throw new Error(`decodeFcPack: unsupported voxelOrder '${order}'`);
}

// Read the packed .bin contents into 7 typed arrays. The arrayBuffer is the
// result of fetch(...).arrayBuffer(). The index JSON is the companion file
// emitted by scripts/build_yeo7_connectome.py.
//
// Returns:
//   { tMaps: Float32Array[7], byNetwork: { [name]: Float32Array }, voxelsPerMap }
//
// Current Yeo7 packs were written by NumPy's ndarray.tofile(), which emits
// C/row-major bytes. The rest of this app uses NIfTI order, so decode at the
// asset boundary before any thresholding or NIfTI serialization happens.
export function decodeFcPack(arrayBuffer, index) {
  if (!index || !Array.isArray(index.shape) || index.shape.length !== 4) {
    throw new Error('decodeFcPack: index.shape must be [7, X, Y, Z]');
  }
  if (index.dtype !== 'float32') {
    throw new Error(`decodeFcPack: only float32 supported, got ${index.dtype}`);
  }
  const [N, X, Y, Z] = index.shape;
  if (N !== 7) throw new Error(`decodeFcPack: pack must have 7 networks, got ${N}`);
  const voxelsPerMap = X * Y * Z;
  if (voxelsPerMap !== index.voxelsPerMap) {
    throw new Error(
      `decodeFcPack: voxelsPerMap mismatch ${voxelsPerMap} vs ${index.voxelsPerMap}`
    );
  }
  const voxelOrder = fcPackVoxelOrder(index);
  const dims = [X, Y, Z];
  const tMaps = [];
  for (let k = 0; k < 7; k++) {
    const raw = new Float32Array(arrayBuffer, k * voxelsPerMap * 4, voxelsPerMap);
    tMaps.push(voxelOrder === 'row-major' ? rowMajorToNiftiOrder(raw, dims) : raw);
  }
  const byNetwork = {};
  if (index.networkLabels) {
    for (const [labelStr, name] of Object.entries(index.networkLabels)) {
      const k = Number(labelStr) - 1;
      if (k >= 0 && k < 7) byNetwork[name] = tMaps[k];
    }
  }
  return { tMaps, byNetwork, voxelsPerMap };
}

// Convert the output of `summarizeNetworkOverlap` (parcel-overlap.js)
// into a length-7 weight vector aligned to the FC pack's network order.
//
//   summary: { totalLesionVoxels, networks: [{network, voxelsInLesion, fractionOfLesion, ...}, ...] }
//   networkOrder: ['Visual', ..., 'Default'] (must match the FC pack's index)
//
// Networks present in the summary but not in networkOrder (typically
// 'Unassigned') are dropped — there's no FC channel for them, so they
// can't contribute to the weighted sum. Their voxelsInLesion silently
// reduces the total fraction; the caller may want to renormalise the
// returned weights to sum to 1, but we don't auto-do that — the
// fraction-of-lesion semantic is preserved by reading directly from
// summary.networks[].fractionOfLesion.
export function summaryToNetworkWeights(summary, networkOrder) {
  if (!summary || !Array.isArray(summary.networks)) {
    throw new Error('summaryToNetworkWeights: bad summary input');
  }
  if (!Array.isArray(networkOrder) || networkOrder.length !== 7) {
    throw new Error('summaryToNetworkWeights: networkOrder must be a length-7 array');
  }
  const weights = new Float32Array(7);
  const byName = {};
  for (const row of summary.networks) byName[row.network] = row;
  for (let k = 0; k < 7; k++) {
    const row = byName[networkOrder[k]];
    weights[k] = row ? Number(row.fractionOfLesion) || 0 : 0;
  }
  return weights;
}
