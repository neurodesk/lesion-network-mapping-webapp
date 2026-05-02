// Phase 16: in-browser affine pre-registration helper. The SynthMorph
// deformable head requires its input at exactly 160x160x192 1mm AND
// roughly MNI-aligned. Real clinical T1s come in arbitrary dims, voxel
// sizes, and orientations.
//
// This module provides a minimal "centroid match" prealigner: given a
// source T1 and a brain mask, compute the source brain centroid in
// world-mm coordinates, then build a destination affine for the
// 160x160x192 1mm MNI grid that places that centroid at MNI voxel
// (80, 80, 96). The orchestrator pairs this with `resampleAffine(...)`
// from resample.js to produce the prealigned T1 and brainmask.
//
// Limitations (documented for future improvement):
//   - No rotation alignment. Assumes the source affine encodes the
//     scanner-to-anatomy transform correctly (i.e. the T1 is at least
//     ACPC-aligned). Pure axial/coronal/sagittal acquisitions usually
//     are; rotated clinical scans may need a follow-up rigid step.
//   - No scale correction. Adult brain dimensions are similar enough
//     (~17x14x16 cm) that 1mm isotropic resampling at the centroid
//     suffices for the deformable refinement stage to take it the
//     rest of the way.
//   - Pathology may bias the centroid (large lesions shift centre of
//     mass toward the unaffected hemisphere). For severe stroke a
//     follow-up affine pass (Phase 16 v2) is the right fix.

// Compute the voxel-space centroid of a binary mask (i.e. average over
// the indices of all non-zero voxels). Throws on empty mask or dim
// mismatch.
export function centroidOfMask(mask, dims) {
  if (!Array.isArray(dims) || dims.length !== 3) {
    throw new Error('centroidOfMask: dims must be [X, Y, Z]');
  }
  const [X, Y, Z] = dims;
  const expected = X * Y * Z;
  if (mask.length !== expected) {
    throw new Error(`centroidOfMask: data length ${mask.length} != ${X}x${Y}x${Z}=${expected}`);
  }
  let sx = 0, sy = 0, sz = 0, n = 0;
  for (let z = 0; z < Z; z++) {
    for (let y = 0; y < Y; y++) {
      for (let x = 0; x < X; x++) {
        if (mask[x + y * X + z * X * Y]) {
          sx += x; sy += y; sz += z; n++;
        }
      }
    }
  }
  if (n === 0) throw new Error('centroidOfMask: mask is empty');
  return [sx / n, sy / n, sz / n];
}

// Apply a 4x4 affine to a [x, y, z] voxel coord, returning the world-mm
// coord (drops the homogeneous component).
export function applyAffineToVoxel(M, voxel) {
  const [x, y, z] = voxel;
  return [
    M[0][0] * x + M[0][1] * y + M[0][2] * z + M[0][3],
    M[1][0] * x + M[1][1] * y + M[1][2] * z + M[1][3],
    M[2][0] * x + M[2][1] * y + M[2][2] * z + M[2][3]
  ];
}

// Build the destination affine for the MNI160 1mm grid, parameterised
// so that the source brain centroid (in source world coords) lands at
// MNI voxel `mniCenterVox` (default [80, 80, 96]). Uses the canonical
// FSL MNI152 orientation: (-1, +1, +1) per voxel mm so x is flipped.
//
// Math: for a destination voxel v, world W = D @ [v;1]. We want
// W(mniCenterVox) = srcCentroidWorld. With D = [[-1,0,0,tx],[0,1,0,ty],
// [0,0,1,tz],[0,0,0,1]]:
//   tx + (-1) * cv[0] = c[0]  -> tx = c[0] + cv[0]
//   ty +   1  * cv[1] = c[1]  -> ty = c[1] - cv[1]
//   tz +   1  * cv[2] = c[2]  -> tz = c[2] - cv[2]
export function computePrealignAffine(srcCentroidWorld, options = {}) {
  if (!Array.isArray(srcCentroidWorld) || srcCentroidWorld.length !== 3) {
    throw new Error('computePrealignAffine: srcCentroidWorld must be [x, y, z] mm');
  }
  const { mniCenterVox = [80, 80, 96] } = options;
  const [cx, cy, cz] = srcCentroidWorld;
  const [vx, vy, vz] = mniCenterVox;
  return [
    [-1, 0, 0, cx + vx],
    [0, 1, 0, cy - vy],
    [0, 0, 1, cz - vz],
    [0, 0, 0, 1]
  ];
}
