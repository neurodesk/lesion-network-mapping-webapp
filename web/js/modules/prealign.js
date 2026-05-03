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

// Phase 26: 3x3 mass-weighted voxel covariance of a binary mask. Used as
// the input to PCA — eigenvectors give the brain's principal axes in
// voxel space, eigenvalues give the variance along each axis.
//
// cov[r][c] = (1/N) * Σ_v (v[r] - centroid[r]) * (v[c] - centroid[c])
// where v ranges over voxels in the mask. N = mask voxel count. We
// divide by N (population) rather than N-1 (sample) because we treat
// the mask as the entire population.
export function covarianceOfMask(mask, dims, centroid) {
  if (!Array.isArray(dims) || dims.length !== 3) {
    throw new Error('covarianceOfMask: dims must be [X, Y, Z]');
  }
  if (!Array.isArray(centroid) || centroid.length !== 3) {
    throw new Error('covarianceOfMask: centroid must be [cx, cy, cz]');
  }
  const [X, Y, Z] = dims;
  const expected = X * Y * Z;
  if (mask.length !== expected) {
    throw new Error(`covarianceOfMask: data length ${mask.length} != ${expected}`);
  }
  const [cx, cy, cz] = centroid;
  let sxx = 0, syy = 0, szz = 0, sxy = 0, sxz = 0, syz = 0, n = 0;
  for (let z = 0; z < Z; z++) {
    const dz = z - cz;
    for (let y = 0; y < Y; y++) {
      const dy = y - cy;
      for (let x = 0; x < X; x++) {
        if (!mask[x + y * X + z * X * Y]) continue;
        const dx = x - cx;
        sxx += dx * dx; syy += dy * dy; szz += dz * dz;
        sxy += dx * dy; sxz += dx * dz; syz += dy * dz;
        n++;
      }
    }
  }
  if (n === 0) throw new Error('covarianceOfMask: mask is empty');
  return [
    [sxx / n, sxy / n, sxz / n],
    [sxy / n, syy / n, syz / n],
    [sxz / n, syz / n, szz / n]
  ];
}

// Phase 26: Jacobi eigendecomposition of a symmetric 3x3 matrix.
// Returns:
//   { eigenvalues: [λ0, λ1, λ2],
//     eigenvectors: [v0, v1, v2]   // each vk is a unit-length [x, y, z] }
//
// Order matches: eigenvectors[i] corresponds to eigenvalues[i]. They are
// NOT sorted; principalAxisAlign sorts by descending magnitude. For 3x3
// the off-diagonal sweep usually converges in <10 iterations; we cap at
// 50 to be safe.
export function jacobiEigen3x3(M) {
  // Working copy.
  let A = [
    [M[0][0], M[0][1], M[0][2]],
    [M[1][0], M[1][1], M[1][2]],
    [M[2][0], M[2][1], M[2][2]]
  ];
  // Eigenvectors accumulated as the rotation matrix V. Columns are the
  // current eigenvector estimates.
  let V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

  for (let iter = 0; iter < 50; iter++) {
    // Find off-diagonal entry of largest magnitude.
    let p = 0, q = 1;
    let maxAbs = Math.abs(A[0][1]);
    if (Math.abs(A[0][2]) > maxAbs) { maxAbs = Math.abs(A[0][2]); p = 0; q = 2; }
    if (Math.abs(A[1][2]) > maxAbs) { maxAbs = Math.abs(A[1][2]); p = 1; q = 2; }
    if (maxAbs < 1e-12) break;

    const apq = A[p][q];
    const app = A[p][p];
    const aqq = A[q][q];
    // Givens rotation angle: choose t such that the (p,q) entry zeroes.
    let t;
    if (Math.abs(apq) < 1e-30) {
      t = 0;
    } else {
      const theta = (aqq - app) / (2 * apq);
      if (Math.abs(theta) > 1e10) {
        t = 1 / (2 * theta);
      } else {
        t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        if (theta === 0) t = 1; // Sign(0) is 0; default to +1.
      }
    }
    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;

    // Apply rotation in the (p, q) plane: A' = R^T A R.
    const newA = [
      [A[0][0], A[0][1], A[0][2]],
      [A[1][0], A[1][1], A[1][2]],
      [A[2][0], A[2][1], A[2][2]]
    ];
    newA[p][p] = app - t * apq;
    newA[q][q] = aqq + t * apq;
    newA[p][q] = newA[q][p] = 0;
    for (let i = 0; i < 3; i++) {
      if (i !== p && i !== q) {
        newA[i][p] = newA[p][i] = c * A[i][p] - s * A[i][q];
        newA[i][q] = newA[q][i] = c * A[i][q] + s * A[i][p];
      }
    }
    A = newA;

    // V' = V R.
    const newV = V.map(row => row.slice());
    for (let i = 0; i < 3; i++) {
      newV[i][p] = c * V[i][p] - s * V[i][q];
      newV[i][q] = c * V[i][q] + s * V[i][p];
    }
    V = newV;
  }

  // Extract: eigenvalues from diagonal; eigenvectors from columns of V.
  const eigenvalues = [A[0][0], A[1][1], A[2][2]];
  const eigenvectors = [
    [V[0][0], V[1][0], V[2][0]],
    [V[0][1], V[1][1], V[2][1]],
    [V[0][2], V[1][2], V[2][2]]
  ];
  return { eigenvalues, eigenvectors };
}

// Phase 26: full principal-axis prealign. Combines centroid match
// (Phase 16 v1) with PCA rotation:
//
//   1. centroidOfMask + applyAffineToVoxel -> source centroid world coord.
//   2. covarianceOfMask + jacobiEigen3x3   -> principal axes in source voxel space.
//   3. Sort eigenvectors by descending eigenvalue. Largest = head's longest extent.
//   4. Build a source-voxel rotation R whose columns are the sorted PCs.
//      R @ e_x = PC1_voxel, R @ e_y = PC2_voxel, R @ e_z = PC3_voxel.
//      Force right-handed (det = +1) by flipping the third column if needed.
//   5. Construct an MNI160 destination affine D such that resampleAffine
//      walking the dst grid samples the source as if rotated by R^-1
//      around the centroid. dst voxel (80, 80, 96) -> source centroid.
//
// Returns:
//   {
//     dstAffine,    // 4x4 destination affine for resampleAffine
//     mniDims,      // [160, 160, 192]
//     mniCenterVox, // [80, 80, 96]
//     eigenvalues,  // sorted descending, by principal axis order
//     R             // 3x3 source-voxel rotation; column k = principal axis k
//   }
//
// Limitations:
//   - PCA alone determines axes but not signs (a brain rotated 180° around
//     an axis has identical covariance). We force the rotation matrix to
//     be right-handed but DO NOT detect superior-inferior or
//     left-right orientation. The deformable SynthMorph stage downstream
//     handles small remaining rotations; full 180° flips are out of
//     scope for this module.
//   - For nearly-isotropic brains (eigenvalues degenerate), the
//     eigenvectors are arbitrary up to a rotation in the eigenspace.
//     We accept whatever Jacobi converges to.
export function principalAxisAlign(mask, dims, srcAffine, options = {}) {
  const { mniDims = [160, 160, 192], mniCenterVox = [80, 80, 96] } = options;

  const centroidVox = centroidOfMask(mask, dims);
  const cov = covarianceOfMask(mask, dims, centroidVox);
  const { eigenvalues, eigenvectors } = jacobiEigen3x3(cov);

  // Sort by descending eigenvalue; carry eigenvectors along.
  const idx = [0, 1, 2].sort((a, b) => eigenvalues[b] - eigenvalues[a]);
  const sortedEigs = idx.map(i => eigenvalues[i]);
  const sortedVecs = idx.map(i => eigenvectors[i]);

  // Build R with sorted PCs as columns. R[:, k] = sortedVecs[k].
  const R = [
    [sortedVecs[0][0], sortedVecs[1][0], sortedVecs[2][0]],
    [sortedVecs[0][1], sortedVecs[1][1], sortedVecs[2][1]],
    [sortedVecs[0][2], sortedVecs[1][2], sortedVecs[2][2]]
  ];
  // Force right-handed: det(R) must be +1, not -1 (mirror).
  // KNOWN LIMITATION (Phase 33 audit): a 3rd-moment sign correction
  // would resolve the 180° ambiguity (heavy half lands on a canonical
  // MNI side regardless of acquisition pose), but a naive flip-then-
  // re-det interacts badly: flipping a column for sign correction can
  // break right-handedness, and the det-fix below undoes it. Proper
  // resolution needs to either (a) accept det(R) = -1 (consistent
  // with MNI's left-handed FSL convention) and audit downstream
  // consumers, or (b) bake an A/P/L/R anatomical prior into the
  // algorithm. Documented as expected-fail in
  // scripts/test_prealign_pca_orientation.cjs.
  const det =
    R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
    R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
    R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
  if (det < 0) {
    // Flip the third column to make a right-handed system.
    R[0][2] = -R[0][2];
    R[1][2] = -R[1][2];
    R[2][2] = -R[2][2];
  }

  // Build the source-voxel-to-source-voxel transform M such that for a
  // destination voxel v (in the MNI160 grid):
  //   src_voxel = R @ (v - mni_center) + src_centroid
  //            = R @ v + (src_centroid - R @ mni_center)
  const Rmc = [
    R[0][0] * mniCenterVox[0] + R[0][1] * mniCenterVox[1] + R[0][2] * mniCenterVox[2],
    R[1][0] * mniCenterVox[0] + R[1][1] * mniCenterVox[1] + R[1][2] * mniCenterVox[2],
    R[2][0] * mniCenterVox[0] + R[2][1] * mniCenterVox[1] + R[2][2] * mniCenterVox[2]
  ];
  const t = [
    centroidVox[0] - Rmc[0],
    centroidVox[1] - Rmc[1],
    centroidVox[2] - Rmc[2]
  ];
  const M = [
    [R[0][0], R[0][1], R[0][2], t[0]],
    [R[1][0], R[1][1], R[1][2], t[1]],
    [R[2][0], R[2][1], R[2][2], t[2]],
    [0,       0,       0,       1]
  ];

  // dstAffine = srcAffine @ M (4x4 multiply). resampleAffine internally
  // does inv(srcAffine) @ dstAffine to compose dst_voxel -> src_voxel; that
  // recovers M.
  const dstAffine = matmul4x4(srcAffine, M);

  return {
    dstAffine, mniDims, mniCenterVox,
    eigenvalues: sortedEigs, R
  };
}

function matmul4x4(A, B) {
  const out = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += A[r][k] * B[k][c];
      out[r][c] = s;
    }
  }
  return out;
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
