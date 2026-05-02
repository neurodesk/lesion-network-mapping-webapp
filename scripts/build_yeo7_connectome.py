#!/usr/bin/env python3
"""Compute the Yeo7 group functional-connectivity (FC) pack from a small
public rsfMRI dataset and emit it as a single packed binary blob + index.

Output:
  /tmp/lnm_yeo7_fc/yeo7_fc_pack.bin           (Float32, 7 brain-wide t-maps
                                               at the Yeo7 atlas grid)
  /tmp/lnm_yeo7_fc/yeo7_fc_pack.index.json    (byte offsets + metadata)

Source dataset
--------------
Earlier plan called for "published Yeo lab group-FC maps". The Yeo lab
released the *parcellations* with their 2011 paper (which we already use as
`yeo7-2mm`), but did **not** release per-network voxel-wise group-FC maps.
This script computes equivalent maps from the **ADHD-200** preprocessed
release (CC-BY equivalent, 30 subjects via `nilearn.datasets.fetch_adhd`).

Pipeline
--------
For each Yeo7 network k:
  1. Extract per-subject ROI mean time-series (network k).
  2. Pearson-correlate against every brain voxel.
  3. Fisher-z transform.
  4. Group-mean / (group-std / sqrt(N)) -> one-sample t-map vs zero.

The 7 t-maps are stacked into a single contiguous Float32 buffer (one
voxel-grid per network). The companion index.json records byte offsets,
network labels, and dtype metadata so the browser worker can lazy-fetch
just the channels it needs (fc-weighted-sum.js, Phase 4.3).

Caveats
-------
- ADHD-200 N=30 is small. Group t-stats are noisier than Lead-DBS GSP1000
  (N=1000) and individual values shouldn't be over-interpreted. Useful for
  the LNM topographic pattern; not a substitute for clinical use.
- Assumes the Yeo7 atlas has been fetched into /tmp/yeo_fetch/ by the
  Phase 1 build (see tests/fixtures/synthstrip-mini/SOURCE.md or rerun the
  Phase 1 atlas fetch).
"""
import hashlib
import json
import os

import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn.image import load_img, resample_to_img
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker

OUT_DIR = "/tmp/lnm_yeo7_fc"
YEO7_PATH = "/tmp/yeo_fetch/Yeo7_LiberalMask_2mm.nii.gz"
NETWORK_NAMES = [
    "Visual", "Somatomotor", "DorsalAttention", "VentralAttention",
    "Limbic", "Frontoparietal", "Default",
]


def main(n_subjects: int = 30):
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(YEO7_PATH):
        raise SystemExit(
            f"Yeo7 atlas missing at {YEO7_PATH}. Re-run the Phase 1 atlas "
            "fetch (see tests/fixtures/synthstrip-mini/build_t1.py for an "
            "example nilearn fetch)."
        )

    yeo7 = load_img(YEO7_PATH)
    print(f"Atlas: {yeo7.shape}, voxsize={np.abs(np.diag(yeo7.affine))[:3]}")

    print(f"Fetching ADHD-200 ({n_subjects} subjects)...")
    adhd = datasets.fetch_adhd(n_subjects=n_subjects)

    roi_masker = NiftiLabelsMasker(labels_img=yeo7, standardize="zscore_sample", verbose=0)
    brain_masker = NiftiMasker(standardize="zscore_sample", verbose=0)
    brain_masker.fit(adhd.func[0])
    n_brain = int(brain_masker.mask_img_.get_fdata().sum())
    print(f"Brain mask voxels: {n_brain}")

    per_subj_z = []
    for i, func in enumerate(adhd.func):
        print(f"  subject {i + 1}/{len(adhd.func)}: {os.path.basename(func)}")
        roi_ts = roi_masker.fit_transform(func, confounds=adhd.confounds[i])
        brain_ts = brain_masker.transform(func, confounds=adhd.confounds[i])
        T = roi_ts.shape[0]
        r = (roi_ts.T @ brain_ts) / T
        z = np.arctanh(np.clip(r, -0.999, 0.999))
        per_subj_z.append(z)

    z_stack = np.stack(per_subj_z, axis=0)
    group_t = z_stack.mean(axis=0) / np.maximum(
        z_stack.std(axis=0) / np.sqrt(z_stack.shape[0]), 1e-6
    )
    print(f"Group t-stat range: [{group_t.min():.2f}, {group_t.max():.2f}]")

    fc_maps_3d = []
    for k in range(7):
        img_k = brain_masker.inverse_transform(group_t[k])
        img_k = resample_to_img(
            img_k, yeo7, interpolation="linear",
            force_resample=True, copy_header=True,
        )
        fc_maps_3d.append(np.asarray(img_k.dataobj).astype(np.float32))

    pack = np.stack(fc_maps_3d, axis=0).astype(np.float32)
    bin_path = os.path.join(OUT_DIR, "yeo7_fc_pack.bin")
    idx_path = os.path.join(OUT_DIR, "yeo7_fc_pack.index.json")
    pack.tofile(bin_path)

    n_per_map = pack.shape[1] * pack.shape[2] * pack.shape[3]
    index = {
        "dtype": "float32",
        "shape": list(pack.shape),
        "voxelsPerMap": int(n_per_map),
        "byteOffsets": [int(k * n_per_map * 4) for k in range(8)],
        "atlasAssetId": "yeo7-2mm",
        "networkLabels": {str(k + 1): name for k, name in enumerate(NETWORK_NAMES)},
        "statistic": "group-tstat",
        "source": f"ADHD-200, {n_subjects} subjects, nilearn fetch_adhd",
        "atlasResolutionMm": 2,
    }
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2)

    sha = hashlib.sha256(open(bin_path, "rb").read()).hexdigest()
    sz = os.path.getsize(bin_path)
    print(f"\nWrote {bin_path}: {sz:,} bytes ({sz/1024/1024:.2f} MB)")
    print(f"  sha256={sha}")
    print(f"Wrote {idx_path}")
    print(
        "\nNext: upload both files to "
        "huggingface.co/datasets/sbollmann/lnm-webapp-models/connectomes/"
    )


if __name__ == "__main__":
    main()
