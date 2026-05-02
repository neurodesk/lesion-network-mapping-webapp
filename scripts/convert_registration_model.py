#!/usr/bin/env python3
"""Convert SynthMorph "brains" registration weights (.h5, Apache-2.0) into
an ONNX file suitable for browser inference.

Source: voxelmorph/synthmorph "brains-dice-vel-0.5-res-16-256f.h5" hosted at
        https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/
        brains-dice-vel-0.5-res-16-256f.h5
        (Hoffmann et al. 2022, "SynthMorph: learning contrast-invariant
         registration without acquired images"; Apache-2.0).

The full model has three custom layers we don't want to convert:
- VecInt              (scaling-and-squaring SVF integration; uses tf.while_loop)
- RescaleTransform    (SVF half-res -> full-res upsample)
- SpatialTransformer  (warps an image with the integrated displacement field)

Only the UNet backbone (input concat -> encoder -> decoder -> final flow
Conv3D) is exported — input shape `(2 x [1, 160, 160, 192, 1])`, output
shape `(1, 80, 80, 96, 3)` = stationary velocity field at half resolution.

The browser side performs:
  1. SVF integration via scaling-and-squaring in pure JS (~30 LOC).
  2. SVF upsample via trilinear interpolation in pure JS.
  3. Spatial warp of the lesion mask via the existing trilinear sampler in
     web/js/modules/volume-utils.js.

This split bypasses the only known ONNX-export pain point (tf.while_loop)
without losing any modelled accuracy.

Usage:
  pip install --user tensorflow tf2onnx voxelmorph
  python3 scripts/convert_registration_model.py
  # then upload /tmp/lnm_synthmorph_svf.onnx to
  # huggingface.co/datasets/sbollmann/lnm-webapp-models/models/lnm-synthmorph-mni.onnx
"""
import hashlib
import os
import subprocess
import sys
import urllib.request as U

WEIGHTS_URL = ("https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/"
               "synthmorph/brains-dice-vel-0.5-res-16-256f.h5")
WEIGHTS_PATH = "/tmp/synthmorph_brains.h5"
SAVED_MODEL_DIR = "/tmp/sm_saved"
ONNX_PATH = "/tmp/lnm_synthmorph_svf.onnx"


def main():
    # voxelmorph 0.2 calls inspect.getargspec which was removed in Python 3.11.
    # If you see AttributeError on import, patch:
    #   sed -i 's/inspect.getargspec(func)/inspect.getfullargspec(func)[:4]/' \
    #     $(python -c "import neurite, os; print(os.path.dirname(neurite.__file__))")/tf/modelio.py
    os.environ["TF_USE_LEGACY_KERAS"] = "1"

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Downloading {WEIGHTS_URL}...")
        U.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    sz = os.path.getsize(WEIGHTS_PATH)
    print(f"Weights: {WEIGHTS_PATH} ({sz:,} bytes)")

    import tensorflow as tf
    import voxelmorph as vxm
    import numpy as np

    full = vxm.networks.VxmDense.load(WEIGHTS_PATH, input_model=None)
    # Cut the model just before VecInt so the exported subgraph is pure
    # convolutions + activations + concatenations + pool/upsample.
    svf_layer = full.get_layer("vxm_dense_flow").output
    sub = tf.keras.Model(inputs=full.inputs, outputs=svf_layer, name="synthmorph_svf")
    print(f"SVF sub-model: inputs={[t.shape for t in sub.inputs]} output={sub.output.shape}")
    print(f"params: {sub.count_params():,}")

    # Forward sanity check.
    np.random.seed(0)
    src = np.random.rand(1, 160, 160, 192, 1).astype(np.float32)
    tgt = np.random.rand(1, 160, 160, 192, 1).astype(np.float32)
    tf_out = sub.predict([src, tgt], verbose=0)
    print(f"forward OK; SVF range: [{tf_out.min():.4f}, {tf_out.max():.4f}]")

    # SavedModel + tf2onnx CLI.
    sub.export(SAVED_MODEL_DIR)
    print(f"SavedModel: {SAVED_MODEL_DIR}")

    print("Running tf2onnx...")
    result = subprocess.run(
        [sys.executable, "-m", "tf2onnx.convert",
         "--saved-model", SAVED_MODEL_DIR,
         "--output", ONNX_PATH,
         "--opset", "17"],
        check=True
    )
    print(f"ONNX: {ONNX_PATH}")

    # Parity check via onnxruntime (Python).
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_names = [x.name for x in sess.get_inputs()]
    feed = {in_names[0]: src, in_names[1]: tgt}
    ort_out = sess.run(None, feed)[0]
    diff_max = float(np.max(np.abs(tf_out - ort_out)))
    diff_mean = float(np.mean(np.abs(tf_out - ort_out)))
    print(f"max abs diff: {diff_max:.3e}")
    print(f"mean abs diff: {diff_mean:.3e}")
    if diff_max > 1e-2:
        sys.exit(f"parity check failed: {diff_max} > 1e-2")

    sha = hashlib.sha256(open(ONNX_PATH, "rb").read()).hexdigest()
    out_sz = os.path.getsize(ONNX_PATH)
    print(f"\nUpload:  /tmp/lnm_synthmorph_svf.onnx -> "
          f"sbollmann/lnm-webapp-models/models/lnm-synthmorph-mni.onnx")
    print(f"size:    {out_sz:,} bytes ({out_sz/1024/1024:.1f} MB)")
    print(f"sha256:  {sha}")


if __name__ == "__main__":
    main()
