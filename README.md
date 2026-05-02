# Lesion Network Mapping Webapp

A fully client-side webapp for **Lesion Network Mapping (LNM)** of stroke
lesions, running entirely in the browser:

1. Auto-segment a stroke lesion from a structural MRI (ONNX model in a Web Worker).
2. Normalize the patient brain to MNI152 with a deep-learning registration model.
3. Compute lesion overlap with a parcellation atlas (Schaefer 400 x 7 networks).
4. Combine per-parcel precomputed normative functional-connectivity maps
   (Lead-DBS / Lead-Mapper style) into a lesion-network map.
5. Threshold and visualize on top of MNI.

No backend. No data upload. Hosted on GitHub Pages.

## Status

**Phase 1 complete (v0.1.0)** — manual-mask Yeo 7-network overlap. Drop a
binary lesion mask aligned to MNI152NLin2009cAsym 2mm, click "Compute
overlap", get a per-network table with voxel counts, % of lesion, an inline
magnitude bar, and a CSV export. Voxels falling outside the Yeo brain mask
are surfaced as a warning. The Yeo7 atlas is fetched live from
[`sbollmann/lnm-webapp-models`](https://huggingface.co/datasets/sbollmann/lnm-webapp-models)
on Hugging Face and cached client-side.

**Phase 2 complete (v0.2.0)** — auto brain extraction + lesion
segmentation on T1.

Drop a structural T1; the app:

1. Runs **SynthStrip brain extraction** automatically in a module worker
   (~7–10 s on M-series, WASM EP, single-pass). Result is rendered as a
   translucent green overlay and downloadable as `lnm-brainmask.nii`.
   Model: [FreeSurfer SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/)
   ONNX export ported from
   [`neurodesk/vesselboost-webapp`](https://github.com/neurodesk/vesselboost-webapp).
2. On click of "Run lesion segmentation", runs **SynthStroke baseline**
   (the closest available openly-licensed model for ATLAS-2-style
   chronic stroke on T1; 3D MONAI UNet, MELBA 2025, MIT). ~5 s per pass
   on M-series. Result is rendered as a translucent red overlay and
   downloadable as `lnm-lesion.nii`. Sliding-window 128³ patches,
   threshold 0.4, min cluster 30, overlap 0.25, no TTA.

Phases 3–5 (MNI registration, parcel-FC weighted sum, thresholding /
network-map overlay) are planned but not yet implemented. Until Phase 3
ships registration, the lesion segmentation produces a mask in the
input image's native space; the manual-mask Yeo overlap flow continues
to require a mask already aligned to MNI152NLin2009cAsym 2mm.

## Attribution

The browser scaffolding (NiiVue viewer integration, ONNX Runtime Web worker
pipeline, NIfTI/DICOM I/O, GitHub Pages deploy workflow) is adapted from
[`neurodesk/spinalcordtoolbox-webapp`](https://github.com/neurodesk/spinalcordtoolbox-webapp).
See `THIRD_PARTY_NOTICES.md` (added in Phase 1) for full credit.

Pipeline-specific dependencies (added incrementally):

- **Brain extraction**: SynthStrip (Hoopes 2022, Apache-2.0); ONNX export ported from `neurodesk/vesselboost-webapp`.
- **Lesion segmentation**: SynthStroke baseline (Chalcroft 2025 MELBA, MIT); 3D MONAI UNet, T1.
- **Registration** (Phase 3): SynthMorph (Hoffmann et al., Apache-2.0), exported to ONNX.
- **Atlas**: Schaefer 2018 400 × 7 networks (CC-BY).
- **Connectome** (Phase 4): Lead-DBS GSP1000 group functional connectome.

## Local development

```sh
npm install
bash web/setup.sh   # downloads ONNX Runtime WASM
bash web/run.sh     # serves http://localhost:8080/
npm test            # 10 Node-only suites: lint, tasks, manifest, parcel-overlap,
                    #                      overlap-export, volume-utils,
                    #                      brain-extraction, worker, app, html
```

### Browser smoke tests

Optional. Run the Phase 1 manual-mask Yeo flow + the Phase 2a.1 auto-fired
SynthStrip flow in headless Chromium. Not in `npm test`; requires a one-off
browser install:

```sh
npx playwright install chromium
npm run test:smoke               # ~10 s on M-series; needs HF access
```

### Node-side parity tests

Drive each ONNX pipeline directly via `onnxruntime-node` (no browser),
against a real MNI152 anatomical T1. Pipeline-correctness checks
(plausibility, not Dice — see commit notes).

```sh
npm run test:synthstrip-parity   # SynthStrip:    ~5 s
npm run test:lesion-seg-parity   # Lesion seg:    ~1 s
```

Both fetch their respective ONNX models live from Hugging Face on first
run (cached under `web/models/_dev_cache/`, gitignored).

## License

TBD; intended to be open-source. Third-party assets retain their own licenses.
