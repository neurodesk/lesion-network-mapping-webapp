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

**Phase 1 complete (v0.1.0)**: manual-mask Yeo 7-network overlap. Drop a
binary lesion mask aligned to MNI152NLin2009cAsym 2mm, click "Compute
overlap", get a per-network table with voxel counts, % of lesion, an inline
magnitude bar, and a CSV export. Voxels falling outside the Yeo brain mask
are surfaced as a warning. The Yeo7 atlas is fetched live from
[`sbollmann/lnm-webapp-models`](https://huggingface.co/datasets/sbollmann/lnm-webapp-models)
on Hugging Face and cached client-side.

Phases 2–5 (auto-segmentation, MNI registration, parcel-FC weighted sum,
thresholding) are planned but not yet implemented.

## Attribution

The browser scaffolding (NiiVue viewer integration, ONNX Runtime Web worker
pipeline, NIfTI/DICOM I/O, GitHub Pages deploy workflow) is adapted from
[`neurodesk/spinalcordtoolbox-webapp`](https://github.com/neurodesk/spinalcordtoolbox-webapp).
See `THIRD_PARTY_NOTICES.md` (added in Phase 1) for full credit.

Pipeline-specific dependencies (added incrementally):

- **Lesion segmentation**: ATLAS-2-trained nnU-Net (Liew et al.), exported to ONNX.
- **Registration**: SynthMorph (Hoffmann et al., Apache-2.0), exported to ONNX.
- **Atlas**: Schaefer 2018 400 x 7 networks (CC-BY).
- **Connectome**: Lead-DBS GSP1000 group functional connectome.

## Local development

```sh
npm install
bash web/setup.sh   # downloads ONNX Runtime WASM
bash web/run.sh     # serves http://localhost:8080/
npm test            # 7 Node-only suites: lint, tasks, manifest, parcel-overlap,
                    #                     overlap-export, app, html
```

### Browser smoke test

Optional, runs the full Phase 1 flow in headless Chromium against a deterministic
phantom. Not in `npm test`; requires a one-off browser install:

```sh
npx playwright install chromium
npm run test:smoke
```

The smoke test fetches the Yeo atlas live from Hugging Face.

## License

TBD; intended to be open-source. Third-party assets retain their own licenses.
