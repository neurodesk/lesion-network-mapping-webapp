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

Early scaffold. The app is being built up phase-by-phase under TDD. Current
state: scaffold imported from
[`neurodesk/spinalcordtoolbox-webapp`](https://github.com/neurodesk/spinalcordtoolbox-webapp)
at SHA `40d0f5451ec38ee402f7c62186a6614f19eb2304`, SCT-specific assets
stripped, awaiting LNM pipeline wiring.

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
bash web/run.sh     # serves http://localhost:8000/web/
npm test            # lint-only at scaffold; tests grow per phase
```

## License

TBD; intended to be open-source. Third-party assets retain their own licenses.
