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

**Phase 4 complete (v0.4.0)** — Yeo7 group functional-connectivity weighted
sum. After running "Compute overlap" on a manual MNI 2 mm lesion mask, click
"Compute network map" — the orchestrator fetches a 30 MB Yeo7 FC pack (7
brain-wide t-maps, computed from 30 ADHD-200 subjects via
[`scripts/build_yeo7_connectome.py`](scripts/build_yeo7_connectome.py)),
weights each network's t-map by the lesion's share of that network, and
emits a Float32 NIfTI on the Yeo7 atlas grid (99×117×95 2 mm). Output
renders as a red-yellow overlay and downloads as `lnm-network-map.nii`.
Pure main-thread JS — no worker round-trip; the math is a per-voxel linear
combination via [`web/js/modules/fc-weighted-sum.js`](web/js/modules/fc-weighted-sum.js).

**Phase 3 complete (v0.3.0)** — deformable MNI registration via
**SynthMorph** (Hoffmann 2022, Apache-2.0). Click "Run MNI registration"
on a 160×160×192 1mm structural T1; the app fetches the SynthMorph
ONNX (81 MB) + the lnm-mni160 reference (8 MB), runs the SVF-only
sub-network in the worker, then performs scaling-and-squaring SVF
integration + half→full upsample + the spatial warp in pure JS
([`web/js/modules/registration.js`](web/js/modules/registration.js)).
WebGPU execution provider when available; falls back to WASM.

The **`lnm-yeo-auto`** pipeline declares the full T1 → SynthStrip →
seg → register → MNI Yeo overlap chain. Stages run individually for
now (each is a button click); a one-shot "auto" runner that also
resamples the warped lesion onto the MNI152 2mm Yeo grid is a
follow-up polish slice.

**Experimental notes**: SynthMorph's deformable head expects roughly-
MNI-aligned input. Without an upstream affine pre-step (FSL FLIRT,
ANTs `antsRegistrationSyNQuick`), deformable registration on raw
clinical T1 may not converge well. Inputs must be exactly 160×160×192
at 1mm; the orchestrator surfaces a clear error otherwise.

**Phase 25 complete (v0.13.2)** — manifest checksum verifier.

`scripts/test_manifest_checksums.cjs` walks every supported asset in
`web/models/manifest.json`, locates its bytes in either
`web/models/_dev_cache/` (developer machine) or
`tests/fixtures/yeo7-mini/atlas.nii.gz` (committed copy), and asserts
the actual sha256 matches the manifest's declared one. Catches:

1. The committed Yeo7 fixture drifting from the runtime asset.
2. A model rebuild landing in `_dev_cache` without a matching manifest
   bump (cacheKey collision → browser serves stale cached bytes).
3. A manifest typo where `filename` / `sourceUrl` doesn't match the
   actual asset id.

Skips assets without a local copy (most CI runners don't fetch the
~200 MB of weights), but requires ≥ 1 verified asset so the gate
doesn't silently no-op. Currently verifies 7 / 7 supported assets
(including independent verification of the committed Yeo7 fixture)
on a developer machine.

**Phase 24 complete (v0.13.1)** — real-data Yeo overlap in `npm test`.

`tests/fixtures/yeo7-mini/atlas.nii.gz` (73 KB, identical bytes to the
runtime asset) committed alongside the existing `ds004884-mini` lesion
fixture. `scripts/test_real_data_bridge.mjs` extended to load the
atlas, run `computeParcelOverlap` + `summarizeNetworkOverlap` on the
prealigned + Yeo-resampled lesion, and assert:

- ≥ 2 Yeo networks hit (real chronic strokes span vascular
  territories that cross network boundaries)
- network voxel sums + out-of-atlas voxels = total Yeo lesion count

Output for the committed ds004884 case (left-hemisphere chronic
stroke): 7 networks hit; Limbic 11.6%, Default 10.5%, Frontoparietal
7.1% dominant; ~63% subcortical / outside the cortical Yeo mask.

**Phase 21+22 complete (v0.13.0)** — UX + deploy budget.

- **Reset state button**: New "Clear results" button in the Results
  section drops every intermediate file (overlap, network map,
  thresholded mask, brain mask, lesion mask, MNI lesion, network map
  raw data) + re-disables every download button + clears the table
  body and threshold summary. Structural file is retained so users
  can quickly re-run on the same input.
- **Deploy-size budget**: New `scripts/test_deploy_budget.cjs` walks
  the static deploy artifact (web/ minus _dev_cache) and tallies
  every supported manifest entry's `sizeBytes` for runtime fetches.
  Asserts static < 60 MB (currently 38) and total cold-load < 300 MB
  (currently 247). Locks the original plan's "<200 MB cold load"
  acceptance criterion against future model swaps.

**Phase 18 complete (v0.12.1)** — real-data bridge integration test.

The unit suites covered the resample math on synthetic phantoms and
the prealign math on synthetic centroids; this slice connects them on
real anatomical shape so a regression that fires only on a real lesion
surfaces in CI. New `scripts/test_real_data_bridge.mjs` runs in `npm
test`:

1. Decodes the committed `ds004884-mini` lesion mask (160×256×256 1mm
   real chronic stroke, ~131k voxels).
2. Runs `centroidOfMask` → `applyAffineToVoxel` → `computePrealignAffine`.
3. Resamples the lesion onto MNI160 1mm via `resampleAffine(...)`.
4. Resamples again onto the canonical Yeo7 99×117×95 2mm grid.
5. Asserts: source 131k → MNI160 ~131k (within ±15% for a 1mm→1mm
   shift+flip), MNI160 → Yeo7 within ±50% of the expected ⅛
   downsample, and the MNI160 centroid lands at voxel (80, 80, 96)
   within 1 voxel.

**Phase 16 complete (v0.12.0)** — in-browser affine pre-registration.

The SynthMorph deformable head requires its input at exactly 160×160×192
1mm AND roughly MNI-aligned. Real clinical T1s come in arbitrary dims,
voxel sizes, and ACPC orientations, which previously meant external
prep (FSL FLIRT / ANTs) before the auto chain could run. This phase
ships a centroid-match prealigner in pure JS:

- New module [`web/js/modules/prealign.js`](web/js/modules/prealign.js)
  exports `centroidOfMask`, `applyAffineToVoxel`, and
  `computePrealignAffine`. The destination affine places the source
  brain centroid at MNI160 voxel (80, 80, 96) under canonical FSL
  orientation (-x, +y, +z); when fed to `resampleAffine(..., 'trilinear')`
  it produces a 160³ 1mm prealigned T1 + brainmask in one pass.
- New orchestrator method `prealignToMni160()` and a new "Pre-align to
  MNI 160³ 1mm" button in the Lesion section. Runs SynthStrip first
  if the brainmask is absent, then resamples T1 + brainmask in pure JS,
  clears stale downstream state, and rerenders the viewer.
- Test suite: `scripts/test_prealign.cjs` covers the math (cube
  centroid, identity / scaled+translated affines, round-trip from MNI
  voxel back to source world centroid). Wired into `npm test`.

Limitations: centroid match only — no rotation correction, no
intensity-based optimisation. Works well for ACPC-aligned scans
(modern T1s); rotated clinical acquisitions or scans with severe
pathology may need a follow-up rigid pass (Phase 16 v2).

**Phase 20 complete (v0.11.1)** — CI/CD lock-down. Both
`.github/workflows/deploy-pages.yml` and `.github/workflows/release.yml`
were carried over from the SCT scaffold and broken on this fork
(referenced Git LFS we don't use, `web/models/*.onnx` files that don't
exist, and missing helper scripts `get_version.sh` /
`summarize_test_failures.cjs`). Rewritten:

- LFS removed throughout (we fetch ONNX from Hugging Face at runtime).
- Deploy verification now checks `web/wasm/ort-wasm*.wasm` (the only
  weight-class artifact we actually ship).
- New `scripts/get_version.sh` parses `VERSION` out of
  `web/js/app/config.js` for the release-tag bump step.
- Release workflow simplified — runs `npm test`, uploads the log on
  failure, computes the next patch version, tags + creates the GitHub
  release, then the deploy workflow's `workflow_run` trigger redeploys
  production from the new tag.

**Phase 19 complete (v0.11.0)** — per-stage perf instrumentation.

`runFullPipeline()` now wraps each stage in `performance.now()` markers,
logs a `[perf] <stage> (<module>): NN.NN s` line per stage, and emits a
final `=== Pipeline complete in X ===` summary with the total runtime
and stage count. Stage timings are also collected into
`window.app._perfStats` for inspection (e.g. `JSON.stringify(app._perfStats)`
in the dev console). No external deps; same suite stays green.

**Phase 15 complete (v0.10.0)** — pipeline-driven `runFullPipeline`.

- `runFullPipeline()` now iterates `selectedPipeline.stages` instead of
  hard-coding the chain. Selecting a different pipeline truncates or
  extends what runs:
  - **lnm-yeo-only**: overlap (manual mask).
  - **lnm-segment-only**: brain extraction → lesion seg.
  - **lnm-network-map**: overlap → FC → threshold (manual mask).
  - **lnm-yeo-auto**: brain extraction → seg → register → overlap →
    FC → threshold.
- New `_runStage(stage)` dispatches on `stage.module`. Unknown modules
  throw so a manifest typo surfaces immediately.
- Threshold module added to `IMPLEMENTED_MODULES`. Stage-level
  `defaults` are pushed into the threshold UI controls before
  `applyNetworkThreshold()` runs, so each pipeline can ship its own
  threshold preset.
- Test contract pinned: `_runStage` must have a case for every
  implemented module; `runFullPipeline` must iterate `pipeline.stages`.

**Phase 14 complete (v0.9.1)** — cancel-button wiring. The
`#cancelButton` next to the status line now actually terminates the
worker via `executor.cancel()`. Disabled state is driven from
`handleWorkerProgress` (enabled while progress fraction is in [0, 1))
and `handleStepComplete` (disabled at end-of-step), matching the
SCT-era behaviour for which the button was originally drawn.

**Phase 13 complete (v0.9.0)** — UX surface.

- Pipeline dropdown now lists every runnable pipeline declared in
  `lnm-tasks.js`, not just `lnm-yeo-only`. The Schaefer400 / GSP1000
  placeholder (`lnm-default`) is flagged `hidden: true` until those
  assets ship.
- New `isPipelineRunnable(pipeline)` helper + Node test enforces the
  dropdown filter contract.
- About modal now shows the actual `Config.VERSION` instead of an
  empty placeholder.

**Phase 8–11 complete (v0.8.0)** — fixtures + smoke + docs.

- New 160×160×192 1mm fixture
  ([`tests/fixtures/lnm-auto-mini`](tests/fixtures/lnm-auto-mini)):
  MNI152NLin2009cAsym 1mm template + planted hypointensity sphere as a
  smoke-test stand-in for a real stroke T1.
- Browser smoke now covers both branches of the full pipeline button.
  Phase 8 (manual mask, ~15 s) and Phase 10 (auto chain, ~5 min cold)
  exercise the orchestrator end-to-end.
- AGENTS.md "Test surface" + "Key Conventions" sections rewritten
  from the SCT scaffold to the current LNM invariants (module worker
  lazy nifti load, Cache Storage URL fragment trick, NiiVue
  `addVolumeFromUrl` call shape, SynthMorph 160³ hard check,
  bridge/runFullPipeline contract, threshold UI live-update).

**Phase 7 complete (v0.7.0)** — polish + parity guard.

- Citations modal updated with Yeo, SynthStrip (Hoopes 2022),
  SynthStroke (Chalcroft 2025), ATLAS v2.0 (Liew 2022), SynthMorph
  (Hoffmann 2022), ADHD-200, and the canonical lesion-network-mapping
  paper (Boes 2015).
- `runFullPipeline()` now detects a manually-loaded Yeo-grid lesion mask
  and skips segmentation/registration/bridge — going straight to
  overlap → FC → threshold so the manual flow is one click too.
- New `test:resample-parity` suite asserts a Yeo→MNI160→Yeo
  nearest-neighbor roundtrip preserves a 6³ phantom at Dice = 1.0 with
  no centroid drift, locking down the bridge math against future
  changes.

**Phase 6 complete (v0.6.0)** — end-to-end auto-pipeline. The native lesion
mask produced by SynthStroke is now bridged onto the Yeo7 MNI 2 mm grid in
two steps the orchestrator chains internally:

1. The worker applies the SynthMorph integrated displacement field to the
   F-order lesion voxels (`stepWarpMask` → 160×160×192 1 mm).
2. Pure-JS [`web/js/modules/resample.js`](web/js/modules/resample.js)
   performs an affine-aware resample (NIfTI sform → 4×4 inverse →
   per-voxel destination lookup) onto the Yeo atlas grid (99×117×95
   2 mm), nearest mode for binary masks.

The result is adopted as `this.lesionFile` so a downstream
`runYeoOverlap` → `runFcNetworkMap` → `applyNetworkThreshold` chain
runs without any extra plumbing. A new **"Apply registration to lesion"**
button exposes step (1)+(2); a **"Run full pipeline"** button chains
brain extraction, lesion segmentation, registration, the bridge, Yeo
overlap, FC network map, and threshold (defaults) in one click.

**Phase 5 complete (v0.5.0)** — thresholding UI + cluster cleanup. The
"Network map" subsection now exposes a Threshold panel:

- **Mode**: absolute (slider is t-stat) or percentile of |voxels| (slider
  is 0–100).
- **Symmetric** toggle: `|x| > T` instead of `x > T` for positive/negative
  one-sided.
- **Min cluster (voxels)**: post-threshold 26-connected component cleanup
  via the existing `removeSmallComponents` helper.
- A live summary line reports the survivor count; a **Download
  thresholded mask** button emits a `Uint8` NIfTI binary mask
  (`lnm-network-map-thresh.nii`).

Pure JS in [`web/js/modules/threshold.js`](web/js/modules/threshold.js);
the slider re-fires `applyNetworkThreshold` on every input change so the
mask + summary stay in sync with the controls.

## Attribution

The browser scaffolding (NiiVue viewer integration, ONNX Runtime Web worker
pipeline, NIfTI/DICOM I/O, GitHub Pages deploy workflow) is adapted from
[`neurodesk/spinalcordtoolbox-webapp`](https://github.com/neurodesk/spinalcordtoolbox-webapp).
See `THIRD_PARTY_NOTICES.md` (added in Phase 1) for full credit.

Pipeline-specific dependencies (added incrementally):

- **Brain extraction**: SynthStrip (Hoopes 2022, Apache-2.0); ONNX export ported from `neurodesk/vesselboost-webapp`.
- **Lesion segmentation**: SynthStroke baseline (Chalcroft 2025 MELBA, MIT); 3D MONAI UNet, T1.
- **Registration**: SynthMorph (Hoffmann 2022, Apache-2.0); UNet-only ONNX cut (layers 0–33); JS-side SVF integration + warp.
- **Atlas**: Schaefer 2018 400 × 7 networks (CC-BY).
- **Connectome**: ADHD-200 group functional connectivity (computed by `scripts/build_yeo7_connectome.py`, N=30 subjects, Yeo7 ROI seed-to-voxel t-maps); Lead-DBS GSP1000 + Schaefer 400 are a future hardening upgrade.

## Local development

```sh
npm install
bash web/setup.sh   # downloads ONNX Runtime WASM
bash web/run.sh     # serves http://localhost:8080/
npm test            # 12 Node-only suites: lint, tasks, manifest, parcel-overlap,
                    #                      overlap-export, volume-utils,
                    #                      brain-extraction, registration,
                    #                      fc-weighted-sum, worker, app, html
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
npm run test:synthstrip-parity     # SynthStrip:        ~5 s
npm run test:lesion-seg-parity     # Lesion seg:        ~5 s; Dice >= 0.50 vs ds004884 ground truth
npm run test:registration-parity   # SynthMorph:       ~37 s (CPU EP); self-pair near-identity
npm run test:fc-weighted-sum-parity # FC weighted sum: ~1 s; identity case bit-exact
```

All fetch their respective ONNX models live from Hugging Face on first
run (cached under `web/models/_dev_cache/`, gitignored). The
lesion-segmentation parity uses one chronic-stroke subject from
[OpenNeuro ds004884](https://openneuro.org/datasets/ds004884/versions/1.0.1)
(Aphasia Recovery Cohort, Roth et al. 2024 — CC0); see
`tests/fixtures/ds004884-mini/SOURCE.md` for attribution. Observed Dice
on `sub-M2051 ses-284`: 0.5325.

## License

TBD; intended to be open-source. Third-party assets retain their own licenses.
