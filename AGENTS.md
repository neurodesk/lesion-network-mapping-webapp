<!-- SPECKIT START -->
# General Instructions

- after every change to the source code make sure the Agent.md file is updated
- after every new feature added, make sure there is a test for the feature (no tests for removing features)
- after changing the code, start a new dev server and ask the user to check the resulting app functionality


## Linting

Run `npm run lint` before committing JS changes. This parses all `web/**/*.js` files for syntax errors using acorn. The same check runs in CI before deploy.

Common issues it catches:
- `await` in non-async functions
- Mismatched brackets/parens
- Invalid ES module syntax

## Dependency Maintenance

- Keep npm dependencies current with `npm install <package>@latest` so `package.json` and `package-lock.json` stay in sync.
- Keep browser CDN dependencies pinned in `web/index.html`; check the upstream package version before changing those URLs.

## Architecture

- `web/js/lnm-app.js` — Main Lesion Network Mapping app class. Owns the full chain: file load, brain extraction, lesion segmentation, MNI registration, warp+resample bridge, Yeo overlap, FC network map, thresholding, version-label formatting, and the one-click `runFullPipeline()`
- `web/js/app/config.js` — Model config, version (bumped by the manual release workflow)
- `web/js/app/lnm-tasks.js` — LNM pipeline inventory and stage status helpers
- `web/js/app/lnm-labels.js` — Yeo 7-network label table and NiiVue colormap
- `web/js/modules/overlap-export.js` — Pure JS CSV serialization for LNM overlap summaries
- `web/js/modules/overlap-render.js` — DOM rendering for the LNM network-overlap and affected-network results tables
- `web/js/app/sct-tasks.js` — SCT stable task inventory and task status helpers
- `web/js/app/labels.js` — Task labels + NiiVue colormap
- `web/js/modules/parcel-overlap.js` — Pure JS parcel and network overlap reducers
- `web/js/modules/threshold.js` — Threshold + cluster-cleanup for the FC network map (modes: absolute / percentile, optional symmetric, optional minClusterVoxels via existing CC helpers)
- `web/js/modules/brain-extraction.js` — SynthStrip wrapper: RAS→LIA, foreground crop, adaptive fast-mode target spacing, conform/pad, ONNX SDT inference, original-grid SDT thresholding, largest-CC cleanup, and LIA→RAS mask output
- `web/js/modules/resample.js` — Affine-aware 3D resampler that bridges the SynthMorph 160×160×192 1mm warp output onto the Yeo7 MNI2mm 99×117×95 grid; `affineFromHeader` reads sform/qform off a nifti-reader-js header, `resampleAffine` does NN/trilinear sampling under the destination affine
- `web/js/modules/prealign.js` — In-browser affine pre-registration to the SynthMorph-required MNI160 1mm grid. `centroidOfMask` computes the brain-mask voxel centroid; `applyAffineToVoxel` applies a 4×4 affine to a voxel coord; `computePrealignAffine` builds the centroid-only destination affine (Phase 16 v1); `covarianceOfMask` + `jacobiEigen3x3` + `principalAxisAlign` add Phase 26 PCA rotation so the brain's principal axes line up with MNI canonical axes (right-handed enforced via det(R)=+1 sign flip)
- `web/js/inference-worker.js` — Web Worker running the 3D inference pipeline (~700 lines, uses `importScripts`, not ES modules)
- `web/js/controllers/` — FileIO, DICOM, Inference, Viewer controllers
- `web/js/modules/` — UI components and inference pipeline modules

## Key Conventions

- The inference worker is a **module worker** (`type: 'module'`); load nifti-reader-js lazily inside the message handler (`niftiReady` promise) — top-level `await import(...)` causes Chromium to drop the first messages.
- Cache Storage (Atlas + connectome assets) requires URL-shaped keys. Bare strings like `'yeo7-fc-pack-adhd200-n30-v1'` parse as URL schemes and the put fails. Fold the manifest's `cacheKey` into the URL fragment: `${url}#${encodeURIComponent(cacheKey)}`. See `atlas-loader.js fetchCacheFirst`.
- `ViewerController` owns NiiVue overlay lifecycle. `loadVolumeStack()` loads the base via `nv.loadVolumes([single])` and adds each overlay via `nv.addVolumeFromUrl()` — passing multiple entries to `loadVolumes()` is a silent regression in NiiVue 0.68.x (cal_min/cal_max/colormap LUT not initialised on the overlay path).
- SynthMorph deformable still expects the source T1 prealigned to the canonical 160×160×192 1mm MNI grid, and `stepRegister` enforces that input. The browser ONNX graph itself is spatially retargeted to 48×48×64 (`lnm-synthmorph-mni-48x48x64.onnx`) so the first Conv3D activation stays under the browser budget; the worker downsamples source/reference to 48×48×64 before ONNX and upsamples the integrated 24×24×32 SVF-derived displacement back to 160×160×192.
- SynthStrip fast mode must not blindly downsample 1mm clinical T1s to 2mm. `chooseFastTargetSpacing()` uses the foreground bounding box to stay in the smallest safe conform bucket (typically ~192³) while preserving as much native detail as possible, and resampled runs threshold the linearly upsampled SDT on the original grid rather than nearest-upsampling a binary 2mm mask.
- The old full-resolution SynthMorph ONNX graph OOMed browser runtimes because its first activation was ~4.7 GiB. Keep `npm run test:synthmorph-browser-model` green whenever the registration manifest or converter changes; it pins the browser graph dimensions, cache key, provider routing, and activation budget. The current 48×48×64 graph still contains 3D MaxPool nodes, so `browserRuntime.executionProviders` is `['wasm']`; ORT WebGPU rejects NHWC 3D pooling.
- 2-channel softmax outputs (e.g. SynthStroke baseline) must be collapsed to a single logit (`logit_stroke - logit_bg`) before `runPatch` consumes them. Treating them as a 1-channel logit produces whole-brain coverage as a regression.
- F-order NIfTI vs row-major NDHWC tensor layout: registration.js operates on F-order voxel arrays; SynthMorph forward consumes NDHWC; the worker handles both transpositions. Do not change the layout contract without updating `test_registration.cjs` + the parity fixture.
- Yeo7 FC packs are stored as NumPy row-major channels. `decodeFcPack()` transposes each map to NIfTI x-fast order, and `runFcNetworkMap()` writes network-map and threshold-map NIfTIs with the Yeo atlas affine; otherwise NiiVue renders striped or spatially misplaced scalar overlays. Network-map display must use a Yeo-grid atlas-space stack, not the prealigned patient T1 or the smaller-FOV MNI160 template, because the FC map is in Yeo atlas space.
- Bridge from MNI160 1mm warp output to Yeo7 atlas grid: `applyRegistrationToLesion()` invokes `runWarpMask` (worker), awaits the `'mni-lesion'` stage data via a one-shot resolver, and resamples onto the atlas with `resampleAffine(..., 'nearest')`. The output replaces `this.lesionFile` so downstream stages run unchanged.
- Patient-space final threshold display: `applyNetworkThreshold()` still creates the canonical Yeo-grid threshold NIfTI, but if a structural T1 and SynthMorph displacement are available the app resamples that mask back to the fixed `lnm-mni160` registration grid (use the template affine, not the subject/prealign affine) and calls worker `inverse-warp-mask` to emit `threshold-patient`. The viewer then switches to the patient-space stack: structural T1 base, brain mask overlay, native lesion overlay, and threshold overlay.
- `runFullPipeline()` has two branches. (a) Manual: a Yeo-grid (99×117×95) lesion mask is already loaded → skip seg/register/bridge. (b) Auto: structural T1 only → full chain. The dim-probe gate enforces 99×117×95 exactly to match the overlap reducer.
- The `outsideAtlasWarning` DOM id is retained for compatibility, but the visible UI copy is a neutral Yeo cortical-label coverage note. `voxelsOutsideAtlas` means lesion voxels where the atlas label is `0` (unlabeled by the cortical Yeo parcellation), not voxels outside a whole-brain mask.
- Worker-backed stages used by `runFullPipeline()` must resolve only after their emitted output is ready: brain extraction waits for `'brainmask'` stage data, lesion segmentation waits for `'segmentation'` stage data, and registration waits for the `'register'` step-complete event before the warp bridge runs.
- Loading a structural image must never start SynthStrip or any other processing automatically. `setStructural()` may load the viewer and auto-select `lnm-yeo-auto`, but computation starts only from `Run analysis` or an explicit per-stage button.
- Do not expose a visible pipeline selector. `Run analysis` is input-driven: structural T1 selects the full auto chain, and researcher-mode Yeo-grid masks select the hidden manual network-map path; advanced users can still run individual stages with the compact per-stage buttons.
- Keep sidebar help copy behind compact inline `i` help popovers (`.help-icon` / `.help-popover`) rather than always-visible helper paragraphs, matching the QSMbly-style controls.
- Keep advanced per-stage button labels compact. Technical grid details such as MNI160 1 mm belong in `title`/help text, not inline superscript/unit strings that wrap inside small buttons.
- Threshold UI: when the mode flips between absolute / percentile, the slider's `min/max/step/value` are retuned (0..1 / 0..10 with 0.1% steps). Percentile mode is user-facing top-percent semantics (`5` keeps roughly the strongest 5%, `0` keeps none) and `applyNetworkThreshold()` converts that to the quantile cutoff expected by `applyThresholdDetailed`. The slider and min-cluster field re-fire thresholding on every input change; the summary reports how many voxels cluster cleanup removed, so "cluster size" changes that do not affect a large connected component are visible instead of silent. The thresholded mask, final affected Yeo-network table, summary, and live red `threshold-preview` overlay stay in sync while the scalar FC map remains visible.
- Viewer layer toggles are stage-driven and must hide volumes by setting opacity to `0`, not by removing NiiVue volumes. `ViewerController` preserves per-stage visibility for `structural`, `brainmask`, `segmentation`/`lesion`, and `threshold-preview` so toggles survive overlay replacement and threshold-slider refreshes.
- Config version is bumped by the manual GitHub Actions release workflow via `sed`; it increments the patch version — do not bump manually (per-phase commits in this codebase have done so explicitly under TDD).
- `populateVersionLabel()` uses `formatVersionLabel()` to combine `Config.VERSION` with `build-info.json`; staging versions already carry `-staging+<shortsha>`, so the build-info SHA must not be appended a second time.

## Test surface

| Script | What it covers |
| --- | --- |
| `npm run lint` | acorn syntax check on all `web/**/*.js` |
| `npm run test:tasks` | LNM_PIPELINES inventory + getPipelineById helpers |
| `npm run test:manifest` | `web/models/manifest.json` consistency: every pipeline stage's `modelAssetId`/`atlasAssetId`/`connectomeAssetId` resolves; supported assets carry checksum + sizeBytes + sourceUrl; networkLabels coverage |
| `npm run test:synthmorph-browser-model` | Browser-runtime SynthMorph contract: manifest must point at the 48×48×64 graph, declare 24×24×32 SVF dims, and keep the first Conv3D activation under 256 MiB |
| `npm run test:parcel-overlap` | Pure-JS parcel-overlap reducer: voxel counting, network aggregation, network-size denominators, edge cases |
| `npm run test:overlap-export` | CSV schema, ordering, numeric formatting, missing-network-size edge cases |
| `npm run test:volume-utils` | 3D resample / connectedComponents3D / removeSmallComponents reused across pipeline stages |
| `npm run test:brain-extraction` | SynthStrip orchestration helpers (header round-trip, RAS reorientation, conform/center-pad) plus adaptive fast-mode target spacing regression |
| `npm run test:registration` | SVF integration (scaling-and-squaring), displacement-field upsample, warpVolume + inverseWarpVolume — synthetic-warp roundtrip |
| `npm run test:fc-weighted-sum` | fcWeightedSum + decodeFcPack row-major→NIfTI voxel-order conversion + summaryToNetworkWeights, synthetic 3-parcel toy connectome |
| `npm run test:threshold` | applyThreshold (absolute / percentile, one-sided / symmetric, with minClusterVoxels) + quantileAbsValue |
| `npm run test:resample` | affineFromHeader + invertAffine + resampleAffine (5 cases: identity, downsample, oob, trilinear, validation) |
| `npm run test:resample-parity` | Yeo grid → MNI160 1mm → Yeo grid roundtrip on a 6³ phantom: Dice = 1.0 + centroid drift < 1 voxel |
| `npm run test:prealign` | centroidOfMask + applyAffineToVoxel + computePrealignAffine (centroid-only Phase 16 v1 math) |
| `npm run test:prealign-pca` | covarianceOfMask + jacobiEigen3x3 + principalAxisAlign (Phase 26 PCA: rotated phantom recovers principal axis, det(R)=+1) |
| `npm run test:real-data-bridge` | Real ds004884 stroke (160×256×256 1mm) → MNI160 → Yeo7 → parcel overlap. Pinned-fixture parity gate (Phase 30) |
| `npm run test:real-data-pca` | Phase 26 PCA on the ds004884 T1: positive eigenvalues, det(R)=+1, resampled centroid within 1.5 voxels of MNI center |
| `npm run test:deploy-budget` | Static deploy < 60 MB; runtime cold-load < 300 MB (currently 38 + 209) |
| `npm run test:manifest-checksums` | sha256 of every cached/committed asset matches its manifest entry; catches manifest/fixture drift |
| `npm run test:worker` | inference-worker module-worker invariants + ~25 source-grep guards on the message protocol; pins SynthMorph EP introspection and inverse-warp-mask wiring |
| `npm run test:app` | LesionNetworkMappingApp class shape + import surface + per-phase wiring assertions (Yeo, atlas, threshold, resample/bridge, FC, PCA prealign, perf instrumentation, patient-space viewer behavior) |
| `npm run test:app-behavior` | Runtime-stubbed LesionNetworkMappingApp behavior checks for pipeline dispatch, worker-stage waits, threshold-preview/projection scheduling, affected-network labeling, layer toggles, top-percent threshold semantics, min-cluster input recompute, preconditions, explicit-start structural loading, auto-promote, and version-label SHA de-duplication |
| `npm run test:viewer-controller` | ViewerController call-shape and overlay lifecycle checks, including stage-aware replacement used by threshold previews and per-stage visibility toggles |
| `npm run test:ui-modules` | ConsoleOutput log/clear/copy with Clipboard API fallback, ProgressManager, and ModalManager behavior |
| `npm run test:html` | `web/index.html` required-ID lockdown, compact help-popover/button-label guardrails, affected-network table surface, and no surviving SCT branding |
| `npm test` | Full Node suite: lint + browser-model contract + 20 above (no browser, no real-inference fixtures) |
| `npm run test:smoke` | Browser smoke (Playwright + headless Chromium): manual-mask Yeo overlap (Phase 1c.4), SynthStrip (2a.1.5), SynthStroke (2a.2.5), SynthMorph completion (3.7), full-pipeline manual branch (Phase 8), full-pipeline auto branch (Phase 10). Opt-in; requires `npx playwright install chromium`. ~5 min cold for Phase 10 |
| `npm run test:synthstrip-parity` | SynthStrip ONNX parity vs FreeSurfer reference plus ds004884 1mm clinical fast-mode overgrowth regression (Node, slow; opt-in) |
| `npm run test:lesion-seg-parity` | SynthStroke ONNX parity (Node, slow; opt-in) |
| `npm run test:registration-parity` | SynthMorph ONNX parity (Node, slow; opt-in) |
| `npm run test:fc-weighted-sum-parity` | FC weighted-sum parity vs reference (Node, slow; opt-in) |

## CI/CD

- **Release workflow** (`.github/workflows/release.yml`): manual-only promotion. The `validate` job runs the full `npm test` (including heavy ONNX-inference tests), captures failed-test details in the Actions step summary/job outputs, and the `release` job only runs on green and bumps version, creates tag + GitHub release.
- **Deploy workflow** (`.github/workflows/deploy-pages.yml`): deploys staging from `main` immediately on pushes to `main`, and deploys production from the latest release tag after the manual release workflow completes successfully. It downloads ONNX Runtime WASM files and verifies model assets before deploying to GitHub Pages; tests are run by the release workflow, not the deploy workflow.
- GitHub Pages deploys must check out Git LFS assets and verify `web/models/*.onnx` are real model binaries, not LFS pointer files.
- Production deploys build from the latest release tag while the workflow file comes from `main`; asset verification must tolerate older release tags by validating ONNX files and template `.nii.gz` files that exist in the checked-out build, without hard-coding newer template paths.

<!-- SPECKIT END -->
