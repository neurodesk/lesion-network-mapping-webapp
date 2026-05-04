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

- `web/js/lnm-app.js` — Main Lesion Network Mapping app class. Owns the full chain: file load, brain extraction, lesion segmentation, MNI registration, warp+resample bridge, Yeo overlap, FC network map, thresholding, and the one-click `runFullPipeline()`
- `web/js/app/config.js` — Model config, version (bumped by the manual release workflow)
- `web/js/app/lnm-tasks.js` — LNM pipeline inventory and stage status helpers
- `web/js/app/lnm-labels.js` — Yeo 7-network label table and NiiVue colormap
- `web/js/modules/overlap-export.js` — Pure JS CSV serialization for LNM overlap summaries
- `web/js/modules/overlap-render.js` — DOM rendering for the LNM network-overlap results table
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
- The old full-resolution SynthMorph ONNX graph OOMed browser runtimes because its first activation was ~4.7 GiB. Keep `npm run test:synthmorph-browser-model` green whenever the registration manifest or converter changes; it pins the browser graph dimensions, cache key, and activation budget. `executionProviders: ['webgpu', 'wasm']` still prefers WebGPU when available, with WASM fallback now expected to complete for the 48×48×64 graph.
- 2-channel softmax outputs (e.g. SynthStroke baseline) must be collapsed to a single logit (`logit_stroke - logit_bg`) before `runPatch` consumes them. Treating them as a 1-channel logit produces whole-brain coverage as a regression.
- F-order NIfTI vs row-major NDHWC tensor layout: registration.js operates on F-order voxel arrays; SynthMorph forward consumes NDHWC; the worker handles both transpositions. Do not change the layout contract without updating `test_registration.cjs` + the parity fixture.
- Bridge from MNI160 1mm warp output to Yeo7 atlas grid: `applyRegistrationToLesion()` invokes `runWarpMask` (worker), awaits the `'mni-lesion'` stage data via a one-shot resolver, and resamples onto the atlas with `resampleAffine(..., 'nearest')`. The output replaces `this.lesionFile` so downstream stages run unchanged.
- `runFullPipeline()` has two branches. (a) Manual: a Yeo-grid (99×117×95) lesion mask is already loaded → skip seg/register/bridge. (b) Auto: structural T1 only → full chain. The dim-probe gate enforces 99×117×95 exactly to match the overlap reducer.
- Threshold UI: when the mode flips between absolute / percentile, the slider's `min/max/step/value` are retuned (0..1 / 0..100). The slider re-fires `applyNetworkThreshold` on every input change, so the thresholded mask + summary stay in sync.
- Config version is bumped by the manual GitHub Actions release workflow via `sed`; it increments the patch version — do not bump manually (per-phase commits in this codebase have done so explicitly under TDD).

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
| `npm run test:registration` | SVF integration (scaling-and-squaring), displacement-field upsample, warpVolume — synthetic-warp roundtrip |
| `npm run test:fc-weighted-sum` | fcWeightedSum + decodeFcPack + summaryToNetworkWeights, synthetic 3-parcel toy connectome |
| `npm run test:threshold` | applyThreshold (absolute / percentile, one-sided / symmetric, with minClusterVoxels) + quantileAbsValue |
| `npm run test:resample` | affineFromHeader + invertAffine + resampleAffine (5 cases: identity, downsample, oob, trilinear, validation) |
| `npm run test:resample-parity` | Yeo grid → MNI160 1mm → Yeo grid roundtrip on a 6³ phantom: Dice = 1.0 + centroid drift < 1 voxel |
| `npm run test:prealign` | centroidOfMask + applyAffineToVoxel + computePrealignAffine (centroid-only Phase 16 v1 math) |
| `npm run test:prealign-pca` | covarianceOfMask + jacobiEigen3x3 + principalAxisAlign (Phase 26 PCA: rotated phantom recovers principal axis, det(R)=+1) |
| `npm run test:real-data-bridge` | Real ds004884 stroke (160×256×256 1mm) → MNI160 → Yeo7 → parcel overlap. Pinned-fixture parity gate (Phase 30) |
| `npm run test:real-data-pca` | Phase 26 PCA on the ds004884 T1: positive eigenvalues, det(R)=+1, resampled centroid within 1.5 voxels of MNI center |
| `npm run test:deploy-budget` | Static deploy < 60 MB; runtime cold-load < 300 MB (currently 38 + 209) |
| `npm run test:manifest-checksums` | sha256 of every cached/committed asset matches its manifest entry; catches manifest/fixture drift |
| `npm run test:worker` | inference-worker module-worker invariants + ~20 source-grep guards on the message protocol; pins SynthMorph EP introspection (Phase 28) |
| `npm run test:app` | LesionNetworkMappingApp class shape + import surface + per-phase wiring assertions (Yeo, atlas, threshold, resample/bridge, FC, PCA prealign, perf instrumentation) |
| `npm run test:html` | `web/index.html` required-IDs lockdown (27 IDs incl. all phase additions) + no surviving SCT branding |
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
