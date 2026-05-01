<!-- SPECKIT START -->
# General Instructions

- after every change to the source code make sure the Agent.md file is updated
- after every new feature added, make sure there is a test for the feature (no tests for removing features)
- after changing the code, start a new dev server and ask the user to check the resulting app functionality

## Development

- **Start dev server**: `cd web && bash run.sh` (serves on http://localhost:8080)
- `web/run.sh` stops an existing SCT dev server on the requested port before starting a replacement and serves local development files with no-store cache headers; use an alternate port argument when another non-SCT process owns `8080`
- **Setup**: `cd web && bash setup.sh` (downloads ONNX Runtime WASM files)

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

- `web/js/lnm-app.js` — Main Lesion Network Mapping app class, orchestrates Phase 1 file loading, viewer display, and overlap stubs
- `web/js/app/config.js` — Model config, version (bumped by the manual release workflow)
- `web/js/app/lnm-tasks.js` — LNM pipeline inventory and stage status helpers
- `web/js/app/lnm-labels.js` — Yeo 7-network label table and NiiVue colormap
- `web/js/app/sct-tasks.js` — SCT stable task inventory and task status helpers
- `web/js/app/labels.js` — Task labels + NiiVue colormap
- `web/js/modules/parcel-overlap.js` — Pure JS parcel and network overlap reducers
- `web/js/inference-worker.js` — Web Worker running the 3D inference pipeline (~700 lines, uses `importScripts`, not ES modules)
- `web/js/controllers/` — FileIO, DICOM, Inference, Viewer controllers
- `web/js/modules/` — UI components and inference pipeline modules

## Key Conventions

- The inference worker uses `importScripts()` (no ES modules)
- `ViewerController` owns NiiVue overlay lifecycle; keep segmentation as one managed overlay volume, and use segmentation-as-base mode when the input volume is hidden because NiiVue volume 0 is not a reliable hide target.
- Route input/segmentation visibility changes through `renderViewerVolumes()` so the Results eye buttons and toolbar input toggle rebuild a consistent NiiVue volume stack.
- Config version is bumped by the manual GitHub Actions release workflow via `sed`; it increments the patch version — do not bump manually
- Keep model availability metadata internal; user-facing UI copy should describe runnable tasks without release/support commentary.
- SCT Processing controls should not include explanatory copy about synthetic validation fixtures or generated task metadata.
- SCT Processing should only expose operations wired to the current loaded case and real output stages. Do not add pure helper/demo operations to the operation dropdown; `npm run test:routing` enforces that only vertebral labeling is exposed today.
- Run `python scripts/validate_sct_models.py --manifest web/models/manifest.json --all-tasks` after SCT manifest changes
- Gray matter (`graymatter`) uses a 2D nnU-Net wrapper with stronger connected-component cleanup; keep its preprocessing (`modelAxisOrder: 'zyx'`) and defaults (`minComponentSize: 1000`) in sync across `web/js/app/sct-tasks.js` AND `web/models/manifest.json` — the runtime reads `sct-tasks.js`, so missing fields there silently disable axis reordering and produce empty/near-empty masks.
- Spinal cord (`spinalcord`) preprocesses every input by resampling to `targetSpacing: [0.8958333, 0.7, 1.0]` (RAS xyz mm; the nnUNet `3d_fullres` plan ships `[1.0, 0.7, 0.8958]` in SI/AP/RL order, which matches after the `modelAxisOrder: 'zyx'` transpose) before inference. Without resampling, anisotropic axial inputs (e.g. MT 0.9×0.9×5 mm) feed the wrong physical scale and produce empty masks; keep both the targetSpacing AND the zyx axis order synced across `web/js/app/sct-tasks.js` and `web/models/manifest.json`. Covered by parity fixtures for the T2, dMRI, T2s, T1, and MT spinal cord cases.
- Vertebral labeling (`vertebrae`) is a post-segmentation browser step for T2 volumes. It ports SCT's C2-C3 OpenCV HOG/SVM YAML detector in JS and propagates PAM50 vertebral level distances; keep `web/models/c2c3_disc_models/*.yml`, `web/models/templates/PAM50/PAM50_levels.nii.gz`, `web/js/modules/vertebrae.js`, and the `batch_t2_label_vertebrae` multilabel parity gate in sync.
- The vertebrae task is flagged `processingOnly: true` in `web/js/app/sct-tasks.js` AND `web/models/manifest.json`. The SCT Segmentation dropdown filters out `processingOnly` tasks because they have no model assets; routing them through `runInference()` would silently fall back to `Config.MODEL.name` (sct-spinalcord.onnx) and produce no vertebral labels. Vertebrae is selected from the SCT Processing operation dropdown, which calls `runVertebralLabeling()` after a segmentation result exists. `npm run test:routing` enforces the invariant: every supported task in the segmentation dropdown has a primary model asset.
- Label-mask stages (`segmentation`, `vertebrae`) have independent Results visibility tracked in `_stageVisibility`; the Results eye buttons must only toggle their own stage. When input is visible, render visible label masks as overlays in stable order (`segmentation`, then `vertebrae`). When input is hidden, promote the first visible label mask to the NiiVue base volume only for rendering and draw any remaining visible label masks as overlays. Adding a new label-mask stage means: extend `isOverlayStage()`, register its colormap, map it in `getOverlayColormapId()`, and include it in the stable overlay order used by `getVisibleOverlayStages()`.
- `ViewerController.loadVolumeStack()` MUST load the base via `nv.loadVolumes([single])` and then add each overlay via `nv.addVolumeFromUrl()`. Calling `nv.loadVolumes([base, overlay1, overlay2, ...])` with multiple entries is a silent regression in NiiVue 0.68.x: the overlays appear in `nv.volumes` but `cal_min`/`cal_max`/colormap LUT state is not initialised on the overlay path, so binary/label-mask overlays vanish from the viewer even though `_stageVisibility` says they should render. `npm run test:viewer` enforces the call shape.
- Visible label-mask stages MUST also have a current result before rendering; never auto-render a stale sibling result whose stage is missing from the current run. Whenever a new run starts (`runSegmentation`, `clearResults`, `resetAllSteps`), call `resetStageVisibility()` and re-render so the viewer cannot show a previous overlay. `npm run test:routing` enforces these invariants.
- Label-mask NIfTI outputs and viewer overlays must preserve the true maximum label value in `cal_max`. Vertebrae labels use values above 1; forcing `cal_max=1` collapses all nonzero labels to one terminal color even though cursor readout still reports distinct label indices.
- `generateNiivueColormap()` in `web/js/app/labels.js` emits a NiiVue 0..255-indexed step LUT (each label index is scaled into the LUT range and paired with a held stop just below the next label) so NiiVue does not linearly interpolate between adjacent label colors and smear neighbouring vertebrae together. `npm run test:labels` enforces this. Use perceptually-ordered palettes (turbo-style) for ordinal label sets like vertebral levels — neighbouring labels get neighbouring hues, which reads the anatomy correctly.
- `STEP_EPSILON` in `labels.js` MUST be `>= 1.0`. NiiVue's `makeLut()` casts our `I` array through `Uint8ClampedArray.from(...)`, which round-half-to-even. With a sub-Uint8 epsilon (e.g. `1e-3`) the held stop and the next label start collide on the same Uint8 bucket. For binary masks (spinalcord — 2 labels), the only LUT segment is `[0, 255]` interpolating background→background, the trailing zero-range segment produces NaNs clamped to 0, and the entire LUT becomes transparent — the segmentation overlay disappears even though the volume is loaded and the eye toggle is on. Vertebrae masks the bug because later segments overwrite the corrupted bucket. The `test_labels.mjs` spinalcord regression case simulates `Uint8ClampedArray.from(I)` and walks NiiVue's segment-fill loop; LUT[255] must paint the labelled colour or the test fails.
- UI control coverage is enforced by `npm run test:ui`; browser-generated fixture parity for critical supported tasks is enforced by `npm run test:fixtures`. Regenerate fixture outputs with `npm run test:fixtures:generate` when model preprocessing changes.
- SCT batch reference fixtures are downloaded from Hugging Face dataset `sbollmann/sct-webapp-data` by `scripts/huggingface-fixtures.cjs`; only `input.nii.gz`, `batch_output.nii.gz`, and `batch_processing.sh` are downloaded. `browser_output.nii.gz` remains locally generated by `npm run test:fixtures:generate`.
- `batch_t2_label_vertebrae` is included in `test_fixture_parity_outputs.cjs` with multilabel Dice gating; do not replace it with a binary foreground-only gate.

## Test surface

| Script | What it covers |
| --- | --- |
| `npm run lint` | acorn syntax check on all `web/**/*.js` |
| `npm run test:manifest` | Asserts `web/js/app/sct-tasks.js` (read by the browser) and `web/models/manifest.json` (read by fixture-parity scripts) agree on `preprocessing`, `inferenceDefaults`, `patchSize`, `checksum`, `filename` for every supported task. Catches the silent runtime drift class that fixture tests miss. Also asserts every supported, non-`processingOnly` task ships at least one model asset. |
| `npm run test:routing` | Asserts the SCT Segmentation dropdown only offers tasks with a model asset, that `processingOnly` tasks (vertebrae) are gated through SCT Processing → `runVertebralLabeling`, and that label-mask stages render as overlays via `_overlayStage`/`resolveOverlayStage`. Catches the silent fall-back class where `runInference()` would run the default model on a model-less task. |
| `npm run test:labels` | Asserts `generateNiivueColormap()` emits a step LUT (held stops between consecutive label indices) so NiiVue does not interpolate across discrete labels. |
| `npm run test:ui` | Static control-presence audit: every `index.html` interactive control is referenced in JS and assigned a coverage source |
| `npm run test:viewer` | `ViewerController` overlay lifecycle against a fake NiiVue |
| `npm run test:processing` | `sct-processing.js` pure-function unit tests (signal processing, segmentation utilities) |
| `npm run test:vertebrae` | Vertebral labeling unit and fixture-focused checks: OpenCV HOG/SVM YAML parsing, PAM50 distance propagation, and multilabel Dice on `batch_t2_label_vertebrae` |
| `npm run test:batch:webapp` | 62 `batch_processing.sh` SCT commands → browser feature mapping |
| `npm run test:fixtures` | Dice + foreground-ratio gates for segmentation fixtures plus multilabel vertebrae parity; downloads missing SCT batch references from Hugging Face and generates missing/stale `browser_output.nii.gz` files with real ONNX inference |
| `npm run test:fixtures:download` | Downloads `test_data/batch_processing.sh` plus fixture `input.nii.gz` and `batch_output.nii.gz` files from Hugging Face |
| `npm run test:fixtures:generate` | Regenerates `browser_output.nii.gz` by running real ONNX inference (Node-only, no browser) |
| `npm run test:controllers` | `FileIOController`, `DicomController`, `InferenceExecutor` against fake DOM/Worker |
| `npm run test:ui:modules` | `ProgressManager`, `ConsoleOutput`, `ModalManager` against fake DOM |
| `npm run test:ci-summary` | Release-workflow test log summarizer used to publish failed npm script/test details in GitHub Actions summaries |
| `npm run test:inference:e2e` | Inference worker driven via VM shim against 3 fixtures with real ONNX Runtime (slow) |
| `npm run test:worker:protocol` | Worker postMessage protocol invariants (progress order, monotonicity, terminal-message uniqueness, error path) — slow |
| `npm run test:server` | Dev server graceful restart |
| `npm run test:fast` | Lint + manifest consistency + UI + viewer + processing + batch + fixtures + controllers + UI modules (no real-inference tests) |
| `npm test` | Full suite: `test:fast` + `test:inference:e2e` + `test:worker:protocol` + `test:server` |

## CI/CD

- **Release workflow** (`.github/workflows/release.yml`): manual-only promotion. The `validate` job runs the full `npm test` (including heavy ONNX-inference tests), captures failed-test details in the Actions step summary/job outputs, and the `release` job only runs on green and bumps version, creates tag + GitHub release.
- **Deploy workflow** (`.github/workflows/deploy-pages.yml`): deploys staging from `main` immediately on pushes to `main`, and deploys production from the latest release tag after the manual release workflow completes successfully. It downloads ONNX Runtime WASM files and verifies model assets before deploying to GitHub Pages; tests are run by the release workflow, not the deploy workflow.
- GitHub Pages deploys must check out Git LFS assets and verify `web/models/*.onnx` are real model binaries, not LFS pointer files.
- Production deploys build from the latest release tag while the workflow file comes from `main`; asset verification must tolerate older release tags by validating ONNX files and template `.nii.gz` files that exist in the checked-out build, without hard-coding newer template paths.

<!-- SPECKIT END -->
