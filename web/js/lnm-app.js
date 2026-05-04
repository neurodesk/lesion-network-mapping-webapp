import { FileIOController } from './controllers/FileIOController.js';
import { ViewerController } from './controllers/ViewerController.js';
import { InferenceExecutor } from './controllers/InferenceExecutor.js';
import { LNM_PIPELINES, getPipelineById, isPipelineRunnable } from './app/lnm-tasks.js';
import { YEO7_COLORMAP } from './app/lnm-labels.js';
import { computeParcelOverlap, summarizeNetworkOverlap } from './modules/parcel-overlap.js';
import { loadAtlasFromManifest, loadConnectomeFromManifest, decodeNiftiBuffer } from './modules/atlas-loader.js';
import { fcWeightedSum, decodeFcPack, summaryToNetworkWeights } from './modules/fc-weighted-sum.js';
import { applyThreshold } from './modules/threshold.js';
import { affineFromHeader, resampleAffine } from './modules/resample.js';
import { centroidOfMask, applyAffineToVoxel, computePrealignAffine, principalAxisAlign } from './modules/prealign.js';
import { writeNifti1 } from './modules/nifti-writer.js';
import { serializeOverlapCsv } from './modules/overlap-export.js';
import { renderOverlapTable } from './modules/overlap-render.js';
import { ConsoleOutput } from './modules/ui/ConsoleOutput.js';
import { ProgressManager } from './modules/ui/ProgressManager.js';
import { ModalManager } from './modules/ui/ModalManager.js';
import * as Config from './app/config.js';

function splitModelUrl(url) {
  const i = url.lastIndexOf('/');
  return { base: url.slice(0, i), name: url.slice(i + 1) };
}

function arrayBufferToFile(buffer, name) {
  // The worker emits uncompressed NIfTI bytes (createOutputNifti); files
  // ending in .nii are raw, .nii.gz are gzip-compressed. We use .nii.
  const blob = new Blob([buffer], { type: 'application/octet-stream' });
  return new File([blob], name, { type: 'application/octet-stream' });
}

function binarise(typedArray) {
  const out = new Uint8Array(typedArray.length);
  for (let i = 0; i < typedArray.length; i++) {
    out[i] = typedArray[i] > 0 ? 1 : 0;
  }
  return out;
}

function dimsEqual(a, b) {
  return Array.isArray(a) && Array.isArray(b) &&
    a.length === b.length &&
    a.every((value, index) => value === b[index]);
}

function computeNetworkSizes(atlasData, networkLabels) {
  const sizes = {};
  for (let i = 0; i < atlasData.length; i++) {
    const label = atlasData[i];
    if (label === 0 || !Object.prototype.hasOwnProperty.call(networkLabels, label)) continue;
    const network = networkLabels[label];
    sizes[network] = (sizes[network] || 0) + 1;
  }
  return sizes;
}

export class LesionNetworkMappingApp {
  constructor() {
    this.nv = new niivue.Niivue({
      ...Config.VIEWER_CONFIG,
      onLocationChange: (data) => this.updateViewerInfo(data)
    });

    this.console = new ConsoleOutput('consoleOutput');
    this.progress = new ProgressManager(Config.PROGRESS_CONFIG);
    this.structuralFile = null;
    this.lesionFile = null;
    this.overlapResult = null;
    this.brainmaskFile = null;     // populated by handleStageData('brainmask')
    this.lesionMaskFile = null;    // populated by handleStageData('segmentation')
    this.networkMapFile = null;    // Phase 4: populated by runFcNetworkMap
    this.networkMapData = null;    // Phase 5: raw Float32Array for re-thresholding
    this.networkMapDims = null;
    this.networkMapSpacing = null;
    this.thresholdedMaskFile = null; // Phase 5: thresholded binary NIfTI
    this.mniLesionFile = null;       // Phase 6: warped lesion at MNI160 1mm (pre-resample)
    this._mniLesionResolver = null;  // Phase 6: one-shot promise for warp-mask stage data
    this._perfStats = [];            // Phase 19: per-stage runtime markers
    this._perfRunStart = null;       // Phase 19: total runFullPipeline start
    this.manifest = null;          // populated lazily by ensureManifest()
    this.selectedPipeline = getPipelineById('lnm-yeo-only') || LNM_PIPELINES[0];
    // Phase 31: track whether the user has manually picked a pipeline
    // via the dropdown. If not, setStructural/setLesion auto-promote
    // to the pipeline that best matches the loaded input — so clicking
    // "Run full pipeline" with a manual mask gets the full overlap+FC+
    // threshold chain (lnm-network-map), and a structural T1 gets the
    // full auto chain (lnm-yeo-auto).
    this._userPickedPipeline = false;

    this.executor = new InferenceExecutor({
      updateOutput: (msg) => this.updateOutput(msg),
      setProgress: (frac, label) => this.handleWorkerProgress(frac, label),
      onStageData: (data) => this.handleStageData(data),
      onStepComplete: (step) => this.handleStepComplete(step),
      onError: (msg) => this.updateOutput(`Worker error: ${msg}`),
      onInitialized: () => this.updateOutput('Inference worker ready.')
    });
  }

  async init() {
    this.structuralFileIO = new FileIOController({
      updateOutput: (msg) => this.updateOutput(msg),
      onFileLoaded: (file) => this.setStructural(file)
    });
    this.lesionFileIO = new FileIOController({
      updateOutput: (msg) => this.updateOutput(msg),
      onFileLoaded: (file) => this.setLesion(file)
    });
    this.viewerController = new ViewerController({
      nv: this.nv,
      updateOutput: (msg) => this.updateOutput(msg)
    });

    this.aboutModal = new ModalManager('aboutModal');
    this.privacyModal = new ModalManager('privacyModal');
    this.citationsModal = new ModalManager('citationsModal');

    await this.setupViewer();
    this.viewerController.registerSctColormap(YEO7_COLORMAP, 'lnm-yeo7');
    this.bindEvents();
    this.populatePipelineSelect();
    this.populateVersionLabel();
    this.updateOutput('Ready.');
  }

  async setupViewer() {
    await this.nv.attachTo('gl1');
    this.nv.setMultiplanarPadPixels(5);
    this.nv.setSliceType(this.nv.sliceTypeMultiplanar);
    this.nv.setInterpolation(true);
    this.nv.drawScene();
  }

  bindEvents() {
    const structuralInput = document.getElementById('structuralFileInput');
    if (structuralInput) {
      structuralInput.addEventListener('change', (event) => {
        this.structuralFileIO.handleFiles(event.target.files);
      });
    }

    const lesionInput = document.getElementById('lesionFileInput');
    if (lesionInput) {
      lesionInput.addEventListener('change', (event) => {
        this.lesionFileIO.handleFiles(event.target.files);
      });
    }

    const pipelineSelect = document.getElementById('pipelineSelect');
    if (pipelineSelect) {
      pipelineSelect.addEventListener('change', () => {
        this.selectedPipeline = getPipelineById(pipelineSelect.value) || this.selectedPipeline;
        this._userPickedPipeline = true;
      });
    }

    const computeButton = document.getElementById('computeOverlapButton');
    if (computeButton) computeButton.addEventListener('click', () => this.runYeoOverlap());

    const csvButton = document.getElementById('downloadOverlapCsv');
    if (csvButton) {
      csvButton.disabled = true;
      csvButton.addEventListener('click', () => this.exportCsv());
    }

    const runBrainBtn = document.getElementById('runBrainExtractionButton');
    if (runBrainBtn) {
      runBrainBtn.addEventListener('click', () => {
        this.runBrainExtraction().catch(
          err => this.updateOutput(`Brain extraction failed: ${err.message}`)
        );
      });
    }
    const downloadBrainBtn = document.getElementById('downloadBrainMaskButton');
    if (downloadBrainBtn) {
      downloadBrainBtn.disabled = true;
      downloadBrainBtn.addEventListener('click', () => this.downloadBrainMask());
    }

    const runLesionBtn = document.getElementById('runLesionSegmentationButton');
    if (runLesionBtn) {
      runLesionBtn.addEventListener('click', () => {
        this.runLesionSegmentation().catch(
          err => this.updateOutput(`Lesion segmentation failed: ${err.message}`)
        );
      });
    }
    const downloadLesionBtn = document.getElementById('downloadLesionMaskButton');
    if (downloadLesionBtn) {
      downloadLesionBtn.disabled = true;
      downloadLesionBtn.addEventListener('click', () => this.downloadLesionMask());
    }

    const runRegBtn = document.getElementById('runRegistrationButton');
    if (runRegBtn) {
      runRegBtn.addEventListener('click', () => {
        this.runRegistration().catch(
          err => this.updateOutput(`Registration failed: ${err.message}`)
        );
      });
    }

    const applyRegBtn = document.getElementById('applyRegistrationToLesionButton');
    if (applyRegBtn) {
      applyRegBtn.addEventListener('click', () => {
        this.applyRegistrationToLesion().catch(
          err => this.updateOutput(`Apply registration failed: ${err.message}`)
        );
      });
    }

    const prealignBtn = document.getElementById('prealignToMniButton');
    if (prealignBtn) {
      prealignBtn.addEventListener('click', () => {
        this.prealignToMni160().catch(
          err => this.updateOutput(`Prealign failed: ${err.message}`)
        );
      });
    }

    const runFullBtn = document.getElementById('runFullPipelineButton');
    if (runFullBtn) {
      runFullBtn.addEventListener('click', () => {
        this.runFullPipeline().catch(
          err => this.updateOutput(`Full pipeline failed: ${err.message}`)
        );
      });
    }

    // Phase 14: cancel button terminates the worker. The executor's
    // cancel() rejects pending restores + clears running-step state and
    // surfaces a 'Cancelled' status. Disabled state is driven from
    // handleWorkerProgress / handleStepComplete.
    const cancelBtn = document.getElementById('cancelButton');
    if (cancelBtn) {
      cancelBtn.disabled = true;
      cancelBtn.addEventListener('click', () => {
        try { this.executor.cancel(); }
        catch (err) { this.updateOutput(`Cancel failed: ${err.message}`); }
      });
    }

    const clearBtn = document.getElementById('clearResultsButton');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => this.clearResults({ full: false }));
    }

    const runFcBtn = document.getElementById('computeNetworkMapButton');
    if (runFcBtn) {
      runFcBtn.addEventListener('click', () => {
        this.runFcNetworkMap().catch(
          err => this.updateOutput(`Network map failed: ${err.message}`)
        );
      });
    }
    const downloadFcBtn = document.getElementById('downloadNetworkMapButton');
    if (downloadFcBtn) {
      downloadFcBtn.disabled = true;
      downloadFcBtn.addEventListener('click', () => this.downloadNetworkMap());
    }

    const thresholdValue = document.getElementById('networkThresholdValue');
    const thresholdMode = document.getElementById('networkThresholdMode');
    const thresholdSym = document.getElementById('networkThresholdSymmetric');
    const thresholdMinCluster = document.getElementById('networkThresholdMinCluster');
    const thresholdValueLabel = document.getElementById('networkThresholdValueLabel');
    const updateThresholdLabel = () => {
      if (!thresholdValueLabel || !thresholdValue) return;
      const mode = thresholdMode ? thresholdMode.value : 'absolute';
      const v = Number(thresholdValue.value);
      thresholdValueLabel.textContent = mode === 'percentile'
        ? `${v.toFixed(0)}%`
        : v.toFixed(2);
    };
    const triggerRecompute = () => {
      updateThresholdLabel();
      if (this.networkMapData) {
        try { this.applyNetworkThreshold(); }
        catch (err) { this.updateOutput(`Threshold failed: ${err.message}`); }
      }
    };
    if (thresholdMode) {
      thresholdMode.addEventListener('change', () => {
        // Re-tune slider range to match the chosen mode.
        if (thresholdValue) {
          if (thresholdMode.value === 'percentile') {
            thresholdValue.min = '0';
            thresholdValue.max = '100';
            thresholdValue.step = '1';
            thresholdValue.value = '95';
          } else {
            thresholdValue.min = '0';
            thresholdValue.max = '1';
            thresholdValue.step = '0.01';
            thresholdValue.value = '0.5';
          }
        }
        triggerRecompute();
      });
    }
    if (thresholdValue) thresholdValue.addEventListener('input', triggerRecompute);
    if (thresholdSym) thresholdSym.addEventListener('change', triggerRecompute);
    if (thresholdMinCluster) thresholdMinCluster.addEventListener('change', triggerRecompute);
    updateThresholdLabel();

    const downloadThreshBtn = document.getElementById('downloadThresholdedNetworkMapButton');
    if (downloadThreshBtn) {
      downloadThreshBtn.disabled = true;
      downloadThreshBtn.addEventListener('click', () => this.downloadThresholdedNetworkMap());
    }

    const copyConsole = document.getElementById('copyConsole');
    if (copyConsole) copyConsole.addEventListener('click', () => this.console.copyToClipboard());

    const clearConsole = document.getElementById('clearConsole');
    if (clearConsole) clearConsole.addEventListener('click', () => this.console.clear());

    document.querySelectorAll('.view-tab[data-view]').forEach(button => {
      button.addEventListener('click', () => {
        document.querySelectorAll('.view-tab[data-view]').forEach(tab => tab.classList.remove('active'));
        button.classList.add('active');
        this.viewerController.setViewType(button.dataset.view);
      });
    });

    const overlayOpacity = document.getElementById('overlayOpacity');
    if (overlayOpacity) {
      overlayOpacity.addEventListener('input', (event) => {
        const value = parseFloat(event.target.value);
        this.viewerController.setOverlayOpacity(value);
        const display = document.getElementById('overlayOpacityValue');
        if (display) display.textContent = `${Math.round(value * 100)}%`;
      });
    }

    const interpolation = document.getElementById('interpolation');
    if (interpolation) {
      interpolation.addEventListener('change', (event) => {
        this.nv.setInterpolation(!event.target.checked);
        this.nv.drawScene();
      });
    }

    const colorbarToggle = document.getElementById('colorbarToggle');
    if (colorbarToggle) {
      colorbarToggle.addEventListener('change', (event) => {
        this.nv.opts.isColorbar = event.target.checked;
        this.nv.drawScene();
      });
    }

    const crosshairToggle = document.getElementById('crosshairToggle');
    if (crosshairToggle) {
      crosshairToggle.addEventListener('change', (event) => {
        this.nv.setCrosshairWidth(event.target.checked ? 1 : 0);
      });
    }

    const colormapSelect = document.getElementById('colormapSelect');
    if (colormapSelect) {
      colormapSelect.addEventListener('change', (event) => {
        if (this.nv.volumes?.[0]) {
          this.nv.volumes[0].colormap = event.target.value;
          this.nv.updateGLVolume();
        }
      });
    }

    this.bindModalButton('aboutButton', this.aboutModal);
    this.bindModalButton('privacyButton', this.privacyModal);
    this.bindModalButton('citationsButton', this.citationsModal);
    this.bindCloseButton('closeAbout', this.aboutModal);
    this.bindCloseButton('closePrivacy', this.privacyModal);
    this.bindCloseButton('closeCitations', this.citationsModal);
  }

  bindModalButton(buttonId, modal) {
    const button = document.getElementById(buttonId);
    if (button) button.addEventListener('click', () => modal.open());
  }

  bindCloseButton(buttonId, modal) {
    const button = document.getElementById(buttonId);
    if (button) button.addEventListener('click', () => modal.close());
  }

  populatePipelineSelect() {
    const pipelineSelect = document.getElementById('pipelineSelect');
    if (!pipelineSelect) return;

    pipelineSelect.innerHTML = '';
    // Phase 13: surface every fully-runnable pipeline (every required
    // stage's module is implemented + has its asset). The 'Run full
    // pipeline' button auto-detects manual-mask vs auto-T1 input
    // regardless of selection; the dropdown is informational for now.
    const runnable = LNM_PIPELINES.filter(isPipelineRunnable);
    for (const pipeline of runnable) {
      const option = document.createElement('option');
      option.value = pipeline.id;
      option.textContent = pipeline.displayName;
      pipelineSelect.appendChild(option);
    }
    pipelineSelect.value = this.selectedPipeline?.id || 'lnm-yeo-only';
  }

  // Phase 13 + Phase 40: populate every visible version slot from
  // Config.VERSION. Best-effort augment with build-info.json (written
  // by web/run.sh for local dev + by .github/workflows/ for deploys)
  // to surface the commit SHA / branch / dirty flag.
  //
  //   local dev    -> "v0.17.0 (abc1234, main, dirty)"
  //   staging      -> "v0.17.0-staging+abc1234"  (sed'd by deploy-pages.yml)
  //                   plus "(abc1234, main)" from build-info.json
  //   production   -> "v0.17.0" (release-tag build; build-info.json may
  //                   carry the tag SHA)
  //
  // build-info.json is fetched best-effort — a 404 falls back to just
  // VERSION, so the static deploy works whether or not the file exists.
  async populateVersionLabel() {
    let label = Config.VERSION ? `v${Config.VERSION}` : '';
    try {
      const r = await fetch('build-info.json', { cache: 'no-store' });
      if (r.ok) {
        const info = await r.json();
        const bits = [];
        if (info.sha) bits.push(info.sha);
        if (info.branch && info.branch !== 'main') bits.push(info.branch);
        if (info.dirty) bits.push('dirty');
        if (bits.length) label += ` (${bits.join(', ')})`;
      }
    } catch (e) { /* best-effort: silent fallback to VERSION */ }
    const ids = ['aboutAppVersion', 'appVersion', 'footerVersion'];
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el) el.textContent = label;
    }
  }

  async setStructural(file) {
    if (!file) return;
    this.structuralFile = file;
    await this.viewerController.loadBaseVolume(file, { stage: 'structural' });
    this.updateOutput(`Structural image ready: ${file.name}`);
    // Phase 31: auto-promote the pipeline selection. A structural T1
    // means the user wants the full auto chain.
    this._autoPromotePipeline('lnm-yeo-auto');
    // Auto-run brain extraction on every structural drop. The button under
    // #stepLesionSection re-triggers it on demand. Any error is swallowed
    // into the console; the rest of the manual-mask flow continues to work.
    this.runBrainExtraction().catch(
      err => this.updateOutput(`Brain extraction failed: ${err.message}`)
    );
  }

  async setLesion(file) {
    if (!file) return;
    this.lesionFile = file;
    if (this.structuralFile) {
      await this.viewerController.loadOverlay(file, 'red', 0.5, { stage: 'lesion' });
    } else {
      await this.viewerController.loadBaseVolume(file, { stage: 'lesion' });
    }
    this.updateOutput(`Lesion mask ready: ${file.name}`);
    // Phase 31: a manual lesion mask without a structural T1 means the
    // user wants the manual-mask network-map chain (overlap + FC +
    // threshold). With a structural already loaded we leave the auto
    // pipeline selected.
    if (!this.structuralFile) {
      this._autoPromotePipeline('lnm-network-map');
    }
  }

  // Phase 31: promote the dropdown selection if the user hasn't manually
  // overridden it. Keeps the dropdown a source-of-truth when the user
  // cares, but defaults to the right full-chain pipeline based on what
  // they dropped first.
  _autoPromotePipeline(pipelineId) {
    if (this._userPickedPipeline) return;
    const pipeline = getPipelineById(pipelineId);
    if (!pipeline) return;
    this.selectedPipeline = pipeline;
    const pipelineSelect = document.getElementById('pipelineSelect');
    if (pipelineSelect) pipelineSelect.value = pipelineId;
  }

  async runYeoOverlap() {
    if (!this.lesionFile) {
      this.updateOutput('Drop a lesion mask before computing overlap.');
      return;
    }
    this.updateOutput('Loading Yeo7 atlas...');
    const atlas = await loadAtlasFromManifest('yeo7-2mm');
    this.updateOutput('Decoding lesion mask...');
    const lesionBuf = await this.lesionFile.arrayBuffer();
    const lesion = await decodeNiftiBuffer(lesionBuf);

    if (!dimsEqual(lesion.dims, atlas.dims)) {
      this.updateOutput(`Lesion dims ${lesion.dims.join('x')} do not match atlas ${atlas.dims.join('x')}. Re-register the mask to ${atlas.manifestEntry.mniSpace || 'MNI152 2mm'} first.`);
      return;
    }

    const lesionBin = binarise(lesion.data);
    const parcelResult = computeParcelOverlap({
      lesion: lesionBin,
      atlas: atlas.data,
      dims: atlas.dims,
    });
    const summary = summarizeNetworkOverlap(parcelResult, atlas.networkLabels);
    const networkSizes = computeNetworkSizes(atlas.data, atlas.networkLabels);
    this.overlapResult = { parcelResult, summary, atlas, networkSizes };
    this.showOutsideAtlasWarning(parcelResult.voxelsOutsideAtlas, parcelResult.totalLesionVoxels);

    const tableEl = document.getElementById('networkOverlapTable');
    if (tableEl) {
      renderOverlapTable(tableEl, summary, {
        colormap: YEO7_COLORMAP,
        networkLabels: atlas.networkLabels
      });
    }
    const csvButton = document.getElementById('downloadOverlapCsv');
    if (csvButton) csvButton.disabled = false;

    this.updateOutput(
      `Overlap computed for ${summary.networks.length} networks ` +
      `(${parcelResult.voxelsOutsideAtlas} lesion voxels outside atlas).`
    );
  }

  // ---- Phase 2a.1.4b: brain extraction wiring ----

  async ensureManifest() {
    if (this.manifest) return this.manifest;
    const response = await fetch('./models/manifest.json');
    if (!response.ok) {
      throw new Error(`Failed to load manifest: HTTP ${response.status}`);
    }
    this.manifest = await response.json();
    return this.manifest;
  }

  async runBrainExtraction() {
    if (!this.structuralFile) {
      this.updateOutput('Drop a structural image first.');
      return;
    }
    const manifest = await this.ensureManifest();
    const entry = manifest.modelAssets?.find(a => a.id === 'lnm-synthstrip');
    if (!entry) throw new Error("Manifest is missing the 'lnm-synthstrip' model asset.");
    if (entry.supportStatus !== 'supported') {
      throw new Error(`'lnm-synthstrip' is ${entry.supportStatus}; cannot run brain extraction.`);
    }
    const { base, name } = splitModelUrl(entry.sourceUrl);

    this.updateOutput('Starting SynthStrip brain extraction...');
    const inputBuffer = await this.structuralFile.arrayBuffer();
    await this.executor.loadVolume(inputBuffer);
    await this.executor.runSynthStrip({
      modelAssetId: entry.id,
      modelName: name || 'synthstrip.onnx',
      modelBaseUrl: base,
      cacheKey: entry.cacheKey,
      // SynthStrip 'fast' mode caps the resample target at 2mm (instead of
      // 1mm). For 1-2mm inputs this is a no-op resample and brings the
      // conformed inference volume down to ~7-8M voxels — within the WASM
      // 4GB heap. 1mm-mode produces ~12-15M voxels and reliably ORT-OOMs
      // in headless Chromium; the higher-quality path is a future option.
      fast: true,
      dilate: false
    });
  }

  handleWorkerProgress(frac, label) {
    if (!this.progress) return;
    this.progress.setProgress(frac, label);
    // Phase 14: enable the cancel button while the worker is mid-run.
    const cancelBtn = document.getElementById('cancelButton');
    if (cancelBtn) {
      cancelBtn.disabled = !(typeof frac === 'number' && frac >= 0 && frac < 1);
    }
  }

  handleStepComplete(step) {
    this.updateOutput(`Worker step '${step}' complete.`);
    // Phase 14: a completed step ends the cancellable window.
    const cancelBtn = document.getElementById('cancelButton');
    if (cancelBtn) cancelBtn.disabled = true;
  }

  handleStageData(data) {
    if (!data || !data.stage) return;
    if (data.stage === 'brainmask' && data.niftiData) {
      const file = arrayBufferToFile(data.niftiData, 'brainmask.nii');
      this.brainmaskFile = file;
      // Render as a translucent overlay over the structural; if the user
      // dropped a lesion manually before the structural, the lesion stage
      // re-renders normally on its next setLesion() call.
      if (this.structuralFile) {
        this.viewerController
          .loadOverlay(file, 'green', 0.4, { stage: 'brainmask' })
          .catch(err => this.updateOutput(`Brain mask render error: ${err.message}`));
      }
      const btn = document.getElementById('downloadBrainMaskButton');
      if (btn) btn.disabled = false;
      this.updateOutput('Brain mask ready.');
      return;
    }
    if (data.stage === 'segmentation' && data.niftiData) {
      const file = arrayBufferToFile(data.niftiData, 'lesion.nii');
      this.lesionMaskFile = file;
      if (this.structuralFile) {
        this.viewerController
          .loadOverlay(file, 'red', 0.5, { stage: 'segmentation' })
          .catch(err => this.updateOutput(`Lesion mask render error: ${err.message}`));
      }
      const btn = document.getElementById('downloadLesionMaskButton');
      if (btn) btn.disabled = false;
      this.updateOutput('Lesion segmentation ready.');
      return;
    }
    // Phase 6.2: warp-mask emits the lesion warped onto MNI160 1mm. The
    // stage data is the NIfTI ArrayBuffer; applyRegistrationToLesion()
    // awaits it via a one-shot resolver before resampling onto the Yeo grid.
    if (data.stage === 'mni-lesion' && data.niftiData) {
      this.mniLesionFile = arrayBufferToFile(data.niftiData, 'lesion-mni1mm.nii');
      this.updateOutput('Lesion warped to MNI160 1mm.');
      if (this._mniLesionResolver) {
        const r = this._mniLesionResolver;
        this._mniLesionResolver = null;
        r.resolve(data.niftiData);
      }
    }
  }

  downloadBrainMask() {
    if (!this.brainmaskFile) {
      this.updateOutput('No brain mask available yet.');
      return;
    }
    const url = URL.createObjectURL(this.brainmaskFile);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'lnm-brainmask.nii';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  // Phase 2a.2.3: lesion-segmentation entry point. Reads the manifest
  // entry for 'lnm-stroke-lesion', dispatches the SCT-derived
  // run-inference op (see web/js/inference-worker.js stepInference). The
  // worker fetches + runs the SynthStroke baseline ONNX, applies the
  // sliding-window pipeline, and emits a 'segmentation' stageData NIfTI.
  async runLesionSegmentation() {
    if (!this.structuralFile) {
      this.updateOutput('Drop a structural image first.');
      return;
    }
    const manifest = await this.ensureManifest();
    const entry = manifest.modelAssets?.find(a => a.id === 'lnm-stroke-lesion');
    if (!entry) throw new Error("Manifest is missing the 'lnm-stroke-lesion' model asset.");
    if (entry.supportStatus !== 'supported') {
      throw new Error(`'lnm-stroke-lesion' is ${entry.supportStatus}; cannot run lesion segmentation.`);
    }
    const { base, name } = splitModelUrl(entry.sourceUrl);

    this.updateOutput('Starting lesion segmentation...');
    const inputBuffer = await this.structuralFile.arrayBuffer();
    await this.executor.loadVolume(inputBuffer);
    await this.executor.runInference({
      taskId: 'lnm-segment-only',
      modelAssetId: entry.id,
      modelName: name || 'lnm-stroke-lesion.onnx',
      modelBaseUrl: base,
      cacheKey: entry.cacheKey,
      supportStatus: entry.supportStatus,
      patchSize: entry.patchSize || [128, 128, 128],
      threshold: entry.probabilityThreshold ?? 0.4,
      minComponentSize: entry.minComponentSize ?? 30,
      preprocessing: entry.preprocessing || {},
      // Sliding-window overlap and TTA defaults per the locked Phase 2a.2
      // plan: lighter than nnU-Net's 0.5 + 8-axis to keep browser inference
      // bounded; quality toggle is a future polish item.
      overlap: 0.25,
      testTimeAugmentation: false
    });
  }

  downloadLesionMask() {
    if (!this.lesionMaskFile) {
      this.updateOutput('No lesion segmentation available yet.');
      return;
    }
    const url = URL.createObjectURL(this.lesionMaskFile);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'lnm-lesion.nii';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  // Phase 4.4: lesion network map via Yeo7 group-FC weighted sum. Pure
  // main-thread (no worker) — the math is just a per-voxel linear combo
  // of seven precomputed t-maps. Requires runYeoOverlap to have run first
  // (we read the network-overlap result for the weights).
  async runFcNetworkMap() {
    if (!this.overlapResult) {
      this.updateOutput('Run "Compute overlap" first to get network weights.');
      return;
    }
    this.updateOutput('Loading Yeo7 group-FC pack...');
    // Phase 37: surface download progress for the heavy FC pack
    // (~30 MB cold; cache hit is instant). The callback is throttled
    // to one progress message per ~512 KB to avoid spamming the
    // console + status bar.
    let lastTick = 0;
    const { arrayBuffer, index, manifestEntry } =
      await loadConnectomeFromManifest('yeo7-fc-pack', {
        onProgress: ({ received, total, label }) => {
          if (received - lastTick < 512 * 1024 && received !== total) return;
          lastTick = received;
          const mb = (received / 1048576).toFixed(1);
          if (total) {
            const totalMb = (total / 1048576).toFixed(0);
            const pct = Math.round((received / total) * 100);
            this.handleWorkerProgress(pct / 100, `Downloading ${label} (${mb}/${totalMb} MB)`);
          } else {
            this.updateOutput(`Downloading ${label}: ${mb} MB`);
          }
        }
      });
    const pack = decodeFcPack(arrayBuffer, index);

    const NETWORK_ORDER = [
      'Visual', 'Somatomotor', 'DorsalAttention', 'VentralAttention',
      'Limbic', 'Frontoparietal', 'Default'
    ];
    const weights = summaryToNetworkWeights(this.overlapResult.summary, NETWORK_ORDER);
    const dims = index.shape.slice(1);   // shape = [7, X, Y, Z]
    this.updateOutput(
      `Computing network map: weights=[${
        Array.from(weights).map(w => w.toFixed(2)).join(', ')
      }]`
    );
    const fcMap = fcWeightedSum(weights, pack.tMaps, dims);

    // Stash for Phase 5 re-thresholding without recomputing the FC sum.
    this.networkMapData = fcMap;
    this.networkMapDims = dims;
    const spacingMm = manifestEntry.atlasResolutionMm || 2;
    this.networkMapSpacing = [spacingMm, spacingMm, spacingMm];

    // Wrap as a NIfTI for download / overlay. The Yeo atlas's spacing /
    // affine is the canonical pose for the FC pack — manifestEntry from
    // the connectome carries atlasResolutionMm; the Yeo atlas itself is
    // the same grid (99x117x95 2mm).
    const niftiBuffer = writeNifti1(fcMap, {
      dims,
      spacing: this.networkMapSpacing,
      description: 'LNM Yeo7 FC weighted sum'
    });
    this.networkMapFile = arrayBufferToFile(niftiBuffer, 'lnm-network-map.nii');

    // Render as overlay on the structural / lesion view, blue-red diverging.
    if (this.structuralFile || this.lesionFile) {
      this.viewerController
        .loadOverlay(this.networkMapFile, 'redyell', 0.5, { stage: 'network-map' })
        .catch(err => this.updateOutput(`Network-map render error: ${err.message}`));
    }
    const dlBtn = document.getElementById('downloadNetworkMapButton');
    if (dlBtn) dlBtn.disabled = false;

    // Quick stats: range + voxels above |t| > 5 (rough significance bar).
    let mn = Infinity, mx = -Infinity, above = 0;
    for (let i = 0; i < fcMap.length; i++) {
      const v = fcMap[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      if (Math.abs(v) > 5) above += 1;
    }
    this.updateOutput(
      `Network map ready: t-range [${mn.toFixed(1)}, ${mx.toFixed(1)}], ` +
      `${above.toLocaleString()} voxels with |t|>5.`
    );
  }

  downloadNetworkMap() {
    if (!this.networkMapFile) {
      this.updateOutput('No network map available yet.');
      return;
    }
    const url = URL.createObjectURL(this.networkMapFile);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'lnm-network-map.nii';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  // Phase 5: re-threshold the cached network map and update the
  // thresholded-mask download. The 'live' overlay update for the slider is
  // best done via NiiVue cal_min/cal_max (no data swap), but this method
  // is the source of truth for the *binary* thresholded mask used by the
  // download button + the parity tests.
  //
  // Reads the threshold UI controls:
  //   #networkThresholdValue   range slider, 0..100 (% for percentile;
  //                            t-stat for absolute, scaled by /10).
  //   #networkThresholdMode    select: 'absolute' | 'percentile'.
  //   #networkThresholdSymmetric  checkbox.
  //   #networkThresholdMinCluster number input.
  applyNetworkThreshold() {
    if (!this.networkMapData) {
      this.updateOutput('Compute the network map first.');
      return null;
    }
    const valueEl = document.getElementById('networkThresholdValue');
    const modeEl = document.getElementById('networkThresholdMode');
    const symEl = document.getElementById('networkThresholdSymmetric');
    const minClEl = document.getElementById('networkThresholdMinCluster');
    const rawValue = valueEl ? Number(valueEl.value) : 0;
    const mode = modeEl ? modeEl.value : 'absolute';
    const symmetric = symEl ? !!symEl.checked : false;
    const minClusterVoxels = minClEl ? Number(minClEl.value) || 0 : 0;
    // For the slider: 'percentile' mode interprets [0..100] as a percentile;
    // 'absolute' mode interprets the slider value directly as a t-stat.
    const value = mode === 'percentile' ? rawValue / 100 : rawValue;

    const mask = applyThreshold(this.networkMapData, this.networkMapDims, {
      mode, value, symmetric, minClusterVoxels
    });
    let count = 0;
    for (let i = 0; i < mask.length; i++) count += mask[i];
    const niftiBuffer = writeNifti1(mask, {
      dims: this.networkMapDims,
      spacing: this.networkMapSpacing,
      description: `LNM thresholded ${mode}=${value} sym=${symmetric} cluster>=${minClusterVoxels}`
    });
    this.thresholdedMaskFile = arrayBufferToFile(niftiBuffer, 'lnm-network-map-thresh.nii');
    const dlBtn = document.getElementById('downloadThresholdedNetworkMapButton');
    if (dlBtn) dlBtn.disabled = false;
    const summaryEl = document.getElementById('networkThresholdSummary');
    if (summaryEl) {
      summaryEl.textContent =
        `${count.toLocaleString()} voxels survive ${mode}` +
        (symmetric ? ' (|t|)' : ' (t)') +
        ` > ${value}` +
        (minClusterVoxels > 1 ? ` + cluster≥${minClusterVoxels}` : '');
    }
    return mask;
  }

  downloadThresholdedNetworkMap() {
    if (!this.thresholdedMaskFile) {
      // Try to (re)compute first.
      this.applyNetworkThreshold();
      if (!this.thresholdedMaskFile) return;
    }
    const url = URL.createObjectURL(this.thresholdedMaskFile);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'lnm-network-map-thresh.nii';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  // Phase 3.4: SynthMorph MNI registration. Looks up the model + reference
  // in the manifest, posts to the worker. The worker stashes the integrated
  // displacement field on its state for the lnm-yeo-auto bridge (Phase 3.5)
  // to apply via runWarpMask.
  async runRegistration() {
    if (!this.structuralFile) {
      this.updateOutput('Drop a structural image first.');
      return;
    }
    const manifest = await this.ensureManifest();
    const model = manifest.modelAssets?.find(a => a.id === 'lnm-synthmorph-mni');
    const ref = manifest.atlasAssets?.find(a => a.id === 'lnm-mni160');
    if (!model || model.supportStatus !== 'supported') {
      throw new Error("Manifest entry 'lnm-synthmorph-mni' is not supported.");
    }
    if (!ref || ref.supportStatus !== 'supported') {
      throw new Error("Manifest entry 'lnm-mni160' is not supported.");
    }
    const m = splitModelUrl(model.sourceUrl);
    const modelFileName = (model.filename || m.name || 'lnm-synthmorph-mni.onnx').split('/').pop();
    const modelLocalUrl = new URL(`models/_dev_cache/${modelFileName}`, window.location.href).href;

    this.updateOutput('Starting MNI registration (SynthMorph deformable)...');
    const inputBuffer = await this.structuralFile.arrayBuffer();
    await this.executor.loadVolume(inputBuffer);
    await this.executor.runRegistration({
      modelAssetId: model.id,
      modelName: m.name || modelFileName,
      modelBaseUrl: m.base,
      modelCacheKey: model.cacheKey,
      modelLocalUrl,
      modelInputDims: model.browserRuntime?.inputDims || model.inputShape?.slice(1, 4),
      svfDims: model.browserRuntime?.svfDims || model.svfShape?.slice(1, 4),
      referenceAssetId: ref.id,
      referenceUrl: ref.sourceUrl,
      referenceCacheKey: ref.cacheKey,
      nbSteps: 7
    });
  }

  // Phase 16.2: in-browser affine pre-registration (centroid match) so
  // arbitrary clinical T1s can flow into the SynthMorph deformable
  // stage, which hard-requires 160x160x192 1mm input. Runs SynthStrip
  // first if no brainmask is present, computes the brain centroid in
  // source world coords, then resamples the T1 + brainmask onto the
  // MNI160 1mm grid with the centroid placed at MNI voxel (80, 80, 96).
  // The lesion seg + lesion file are cleared because they were
  // computed in the old space; user re-runs them on the aligned grid.
  async prealignToMni160({ skipIfAligned = false } = {}) {
    if (!this.structuralFile) {
      this.updateOutput('Drop a structural T1 first.');
      return;
    }

    // Phase 34: idempotent fast-path. If the structural is already at
    // exactly the SynthMorph-required pose (160x160x192 1mm) and 1mm
    // isotropic, prealign has nothing to do — used by runFullPipeline's
    // auto chain so users with already-aligned T1s don't pay the
    // cost. Probe via a header decode (cheap relative to a resample).
    if (skipIfAligned) {
      const probeBuf = await this.structuralFile.arrayBuffer();
      const probe = await decodeNiftiBuffer(probeBuf);
      const isAligned =
        probe.dims[0] === 160 && probe.dims[1] === 160 && probe.dims[2] === 192;
      if (isAligned) {
        this.updateOutput('Structural already at 160x160x192, skipping prealign.');
        return;
      }
    }

    if (!this.brainmaskFile) {
      this.updateOutput('Running brain extraction (prealign needs the brain mask)...');
      await this.runBrainExtraction();
      if (!this.brainmaskFile) {
        throw new Error('Brain extraction did not produce a mask; prealign aborted.');
      }
    }

    this.updateOutput('Decoding T1 + brainmask for prealign...');
    const t1Buf = await this.structuralFile.arrayBuffer();
    const t1 = await decodeNiftiBuffer(t1Buf);
    const t1Affine = affineFromHeader(t1.header);

    const maskBuf = await this.brainmaskFile.arrayBuffer();
    const mask = await decodeNiftiBuffer(maskBuf);
    if (!dimsEqual(mask.dims, t1.dims)) {
      throw new Error(
        `prealign: brain mask dims ${mask.dims.join('x')} != T1 dims ${t1.dims.join('x')}`
      );
    }

    // PCA principal-axis alignment (Phase 26): rotates the brain so its
    // principal axes line up with MNI canonical axes, plus the centroid
    // translation. For nearly-isotropic brains the rotation is small;
    // for clinical T1s acquired off-axis it can be substantial.
    const { dstAffine, mniDims, eigenvalues } = principalAxisAlign(
      mask.data, t1.dims, t1Affine
    );
    const cVox = centroidOfMask(mask.data, t1.dims);
    const cWorld = applyAffineToVoxel(t1Affine, cVox);
    this.updateOutput(
      `Prealign (PCA): centroid src voxel (${cVox.map(v => v.toFixed(1)).join(', ')}) ` +
      `-> world (${cWorld.map(v => v.toFixed(1)).join(', ')}) mm; ` +
      `eigenvalues=[${eigenvalues.map(e => e.toFixed(1)).join(', ')}].`
    );

    // Resample T1 (trilinear) and brainmask (nearest, binary).
    const t1Resampled = resampleAffine(
      t1.data, t1.dims, t1Affine, mniDims, dstAffine, 'trilinear'
    );
    const maskResampled = resampleAffine(
      mask.data, t1.dims, t1Affine, mniDims, dstAffine, 'nearest'
    );
    const maskBin = new Uint8Array(maskResampled.length);
    for (let i = 0; i < maskResampled.length; i++) maskBin[i] = maskResampled[i] > 0.5 ? 1 : 0;

    const flatAff = [
      dstAffine[0][0], dstAffine[0][1], dstAffine[0][2], dstAffine[0][3],
      dstAffine[1][0], dstAffine[1][1], dstAffine[1][2], dstAffine[1][3],
      dstAffine[2][0], dstAffine[2][1], dstAffine[2][2], dstAffine[2][3]
    ];
    const t1Nifti = writeNifti1(t1Resampled, {
      dims: mniDims, spacing: [1, 1, 1], affine: flatAff,
      description: 'LNM prealign: centroid match to MNI160 1mm'
    });
    const maskNifti = writeNifti1(maskBin, {
      dims: mniDims, spacing: [1, 1, 1], affine: flatAff,
      description: 'LNM prealign brainmask resampled to MNI160 1mm'
    });

    this.structuralFile = arrayBufferToFile(t1Nifti, 'lnm-prealign-t1.nii');
    this.brainmaskFile = arrayBufferToFile(maskNifti, 'lnm-prealign-brainmask.nii');
    // Stale results from the pre-prealign space.
    this.lesionMaskFile = null;
    this.lesionFile = null;
    this.mniLesionFile = null;
    this.overlapResult = null;
    this.networkMapFile = null;
    this.networkMapData = null;
    this.thresholdedMaskFile = null;

    // Refresh viewer with the aligned T1 + brainmask overlay.
    await this.viewerController.loadBaseVolume(this.structuralFile, { stage: 'structural' });
    try {
      await this.viewerController.loadOverlay(this.brainmaskFile, 'green', 0.4, { stage: 'brainmask' });
    } catch (err) {
      // Non-fatal: the overlay is cosmetic.
      this.updateOutput(`Brainmask overlay re-render warning: ${err.message}`);
    }
    this.updateOutput('Prealign complete: T1 + brainmask resampled to 160x160x192 1mm.');
    return this.structuralFile;
  }

  // Phase 6.2: bridge Register -> Yeo overlap. Decodes the segmentation NIfTI
  // produced by runLesionSegmentation, hands the F-order Uint8 voxels to the
  // worker (which applies the integrated displacement field stashed by
  // runRegistration), then resamples the warped 1mm output onto the Yeo7
  // atlas grid via affine resample. Sets `this.lesionFile` so a follow-up
  // runYeoOverlap()/runFcNetworkMap()/applyNetworkThreshold() chain runs
  // unmodified.
  async applyRegistrationToLesion() {
    if (!this.lesionMaskFile) {
      this.updateOutput('Run lesion segmentation first.');
      return null;
    }
    this.updateOutput('Decoding lesion mask for warp...');
    const lesionBuf = await this.lesionMaskFile.arrayBuffer();
    const decoded = await decodeNiftiBuffer(lesionBuf);
    if (decoded.dims[0] !== 160 || decoded.dims[1] !== 160 || decoded.dims[2] !== 192) {
      throw new Error(
        `applyRegistrationToLesion: expected 160x160x192 lesion mask; got ${decoded.dims.join('x')}. ` +
        `Run registration on a 160x160x192 1mm structural first.`
      );
    }
    const maskU8 = new Uint8Array(decoded.data.length);
    for (let i = 0; i < decoded.data.length; i++) maskU8[i] = decoded.data[i] > 0 ? 1 : 0;
    // Copy to a transferable ArrayBuffer (worker takes ownership).
    const transferBuf = maskU8.buffer.slice(0);

    this.mniLesionFile = null;
    const mniLesionPromise = new Promise((resolve, reject) => {
      this._mniLesionResolver = { resolve, reject };
    });
    await this.executor.runWarpMask({
      maskBuffer: transferBuf,
      maskDims: [160, 160, 192]
    });
    const mniLesionBuf = await mniLesionPromise;

    // Resample the 1mm warped lesion onto the Yeo atlas grid.
    this.updateOutput('Resampling warped lesion onto Yeo atlas grid...');
    const warped = await decodeNiftiBuffer(mniLesionBuf);
    const warpedAffine = affineFromHeader(warped.header);
    const atlas = await loadAtlasFromManifest('yeo7-2mm');
    const atlasAffine = affineFromHeader(atlas.header);
    const warpedU8 = warped.data instanceof Uint8Array
      ? warped.data
      : binarise(warped.data);
    const yeoMask = resampleAffine(
      warpedU8,
      warped.dims, warpedAffine,
      atlas.dims, atlasAffine,
      'nearest'
    );

    // Wrap the Yeo-grid mask as a NIfTI and adopt it as the lesion file.
    const flatAffine = [
      atlasAffine[0][0], atlasAffine[0][1], atlasAffine[0][2], atlasAffine[0][3],
      atlasAffine[1][0], atlasAffine[1][1], atlasAffine[1][2], atlasAffine[1][3],
      atlasAffine[2][0], atlasAffine[2][1], atlasAffine[2][2], atlasAffine[2][3]
    ];
    const yeoNifti = writeNifti1(yeoMask, {
      dims: atlas.dims,
      spacing: [2, 2, 2],
      affine: flatAffine,
      description: 'LNM lesion warped + resampled to MNI Yeo grid'
    });
    const yeoFile = arrayBufferToFile(yeoNifti, 'lnm-lesion-yeo.nii');
    await this.setLesion(yeoFile);
    this.updateOutput('Lesion ready on Yeo grid.');
    return yeoFile;
  }

  // Phase 15: pipeline-driven runFullPipeline. Iterates the selected
  // pipeline's stages and dispatches each via _runStage. The auto-detect
  // shortcut for manually-dropped Yeo-grid lesion masks survives as a
  // precondition gate when the pipeline starts with parcel-overlap.
  //
  // Stages declared as required:false are still run when their module is
  // implemented (e.g. threshold runs with the pipeline's `defaults` if
  // present, falling back to whatever the UI has set).
  async runFullPipeline() {
    const pipeline = this.selectedPipeline;
    if (!pipeline || !Array.isArray(pipeline.stages) || pipeline.stages.length === 0) {
      this.updateOutput('No pipeline selected.');
      return;
    }
    this.updateOutput(`=== Running ${pipeline.displayName} ===`);

    // Precondition gate based on the first stage's input expectation.
    const firstModule = pipeline.stages[0]?.module;
    if (firstModule === 'parcel-overlap') {
      // Manual-mask path — needs a Yeo-grid lesion already loaded.
      if (!this.lesionFile) {
        this.updateOutput('Drop a Yeo-grid lesion mask first.');
        return;
      }
      const onYeoGrid = await this._lesionFileMatchesYeoGrid();
      if (!onYeoGrid) {
        this.updateOutput(
          'Lesion mask is not on the Yeo7 grid (99x117x95). Use a pipeline ' +
          'that includes registration, or pre-register the mask externally.'
        );
        return;
      }
    } else if (firstModule === 'brain-extraction' || firstModule === 'inference-pipeline') {
      // Auto path — needs a structural T1.
      if (!this.structuralFile) {
        this.updateOutput('Drop a structural T1 first.');
        return;
      }
    }

    this._perfStats = [];
    this._perfRunStart = this._now();
    for (const stage of pipeline.stages) {
      const stageStart = this._now();
      try {
        await this._runStage(stage);
      } catch (err) {
        this.updateOutput(`Stage '${stage.id}' (${stage.module}) failed: ${err.message}`);
        return;
      }
      const elapsedMs = this._now() - stageStart;
      this._perfStats.push({ id: stage.id, module: stage.module, ms: elapsedMs });
      this.updateOutput(`[perf] ${stage.id} (${stage.module}): ${this._formatMs(elapsedMs)}`);
    }
    const totalMs = this._now() - this._perfRunStart;
    this.updateOutput(
      `=== Pipeline complete in ${this._formatMs(totalMs)} ` +
      `(${this._perfStats.length} stage${this._perfStats.length === 1 ? '' : 's'}) ===`
    );
  }

  // Phase 19: monotonic clock; falls back to Date.now in non-browser
  // environments (the contract test imports the module under Node, so
  // performance.now is always defined there too — but we keep the guard
  // for older runtimes / shims).
  _now() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  _formatMs(ms) {
    if (ms < 1000) return `${ms.toFixed(0)} ms`;
    if (ms < 60_000) return `${(ms / 1000).toFixed(2)} s`;
    return `${(ms / 60_000).toFixed(2)} min`;
  }

  // Phase 15: stage dispatch. Maps a pipeline stage's `module` to the
  // existing orchestrator method. Unknown modules throw so a manifest
  // typo surfaces immediately rather than silently skipping.
  async _runStage(stage) {
    if (!stage || !stage.module) {
      throw new Error('_runStage: stage must declare a module');
    }
    switch (stage.module) {
      case 'brain-extraction':
        if (this.brainmaskFile) {
          this.updateOutput('Brain mask already present, skipping brain-extraction.');
          return;
        }
        return this.runBrainExtraction();
      case 'prealign':
        // Phase 34: idempotent prealign. Skip if the structural is
        // already at the SynthMorph-required pose (160x160x192 1mm).
        // prealignToMni160() handles the dim probe + early return.
        return this.prealignToMni160({ skipIfAligned: true });
      case 'inference-pipeline':
        if (this.lesionMaskFile) {
          this.updateOutput('Lesion mask already present, skipping segmentation.');
          return;
        }
        return this.runLesionSegmentation();
      case 'registration':
        // The bridge (apply-warp + Yeo-grid resample) is the natural
        // companion of the SynthMorph registration step. We chain them
        // here so a pipeline doesn't have to declare the bridge as a
        // separate stage.
        await this.runRegistration();
        return this.applyRegistrationToLesion();
      case 'parcel-overlap':
        return this.runYeoOverlap();
      case 'fc-weighted-sum':
        return this.runFcNetworkMap();
      case 'threshold':
        // applyNetworkThreshold is sync and reads UI controls. If the
        // stage carries `defaults`, push them into the controls before
        // computing so the dropdown's choice is honoured even when the
        // user hasn't touched the threshold UI.
        this._applyThresholdDefaults(stage.defaults);
        this.applyNetworkThreshold();
        return;
      default:
        throw new Error(`_runStage: unknown module '${stage.module}'`);
    }
  }

  // Phase 15: copy the pipeline stage's threshold defaults into the live
  // UI controls so applyNetworkThreshold (which reads from the DOM) honours
  // them.  No-op if the controls are missing.
  _applyThresholdDefaults(defaults) {
    if (!defaults || typeof defaults !== 'object') return;
    const modeEl = document.getElementById('networkThresholdMode');
    const valueEl = document.getElementById('networkThresholdValue');
    const symEl = document.getElementById('networkThresholdSymmetric');
    const minClEl = document.getElementById('networkThresholdMinCluster');
    if (modeEl && defaults.mode) modeEl.value = defaults.mode;
    if (valueEl && typeof defaults.value === 'number') {
      // Slider stores the raw value; percentile mode interprets [0,100],
      // absolute mode interprets the value directly. Keep the manifest's
      // convention (percentile = 0..100) consistent.
      valueEl.value = String(defaults.value);
    }
    if (symEl && typeof defaults.symmetric === 'boolean') symEl.checked = defaults.symmetric;
    if (minClEl && typeof defaults.minClusterVoxels === 'number') {
      minClEl.value = String(defaults.minClusterVoxels);
    }
  }

  // Phase 21: clear all intermediate pipeline state so the user can start
  // a fresh run without reloading the page. Resets every result slot, the
  // viewer, and the threshold UI surface; preserves the structural file
  // (the user usually wants to re-run on the same input) unless `full`
  // is set.
  clearResults({ full = false } = {}) {
    this.overlapResult = null;
    this.brainmaskFile = null;
    this.lesionMaskFile = null;
    this.networkMapFile = null;
    this.networkMapData = null;
    this.networkMapDims = null;
    this.networkMapSpacing = null;
    this.thresholdedMaskFile = null;
    this.mniLesionFile = null;
    if (full) {
      this.structuralFile = null;
      this.lesionFile = null;
    }

    // Re-disable every download / threshold-output button so the UI matches
    // the cleared state.
    const buttonIds = [
      'downloadOverlapCsv',
      'downloadBrainMaskButton',
      'downloadLesionMaskButton',
      'downloadNetworkMapButton',
      'downloadThresholdedNetworkMapButton'
    ];
    for (const id of buttonIds) {
      const el = document.getElementById(id);
      if (el) el.disabled = true;
    }
    // Wipe the overlap table body if present.
    const tbody = document.querySelector('#networkOverlapTable tbody');
    if (tbody) tbody.innerHTML = '';
    // Reset the threshold summary.
    const summaryEl = document.getElementById('networkThresholdSummary');
    if (summaryEl) summaryEl.textContent = 'Compute a network map first to enable thresholding.';
    // Hide outside-atlas warning.
    this.showOutsideAtlasWarning(0, 0);

    if (this.executor && typeof this.executor.clearResults === 'function') {
      this.executor.clearResults();
    }

    // Restore the structural file in the viewer (or clear entirely on full reset).
    if (full || !this.structuralFile) {
      try { this.viewerController.clearAll?.(); } catch (e) { /* non-fatal */ }
    } else if (this.structuralFile) {
      this.viewerController.loadBaseVolume(this.structuralFile, { stage: 'structural' })
        .catch(err => this.updateOutput(`Viewer reload after reset failed: ${err.message}`));
    }
    this.updateOutput(full ? 'All state cleared.' : 'Results cleared (structural retained).');
  }

  async _lesionFileMatchesYeoGrid() {
    if (!this.lesionFile) return false;
    try {
      const buf = await this.lesionFile.arrayBuffer();
      const decoded = await decodeNiftiBuffer(buf);
      // Yeo7 atlas is 99x117x95 (MNI152NLin2009cAsym 2mm). The overlap
      // reducer enforces this dim-match anyway; this gate is just for
      // routing.
      return decoded.dims[0] === 99 && decoded.dims[1] === 117 && decoded.dims[2] === 95;
    } catch (err) {
      this.updateOutput(`Could not inspect lesion file: ${err.message}`);
      return false;
    }
  }

  showOutsideAtlasWarning(outside, total) {
    const el = document.getElementById('outsideAtlasWarning');
    if (!el) return;
    if (outside > 0 && total > 0) {
      const pct = ((outside / total) * 100).toFixed(1);
      el.textContent = `${outside} of ${total} lesion voxels (${pct}%) fall outside the Yeo atlas brain mask.`;
      el.classList.remove('hidden');
    } else {
      el.textContent = '';
      el.classList.add('hidden');
    }
  }

  exportCsv() {
    if (!this.overlapResult) return;
    const csv = serializeOverlapCsv(this.overlapResult.summary, {
      networkSizes: this.overlapResult.networkSizes
    });
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'lnm-overlap.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  updateOutput(message) {
    this.console.log(message);
  }

  updateViewerInfo(data) {
    const primary = document.getElementById('viewerInfoPrimary');
    if (primary) primary.textContent = data?.string || '';
    const label = document.getElementById('viewerInfoLabel');
    if (label) label.textContent = '';
  }
}

const app = new LesionNetworkMappingApp();
if (typeof window !== 'undefined') window.app = app;
document.addEventListener('DOMContentLoaded', () => app.init());
