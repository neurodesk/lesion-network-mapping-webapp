import { FileIOController } from './controllers/FileIOController.js';
import { ViewerController } from './controllers/ViewerController.js';
import { InferenceExecutor } from './controllers/InferenceExecutor.js';
import { LNM_PIPELINES, getPipelineById } from './app/lnm-tasks.js';
import { YEO7_COLORMAP } from './app/lnm-labels.js';
import { computeParcelOverlap, summarizeNetworkOverlap } from './modules/parcel-overlap.js';
import { loadAtlasFromManifest, loadConnectomeFromManifest, decodeNiftiBuffer } from './modules/atlas-loader.js';
import { fcWeightedSum, decodeFcPack, summaryToNetworkWeights } from './modules/fc-weighted-sum.js';
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
    this.manifest = null;          // populated lazily by ensureManifest()
    this.selectedPipeline = getPipelineById('lnm-yeo-only') || LNM_PIPELINES[0];

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
    for (const pipeline of LNM_PIPELINES.filter(p => p.id === 'lnm-yeo-only')) {
      const option = document.createElement('option');
      option.value = pipeline.id;
      option.textContent = pipeline.displayName;
      pipelineSelect.appendChild(option);
    }
    pipelineSelect.value = this.selectedPipeline?.id || 'lnm-yeo-only';
  }

  async setStructural(file) {
    if (!file) return;
    this.structuralFile = file;
    await this.viewerController.loadBaseVolume(file, { stage: 'structural' });
    this.updateOutput(`Structural image ready: ${file.name}`);
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
  }

  handleStepComplete(step) {
    this.updateOutput(`Worker step '${step}' complete.`);
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
    const { arrayBuffer, index, manifestEntry } =
      await loadConnectomeFromManifest('yeo7-fc-pack');
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

    // Wrap as a NIfTI for download / overlay. The Yeo atlas's spacing /
    // affine is the canonical pose for the FC pack — manifestEntry from
    // the connectome carries atlasResolutionMm; the Yeo atlas itself is
    // the same grid (99x117x95 2mm).
    const spacingMm = manifestEntry.atlasResolutionMm || 2;
    const niftiBuffer = writeNifti1(fcMap, {
      dims,
      spacing: [spacingMm, spacingMm, spacingMm],
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

    this.updateOutput('Starting MNI registration (SynthMorph deformable)...');
    const inputBuffer = await this.structuralFile.arrayBuffer();
    await this.executor.loadVolume(inputBuffer);
    await this.executor.runRegistration({
      modelAssetId: model.id,
      modelName: m.name || 'lnm-synthmorph-mni.onnx',
      modelBaseUrl: m.base,
      modelCacheKey: model.cacheKey,
      referenceAssetId: ref.id,
      referenceUrl: ref.sourceUrl,
      referenceCacheKey: ref.cacheKey,
      nbSteps: 7
    });
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
