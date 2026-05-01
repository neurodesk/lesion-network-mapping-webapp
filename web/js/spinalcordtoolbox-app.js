/**
 * SpinalCordToolbox - Browser-based spinal cord segmentation
 *
 * Main application class. Orchestrates controllers, viewer, and inference.
 * Pipeline is split into interactive steps that the user runs sequentially.
 */

import { FileIOController } from './controllers/FileIOController.js';
import { ViewerController } from './controllers/ViewerController.js';
import { InferenceExecutor } from './controllers/InferenceExecutor.js';
import { ConsoleOutput } from './modules/ui/ConsoleOutput.js';
import { ProgressManager } from './modules/ui/ProgressManager.js';
import { ModalManager } from './modules/ui/ModalManager.js';
import * as Config from './app/config.js';
import { generateNiivueColormap, getLabelName } from './app/labels.js';
import { DEFAULT_TASK_ID, SCT_TASKS, getDefaultTask, getPrimaryModelAsset, getTaskById, getModelCacheKey, isTaskRunnable } from './app/sct-tasks.js';
import { computeAutoWindow } from './modules/ui/percentile.js';
import './modules/sct-processing.js';

class SpinalCordToolboxApp {
  constructor() {
    // NiiVue
    this.nv = new niivue.Niivue({
      ...Config.VIEWER_CONFIG,
      onLocationChange: (data) => {
        this._lastLocationData = data;
        this.updateViewerInfo(data);
      }
    });

    // UI modules
    this.console = new ConsoleOutput('consoleOutput');
    this.progress = new ProgressManager(Config.PROGRESS_CONFIG);

    // State
    this.inputFile = null;
    this.currentResultTab = 'input';
    this.currentRunningStep = null;
    this.abortUICheckpoint = null;
    this._inputVisible = true;
    this._overlaySliderValue = 0.7;
    this._stageVisibility = {
      input: true,
      segmentation: true,
      vertebrae: true
    };
    this._renderViewerPromise = Promise.resolve();
    this._renderViewerRequested = false;
    this._lastLocationData = null;
    this.selectedTask = getDefaultTask();

    this.init();
  }

  async init() {
    // Version display
    const versionEl = document.getElementById('appVersion');
    if (versionEl) versionEl.textContent = `v${Config.VERSION}`;
    const footerVersionEl = document.getElementById('footerVersion');
    if (footerVersionEl) footerVersionEl.textContent = `v${Config.VERSION}`;
    const aboutVersionEl = document.getElementById('aboutAppVersion');
    if (aboutVersionEl) aboutVersionEl.textContent = `v${Config.VERSION}`;

    // Controllers
    this.fileIOController = new FileIOController({
      updateOutput: (msg) => this.updateOutput(msg),
      onFileLoaded: (file) => this.onFileLoaded(file)
    });

    this.viewerController = new ViewerController({
      nv: this.nv,
      updateOutput: (msg) => this.updateOutput(msg)
    });

    this.inferenceExecutor = new InferenceExecutor({
      updateOutput: (msg) => this.updateOutput(msg),
      setProgress: (val, text) => this.setProgress(val, text),
      onStageData: (data) => this.handleStageData(data),
      onComplete: () => this.onInferenceComplete(),
      onError: (msg) => this.onInferenceError(msg),
      onInitialized: () => this.onWorkerInitialized(),
      onStepComplete: (step) => this.onStepComplete(step),
      onVolumeInfo: (info) => this.onVolumeInfo(info)
    });

    // Modals
    this.aboutModal = new ModalManager('aboutModal');
    this.citationsModal = new ModalManager('citationsModal');
    this.privacyModal = new ModalManager('privacyModal');

    // Register custom colormap
    const colormapData = generateNiivueColormap(this.selectedTask.id);

    // Setup
    await this.setupViewer();

    // Register colormap after viewer is ready
    this.viewerController.registerSctColormap(colormapData, this.getSelectedColormapId());
    this.viewerController.registerSctColormap(generateNiivueColormap('vertebrae'), 'sct-vertebrae');

    this.setupEventListeners();
    this.populateTaskSelector();
    this.setupInfoTooltips();

    // Start ONNX initialization in background
    this.inferenceExecutor.initialize();
  }

  async setupViewer() {
    await this.nv.attachTo('gl1');
    this.nv.setMultiplanarPadPixels(5);
    this.nv.setSliceType(this.nv.sliceTypeMultiplanar);
    this.nv.setInterpolation(true);
    this.nv.drawScene();
  }

  // ==================== Viewer Footer ====================

  updateViewerInfo(data) {
    const primaryEl = document.getElementById('viewerInfoPrimary');
    if (primaryEl) {
      primaryEl.textContent = data?.string || '';
    }

    const labelEl = document.getElementById('viewerInfoLabel');
    if (labelEl) {
      labelEl.textContent = this.getOverlayLabelText(data);
    }
  }

  getOverlayLabelText(data) {
    if (!this.nv?.volumes?.length) return '';

    const visibleLabelStages = this.getVisibleOverlayStages().slice().reverse();
    for (const stage of visibleLabelStages) {
      const volumeIndex = this.viewerController?.getVolumeIndexForStage?.(stage);
      if (volumeIndex === null || volumeIndex === undefined) continue;

      const rawValue = data?.values?.[volumeIndex]?.value;
      if (!Number.isFinite(rawValue)) continue;

      const labelIndex = Math.round(rawValue);
      if (labelIndex <= 0) continue;

      const taskId = stage === 'vertebrae'
        ? 'vertebrae'
        : (this.selectedTask?.id || DEFAULT_TASK_ID);
      return getLabelName(labelIndex, taskId);
    }

    return '';
  }

  // ==================== Event Listeners ====================

  setupEventListeners() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
      fileInput.addEventListener('change', (e) => {
        this.fileIOController.handleFiles(e.target.files);
      });
    }

    this.setupDropZone();

    const runBtn = document.getElementById('runSegmentation');
    if (runBtn) runBtn.addEventListener('click', () => this.runSegmentation());

    const runProcessingBtn = document.getElementById('runProcessingBtn');
    if (runProcessingBtn) runProcessingBtn.addEventListener('click', () => this.runProcessingOperation());

    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect) {
      modelSelect.addEventListener('change', () => this.onTaskSelectionChanged(modelSelect.value));
    }

    ['abortInferenceBtn'].forEach(id => {
      const btn = document.getElementById(id);
      if (btn) btn.addEventListener('click', () => this.abortCurrentStep());
    });

    const cancelBtn = document.getElementById('cancelButton');
    if (cancelBtn) cancelBtn.addEventListener('click', () => this.abortCurrentStep());

    const copyConsole = document.getElementById('copyConsole');
    if (copyConsole) copyConsole.addEventListener('click', () => this.console.copyToClipboard());

    const clearConsole = document.getElementById('clearConsole');
    if (clearConsole) clearConsole.addEventListener('click', () => this.console.clear());

    document.querySelectorAll('.view-tab[data-view]').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.view-tab[data-view]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.viewerController.setViewType(btn.dataset.view);
      });
    });

    const opacitySlider = document.getElementById('overlayOpacity');
    if (opacitySlider) {
      opacitySlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        this._overlaySliderValue = val;
        if (this.getVisibleOverlayStages().length > 0) {
          this.viewerController.setOverlayOpacity(val);
        }
        const display = document.getElementById('overlayOpacityValue');
        if (display) display.textContent = `${Math.round(val * 100)}%`;
      });
    }

    const inputVisibilityToggle = document.getElementById('inputVisibilityToggle');
    if (inputVisibilityToggle) {
      inputVisibilityToggle.addEventListener('change', (e) => {
        void this.toggleInputVisibility(e.target.checked);
      });
    }

    this.setupWindowControls();

    const interpToggle = document.getElementById('interpolation');
    if (interpToggle) {
      interpToggle.addEventListener('change', (e) => {
        this.nv.setInterpolation(!e.target.checked);
        this.nv.drawScene();
      });
    }

    const colorbarToggle = document.getElementById('colorbarToggle');
    if (colorbarToggle) {
      colorbarToggle.addEventListener('change', (e) => {
        this.nv.opts.isColorbar = e.target.checked;
        this.nv.drawScene();
      });
    }

    const crosshairToggle = document.getElementById('crosshairToggle');
    if (crosshairToggle) {
      crosshairToggle.addEventListener('change', (e) => {
        this.nv.setCrosshairWidth(e.target.checked ? 1 : 0);
      });
    }

    const downloadBtn = document.getElementById('downloadCurrentVolume');
    if (downloadBtn) {
      downloadBtn.addEventListener('click', () => this.downloadCurrentVolume());
    }

    const screenshotBtn = document.getElementById('screenshotViewer');
    if (screenshotBtn) {
      screenshotBtn.addEventListener('click', () => this.saveScreenshot());
    }

    const colormapSelect = document.getElementById('colormapSelect');
    if (colormapSelect) {
      colormapSelect.addEventListener('change', (e) => {
        if (this.nv.volumes?.length) {
          this.nv.volumes[0].colormap = e.target.value;
          this.nv.updateGLVolume();
        }
      });
    }

    const clearResults = document.getElementById('clearResults');
    if (clearResults) clearResults.addEventListener('click', () => this.clearResults());

    // Modal buttons
    const aboutBtn = document.getElementById('aboutButton');
    if (aboutBtn) aboutBtn.addEventListener('click', () => this.aboutModal.open());
    const closeAbout = document.getElementById('closeAbout');
    if (closeAbout) closeAbout.addEventListener('click', () => this.aboutModal.close());

    const citationsBtn = document.getElementById('citationsButton');
    if (citationsBtn) citationsBtn.addEventListener('click', () => this.citationsModal.open());
    const closeCitations = document.getElementById('closeCitations');
    if (closeCitations) closeCitations.addEventListener('click', () => this.citationsModal.close());

    const privacyBtn = document.getElementById('privacyButton');
    if (privacyBtn) privacyBtn.addEventListener('click', () => this.privacyModal.open());
    const closePrivacy = document.getElementById('closePrivacy');
    if (closePrivacy) closePrivacy.addEventListener('click', () => this.privacyModal.close());
  }

  populateTaskSelector() {
    const modelSelect = document.getElementById('modelSelect');
    if (!modelSelect) return;

    modelSelect.innerHTML = '';
    for (const task of SCT_TASKS) {
      if (!isTaskRunnable(task)) continue;
      if (task.processingOnly) continue;
      const option = document.createElement('option');
      option.value = task.id;
      option.textContent = task.displayName;
      if (task.id === this.selectedTask.id) option.selected = true;
      modelSelect.appendChild(option);
    }
    this.applyTaskInferenceDefaults();
    this.updateTaskDetails();
  }

  onTaskSelectionChanged(taskId) {
    this.selectedTask = getTaskById(taskId);
    this.viewerController.registerSctColormap(generateNiivueColormap(this.selectedTask.id), this.getSelectedColormapId());
    this.applyTaskInferenceDefaults();
    this.updateTaskDetails();
  }

  applyTaskInferenceDefaults() {
    const assetDefaults = getPrimaryModelAsset(this.selectedTask)?.inferenceDefaults || {};
    const thresholdInput = document.getElementById('thresholdInput');
    if (thresholdInput) {
      thresholdInput.value = String(assetDefaults.probabilityThreshold ?? Config.INFERENCE_DEFAULTS.probabilityThreshold);
    }

    const minSizeInput = document.getElementById('minSizeInput');
    if (minSizeInput) {
      minSizeInput.value = String(assetDefaults.minComponentSize ?? Config.INFERENCE_DEFAULTS.minComponentSize);
    }
  }

  updateTaskDetails() {
    const task = this.selectedTask || getDefaultTask();
    const details = document.getElementById('taskDetails');
    const runBtn = document.getElementById('runSegmentation');
    if (details) {
      const contrasts = (task.inputContrasts || []).join(', ') || 'See SCT documentation';
      details.textContent = `${task.description} Input: ${contrasts}.`;
      details.classList.toggle('task-supported', task.supportStatus === 'supported');
      details.classList.toggle('task-disabled', task.supportStatus !== 'supported');
    }
    if (runBtn) {
      runBtn.disabled = !isTaskRunnable(task);
      runBtn.title = isTaskRunnable(task) ? 'Run SCT segmentation' : 'Task unavailable';
    }
  }

  getSelectedColormapId() {
    return `sct-${this.selectedTask?.id || DEFAULT_TASK_ID}`;
  }

  setupDropZone() {
    const zone = document.getElementById('inputDropZone');
    if (!zone) return;

    zone.addEventListener('dragover', (e) => {
      e.preventDefault();
      zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => {
      zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('dragover');
      this.fileIOController.handleDropItems(e.dataTransfer.items);
    });
  }

  setupInfoTooltips() {
    document.querySelectorAll('.info-icon').forEach(icon => {
      const tooltip = icon.querySelector('.info-tooltip');
      if (!tooltip) return;

      icon.addEventListener('mouseenter', () => {
        tooltip.style.display = 'block';
        const iconRect = icon.getBoundingClientRect();
        const tipRect = tooltip.getBoundingClientRect();
        let top = iconRect.top - tipRect.height - 6;
        let left = iconRect.left + iconRect.width / 2 - tipRect.width / 2;
        if (top < 4) top = iconRect.bottom + 6;
        left = Math.max(4, Math.min(left, window.innerWidth - tipRect.width - 4));
        tooltip.style.top = `${top}px`;
        tooltip.style.left = `${left}px`;
      });

      icon.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
      });
    });
  }

  // ==================== Viewer Controls ====================

  setupWindowControls() {
    const rangeMin = document.getElementById('rangeMin');
    const rangeMax = document.getElementById('rangeMax');
    const windowMin = document.getElementById('windowMin');
    const windowMax = document.getElementById('windowMax');
    const resetBtn = document.getElementById('resetWindow');
    if (!rangeMin || !rangeMax || !windowMin || !windowMax) return;

    const updateSelected = () => {
      const selected = document.getElementById('rangeSelected');
      if (!selected) return;
      const min = parseFloat(rangeMin.value);
      const max = parseFloat(rangeMax.value);
      selected.style.left = `${min}%`;
      selected.style.width = `${max - min}%`;
    };

    const applyFromSliders = () => {
      if (!this.nv.volumes.length) return;
      const vol = this.nv.volumes[0];
      const dataMin = vol.global_min ?? 0;
      const dataMax = vol.global_max ?? 1;
      const range = dataMax - dataMin || 1;
      const newMin = dataMin + (parseFloat(rangeMin.value) / 100) * range;
      const newMax = dataMin + (parseFloat(rangeMax.value) / 100) * range;
      windowMin.value = newMin.toPrecision(4);
      windowMax.value = newMax.toPrecision(4);
      vol.cal_min = newMin;
      vol.cal_max = newMax;
      this.nv.updateGLVolume();
      updateSelected();
    };

    const applyFromInputs = () => {
      if (!this.nv.volumes.length) return;
      const vol = this.nv.volumes[0];
      const newMin = parseFloat(windowMin.value);
      const newMax = parseFloat(windowMax.value);
      if (isNaN(newMin) || isNaN(newMax)) return;
      vol.cal_min = newMin;
      vol.cal_max = newMax;
      this.nv.updateGLVolume();
      this.syncSlidersToVolume();
    };

    rangeMin.addEventListener('input', () => {
      if (parseFloat(rangeMin.value) > parseFloat(rangeMax.value) - 1) {
        rangeMin.value = parseFloat(rangeMax.value) - 1;
      }
      applyFromSliders();
    });

    rangeMax.addEventListener('input', () => {
      if (parseFloat(rangeMax.value) < parseFloat(rangeMin.value) + 1) {
        rangeMax.value = parseFloat(rangeMin.value) + 1;
      }
      applyFromSliders();
    });

    windowMin.addEventListener('change', applyFromInputs);
    windowMax.addEventListener('change', applyFromInputs);

    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        if (!this.nv.volumes.length) return;
        this.applyAutoContrast();
      });
    }
  }

  applyAutoContrast() {
    if (!this.nv.volumes.length) return;
    const vol = this.nv.volumes[0];

    // computeAutoWindow operates on vol.img which may be raw typed data
    // (e.g. Int16Array), but vol.cal_min/cal_max are in scaled space
    // (after NIfTI scl_slope/scl_inter). Convert using global_min/max.
    const { low, high, min: rawMin, max: rawMax } = computeAutoWindow(vol.img);

    let scaledLow = low;
    let scaledHigh = high;
    const rawRange = rawMax - rawMin;
    const scaledRange = vol.global_max - vol.global_min;
    if (rawRange > 0 && scaledRange > 0) {
      // Linear mapping: raw → scaled
      const slope = scaledRange / rawRange;
      const inter = vol.global_min - rawMin * slope;
      scaledLow = low * slope + inter;
      scaledHigh = high * slope + inter;
    }

    vol.cal_min = scaledLow;
    vol.cal_max = scaledHigh;
    this.nv.updateGLVolume();
    this.syncWindowControls();
  }

  syncWindowControls() {
    if (!this.nv.volumes.length) return;
    const vol = this.nv.volumes[0];
    const windowMin = document.getElementById('windowMin');
    const windowMax = document.getElementById('windowMax');
    if (windowMin) windowMin.value = (vol.cal_min ?? 0).toPrecision(4);
    if (windowMax) windowMax.value = (vol.cal_max ?? 1).toPrecision(4);
    this.syncSlidersToVolume();
    const dlBtn = document.getElementById('downloadCurrentVolume');
    if (dlBtn) dlBtn.disabled = false;
  }

  syncSlidersToVolume() {
    if (!this.nv.volumes.length) return;
    const vol = this.nv.volumes[0];
    const dataMin = vol.global_min ?? 0;
    const dataMax = vol.global_max ?? 1;
    const range = dataMax - dataMin || 1;
    const rangeMin = document.getElementById('rangeMin');
    const rangeMax = document.getElementById('rangeMax');
    const selected = document.getElementById('rangeSelected');
    if (!rangeMin || !rangeMax) return;
    const pctMin = Math.max(0, Math.min(100, ((vol.cal_min - dataMin) / range) * 100));
    const pctMax = Math.max(0, Math.min(100, ((vol.cal_max - dataMin) / range) * 100));
    rangeMin.value = pctMin;
    rangeMax.value = pctMax;
    if (selected) {
      selected.style.left = `${pctMin}%`;
      selected.style.width = `${pctMax - pctMin}%`;
    }
  }

  downloadCurrentVolume() {
    if (!this.nv.volumes?.length) {
      this.updateOutput('No volume loaded');
      return;
    }
    const vol = this.nv.volumes[0];
    const name = (vol.name || 'volume').replace(/\.(nii|nii\.gz)$/i, '');
    const niftiBuffer = this.createNiftiFromVolume(vol);
    const blob = new Blob([niftiBuffer], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${name}.nii`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    this.updateOutput(`Downloaded: ${name}.nii`);
  }

  createNiftiFromVolume(vol) {
    const hdr = vol.hdr;
    const img = vol.img;
    let datatype = 16, bitpix = 32, bytesPerVoxel = 4;
    if (img instanceof Float64Array) { datatype = 64; bitpix = 64; bytesPerVoxel = 8; }
    else if (img instanceof Int16Array) { datatype = 4; bitpix = 16; bytesPerVoxel = 2; }
    else if (img instanceof Uint8Array) { datatype = 2; bitpix = 8; bytesPerVoxel = 1; }

    const headerSize = 352;
    const buffer = new ArrayBuffer(headerSize + img.length * bytesPerVoxel);
    const view = new DataView(buffer);

    view.setInt32(0, 348, true);
    const dims = hdr.dims || [3, vol.dims[1], vol.dims[2], vol.dims[3], 1, 1, 1, 1];
    for (let i = 0; i < 8; i++) view.setInt16(40 + i * 2, dims[i] || 0, true);
    view.setInt16(70, datatype, true);
    view.setInt16(72, bitpix, true);
    const pixdim = hdr.pixDims || [1, 1, 1, 1, 1, 1, 1, 1];
    for (let i = 0; i < 8; i++) view.setFloat32(76 + i * 4, pixdim[i] || 1, true);
    view.setFloat32(108, headerSize, true);
    view.setFloat32(112, hdr.scl_slope || 1, true);
    view.setFloat32(116, hdr.scl_inter || 0, true);
    view.setUint8(123, 10);
    view.setInt16(252, hdr.qform_code || 1, true);
    view.setInt16(254, hdr.sform_code || 1, true);
    if (hdr.affine) {
      for (let i = 0; i < 4; i++) {
        view.setFloat32(280 + i * 4, hdr.affine[0][i] || 0, true);
        view.setFloat32(296 + i * 4, hdr.affine[1][i] || 0, true);
        view.setFloat32(312 + i * 4, hdr.affine[2][i] || 0, true);
      }
    }
    view.setUint8(344, 0x6E);
    view.setUint8(345, 0x2B);
    view.setUint8(346, 0x31);
    view.setUint8(347, 0x00);

    new Uint8Array(buffer, headerSize).set(new Uint8Array(img.buffer, img.byteOffset, img.byteLength));
    return buffer;
  }

  saveScreenshot() {
    let filename = 'spinalcordtoolbox_screenshot.png';
    if (this.nv.volumes?.length) {
      const name = (this.nv.volumes[0].name || 'volume').replace(/\.(nii|nii\.gz)$/i, '');
      filename = `${name}_screenshot.png`;
    }
    this.nv.saveScene(filename);
    this.updateOutput(`Screenshot saved: ${filename}`);
  }

  // ==================== File Handling ====================

  async onFileLoaded(file) {
    await this.resetForNewFile();
    this.inputFile = file;
    this.setStageVisible('input', true);
    const inputVisibilityToggle = document.getElementById('inputVisibilityToggle');
    if (inputVisibilityToggle) inputVisibilityToggle.checked = true;
    await this.viewerController.loadBaseVolume(file, { stage: 'input' });
    this.applyDefaultBaseColormap();
    this.syncWindowControls();
    this.applyAutoContrast();

    // Send data to worker for loading
    const inputData = await file.arrayBuffer();
    this.setStepRunning('load');
    await this.inferenceExecutor.loadVolume(inputData);
  }

  async resetForNewFile() {
    if (this.inferenceExecutor.isRunning()) {
      this.inferenceExecutor.cancel();
    }

    this.inputFile = null;
    this.currentResultTab = 'input';
    this.currentRunningStep = null;
    this.abortUICheckpoint = null;
    this._inputVisible = true;
    this.resetStageVisibility();
    this._overlaySliderValue = 0.7;
    this._lastLocationData = null;

    this.console.clear();
    this.progress.reset();
    this.resetStatusDisplay();
    this.resetProcessingInputs();
    this.resetViewerControls();

    await this.resetAllSteps();
    this.updateViewerInfo(null);
  }

  captureAbortUICheckpoint(step) {
    const sectionEnabled = {};
    const buttonsEnabled = {};

    for (const pipelineStep of ['inference', 'processing']) {
      sectionEnabled[pipelineStep] = this.isStepEnabled(pipelineStep);
      buttonsEnabled[pipelineStep] = this.areStepButtonsEnabled(pipelineStep);
    }

    return {
      step,
      sectionEnabled,
      buttonsEnabled,
      currentResultTab: this.currentResultTab || 'input',
      inputVisible: this._inputVisible,
      stageVisibility: { ...this._stageVisibility },
      overlaySliderValue: this._overlaySliderValue
    };
  }

  beginAbortableStep(step) {
    this.currentRunningStep = step;
    this.abortUICheckpoint = this.captureAbortUICheckpoint(step);
    this.inferenceExecutor.captureCheckpoint(step);
  }

  async abortCurrentStep() {
    if (!this.currentRunningStep || !this.inferenceExecutor.isRunning()) return;

    const abortedStep = this.currentRunningStep;
    const checkpoint = this.abortUICheckpoint;
    const statusText = document.getElementById('statusText');
    if (statusText) statusText.textContent = 'Aborting...';

    this.resetAbortControls();

    try {
      const restoreResult = await this.inferenceExecutor.abortCurrentStep();
      if (!restoreResult) return;
      await this.restoreUIFromAbortCheckpoint(checkpoint, abortedStep);
    } catch (error) {
      this.onInferenceError(error?.message || String(error));
    } finally {
      this.currentRunningStep = null;
      this.abortUICheckpoint = null;
    }
  }

  async restoreUIFromAbortCheckpoint(checkpoint, abortedStep) {
    this.progress.reset();
    this.resetAbortControls();

    for (const step of Config.PIPELINE_STEPS) {
      this.updateStepBadge(step, this.inferenceExecutor.getStepStatus(step));
      if (checkpoint?.sectionEnabled?.[step] !== undefined) {
        this.setStepEnabled(step, checkpoint.sectionEnabled[step]);
      }
      if (checkpoint?.buttonsEnabled?.[step] !== undefined) {
        this.setStepButtonsEnabled(step, checkpoint.buttonsEnabled[step]);
      }
    }

    if (abortedStep) {
      this.updateStepBadge(abortedStep, 'pending');
      this.setStepButtonsEnabled(abortedStep, true);
    }

    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
      if (this.inferenceExecutor.getStageOrder().length > 0) {
        resultsSection.classList.remove('hidden');
        resultsSection.classList.remove('collapsed');
      } else {
        resultsSection.classList.add('hidden');
        resultsSection.classList.add('collapsed');
      }
    }

    const overlayControl = document.getElementById('overlayControl');
    if (overlayControl) {
      overlayControl.classList.toggle('hidden', !this.inferenceExecutor.getResult('segmentation'));
    }

    this.currentResultTab = checkpoint?.currentResultTab || 'input';
    this._stageVisibility = {
      ...this.getDefaultStageVisibility(),
      ...(checkpoint?.stageVisibility || {})
    };
    this._overlaySliderValue = checkpoint?.overlaySliderValue ?? 0.5;

    this.rebuildResultsList();

    const targetStage = (this.currentResultTab === 'input' || this.inferenceExecutor.getResult(this.currentResultTab))
      ? this.currentResultTab
      : 'input';
    this.currentResultTab = targetStage;

    this.setStageVisible('input', checkpoint?.inputVisible ?? true);
    const inputVisibilityToggle = document.getElementById('inputVisibilityToggle');
    if (inputVisibilityToggle) inputVisibilityToggle.checked = this.isStageVisible('input');

    const opacitySlider = document.getElementById('overlayOpacity');
    if (opacitySlider) {
      opacitySlider.disabled = this.getVisibleOverlayStages().length === 0;
      opacitySlider.value = String(this._overlaySliderValue);
    }
    const opacityDisplay = document.getElementById('overlayOpacityValue');
    if (opacityDisplay) opacityDisplay.textContent = `${Math.round(this._overlaySliderValue * 100)}%`;

    if (this.inputFile && targetStage) {
      await this.renderViewerVolumes();
    }
    this.syncResultViewButtons();

    const statusText = document.getElementById('statusText');
    if (statusText) statusText.textContent = 'Ready';
    this.updateViewerInfo(this._lastLocationData);
  }

  // ==================== Pipeline Step Methods ====================

  async runSegmentation() {
    if (this.inferenceExecutor.isRunning()) return;

    const overlapSelect = document.getElementById('overlapSelect');
    const overlap = overlapSelect ? parseFloat(overlapSelect.value) : Config.INFERENCE_DEFAULTS.overlap;

    const thresholdInput = document.getElementById('thresholdInput');
    const threshold = thresholdInput ? parseFloat(thresholdInput.value) : Config.INFERENCE_DEFAULTS.probabilityThreshold;

    const minSizeInput = document.getElementById('minSizeInput');
    const minComponentSize = minSizeInput ? parseInt(minSizeInput.value, 10) : Config.INFERENCE_DEFAULTS.minComponentSize;

    const modelSelect = document.getElementById('modelSelect');
    const selectedTaskId = modelSelect ? modelSelect.value : DEFAULT_TASK_ID;
    const selectedTask = getTaskById(selectedTaskId);
    const selectedAsset = getPrimaryModelAsset(selectedTask);
    const assetDefaults = selectedAsset?.inferenceDefaults || {};
    const effectivePatchSize = selectedAsset?.patchSize || selectedTask.patchSize || Config.MODEL.patchSize;

    if (!isTaskRunnable(selectedTask)) {
      this.updateOutput(`SCT task "${selectedTask.displayName}" is unavailable.`);
      this.updateTaskDetails();
      return;
    }

    if (selectedTask.processingOnly || !selectedAsset) {
      this.updateOutput(`SCT task "${selectedTask.displayName}" is a post-processing step. Run it from the SCT Processing section after a segmentation completes.`);
      this.updateTaskDetails();
      return;
    }

    const modelBaseUrl = new URL(Config.MODEL_BASE_URL, window.location.href).href;
    this.beginAbortableStep('inference');

    // Clear previous results — including any vertebrae mask, which is derived
    // from the previous segmentation and would otherwise be auto-rendered as a
    // stale overlay on the new run.
    this.inferenceExecutor.clearResults();
    this.disableAllResultTabs();
    this.resetStageVisibility();
    await this.renderViewerVolumes();

    this.setStepRunning('inference');
    await this.inferenceExecutor.runInference({
      overlap,
      threshold,
      minComponentSize,
      taskId: selectedTask.id,
      modelAssetId: selectedAsset?.id || selectedTask.id,
      supportStatus: selectedTask.supportStatus,
      cacheKey: getModelCacheKey(selectedTask, selectedAsset),
      provenance: {
        taskId: selectedTask.id,
        modelAssetId: selectedAsset?.id || null,
        sourceVersion: selectedAsset?.sourceVersion || 'unknown',
        appVersion: Config.VERSION
      },
      modelName: selectedAsset?.filename || Config.MODEL.name,
      patchSize: effectivePatchSize,
      preprocessing: selectedAsset?.preprocessing || {},
      testTimeAugmentation: !!document.getElementById('ttaToggle')?.checked,
      modelBaseUrl
    });
  }

  runProcessingOperation() {
    const select = document.getElementById('processingOperationSelect');
    const output = document.getElementById('processingOutput');
    const operation = select?.value || 'vertebrae';
    if (operation === 'vertebrae') {
      if (!this.inferenceExecutor.hasResult('segmentation')) {
        this.updateOutput('Run spinal cord segmentation before vertebral labeling');
        return;
      }
      const modelBaseUrl = new URL(Config.MODEL_BASE_URL, window.location.href).href;
      if (output) output.textContent = '';
      this.beginAbortableStep('processing');
      this.setStepRunning('processing');
      this.inferenceExecutor.runVertebralLabeling({
        modelBaseUrl,
        scaleDist: 0.55,
        detectorMinScore: 0.1
      }).catch(error => this.onInferenceError(error.message));
      return;
    }
    if (output) output.textContent = '';
    this.updateOutput(`Unsupported SCT Processing operation: ${operation}`);
  }

  // ==================== Step UI Management ====================

  setStepRunning(step) {
    this.updateStepBadge(step, 'running');
    this.setStepButtonsEnabled(step, false);
    this.setStepAbortVisible(step, true);
    if (this.currentRunningStep === step) {
      const cancelBtn = document.getElementById('cancelButton');
      if (cancelBtn) cancelBtn.disabled = false;
    }
  }

  getStepSectionId(step) {
    const sectionMap = {
      'load': null,
      'inference': 'stepInferenceSection',
      'processing': 'stepProcessingSection'
    };
    return sectionMap[step] || null;
  }

  getStepButtonIds(step) {
    const buttonMap = {
      'inference': ['runSegmentation'],
      'processing': ['runProcessingBtn']
    };
    return buttonMap[step] || [];
  }

  getStepAbortButtonId(step) {
    const abortButtonMap = {
      'inference': 'abortInferenceBtn'
    };
    return abortButtonMap[step] || null;
  }

  isStepEnabled(step) {
    const sectionId = this.getStepSectionId(step);
    if (!sectionId) return false;
    const section = document.getElementById(sectionId);
    return !!section && !section.classList.contains('step-disabled');
  }

  areStepButtonsEnabled(step) {
    const buttonIds = this.getStepButtonIds(step);
    if (buttonIds.length === 0) return false;
    return buttonIds.every(id => {
      const btn = document.getElementById(id);
      return !!btn && !btn.disabled;
    });
  }

  setStepAbortVisible(step, visible) {
    const abortButtonId = this.getStepAbortButtonId(step);
    if (!abortButtonId) return;

    const abortBtn = document.getElementById(abortButtonId);
    if (!abortBtn) return;

    abortBtn.classList.toggle('hidden', !visible);
    abortBtn.disabled = !visible;
  }

  resetAbortControls() {
    for (const step of ['inference', 'processing']) {
      this.setStepAbortVisible(step, false);
    }

    const cancelBtn = document.getElementById('cancelButton');
    if (cancelBtn) cancelBtn.disabled = true;
  }

  async resetAllSteps() {
    // Reset worker state
    if (this.inferenceExecutor.isReady()) {
      await this.inferenceExecutor.resetWorkerState();
    }

    this.currentRunningStep = null;
    this.abortUICheckpoint = null;

    // Reset all UI step sections
    for (const step of Config.PIPELINE_STEPS) {
      this.updateStepBadge(step, '');
      this.setStepEnabled(step, false);
      this.setStepButtonsEnabled(step, false);
      this.setStepAbortVisible(step, false);
    }

    // Reset results
    this.inferenceExecutor.clearResults();
    this.disableAllResultTabs();
    this.resetStageVisibility();

    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
      resultsSection.classList.add('hidden');
      resultsSection.classList.add('collapsed');
    }

    const overlayControl = document.getElementById('overlayControl');
    if (overlayControl) overlayControl.classList.add('hidden');

    this.resetAbortControls();
  }

  resetStatusDisplay() {
    const statusText = document.getElementById('statusText');
    if (statusText) statusText.textContent = 'Ready';
    this.resetAbortControls();
  }

  resetProcessingInputs() {
    const overlapSelect = document.getElementById('overlapSelect');
    if (overlapSelect) overlapSelect.value = String(Config.INFERENCE_DEFAULTS.overlap);

    const thresholdInput = document.getElementById('thresholdInput');
    if (thresholdInput) thresholdInput.value = String(Config.INFERENCE_DEFAULTS.probabilityThreshold);

    const minSizeInput = document.getElementById('minSizeInput');
    if (minSizeInput) minSizeInput.value = String(Config.INFERENCE_DEFAULTS.minComponentSize);

    const ttaToggle = document.getElementById('ttaToggle');
    if (ttaToggle) ttaToggle.checked = false;

    this.applyTaskInferenceDefaults();
  }

  resetViewerControls() {
    document.querySelectorAll('.view-tab[data-view]').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.view === 'multiplanar');
    });
    this.viewerController.setViewType('multiplanar');

    const rangeMin = document.getElementById('rangeMin');
    if (rangeMin) rangeMin.value = '0';
    const rangeMax = document.getElementById('rangeMax');
    if (rangeMax) rangeMax.value = '100';
    const rangeSelected = document.getElementById('rangeSelected');
    if (rangeSelected) {
      rangeSelected.style.left = '0%';
      rangeSelected.style.width = '100%';
    }

    const windowMin = document.getElementById('windowMin');
    if (windowMin) windowMin.value = '';
    const windowMax = document.getElementById('windowMax');
    if (windowMax) windowMax.value = '';

    const overlayControl = document.getElementById('overlayControl');
    if (overlayControl) overlayControl.classList.add('hidden');

    const opacitySlider = document.getElementById('overlayOpacity');
    if (opacitySlider) {
      opacitySlider.disabled = false;
      opacitySlider.value = '0.5';
    }
    const opacityDisplay = document.getElementById('overlayOpacityValue');
    if (opacityDisplay) opacityDisplay.textContent = '50%';

    const inputVisibilityToggle = document.getElementById('inputVisibilityToggle');
    if (inputVisibilityToggle) inputVisibilityToggle.checked = true;

    const colormapSelect = document.getElementById('colormapSelect');
    if (colormapSelect) colormapSelect.value = 'gray';

    const interpolationToggle = document.getElementById('interpolation');
    if (interpolationToggle) interpolationToggle.checked = false;
    this.nv.setInterpolation(true);

    const colorbarToggle = document.getElementById('colorbarToggle');
    if (colorbarToggle) colorbarToggle.checked = false;
    this.nv.opts.isColorbar = false;

    const crosshairToggle = document.getElementById('crosshairToggle');
    if (crosshairToggle) crosshairToggle.checked = true;
    this.nv.setCrosshairWidth(Config.VIEWER_CONFIG.crosshairWidth ?? 1);

    const downloadBtn = document.getElementById('downloadCurrentVolume');
    if (downloadBtn) downloadBtn.disabled = true;

    this.nv.drawScene();
  }

  applyDefaultBaseColormap() {
    const colormapSelect = document.getElementById('colormapSelect');
    const colormap = this.currentResultTab === 'vertebrae'
      ? 'sct-vertebrae'
      : (colormapSelect?.value || 'gray');
    if (!this.nv.volumes?.length) return;
    this.nv.volumes[0].colormap = colormap;
    this.nv.updateGLVolume();
  }

  async onStepComplete(step) {
    const status = this.inferenceExecutor.getStepStatus(step);
    this.updateStepBadge(step, status);
    this.setStepButtonsEnabled(step, true);
    this.setStepAbortVisible(step, false);
    this.resetAbortControls();

    if (this.currentRunningStep === step) {
      this.currentRunningStep = null;
      this.abortUICheckpoint = null;
    }

    const statusText = document.getElementById('statusText');
    if (statusText) statusText.textContent = 'Ready';

    // Enable next step section
    switch (step) {
      case 'load':
        this.setStepEnabled('inference', true);
        this.setStepButtonsEnabled('inference', true);
        this.setStepEnabled('processing', true);
        this.setStepButtonsEnabled('processing', true);
        this.updateTaskDetails();
        break;
      case 'inference':
        break;
    }

    // Load stage data into viewer for preprocessing steps
    // (stageData is already handled in handleStageData)
  }

  onVolumeInfo(info) {
    void info;
  }

  updateStepBadge(step, status) {
    const badgeMap = {
      'load': null,
      'inference': 'stepInferenceBadge',
      'processing': 'stepProcessingBadge'
    };
    // Load step doesn't have a visible badge
    if (step === 'load') return;

    const badge = document.getElementById(badgeMap[step]);
    if (!badge) return;

    badge.className = 'step-badge';
    badge.textContent = '';

    switch (status) {
      case 'running':
        badge.classList.add('badge-running');
        badge.textContent = 'Running';
        break;
      case 'complete':
        badge.classList.add('badge-complete');
        badge.textContent = 'Done';
        break;
      case 'skipped':
        badge.classList.add('badge-skipped');
        badge.textContent = 'Skipped';
        break;
      case 'pending':
        badge.classList.add('badge-pending');
        badge.textContent = 'Pending';
        break;
    }
  }

  setStepEnabled(step, enabled) {
    const sectionId = this.getStepSectionId(step);
    if (!sectionId) return;

    const section = document.getElementById(sectionId);
    if (!section) return;

    if (enabled) {
      section.classList.remove('step-disabled');
    } else {
      section.classList.add('step-disabled');
    }
  }

  setStepButtonsEnabled(step, enabled) {
    const buttons = this.getStepButtonIds(step);
    if (!buttons) return;

    for (const id of buttons) {
      const btn = document.getElementById(id);
      if (btn) btn.disabled = !enabled;
    }
  }

  resetUIDownstream(fromStep) {
    const steps = ['inference', 'processing'];
    const idx = steps.indexOf(fromStep);
    if (idx < 0) return;

    for (let i = idx + 1; i < steps.length; i++) {
      this.updateStepBadge(steps[i], '');
      this.setStepEnabled(steps[i], false);
      this.setStepButtonsEnabled(steps[i], false);
    }
  }

  // ==================== Results ====================

  async handleStageData(data) {
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
      resultsSection.classList.remove('hidden');
      resultsSection.classList.remove('collapsed');
    }

    if (this.isOverlayStage(data.stage)) {
      if (data.stage === 'vertebrae') {
        this.viewerController.registerSctColormap(generateNiivueColormap('vertebrae'), 'sct-vertebrae');
      }
      this.setStageVisible(data.stage, true);
      this.setStageVisible('input', true);
      const overlayControl = document.getElementById('overlayControl');
      if (overlayControl) overlayControl.classList.remove('hidden');
      await this.renderViewerVolumes();
    } else {
      const result = this.inferenceExecutor.getResult(data.stage);
      if (result?.file) {
        this.currentResultTab = data.stage;
        this.setStageVisible('input', true);
        await this.renderViewerVolumes();
      }
    }

    this.rebuildResultsList();
  }

  rebuildResultsList() {
    const container = document.getElementById('stageButtons');
    if (!container) return;
    container.innerHTML = '';

    const stages = this.inferenceExecutor.getStageOrder();
    const dlSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>';
    const viewSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';

    // Build all rows with uniform layout: eye icon + label + download button
    const allStages = this.inputFile
      ? ['input', ...stages.filter(stage => stage !== 'input')]
      : [...stages];

    for (const stage of allStages) {
      const row = document.createElement('div');
      row.className = 'volume-toggle';

      const viewBtn = document.createElement('button');
      viewBtn.className = 'view-btn';
      viewBtn.title = `View ${Config.STAGE_NAMES[stage] || stage}`;
      viewBtn.innerHTML = viewSvg;
      viewBtn.dataset.stage = stage;

      if (this.isOverlayStage(stage)) {
        viewBtn.classList.toggle('active', this.isStageVisible(stage));
        viewBtn.addEventListener('click', () => {
          void this.toggleStageVisibility(stage, !this.isStageVisible(stage));
        });
      } else {
        // Initialize active state based on what's currently displayed
        viewBtn.classList.toggle('active', this.currentResultTab === stage && this.isStageVisible('input'));
        // Base volume stages: toggle load/unload as base volume
        viewBtn.addEventListener('click', () => {
          if (this.currentResultTab === stage && this.isStageVisible('input')) {
            // Already showing this stage — hide it
            void this.toggleInputVisibility(false);
            viewBtn.classList.remove('active');
          } else {
            void this.viewStage(stage);
          }
        });
      }
      row.appendChild(viewBtn);

      const label = document.createElement('span');
      label.className = 'stage-label';
      label.textContent = Config.STAGE_NAMES[stage] || stage;
      row.appendChild(label);

      // Download button (not for input — user already has the file)
      if (stage !== 'input') {
        const dlBtn = document.createElement('button');
        dlBtn.className = 'download-btn';
        dlBtn.title = `Download ${Config.STAGE_NAMES[stage] || stage}`;
        dlBtn.innerHTML = dlSvg;
        dlBtn.addEventListener('click', () => this.inferenceExecutor.downloadStage(stage));
        row.appendChild(dlBtn);
      }

      container.appendChild(row);
    }
  }

  async viewStage(stage) {
    const file = stage === 'input'
      ? this.inputFile
      : this.inferenceExecutor.getResult(stage)?.file;
    if (!file) return;

    await this.viewerController.loadBaseVolume(file);
    this.currentResultTab = stage;
    this.setStageVisible('input', true);
    const inputVisibilityToggle = document.getElementById('inputVisibilityToggle');
    if (inputVisibilityToggle) inputVisibilityToggle.checked = true;
    this.applyDefaultBaseColormap();
    this.syncWindowControls();
    this.applyAutoContrast();

    await this.renderViewerVolumes();

    const container = document.getElementById('stageButtons');
    if (container) {
      container.querySelectorAll('.view-btn').forEach(btn => {
        const btnStage = btn.dataset.stage;
        if (this.isOverlayStage(btnStage)) {
          btn.classList.toggle('active', this.isStageVisible(btnStage));
        } else {
          btn.classList.toggle('active', btnStage === stage && this.isStageVisible('input'));
        }
      });
    }
  }

  getCurrentBaseFile() {
    if (this.currentResultTab === 'input') return this.inputFile;
    if (this.isOverlayStage(this.currentResultTab)) return this.inputFile;
    return this.inferenceExecutor.getResult(this.currentResultTab)?.file || this.inputFile;
  }

  isOverlayStage(stage) {
    return stage === 'segmentation' || stage === 'vertebrae';
  }

  getOverlayColormapId(stage) {
    if (stage === 'vertebrae') return 'sct-vertebrae';
    return this.getSelectedColormapId();
  }

  getDefaultStageVisibility() {
    return {
      input: true,
      segmentation: true,
      vertebrae: true
    };
  }

  resetStageVisibility() {
    this._stageVisibility = this.getDefaultStageVisibility();
    this._inputVisible = this._stageVisibility.input;
  }

  isStageVisible(stage) {
    if (stage === 'input') return this._inputVisible;
    return this._stageVisibility?.[stage] ?? true;
  }

  setStageVisible(stage, visible) {
    if (!this._stageVisibility) this.resetStageVisibility();
    this._stageVisibility[stage] = visible;
    if (stage === 'input') this._inputVisible = visible;
  }

  getVisibleOverlayStages() {
    return ['segmentation', 'vertebrae'].filter(stage => (
      this.isStageVisible(stage) && this.inferenceExecutor.hasResult(stage)
    ));
  }

  async renderViewerVolumes() {
    this._renderViewerRequested = true;
    this._renderViewerPromise = this._renderViewerPromise.then(async () => {
      if (!this._renderViewerRequested) return;
      this._renderViewerRequested = false;
      await this._renderViewerVolumesNow();
    });
    return this._renderViewerPromise;
  }

  async _renderViewerVolumesNow() {
    const visibleOverlayStages = this.getVisibleOverlayStages();

    if (!this.isStageVisible('input')) {
      const [baseOverlayStage, ...remainingOverlayStages] = visibleOverlayStages;
      if (baseOverlayStage) {
        const baseOverlayFile = this.inferenceExecutor.getResult(baseOverlayStage)?.file;
        const stackEntries = [{
          file: baseOverlayFile,
          stage: baseOverlayStage,
          colormap: this.getOverlayColormapId(baseOverlayStage),
          labelMask: true
        }];
        for (const overlayStage of remainingOverlayStages) {
          const overlayFile = this.inferenceExecutor.getResult(overlayStage)?.file;
          stackEntries.push({
            file: overlayFile,
            stage: overlayStage,
            colormap: this.getOverlayColormapId(overlayStage),
            opacity: this._overlaySliderValue,
            labelMask: true
          });
        }
        await this.viewerController.loadVolumeStack(stackEntries);
        this.syncWindowControls();
      } else {
        this.viewerController.clearVolumes();
      }
      return;
    }

    const baseFile = this.getCurrentBaseFile();
    if (!baseFile) return;

    const stackEntries = [{
      file: baseFile,
      stage: this.currentResultTab || 'input',
      colormap: null
    }];
    for (const overlayStage of visibleOverlayStages) {
      const overlayFile = this.inferenceExecutor.getResult(overlayStage)?.file;
      stackEntries.push({
        file: overlayFile,
        stage: overlayStage,
        colormap: this.getOverlayColormapId(overlayStage),
        opacity: this._overlaySliderValue,
        labelMask: true
      });
    }

    await this.viewerController.loadVolumeStack(stackEntries);
    this.applyDefaultBaseColormap();
    this.syncWindowControls();
    this.applyAutoContrast();
  }

  syncResultViewButtons() {
    const container = document.getElementById('stageButtons');
    if (!container) return;

    container.querySelectorAll('.view-btn').forEach(btn => {
      const stage = btn.dataset.stage;
      if (this.isOverlayStage(stage)) {
        btn.classList.toggle('active', this.isStageVisible(stage));
      } else {
        btn.classList.toggle('active', stage === this.currentResultTab && this.isStageVisible('input'));
      }
    });
  }

  async toggleInputVisibility(visible) {
    this.setStageVisible('input', visible);
    const inputVisibilityToggle = document.getElementById('inputVisibilityToggle');
    if (inputVisibilityToggle) inputVisibilityToggle.checked = visible;
    await this.renderViewerVolumes();
    this.rebuildResultsList();
    this.updateViewerInfo(this._lastLocationData);
  }

  async toggleStageVisibility(stage, visible) {
    this.setStageVisible(stage, visible);
    const opacitySlider = document.getElementById('overlayOpacity');
    await this.renderViewerVolumes();
    if (opacitySlider) opacitySlider.disabled = this.getVisibleOverlayStages().length === 0;
    this.syncResultViewButtons();
    this.updateViewerInfo(this._lastLocationData);
  }

  onWorkerInitialized() {}

  async onInferenceComplete() {
    const statusText = document.getElementById('statusText');
    this.resetAbortControls();
    this.currentRunningStep = null;
    this.abortUICheckpoint = null;
    if (statusText) statusText.textContent = 'Ready';

    if (this.getVisibleOverlayStages().length > 0) {
      await this.renderViewerVolumes();

      const opacitySlider = document.getElementById('overlayOpacity');
      if (opacitySlider) {
        opacitySlider.disabled = this.getVisibleOverlayStages().length === 0;
        opacitySlider.value = String(this._overlaySliderValue);
      }
      const opacityDisplay = document.getElementById('overlayOpacityValue');
      if (opacityDisplay) opacityDisplay.textContent = `${Math.round(this._overlaySliderValue * 100)}%`;
      const inputVisibilityToggle = document.getElementById('inputVisibilityToggle');
      if (inputVisibilityToggle) inputVisibilityToggle.checked = this.isStageVisible('input');
      this.rebuildResultsList();
    }
  }

  onInferenceError(msg) {
    const statusText = document.getElementById('statusText');
    this.resetAbortControls();
    this.currentRunningStep = null;
    this.abortUICheckpoint = null;
    if (statusText) statusText.textContent = 'Error';

    // Reset any running badges back
    for (const step of Config.PIPELINE_STEPS) {
      const status = this.inferenceExecutor.getStepStatus(step);
      if (status === 'running') {
        this.updateStepBadge(step, 'pending');
        this.setStepButtonsEnabled(step, true);
      }
    }
  }

  disableAllResultTabs() {
    const container = document.getElementById('stageButtons');
    if (container) container.innerHTML = '';
    this.viewerController.clearVolumes();
    this.resetStageVisibility();
    this._overlaySliderValue = 0.7;
  }

  clearResults() {
    this.inferenceExecutor.clearResults();
    this.disableAllResultTabs();
    this.currentResultTab = 'input';
    this.setStageVisible('input', true);
    const inputVisibilityToggle = document.getElementById('inputVisibilityToggle');
    if (inputVisibilityToggle) inputVisibilityToggle.checked = true;

    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
      resultsSection.classList.add('hidden');
      resultsSection.classList.add('collapsed');
    }

    const overlayControl = document.getElementById('overlayControl');
    if (overlayControl) overlayControl.classList.add('hidden');

    const opacitySlider = document.getElementById('overlayOpacity');
    if (opacitySlider) {
      opacitySlider.disabled = false;
      opacitySlider.value = 0.5;
    }
    const opacityDisplay = document.getElementById('overlayOpacityValue');
    if (opacityDisplay) opacityDisplay.textContent = '50%';

    if (this.inputFile) {
      this.viewerController.loadBaseVolume(this.inputFile);
    }

    this.updateViewerInfo(this._lastLocationData);
  }

  // ==================== UI Helpers ====================

  updateOutput(msg) {
    this.console.log(msg);
  }

  setProgress(value, text) {
    this.progress.setProgress(value);
    const statusText = document.getElementById('statusText');
    if (statusText) {
      if (value >= 1) statusText.textContent = 'Complete';
      else if (text) statusText.textContent = text;
      else if (value > 0) statusText.textContent = 'Processing...';
    }
  }

  clearFiles() {
    this.fileIOController.clearFiles();
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.app = new SpinalCordToolboxApp();
});
