import { FileIOController } from './controllers/FileIOController.js';
import { ViewerController } from './controllers/ViewerController.js';
import { InferenceExecutor } from './controllers/InferenceExecutor.js';
import { MaskDrawingController } from './controllers/MaskDrawingController.js';
import { LNM_PIPELINES, getPipelineById } from './app/lnm-tasks.js';
import { YEO7_COLORMAP } from './app/lnm-labels.js';
import { computeParcelOverlap, summarizeNetworkOverlap } from './modules/parcel-overlap.js';
import { loadAtlasFromManifest, loadConnectomeFromManifest, decodeNiftiBuffer } from './modules/atlas-loader.js';
import { fcWeightedSum, decodeFcPack, summaryToNetworkWeights } from './modules/fc-weighted-sum.js';
import { applyThresholdDetailed } from './modules/threshold.js';
import { affineFromHeader, resampleAffine } from './modules/resample.js';
import { centroidOfMask, applyAffineToVoxel, computePrealignAffine, principalAxisAlign } from './modules/prealign.js';
import { writeNifti1 } from './modules/nifti-writer.js';
import { resampleBinaryMask, writeBinaryMaskNifti } from './modules/mask-transform.js';
import { serializeOverlapCsv } from './modules/overlap-export.js';
import { renderOverlapTable } from './modules/overlap-render.js';
import {
  loadFunctionProfilesFromManifest,
  rankFunctionalTerms,
  renderFunctionalProfileTable
} from './modules/function-profiles.js';
import { ConsoleOutput } from './modules/ui/ConsoleOutput.js';
import { ProgressManager } from './modules/ui/ProgressManager.js';
import { ModalManager } from './modules/ui/ModalManager.js';
import * as Config from './app/config.js';

const NETWORK_TOP_PERCENT_MAX = 10;
const NETWORK_TOP_PERCENT_STEP = 0.1;

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

function affineNearlyEqual(a, b, tolerance = 1e-3) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length < 3 || b.length < 3) return false;
  for (let r = 0; r < 3; r++) {
    if (!Array.isArray(a[r]) || !Array.isArray(b[r]) || a[r].length < 4 || b[r].length < 4) {
      return false;
    }
    for (let c = 0; c < 4; c++) {
      if (Math.abs(a[r][c] - b[r][c]) > tolerance) return false;
    }
  }
  return true;
}

function flattenAffine3Rows(affine) {
  return [
    affine[0][0], affine[0][1], affine[0][2], affine[0][3],
    affine[1][0], affine[1][1], affine[1][2], affine[1][3],
    affine[2][0], affine[2][1], affine[2][2], affine[2][3]
  ];
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

export function formatVersionLabel(version, buildInfo = null) {
  let label = version ? `v${version}` : '';
  const versionText = version || '';
  const bits = [];
  const sha = buildInfo?.sha || '';
  if (sha && !versionText.includes(sha)) bits.push(sha);
  if (buildInfo?.branch && buildInfo.branch !== 'main') bits.push(buildInfo.branch);
  if (buildInfo?.dirty) bits.push('dirty');
  if (bits.length) label += ` (${bits.join(', ')})`;
  return label;
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
    this.nativeStructuralFile = null;
    this.nativeStructuralInfo = null;
    this.fixedMni160Info = null;
    this.prealignSamplingAffine = null;
    this.lesionFile = null;
    this.overlapResult = null;
    this.brainmaskFile = null;     // populated by handleStageData('brainmask')
    this.lesionMaskFile = null;    // confirmed edited mask on fixed lnm-mni160
    this.lesionMaskConfirmed = false;
    this.autoLesionSeedFile = null;
    this.nativeLesionSeedFile = null;
    this.confirmedNativeLesionFile = null;
    this.maskReviewActive = false;
    this._pendingMaskResume = null;
    this.networkMapFile = null;    // Phase 4: populated by runFcNetworkMap
    this.networkMapData = null;    // Phase 5: raw Float32Array for re-thresholding
    this.networkMapDims = null;
    this.networkMapSpacing = null;
    this.networkMapAffine = null;
    this.networkMapBaseFile = null;
    this.thresholdedMaskFile = null; // Phase 5: thresholded binary NIfTI
    this.patientThresholdedMaskFile = null;
    this.patientAtlasFile = null;
    this.registrationTemplateFile = null;
    this.registeredT1MniFile = null;
    this.displacementMagnitudeFile = null;
    this.yeoAtlasMni160File = null;
    this.registrationCheckerboardFile = null;
    this.registrationQcMode = 'mni';
    this.registrationBlendValue = 0.5;
    this.affectedNetworkResult = null;
    this.functionProfiles = null;
    this._functionalProfileRenderPromise = Promise.resolve();
    this._thresholdPreviewTimer = null;
    this._thresholdPreviewRenderPromise = Promise.resolve();
    this._thresholdPreviewVersion = 0;
    this._thresholdProjectionWarningShown = false;
    this._inverseWarpQueue = Promise.resolve();
    this.hasRegistrationDisplacement = false;
    this.mniLesionFile = null;       // Phase 6: warped lesion at MNI160 1mm (pre-resample)
    this._mniLesionResolver = null;  // Phase 6: one-shot promise for warp-mask stage data
    this._perfStats = [];            // Phase 19: per-stage runtime markers
    this._perfRunStart = null;       // Phase 19: total runFullPipeline start
    this._stageDataResolvers = new Map();
    this._stepCompleteResolvers = new Map();
    this.manifest = null;          // populated lazily by ensureManifest()
    // Run analysis is input-driven: structural T1 uses the full auto chain,
    // while researcher-mode Yeo masks auto-select the manual network-map path.
    this.selectedPipeline = getPipelineById('lnm-yeo-auto') || LNM_PIPELINES[0];
    this.viewerLayerVisibility = {
      structural: true,
      brainmask: true,
      lesion: true,
      threshold: true,
      atlasQc: true
    };

    this.executor = new InferenceExecutor({
      updateOutput: (msg) => this.updateOutput(msg),
      setProgress: (frac, label) => this.handleWorkerProgress(frac, label),
      onStageData: (data) => this.handleStageData(data),
      onStepComplete: (step) => this.handleStepComplete(step),
      onError: (msg) => {
        this.updateOutput(`Worker error: ${msg}`);
        this._rejectPendingWorkerWaits(msg);
      },
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
    this.maskDrawingController = new MaskDrawingController({
      nv: this.nv,
      updateOutput: (msg) => this.updateOutput(msg)
    });

    this.aboutModal = new ModalManager('aboutModal');
    this.privacyModal = new ModalManager('privacyModal');
    this.citationsModal = new ModalManager('citationsModal');

    await this.setupViewer();
    this.viewerController.registerSctColormap(YEO7_COLORMAP, 'lnm-yeo7');
    this.bindEvents();
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
        this.runLesionSegmentation()
          .then(() => this.startLesionMaskReview({ seedFile: this.autoLesionSeedFile }))
          .catch(err => this.updateOutput(`Lesion segmentation failed: ${err.message}`));
      });
    }
    const manualMaskBtn = document.getElementById('startManualMaskButton');
    if (manualMaskBtn) {
      manualMaskBtn.addEventListener('click', () => {
        this.startLesionMaskReview({ blank: true })
          .catch(err => this.updateOutput(`Manual mask failed: ${err.message}`));
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

    const checkAtlasAlignmentBtn = document.getElementById('checkAtlasAlignmentButton');
    if (checkAtlasAlignmentBtn) {
      checkAtlasAlignmentBtn.disabled = true;
      checkAtlasAlignmentBtn.addEventListener('click', () => {
        this.showRegistrationQc().catch(
          err => this.updateOutput(`Atlas alignment QC failed: ${err.message}`)
        );
      });
    }

    const registrationQcMode = document.getElementById('registrationQcMode');
    if (registrationQcMode) {
      this.registrationQcMode = registrationQcMode.value || this.registrationQcMode;
      registrationQcMode.addEventListener('change', () => {
        this.registrationQcMode = registrationQcMode.value || 'patient';
        if (this.hasRegistrationDisplacement) {
          this.showRegistrationQc().catch(
            err => this.updateOutput(`Registration QC failed: ${err.message}`)
          );
        }
      });
    }
    const registrationBlendValue = document.getElementById('registrationBlendValue');
    if (registrationBlendValue) {
      this.registrationBlendValue = this.getRegistrationBlendValue();
      this.updateRegistrationBlendLabel(this.registrationBlendValue);
      const handleRegistrationBlendInput = () => {
        this.handleRegistrationBlendInput().catch(
          err => this.updateOutput(`Registration blend update failed: ${err.message}`)
        );
      };
      registrationBlendValue.addEventListener('input', handleRegistrationBlendInput);
      registrationBlendValue.addEventListener('change', handleRegistrationBlendInput);
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
    const thresholdSym = document.getElementById('networkThresholdSymmetric');
    const thresholdMinCluster = document.getElementById('networkThresholdMinCluster');
    const triggerRecompute = () => {
      this.updateThresholdValueLabel();
      if (this.networkMapData) {
        try { this.applyNetworkThreshold(); }
        catch (err) { this.updateOutput(`Threshold failed: ${err.message}`); }
      }
    };
    if (thresholdValue) thresholdValue.addEventListener('input', triggerRecompute);
    if (thresholdSym) thresholdSym.addEventListener('change', triggerRecompute);
    if (thresholdMinCluster) {
      thresholdMinCluster.addEventListener('input', triggerRecompute);
      thresholdMinCluster.addEventListener('change', triggerRecompute);
    }
    this.configureTopPercentThresholdSlider();
    this.updateThresholdValueLabel();

    const downloadThreshBtn = document.getElementById('downloadThresholdedNetworkMapButton');
    if (downloadThreshBtn) {
      downloadThreshBtn.disabled = true;
      downloadThreshBtn.addEventListener('click', () => this.downloadThresholdedNetworkMap());
    }

    const showAtlasQcBtn = document.getElementById('showSubjectAtlasButton');
    if (showAtlasQcBtn) {
      showAtlasQcBtn.disabled = true;
      showAtlasQcBtn.addEventListener('click', () => {
        this.showSubjectSpaceAtlas().catch(
          err => this.updateOutput(`Subject-space atlas failed: ${err.message}`)
        );
      });
    }

    const downloadAtlasQcBtn = document.getElementById('downloadSubjectAtlasButton');
    if (downloadAtlasQcBtn) {
      downloadAtlasQcBtn.disabled = true;
      downloadAtlasQcBtn.addEventListener('click', () => this.downloadSubjectSpaceAtlas());
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

    this.bindViewerLayerToggles();

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

    this.bindMaskDrawingControls();
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

  bindMaskDrawingControls() {
    const bindClick = (id, fn) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('click', fn);
    };
    bindClick('maskPaintButton', () => this.setMaskDrawingTool('paint'));
    bindClick('maskEraseButton', () => this.setMaskDrawingTool('erase'));
    bindClick('maskEraseClusterButton', () => this.setMaskDrawingTool('eraseCluster'));
    bindClick('maskUndoButton', () => this.maskDrawingController?.undo());
    bindClick('maskBlankButton', () => {
      this.startLesionMaskReview({ blank: true })
        .catch(err => this.updateOutput(`Blank mask failed: ${err.message}`));
    });
    bindClick('maskSmoothButton', () => {
      if (!this.maskDrawingController?.smoothDrawing()) {
        this.updateOutput('Smooth mask needs an editable drawing.');
      }
    });
    bindClick('maskInterpolateButton', () => {
      const axis = Number(document.getElementById('maskInterpolateAxis')?.value || 0);
      if (!this.maskDrawingController?.interpolateAcrossSlices(axis)) {
        this.updateOutput('Interpolate needs at least two drawn slices on the selected axis.');
      }
    });
    bindClick('confirmLesionMaskButton', () => {
      this.confirmLesionDrawing({ resumePipeline: true })
        .catch(err => this.updateOutput(`Confirm lesion mask failed: ${err.message}`));
    });
    bindClick('downloadEditedLesionMaskButton', () => {
      this.downloadEditedLesionMask()
        .catch(err => this.updateOutput(`Edited mask download failed: ${err.message}`));
    });

    const brush = document.getElementById('maskBrushSize');
    if (brush) {
      brush.addEventListener('input', () => {
        const size = this.maskDrawingController?.setBrushSize(brush.value) || 1;
        const label = document.getElementById('maskBrushSizeLabel');
        if (label) label.textContent = `${size} vox`;
      });
    }
    const filled = document.getElementById('maskFilledToggle');
    if (filled) {
      filled.addEventListener('change', () => {
        this.maskDrawingController?.setFilled(filled.checked);
      });
    }
    const shape = document.getElementById('maskShapeSelect');
    if (shape) {
      shape.addEventListener('change', () => {
        this.maskDrawingController?.setPenShape(shape.value);
      });
    }
    this.refreshMaskDrawingControls();
  }

  setMaskDrawingTool(tool) {
    this.maskDrawingController?.ensureDrawing();
    this.maskDrawingController?.setTool(tool);
    for (const id of ['maskPaintButton', 'maskEraseButton', 'maskEraseClusterButton']) {
      const el = document.getElementById(id);
      if (el) el.classList.remove('active');
    }
    const activeId = tool === 'erase'
      ? 'maskEraseButton'
      : tool === 'eraseCluster'
        ? 'maskEraseClusterButton'
        : 'maskPaintButton';
    document.getElementById(activeId)?.classList.add('active');
  }

  refreshMaskDrawingControls() {
    const toolbar = document.getElementById('maskDrawingToolbar');
    const available = !!(this.nativeStructuralFile || this.structuralFile || this.autoLesionSeedFile || this.confirmedNativeLesionFile);
    if (toolbar) toolbar.classList.toggle('hidden', !available);
    const confirm = document.getElementById('confirmLesionMaskButton');
    if (confirm) confirm.disabled = !available;
    const download = document.getElementById('downloadEditedLesionMaskButton');
    if (download) download.disabled = !(this.confirmedNativeLesionFile || this.maskDrawingController?.hasDrawing);
    const status = document.getElementById('maskReviewStatus');
    if (status) {
      status.textContent = this.maskReviewActive
        ? 'Review lesion mask'
        : this.lesionMaskConfirmed
          ? 'Mask confirmed'
          : available
            ? 'Mask tools ready'
            : '';
    }
  }

  getViewerLayerControlConfig() {
    return [
      { layer: 'structural', id: 'layerToggleT1', stages: ['structural'] },
      { layer: 'brainmask', id: 'layerToggleBrainMask', stages: ['brainmask'] },
      { layer: 'lesion', id: 'layerToggleLesionMask', stages: ['segmentation', 'lesion'] },
      { layer: 'threshold', id: 'layerToggleThresholdMap', stages: ['threshold-preview'] },
      { layer: 'atlasQc', id: 'layerToggleAtlasQc', stages: ['atlas-qc'] }
    ];
  }

  bindViewerLayerToggles() {
    for (const config of this.getViewerLayerControlConfig()) {
      const el = document.getElementById(config.id);
      if (!el) continue;
      el.addEventListener('change', (event) => {
        this.viewerLayerVisibility[config.layer] = !!event.target.checked;
        this.applyViewerLayerVisibility(config.layer);
        this.refreshViewerLayerControls();
      });
    }
    this.refreshViewerLayerControls();
  }

  getViewerLayerAvailable(layer) {
    switch (layer) {
      case 'structural':
        return !!this.structuralFile;
      case 'brainmask':
        return !!this.brainmaskFile;
      case 'lesion':
        return !!(this.lesionMaskFile || this.autoLesionSeedFile || this.confirmedNativeLesionFile || this.lesionFile);
      case 'threshold':
        return !!(this.patientThresholdedMaskFile || this.thresholdedMaskFile);
      case 'atlasQc':
        return !!(this.patientAtlasFile || (this.hasRegistrationDisplacement && this.yeoAtlasMni160File));
      default:
        return false;
    }
  }

  refreshViewerLayerControls() {
    for (const config of this.getViewerLayerControlConfig()) {
      const el = document.getElementById(config.id);
      if (!el) continue;
      const available = this.getViewerLayerAvailable(config.layer);
      el.disabled = !available;
      el.checked = this.viewerLayerVisibility[config.layer] !== false;
    }
    this.refreshSubjectAtlasControls();
  }

  refreshSubjectAtlasControls() {
    const canProject = !!(
      this.structuralFile &&
      this.hasRegistrationDisplacement &&
      this.executor?.runInverseWarpMask
    );
    const showBtn = document.getElementById('showSubjectAtlasButton');
    if (showBtn) showBtn.disabled = !canProject;
    const checkBtn = document.getElementById('checkAtlasAlignmentButton');
    if (checkBtn) checkBtn.disabled = !canProject;
    const downloadBtn = document.getElementById('downloadSubjectAtlasButton');
    if (downloadBtn) downloadBtn.disabled = !this.patientAtlasFile;
  }

  applyViewerLayerVisibility(layer = null) {
    if (!this.viewerController?.setStageVisible) return;
    for (const config of this.getViewerLayerControlConfig()) {
      if (layer && config.layer !== layer) continue;
      const visible = this.viewerLayerVisibility[config.layer] !== false;
      for (const stage of config.stages) {
        this.viewerController.setStageVisible(stage, visible);
      }
    }
  }

  layerVisible(layer) {
    return this.viewerLayerVisibility[layer] !== false;
  }

  // Phase 13 + Phase 40: populate every visible version slot from
  // Config.VERSION. Best-effort augment with build-info.json (written
  // by web/run.sh for local dev + by .github/workflows/ for deploys)
  // to surface the commit SHA / branch / dirty flag.
  //
  //   local dev    -> "v0.17.0 (abc1234, dirty)" on main
  //   staging      -> "v0.17.0-staging+abc1234"  (sed'd by deploy-pages.yml)
  //                   with build-info.json SHA suppressed as duplicate
  //   production   -> "v0.17.0" (release-tag build; build-info.json may
  //                   carry the tag SHA)
  //
  // build-info.json is fetched best-effort — a 404 falls back to just
  // VERSION, so the static deploy works whether or not the file exists.
  async populateVersionLabel() {
    let buildInfo = null;
    try {
      const r = await fetch('build-info.json', { cache: 'no-store' });
      if (r.ok) {
        buildInfo = await r.json();
      }
    } catch (e) { /* best-effort: silent fallback to VERSION */ }
    const label = formatVersionLabel(Config.VERSION, buildInfo);
    const ids = ['aboutAppVersion', 'appVersion', 'footerVersion'];
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el) el.textContent = label;
    }
  }

  async setStructural(file) {
    if (!file) return;
    this.structuralFile = file;
    this.nativeStructuralFile = file;
    this.nativeStructuralInfo = null;
    this.fixedMni160Info = null;
    this.prealignSamplingAffine = null;
    this.autoLesionSeedFile = null;
    this.nativeLesionSeedFile = null;
    this.confirmedNativeLesionFile = null;
    this.lesionMaskFile = null;
    this.lesionMaskConfirmed = false;
    this.maskReviewActive = false;
    this._pendingMaskResume = null;
    this.hasRegistrationDisplacement = false;
    this.patientThresholdedMaskFile = null;
    this.patientAtlasFile = null;
    this.registeredT1MniFile = null;
    this.displacementMagnitudeFile = null;
    this.registrationCheckerboardFile = null;
    this._thresholdProjectionWarningShown = false;
    await this.viewerController.loadBaseVolume(file, {
      stage: 'structural',
      visible: this.layerVisible('structural')
    });
    this.refreshViewerLayerControls();
    this.refreshMaskDrawingControls();
    this.updateOutput(`Structural image ready: ${file.name}`);
    // Phase 31: auto-promote the pipeline selection. A structural T1
    // means the explicit Run analysis action should use the full auto chain.
    this._autoPromotePipeline('lnm-yeo-auto');
  }

  async setLesion(file) {
    if (!file) return;
    this.lesionFile = file;
    if (this.structuralFile) {
      await this.viewerController.loadOverlay(file, 'red', 0.5, {
        stage: 'lesion',
        visible: this.layerVisible('lesion')
      });
    } else {
      await this.viewerController.loadBaseVolume(file, {
        stage: 'lesion',
        visible: this.layerVisible('lesion')
      });
    }
    this.refreshViewerLayerControls();
    this.updateOutput(`Lesion mask ready: ${file.name}`);
    // Phase 31: a manual lesion mask without a structural T1 means the
    // user wants the manual-mask network-map chain (overlap + FC +
    // threshold). With a structural already loaded we leave the auto
    // pipeline selected.
    if (!this.structuralFile) {
      this._autoPromotePipeline('lnm-network-map');
    }
  }

  // Phase 31 + selector cleanup: Run analysis is driven by the loaded input.
  // Structural T1 promotes to the full auto chain; a researcher-mode Yeo mask
  // promotes to the manual network-map path.
  _autoPromotePipeline(pipelineId) {
    const pipeline = getPipelineById(pipelineId);
    if (!pipeline) return;
    this.selectedPipeline = pipeline;
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
    this.showYeoLabelCoverageNote(parcelResult.voxelsOutsideAtlas, parcelResult.totalLesionVoxels);

    const tableEl = document.getElementById('networkOverlapTable');
    if (tableEl) {
      renderOverlapTable(tableEl, summary, {
        colormap: YEO7_COLORMAP,
        networkLabels: atlas.networkLabels
      });
    }
    this.clearAffectedNetworkTable();
    const csvButton = document.getElementById('downloadOverlapCsv');
    if (csvButton) csvButton.disabled = false;

    this.updateOutput(
      `Overlap computed for ${summary.networks.length} networks ` +
      `(${parcelResult.totalLesionVoxels - parcelResult.voxelsOutsideAtlas} of ` +
      `${parcelResult.totalLesionVoxels} lesion voxels assigned to Yeo cortical ` +
      `network labels; ${parcelResult.voxelsOutsideAtlas} unlabeled).`
    );
    await this.updateDirectFunctionProfile();
  }

  async ensureFunctionProfiles() {
    if (this.functionProfiles) return this.functionProfiles;
    const manifest = await this.ensureManifest();
    const { profiles } = await loadFunctionProfilesFromManifest(
      'yeo7-neurosynth-v7-function-profiles',
      { manifest }
    );
    this.functionProfiles = profiles;
    return profiles;
  }

  clearFunctionProfileTable(resultId, tableId) {
    const resultEl = document.getElementById(resultId);
    if (resultEl) resultEl.classList.add('hidden');
    const tableEl = document.getElementById(tableId);
    if (tableEl) tableEl.innerHTML = '';
  }

  async renderFunctionProfileForSummary(summary, {
    resultId,
    tableId,
    emptyLabel = 'No functional associations'
  }) {
    const resultEl = document.getElementById(resultId);
    const tableEl = document.getElementById(tableId);
    if (!resultEl || !tableEl) return null;
    if (!summary || !Array.isArray(summary.networks) || summary.networks.length === 0) {
      this.clearFunctionProfileTable(resultId, tableId);
      return null;
    }

    const profiles = await this.ensureFunctionProfiles();
    const ranked = rankFunctionalTerms(summary, profiles, {
      topN: 8,
      minScore: 0.01
    });
    renderFunctionalProfileTable(tableEl, ranked, {
      sourceLabel: profiles.sourceLabel || 'Neurosynth v7 via NiMARE',
      emptyLabel
    });
    resultEl.classList.remove('hidden');
    return ranked;
  }

  updateDirectFunctionProfile() {
    this._functionalProfileRenderPromise = this._functionalProfileRenderPromise
      .then(() => this.renderFunctionProfileForSummary(this.overlapResult?.summary, {
        resultId: 'directFunctionProfileResults',
        tableId: 'directFunctionProfileTable',
        emptyLabel: 'No direct-overlap functional associations'
      }))
      .catch(err => {
        this.clearFunctionProfileTable('directFunctionProfileResults', 'directFunctionProfileTable');
        this.updateOutput(`Functional profiles unavailable: ${err.message}`);
        return null;
      });
    return this._functionalProfileRenderPromise;
  }

  updateAffectedFunctionProfile() {
    this._functionalProfileRenderPromise = this._functionalProfileRenderPromise
      .then(() => this.renderFunctionProfileForSummary(this.affectedNetworkResult?.summary, {
        resultId: 'mapFunctionProfileResults',
        tableId: 'mapFunctionProfileTable',
        emptyLabel: 'No connectivity-map functional associations'
      }))
      .catch(err => {
        this.clearFunctionProfileTable('mapFunctionProfileResults', 'mapFunctionProfileTable');
        this.updateOutput(`Functional profiles unavailable: ${err.message}`);
        return null;
      });
    return this._functionalProfileRenderPromise;
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

  async ensureNativeStructuralInfo() {
    if (this.nativeStructuralInfo) return this.nativeStructuralInfo;
    const file = this.nativeStructuralFile || this.structuralFile;
    if (!file || typeof file.arrayBuffer !== 'function') {
      throw new Error('Native structural image is not available.');
    }
    const decoded = await decodeNiftiBuffer(await file.arrayBuffer());
    this.nativeStructuralFile = file;
    this.nativeStructuralInfo = {
      dims: decoded.dims,
      affine: affineFromHeader(decoded.header)
    };
    return this.nativeStructuralInfo;
  }

  async ensureFixedMni160Info() {
    if (this.fixedMni160Info) return this.fixedMni160Info;
    const mni160 = await loadAtlasFromManifest('lnm-mni160');
    this.fixedMni160Info = {
      dims: mni160.dims,
      affine: affineFromHeader(mni160.header),
      spacing: [1, 1, 1]
    };
    return this.fixedMni160Info;
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
    const brainmaskReady = this._waitForStageData('brainmask');
    const brainmaskStepDone = this._waitForStepComplete('brainmask');
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
    await Promise.all([brainmaskReady, brainmaskStepDone]);
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

  _waitForStageData(stage) {
    return new Promise((resolve, reject) => {
      const waiters = this._stageDataResolvers.get(stage) || [];
      waiters.push({ resolve, reject });
      this._stageDataResolvers.set(stage, waiters);
    });
  }

  _waitForStepComplete(step) {
    return new Promise((resolve, reject) => {
      const waiters = this._stepCompleteResolvers.get(step) || [];
      waiters.push({ resolve, reject });
      this._stepCompleteResolvers.set(step, waiters);
    });
  }

  _resolveStageData(stage, data) {
    const waiters = this._stageDataResolvers.get(stage);
    if (!waiters?.length) return;
    this._stageDataResolvers.delete(stage);
    for (const waiter of waiters) waiter.resolve(data);
  }

  _resolveStepComplete(step) {
    const waiters = this._stepCompleteResolvers.get(step);
    if (!waiters?.length) return;
    this._stepCompleteResolvers.delete(step);
    for (const waiter of waiters) waiter.resolve(step);
  }

  _rejectPendingWorkerWaits(message) {
    const err = message instanceof Error ? message : new Error(String(message || 'Worker failed'));
    for (const waiters of this._stageDataResolvers.values()) {
      for (const waiter of waiters) waiter.reject(err);
    }
    for (const waiters of this._stepCompleteResolvers.values()) {
      for (const waiter of waiters) waiter.reject(err);
    }
    this._stageDataResolvers.clear();
    this._stepCompleteResolvers.clear();
    if (this._mniLesionResolver) {
      const resolver = this._mniLesionResolver;
      this._mniLesionResolver = null;
      resolver.reject(err);
    }
  }

  handleStepComplete(step) {
    this.updateOutput(`Worker step '${step}' complete.`);
    // Phase 14: a completed step ends the cancellable window.
    const cancelBtn = document.getElementById('cancelButton');
    if (cancelBtn) cancelBtn.disabled = true;
    this._resolveStepComplete(step);
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
          .loadOverlay(file, 'green', 0.4, {
            stage: 'brainmask',
            visible: this.layerVisible('brainmask')
          })
          .catch(err => this.updateOutput(`Brain mask render error: ${err.message}`));
      }
      const btn = document.getElementById('downloadBrainMaskButton');
      if (btn) btn.disabled = false;
      this.refreshViewerLayerControls();
      this.updateOutput('Brain mask ready.');
      this._resolveStageData(data.stage, data);
      return;
    }
    if (data.stage === 'segmentation' && data.niftiData) {
      const file = arrayBufferToFile(data.niftiData, 'lesion.nii');
      this.autoLesionSeedFile = file;
      this.lesionMaskFile = null;
      this.lesionMaskConfirmed = false;
      const btn = document.getElementById('downloadLesionMaskButton');
      if (btn) btn.disabled = false;
      this.refreshViewerLayerControls();
      this.refreshMaskDrawingControls();
      this.updateOutput('Automatic lesion seed ready for manual review.');
      this._resolveStageData(data.stage, data);
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
      this._resolveStageData(data.stage, data);
      return;
    }
    if (data.stage === 'registered-t1-mni160' && data.niftiData) {
      this.registeredT1MniFile = arrayBufferToFile(data.niftiData, 'lnm-registered-t1-mni160.nii');
      this.registrationCheckerboardFile = null;
      this.updateOutput('Registered T1 QC volume ready on the MNI160 grid.');
      this._resolveStageData(data.stage, data);
      return;
    }
    if (data.stage === 'registration-displacement-mag' && data.niftiData) {
      this.displacementMagnitudeFile = arrayBufferToFile(data.niftiData, 'lnm-registration-displacement-mag.nii');
      this.updateOutput('Registration displacement QC map ready.');
      this._resolveStageData(data.stage, data);
      return;
    }
    if (data.stage === 'threshold-patient' && data.niftiData) {
      this.patientThresholdedMaskFile = arrayBufferToFile(data.niftiData, 'lnm-network-map-thresh-patient.nii');
      this.refreshViewerLayerControls();
      this.updateOutput('Threshold map projected to patient T1 space.');
      this._resolveStageData(data.stage, data);
      return;
    }
    if (data.stage === 'atlas-patient' && data.niftiData) {
      this.patientAtlasFile = arrayBufferToFile(data.niftiData, 'lnm-yeo7-atlas-patient.nii');
      this.refreshViewerLayerControls();
      this.updateOutput('Yeo atlas projected to patient T1 space.');
      this._resolveStageData(data.stage, data);
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
    const segmentationReady = this._waitForStageData('segmentation');
    const segmentationStepDone = this._waitForStepComplete('inference');
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
    await Promise.all([segmentationReady, segmentationStepDone]);
  }

  downloadLesionMask() {
    const file = this.confirmedNativeLesionFile || this.lesionMaskFile || this.autoLesionSeedFile;
    if (!file) {
      this.updateOutput('No lesion mask available yet.');
      return;
    }
    const url = URL.createObjectURL(file);
    const a = document.createElement('a');
    a.href = url;
    a.download = this.confirmedNativeLesionFile ? 'lnm-lesion-edited-native.nii' : 'lnm-lesion-seed.nii';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  async downloadEditedLesionMask() {
    if (this.confirmedNativeLesionFile) {
      const url = URL.createObjectURL(this.confirmedNativeLesionFile);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'lnm-lesion-edited-native.nii';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      return;
    }
    await this.maskDrawingController.downloadDrawing('lnm-lesion-edited-native.nii');
  }

  async resampleSeedMaskToNative(seedFile) {
    if (!seedFile) return null;
    const native = await this.ensureNativeStructuralInfo();
    const seed = await decodeNiftiBuffer(await seedFile.arrayBuffer());
    const seedAffine = this.prealignSamplingAffine || affineFromHeader(seed.header);
    const nativeMask = resampleBinaryMask({
      data: seed.data,
      srcDims: seed.dims,
      srcAffine: seedAffine,
      dstDims: native.dims,
      dstAffine: native.affine
    });
    const nativeNifti = writeBinaryMaskNifti(nativeMask, {
      dims: native.dims,
      affine: native.affine,
      spacing: [1, 1, 1],
      description: 'LNM editable lesion seed projected to native T1 grid'
    });
    this.nativeLesionSeedFile = arrayBufferToFile(nativeNifti, 'lnm-lesion-seed-native.nii');
    return this.nativeLesionSeedFile;
  }

  async startLesionMaskReview({ seedFile = null, blank = false } = {}) {
    const baseFile = this.nativeStructuralFile || this.structuralFile;
    if (!baseFile) {
      this.updateOutput('Drop a structural T1 before editing a lesion mask.');
      return null;
    }
    this.nativeStructuralFile = baseFile;
    await this.ensureNativeStructuralInfo();
    await this.viewerController.loadBaseVolume(baseFile, {
      stage: 'structural',
      visible: this.layerVisible('structural')
    });

    const seed = blank ? null : await this.resampleSeedMaskToNative(seedFile || this.autoLesionSeedFile);
    if (seed) await this.maskDrawingController.loadSeedFile(seed);
    else this.maskDrawingController.startBlank();

    this.maskReviewActive = true;
    this.lesionMaskConfirmed = false;
    this.lesionMaskFile = null;
    this.setMaskDrawingTool('paint');
    this.refreshViewerLayerControls();
    this.refreshMaskDrawingControls();
    this.updateOutput('Review/edit the lesion mask, then confirm it to continue analysis.');
    return seed;
  }

  async confirmLesionDrawing({ resumePipeline = false } = {}) {
    if (!this.maskDrawingController?.hasDrawing) {
      await this.startLesionMaskReview({ seedFile: this.autoLesionSeedFile, blank: !this.autoLesionSeedFile });
    }
    const nativeFile = await this.maskDrawingController.exportDrawingFile('lnm-lesion-edited-native.nii');

    if (!this.prealignSamplingAffine || !this.fixedMni160Info) {
      await this.prealignToMni160({ skipIfAligned: true });
    }
    this.confirmedNativeLesionFile = nativeFile;
    const fixed = await this.ensureFixedMni160Info();
    const native = await decodeNiftiBuffer(await nativeFile.arrayBuffer());
    const nativeAffine = affineFromHeader(native.header);
    const mniMask = resampleBinaryMask({
      data: native.data,
      srcDims: native.dims,
      srcAffine: nativeAffine,
      dstDims: fixed.dims,
      dstAffine: this.prealignSamplingAffine || fixed.affine
    });
    const mniNifti = writeBinaryMaskNifti(mniMask, {
      dims: fixed.dims,
      affine: fixed.affine,
      spacing: fixed.spacing || [1, 1, 1],
      description: 'LNM confirmed edited lesion mask on fixed lnm-mni160 grid'
    });

    this.lesionMaskFile = arrayBufferToFile(mniNifti, 'lnm-lesion-confirmed-mni160.nii');
    this.lesionMaskConfirmed = true;
    this.maskReviewActive = false;
    this.maskDrawingController.close();
    const btn = document.getElementById('downloadLesionMaskButton');
    if (btn) btn.disabled = false;
    this.refreshViewerLayerControls();
    this.refreshMaskDrawingControls();
    this.updateOutput('Edited lesion mask confirmed on the fixed MNI160 grid.');

    if (resumePipeline) await this.resumePipelineAfterMaskConfirmation();
    return this.lesionMaskFile;
  }

  async resumePipelineAfterMaskConfirmation() {
    if (!this._pendingMaskResume) return;
    const pending = this._pendingMaskResume;
    this._pendingMaskResume = null;
    this.updateOutput('Resuming analysis with confirmed lesion mask...');
    const status = await this._runPipelineStages(pending.pipeline, pending.nextStageIndex);
    if (status === 'complete') this.logPipelineComplete();
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
    const pack = decodeFcPack(arrayBuffer, {
      voxelOrder: manifestEntry.voxelOrder,
      ...index
    });

    const NETWORK_ORDER = [
      'Visual', 'Somatomotor', 'DorsalAttention', 'VentralAttention',
      'Limbic', 'Frontoparietal', 'Default'
    ];
    const weights = summaryToNetworkWeights(this.overlapResult.summary, NETWORK_ORDER);
    const dims = index.shape.slice(1);   // shape = [7, X, Y, Z]
    const atlasAssetId = index.atlasAssetId || manifestEntry.atlasAssetId || 'yeo7-2mm';
    const atlas = await loadAtlasFromManifest(atlasAssetId);
    if (!dimsEqual(dims, atlas.dims)) {
      throw new Error(
        `FC pack grid ${dims.join('x')} does not match ${atlasAssetId} atlas ` +
        `${atlas.dims.join('x')}`
      );
    }
    const atlasAffine = affineFromHeader(atlas.header);
    const flatAffine = flattenAffine3Rows(atlasAffine);
    this.updateOutput(
      `Computing network map: weights=[${
        Array.from(weights).map(w => w.toFixed(2)).join(', ')
      }]`
    );
    const fcMap = fcWeightedSum(weights, pack.tMaps, dims);

    // Stash for Phase 5 re-thresholding without recomputing the FC sum.
    this.networkMapData = fcMap;
    this.networkMapDims = dims;
    const spacingMm = manifestEntry.atlasResolutionMm || atlas.manifestEntry?.resolutionMm || 2;
    this.networkMapSpacing = [spacingMm, spacingMm, spacingMm];
    this.networkMapAffine = flatAffine;

    // Wrap as a NIfTI for download / overlay. The Yeo atlas's spacing /
    // affine is the canonical pose for the FC pack — manifestEntry from
    // the connectome carries atlasResolutionMm; the Yeo atlas itself is
    // the same grid (99x117x95 2mm).
    const niftiBuffer = writeNifti1(fcMap, {
      dims,
      spacing: this.networkMapSpacing,
      affine: this.networkMapAffine,
      description: 'LNM Yeo7 FC weighted sum'
    });
    this.networkMapFile = arrayBufferToFile(niftiBuffer, 'lnm-network-map.nii');

    await this.displayNetworkMapOnYeoTemplate(atlas, flatAffine);
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

  buildYeoBrainMaskBaseFile(atlas, flatAffine) {
    const base = new Float32Array(atlas.data.length);
    for (let i = 0; i < atlas.data.length; i++) {
      base[i] = atlas.data[i] > 0 ? 1 : 0;
    }
    const niftiBuffer = writeNifti1(base, {
      dims: atlas.dims,
      spacing: this.networkMapSpacing,
      affine: flatAffine,
      description: 'LNM Yeo7 atlas brain mask display base'
    });
    return arrayBufferToFile(niftiBuffer, 'lnm-yeo-brain-mask.nii');
  }

  async displayNetworkMapOnYeoTemplate(atlas, flatAffine) {
    if (!this.networkMapFile) return;
    try {
      this.networkMapBaseFile = this.buildYeoBrainMaskBaseFile(atlas, flatAffine);
      const entries = [
        { file: this.networkMapBaseFile, stage: 'yeo-brain-mask' }
      ];
      if (this.lesionFile) {
        entries.push({
          file: this.lesionFile,
          colormap: 'red',
          opacity: 0.35,
          stage: 'lesion',
          visible: this.layerVisible('lesion')
        });
      }
      entries.push({
        file: this.networkMapFile,
        colormap: 'blue2red',
        opacity: 0.5,
        stage: 'network-map',
        scalar: true,
        symmetricCal: true
      });
      await this.viewerController.loadVolumeStack(entries);
      this.applyViewerLayerVisibility();
      this.refreshViewerLayerControls();
      this.updateOutput('Network map displayed on the Yeo atlas grid.');
    } catch (err) {
      this.updateOutput(`Network-map render error: ${err.message}`);
    }
  }

  configureTopPercentThresholdSlider({ resetValue = false } = {}) {
    const valueEl = document.getElementById('networkThresholdValue');
    if (!valueEl) return;
    valueEl.min = '0';
    valueEl.max = String(NETWORK_TOP_PERCENT_MAX);
    valueEl.step = String(NETWORK_TOP_PERCENT_STEP);
    if (resetValue) valueEl.value = '5';
  }

  updateThresholdValueLabel() {
    const valueEl = document.getElementById('networkThresholdValue');
    const labelEl = document.getElementById('networkThresholdValueLabel');
    if (!valueEl || !labelEl) return;
    const v = Number(valueEl.value);
    labelEl.textContent = `${v.toFixed(Number.isInteger(v) ? 0 : 1)}%`;
  }

  // Phase 5: re-threshold the cached network map, update the
  // thresholded-mask download, and schedule a live binary preview overlay.
  // The scalar FC t-map stays visible as context; the red preview overlay
  // is replaced as the top-percent slider / cluster controls change.
  //
  // Reads the threshold UI controls:
  //   #networkThresholdValue   range slider, 0..10 top % of voxels.
  //   #networkThresholdSymmetric  checkbox: rank by |t| magnitude.
  //   #networkThresholdMinCluster number input.
  applyNetworkThreshold() {
    if (!this.networkMapData) {
      this.clearAffectedNetworkTable();
      this.updateOutput('Compute the network map first.');
      return null;
    }
    const valueEl = document.getElementById('networkThresholdValue');
    const symEl = document.getElementById('networkThresholdSymmetric');
    const minClEl = document.getElementById('networkThresholdMinCluster');
    const rawValue = valueEl ? Number(valueEl.value) : 5;
    const symmetric = symEl ? !!symEl.checked : true;
    const minClusterVoxels = minClEl ? Number(minClEl.value) || 0 : 0;
    // The UI label is "Top voxels": 5 means keep the strongest 5%.
    // The pure threshold engine takes a percentile cutoff q where q=0.95
    // keeps roughly the top 5%, so invert the UI value here.
    const topPercent = Math.max(0, Math.min(NETWORK_TOP_PERCENT_MAX, rawValue));
    const value = 1 - (topPercent / 100);
    const thresholdResult = applyThresholdDetailed(this.networkMapData, this.networkMapDims, {
      mode: 'percentile', value, symmetric, minClusterVoxels
    });
    const mask = thresholdResult.mask;
    const count = thresholdResult.count;
    const cutoff = thresholdResult.threshold;
    const niftiBuffer = writeNifti1(mask, {
      dims: this.networkMapDims,
      spacing: this.networkMapSpacing,
      affine: this.networkMapAffine,
      description: `LNM thresholded topPercent=${topPercent} q=${value} magnitude=${symmetric} cluster>=${minClusterVoxels}`
    });
    this.thresholdedMaskFile = arrayBufferToFile(niftiBuffer, 'lnm-network-map-thresh.nii');
    this.patientThresholdedMaskFile = null;
    const dlBtn = document.getElementById('downloadThresholdedNetworkMapButton');
    if (dlBtn) dlBtn.disabled = false;
    const summaryEl = document.getElementById('networkThresholdSummary');
    if (summaryEl) {
      const clusterText = minClusterVoxels > 1
        ? `; cluster≥${minClusterVoxels} removed ${thresholdResult.removedByCluster.toLocaleString()} voxels`
        : '';
      const topLabel = topPercent.toFixed(Number.isInteger(topPercent) ? 0 : 1);
      summaryEl.textContent =
        `${count.toLocaleString()} voxels survive top ${topLabel}%` +
        (symmetric ? ' (|t|)' : ' (t)') +
        `; cutoff ${cutoff.toPrecision(3)}` +
        clusterText;
    }
    this.updateAffectedNetworkTable(mask);
    this.refreshViewerLayerControls();
    this.scheduleThresholdPreviewOverlay();
    return mask;
  }

  updateAffectedNetworkTable(mask) {
    this.affectedNetworkResult = null;
    const resultEl = document.getElementById('affectedNetworkResults');
    const tableEl = document.getElementById('affectedNetworkTable');
    const atlas = this.overlapResult?.atlas;

    if (!mask || !atlas) {
      this.clearAffectedNetworkTable();
      return null;
    }
    if (!dimsEqual(this.networkMapDims, atlas.dims)) {
      this.clearAffectedNetworkTable();
      this.updateOutput(
        `Affected-network labels unavailable: threshold map dims ` +
        `${this.networkMapDims?.join('x') || 'unknown'} do not match atlas ` +
        `${atlas.dims.join('x')}.`
      );
      return null;
    }

    const parcelResult = computeParcelOverlap({
      lesion: mask,
      atlas: atlas.data,
      dims: atlas.dims
    });
    const summary = summarizeNetworkOverlap(parcelResult, atlas.networkLabels);
    this.affectedNetworkResult = { parcelResult, summary, atlas };

    if (tableEl) {
      renderOverlapTable(tableEl, summary, {
        colormap: YEO7_COLORMAP,
        percentHeader: '% of map',
        emptyLabel: 'No affected voxels'
      });
    }
    if (resultEl) resultEl.classList.remove('hidden');
    this.updateAffectedFunctionProfile();
    return this.affectedNetworkResult;
  }

  clearAffectedNetworkTable() {
    this.affectedNetworkResult = null;
    const resultEl = document.getElementById('affectedNetworkResults');
    if (resultEl) resultEl.classList.add('hidden');
    const tableEl = document.getElementById('affectedNetworkTable');
    if (tableEl) tableEl.innerHTML = '';
    this.clearFunctionProfileTable('mapFunctionProfileResults', 'mapFunctionProfileTable');
  }

  cancelThresholdPreviewOverlay({ removeOverlay = false } = {}) {
    this._thresholdPreviewVersion += 1;
    if (this._thresholdPreviewTimer !== null) {
      clearTimeout(this._thresholdPreviewTimer);
      this._thresholdPreviewTimer = null;
    }
    if (removeOverlay) {
      try { this.viewerController?.removeVolumeForStage?.('threshold-preview'); }
      catch (e) { /* non-fatal: stale viewer state is cosmetic */ }
    }
  }

  scheduleThresholdPreviewOverlay() {
    if (!this.thresholdedMaskFile) return;
    this.cancelThresholdPreviewOverlay();
    const version = this._thresholdPreviewVersion;
    this._thresholdPreviewTimer = setTimeout(() => {
      this._thresholdPreviewTimer = null;
      this._thresholdPreviewRenderPromise = this._thresholdPreviewRenderPromise
        .catch(() => {})
        .then(() => this.renderThresholdPreviewOverlay(version));
    }, 75);
  }

  async renderThresholdPreviewOverlay(version = this._thresholdPreviewVersion) {
    const file = this.thresholdedMaskFile;
    if (version !== this._thresholdPreviewVersion || !file) return;
    if (!this.viewerController) return;
    if (this.canProjectThresholdToPatientSpace()) {
      try {
        await this.projectThresholdToPatientSpace(version);
        if (version !== this._thresholdPreviewVersion || !this.patientThresholdedMaskFile) return;
        await this.renderPatientLayerStack();
        return;
      } catch (err) {
        this.patientThresholdedMaskFile = null;
        this.refreshViewerLayerControls();
        this.updateOutput(`Patient-space threshold projection failed: ${err.message}`);
      }
    } else {
      this.noteThresholdProjectionUnavailable();
    }
    await this.renderAtlasThresholdPreviewOverlay(version);
  }

  canProjectThresholdToPatientSpace() {
    return !!(
      this.structuralFile &&
      this.thresholdedMaskFile &&
      this.networkMapAffine &&
      this.hasRegistrationDisplacement &&
      this.executor?.runInverseWarpMask
    );
  }

  noteThresholdProjectionUnavailable() {
    this.patientThresholdedMaskFile = null;
    this.refreshViewerLayerControls();
    if (!this.thresholdedMaskFile || this._thresholdProjectionWarningShown) return;
    if (!this.structuralFile) {
      this.updateOutput('Patient-space threshold map unavailable: no structural T1 is loaded; showing atlas-space threshold preview.');
    } else if (!this.hasRegistrationDisplacement) {
      this.updateOutput('Patient-space threshold map unavailable: run registration first; showing atlas-space threshold preview.');
    }
    this._thresholdProjectionWarningShown = true;
  }

  async projectThresholdToPatientSpace(version) {
    const { mask, dims } = await this.resampleThresholdMaskToStructuralGrid();
    if (version !== this._thresholdPreviewVersion) return null;

    const maskBuffer = mask.buffer.slice(mask.byteOffset, mask.byteOffset + mask.byteLength);
    await this.runInverseWarpStage({
      maskBuffer,
      maskDims: dims,
      stage: 'threshold-patient',
      description: 'Threshold map projected to patient T1 space'
    }, 'threshold-patient');
    return this.patientThresholdedMaskFile;
  }

  async runInverseWarpStage(settings, stage) {
    const run = this._inverseWarpQueue.catch(() => {}).then(async () => {
      const stageReady = this._waitForStageData(stage);
      const inverseWarpDone = this._waitForStepComplete('inverse-warp-mask');
      await this.executor.runInverseWarpMask(settings);
      await Promise.all([stageReady, inverseWarpDone]);
    });
    this._inverseWarpQueue = run.catch(() => {});
    await run;
  }

  async resampleThresholdMaskToStructuralGrid() {
    if (!this.thresholdedMaskFile) {
      throw new Error('No thresholded network map is available.');
    }
    if (!this.structuralFile) {
      throw new Error('No structural T1 is available.');
    }
    const [thresholdBuf, structuralBuf, mni160] = await Promise.all([
      this.thresholdedMaskFile.arrayBuffer(),
      this.structuralFile.arrayBuffer(),
      loadAtlasFromManifest('lnm-mni160')
    ]);
    const threshold = await decodeNiftiBuffer(thresholdBuf);
    const structural = await decodeNiftiBuffer(structuralBuf);
    if (!dimsEqual(structural.dims, mni160.dims)) {
      throw new Error(
        `Patient-space threshold projection requires structural dims to match ` +
        `the lnm-mni160 registration grid (${mni160.dims.join('x')}); ` +
        `got ${structural.dims.join('x')}.`
      );
    }
    const thresholdMask = threshold.data instanceof Uint8Array
      ? threshold.data
      : binarise(threshold.data);
    const thresholdAffine = affineFromHeader(threshold.header);
    const mni160Affine = affineFromHeader(mni160.header);
    const resampled = resampleAffine(
      thresholdMask,
      threshold.dims, thresholdAffine,
      mni160.dims, mni160Affine,
      'nearest'
    );
    const mask = resampled instanceof Uint8Array ? resampled : binarise(resampled);
    return { mask, dims: mni160.dims };
  }

  async projectYeoAtlasToMni160Grid() {
    const [atlas, mni160] = await Promise.all([
      loadAtlasFromManifest('yeo7-2mm'),
      loadAtlasFromManifest('lnm-mni160')
    ]);
    const atlasAffine = affineFromHeader(atlas.header);
    const mni160Affine = affineFromHeader(mni160.header);
    const atlasLabels = new Uint8Array(atlas.data.length);
    for (let i = 0; i < atlas.data.length; i++) {
      const label = Math.round(Number(atlas.data[i]) || 0);
      atlasLabels[i] = label > 0 ? Math.min(label, 255) : 0;
    }
    const resampled = resampleAffine(
      atlasLabels,
      atlas.dims, atlasAffine,
      mni160.dims, mni160Affine,
      'nearest'
    );
    return {
      labels: resampled instanceof Uint8Array ? resampled : new Uint8Array(resampled),
      dims: mni160.dims,
      affine: mni160Affine
    };
  }

  async projectAtlasToPatientSpace() {
    if (!this.structuralFile) {
      throw new Error('No structural T1 is available.');
    }
    if (!this.hasRegistrationDisplacement) {
      throw new Error('Run MNI registration first.');
    }
    const { labels, dims } = await this.projectYeoAtlasToMni160Grid();
    const maskBuffer = labels.buffer.slice(labels.byteOffset, labels.byteOffset + labels.byteLength);
    await this.runInverseWarpStage({
      maskBuffer,
      maskDims: dims,
      stage: 'atlas-patient',
      description: 'Yeo atlas projected to patient T1 space',
      labelMap: true
    }, 'atlas-patient');
    return this.patientAtlasFile;
  }

  async showSubjectSpaceAtlas() {
    if (!this.patientAtlasFile) {
      this.updateOutput('Projecting Yeo atlas to patient T1 space for visual alignment QC...');
      await this.projectAtlasToPatientSpace();
    }
    this.viewerLayerVisibility.atlasQc = true;
    await this.renderPatientLayerStack();
    this.refreshViewerLayerControls();
    this.updateOutput('Atlas alignment QC overlay displayed. This is a visual check, not an automated pass/fail score.');
    return this.patientAtlasFile;
  }

  getRegistrationBlendValue() {
    const el = document.getElementById('registrationBlendValue');
    const raw = el?.value ?? this.registrationBlendValue ?? 0.5;
    const value = Number(raw);
    if (!Number.isFinite(value)) return 0.5;
    return Math.max(0, Math.min(1, value));
  }

  formatRegistrationBlendLabel(value = this.registrationBlendValue) {
    const numeric = Number(value);
    const blend = Number.isFinite(numeric) ? Math.max(0, Math.min(1, numeric)) : 0.5;
    if (blend <= 0) return 'MNI template';
    if (blend >= 1) return 'Registered patient';
    return `${Math.round(blend * 100)}% patient`;
  }

  updateRegistrationBlendLabel(value = this.registrationBlendValue) {
    const el = document.getElementById('registrationBlendLabel');
    if (el) el.textContent = this.formatRegistrationBlendLabel(value);
  }

  applyRegistrationBlend(value = this.getRegistrationBlendValue()) {
    const numeric = Number(value);
    const blend = Number.isFinite(numeric) ? Math.max(0, Math.min(1, numeric)) : 0.5;
    this.registrationBlendValue = blend;
    this.updateRegistrationBlendLabel(blend);
    if (!this.viewerController?.setStageOpacity) return false;
    const applied = this.viewerController.setStageOpacity('registered-t1-mni160', blend, {
      apply: true,
      redraw: true
    });
    return !!applied;
  }

  async handleRegistrationBlendInput() {
    const blend = this.getRegistrationBlendValue();
    if (this.applyRegistrationBlend(blend)) return;
    if (!this.hasRegistrationDisplacement || !this.registeredT1MniFile) return;
    const modeEl = document.getElementById('registrationQcMode');
    if (modeEl) modeEl.value = 'mni';
    this.registrationQcMode = 'mni';
    await this.renderMniRegistrationQc();
  }

  getRegistrationQcMode() {
    const el = document.getElementById('registrationQcMode');
    const mode = el?.value || this.registrationQcMode || 'mni';
    return ['patient', 'mni', 'checkerboard', 'displacement'].includes(mode) ? mode : 'mni';
  }

  async showRegistrationQc(mode = this.getRegistrationQcMode()) {
    this.registrationQcMode = mode;
    if (mode === 'patient') {
      return this.showSubjectSpaceAtlas();
    }
    if (!this.hasRegistrationDisplacement) {
      throw new Error('Run MNI registration first.');
    }
    if (mode === 'mni') return this.renderMniRegistrationQc();
    if (mode === 'checkerboard') return this.renderCheckerboardRegistrationQc();
    if (mode === 'displacement') return this.renderDisplacementRegistrationQc();
    return this.showSubjectSpaceAtlas();
  }

  async ensureRegistrationTemplateFile() {
    if (this.registrationTemplateFile) return this.registrationTemplateFile;
    const mni160 = await loadAtlasFromManifest('lnm-mni160');
    const affine = flattenAffine3Rows(affineFromHeader(mni160.header));
    const data = mni160.data instanceof Float32Array
      ? Float32Array.from(mni160.data)
      : Float32Array.from(mni160.data, Number);
    const niftiBuffer = writeNifti1(data, {
      dims: mni160.dims,
      spacing: [1, 1, 1],
      affine,
      description: 'LNM fixed MNI160 registration QC template'
    });
    this.registrationTemplateFile = arrayBufferToFile(niftiBuffer, 'lnm-mni160-template.nii');
    return this.registrationTemplateFile;
  }

  async ensureYeoAtlasMni160File() {
    if (this.yeoAtlasMni160File) return this.yeoAtlasMni160File;
    const { labels, dims, affine } = await this.projectYeoAtlasToMni160Grid();
    const niftiBuffer = writeNifti1(labels, {
      dims,
      spacing: [1, 1, 1],
      affine: flattenAffine3Rows(affine),
      description: 'LNM Yeo7 label atlas resampled to fixed MNI160 grid'
    });
    this.yeoAtlasMni160File = arrayBufferToFile(niftiBuffer, 'lnm-yeo7-atlas-mni160.nii');
    return this.yeoAtlasMni160File;
  }

  async ensureRegistrationCheckerboardFile(blockSize = 8) {
    if (this.registrationCheckerboardFile) return this.registrationCheckerboardFile;
    if (!this.registeredT1MniFile) {
      throw new Error('Registered T1 QC volume is not available; rerun MNI registration.');
    }
    await this.ensureRegistrationTemplateFile();
    const [templateBuf, registeredBuf] = await Promise.all([
      this.registrationTemplateFile.arrayBuffer(),
      this.registeredT1MniFile.arrayBuffer()
    ]);
    const template = await decodeNiftiBuffer(templateBuf);
    const registered = await decodeNiftiBuffer(registeredBuf);
    if (!dimsEqual(template.dims, registered.dims)) {
      throw new Error(
        `Registration checkerboard requires matching dims; ` +
        `template=${template.dims.join('x')} registered=${registered.dims.join('x')}`
      );
    }
    const [X, Y, Z] = template.dims;
    const out = new Float32Array(template.data.length);
    for (let z = 0; z < Z; z++) {
      for (let y = 0; y < Y; y++) {
        for (let x = 0; x < X; x++) {
          const i = x + y * X + z * X * Y;
          const useRegistered = (
            Math.floor(x / blockSize) +
            Math.floor(y / blockSize) +
            Math.floor(z / blockSize)
          ) % 2 === 0;
          out[i] = Number(useRegistered ? registered.data[i] : template.data[i]) || 0;
        }
      }
    }
    const niftiBuffer = writeNifti1(out, {
      dims: template.dims,
      spacing: [1, 1, 1],
      affine: flattenAffine3Rows(affineFromHeader(template.header)),
      description: 'LNM registration QC checkerboard: fixed MNI template and registered T1'
    });
    this.registrationCheckerboardFile = arrayBufferToFile(niftiBuffer, 'lnm-registration-checkerboard.nii');
    return this.registrationCheckerboardFile;
  }

  async renderMniRegistrationQc() {
    if (!this.registeredT1MniFile) {
      throw new Error('Registered T1 QC volume is not available; rerun MNI registration.');
    }
    const [templateFile, atlasFile] = await Promise.all([
      this.ensureRegistrationTemplateFile(),
      this.ensureYeoAtlasMni160File()
    ]);
    const blend = this.getRegistrationBlendValue();
    this.registrationBlendValue = blend;
    this.updateRegistrationBlendLabel(blend);
    const entries = [
      { file: templateFile, stage: 'registration-template' },
      {
        file: this.registeredT1MniFile,
        colormap: 'gray',
        opacity: blend,
        scalar: true,
        stage: 'registered-t1-mni160'
      },
      {
        file: atlasFile,
        colormap: 'lnm-yeo7',
        opacity: 0.35,
        stage: 'atlas-qc',
        visible: this.layerVisible('atlasQc')
      }
    ];
    await this.viewerController.loadVolumeStack(entries);
    this.applyViewerLayerVisibility();
    this.applyRegistrationBlend(blend);
    this.refreshViewerLayerControls();
    this.updateOutput('Registration QC: MNI-space registered T1, fixed template, and full Yeo atlas displayed. Use the Patient/MNI blend slider for visual QC.');
  }

  async renderCheckerboardRegistrationQc() {
    const [checkerboardFile, atlasFile] = await Promise.all([
      this.ensureRegistrationCheckerboardFile(),
      this.ensureYeoAtlasMni160File()
    ]);
    await this.viewerController.loadVolumeStack([
      { file: checkerboardFile, stage: 'registration-checkerboard' },
      {
        file: atlasFile,
        colormap: 'lnm-yeo7',
        opacity: 0.25,
        stage: 'atlas-qc',
        visible: this.layerVisible('atlasQc')
      }
    ]);
    this.applyViewerLayerVisibility();
    this.refreshViewerLayerControls();
    this.updateOutput('Registration QC: checkerboard of fixed MNI template and registered T1 displayed.');
  }

  async renderDisplacementRegistrationQc() {
    if (!this.displacementMagnitudeFile) {
      throw new Error('Displacement magnitude QC map is not available; rerun MNI registration.');
    }
    const entries = [
      { file: await this.ensureRegistrationTemplateFile(), stage: 'registration-template' }
    ];
    if (this.registeredT1MniFile) {
      const blend = this.getRegistrationBlendValue();
      this.registrationBlendValue = blend;
      this.updateRegistrationBlendLabel(blend);
      entries.push({
        file: this.registeredT1MniFile,
        colormap: 'gray',
        opacity: blend,
        scalar: true,
        stage: 'registered-t1-mni160'
      });
    }
    entries.push({
      file: this.displacementMagnitudeFile,
      colormap: 'blue2red',
      opacity: 0.65,
      scalar: true,
      stage: 'registration-displacement'
    });
    await this.viewerController.loadVolumeStack(entries);
    this.applyRegistrationBlend(this.registrationBlendValue);
    this.refreshViewerLayerControls();
    this.updateOutput('Registration QC: SynthMorph displacement magnitude displayed on the fixed MNI template.');
  }

  async renderPatientLayerStack() {
    if (!this.structuralFile) return;
    const entries = [{
      file: this.structuralFile,
      stage: 'structural',
      visible: this.layerVisible('structural')
    }];
    if (this.brainmaskFile) {
      entries.push({
        file: this.brainmaskFile,
        colormap: 'green',
        opacity: 0.4,
        stage: 'brainmask',
        visible: this.layerVisible('brainmask')
      });
    }
    if (this.patientAtlasFile) {
      entries.push({
        file: this.patientAtlasFile,
        colormap: 'lnm-yeo7',
        opacity: 0.45,
        stage: 'atlas-qc',
        visible: this.layerVisible('atlasQc')
      });
    }
    const lesionOverlay = this.lesionMaskFile || this.lesionFile;
    if (lesionOverlay) {
      entries.push({
        file: lesionOverlay,
        colormap: 'red',
        opacity: 0.5,
        stage: this.lesionMaskFile ? 'segmentation' : 'lesion',
        visible: this.layerVisible('lesion')
      });
    }
    if (this.patientThresholdedMaskFile) {
      entries.push({
        file: this.patientThresholdedMaskFile,
        colormap: 'red',
        opacity: 0.65,
        stage: 'threshold-preview',
        visible: this.layerVisible('threshold')
      });
    }
    await this.viewerController.loadVolumeStack(entries);
    this.applyViewerLayerVisibility();
    this.refreshViewerLayerControls();
    this.updateOutput('Patient-space viewer stack displayed.');
  }

  async renderAtlasThresholdPreviewOverlay(version) {
    const file = this.thresholdedMaskFile;
    if (version !== this._thresholdPreviewVersion || !file) return;
    if (!this.viewerController?.replaceOverlayForStage) return;
    try {
      await this.viewerController.replaceOverlayForStage(
        'threshold-preview',
        file,
        'red',
        0.65,
        { visible: this.layerVisible('threshold') }
      );
      if (version !== this._thresholdPreviewVersion) {
        this.viewerController?.removeVolumeForStage?.('threshold-preview');
      }
      this.refreshViewerLayerControls();
    } catch (err) {
      this.updateOutput(`Threshold preview render error: ${err.message}`);
    }
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

  downloadSubjectSpaceAtlas() {
    if (!this.patientAtlasFile) {
      this.updateOutput('No patient-space Yeo atlas available yet.');
      return;
    }
    const url = URL.createObjectURL(this.patientAtlasFile);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'lnm-yeo7-atlas-patient.nii';
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
    this.patientAtlasFile = null;
    this.registeredT1MniFile = null;
    this.displacementMagnitudeFile = null;
    this.registrationCheckerboardFile = null;
    this.refreshViewerLayerControls();
    const registrationReady = this._waitForStepComplete('register');
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
      executionProviders: model.browserRuntime?.executionProviders,
      referenceAssetId: ref.id,
      referenceUrl: ref.sourceUrl,
      referenceCacheKey: ref.cacheKey,
      nbSteps: 7
    });
    await registrationReady;
    this.hasRegistrationDisplacement = true;
    this._thresholdProjectionWarningShown = false;
    this.refreshViewerLayerControls();
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
    let mni160Ref = null;
    const getMni160Ref = async () => {
      if (!mni160Ref) mni160Ref = await loadAtlasFromManifest('lnm-mni160');
      return mni160Ref;
    };

    // Phase 34: idempotent fast-path. If the structural is already at
    // exactly the SynthMorph-required fixed lnm-mni160 pose, prealign
    // has nothing to do — used by runFullPipeline's auto chain so users
    // with already-aligned T1s don't pay the cost. Dims alone are not
    // sufficient: an oblique 160x160x192 prealign output must still be
    // canonicalised onto the fixed template affine.
    if (skipIfAligned) {
      const [probeBuf, mni160] = await Promise.all([
        this.structuralFile.arrayBuffer(),
        getMni160Ref()
      ]);
      const probe = await decodeNiftiBuffer(probeBuf);
      const probeAffine = affineFromHeader(probe.header);
      const mni160Affine = affineFromHeader(mni160.header);
      const isAligned = dimsEqual(probe.dims, mni160.dims) &&
        affineNearlyEqual(probeAffine, mni160Affine);
      if (isAligned) {
        if (!this.nativeStructuralFile) this.nativeStructuralFile = this.structuralFile;
        if (!this.nativeStructuralInfo) {
          this.nativeStructuralInfo = { dims: probe.dims, affine: probeAffine };
        }
        this.fixedMni160Info = { dims: mni160.dims, affine: mni160Affine, spacing: [1, 1, 1] };
        this.prealignSamplingAffine = mni160Affine;
        this.updateOutput('Structural already matches lnm-mni160, skipping prealign.');
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
    if (!this.nativeStructuralFile) this.nativeStructuralFile = this.structuralFile;
    if (!this.nativeStructuralInfo) {
      this.nativeStructuralInfo = { dims: t1.dims, affine: t1Affine };
    }

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
    const mni160 = await getMni160Ref();
    const mni160Affine = affineFromHeader(mni160.header);
    this.fixedMni160Info = { dims: mni160.dims, affine: mni160Affine, spacing: [1, 1, 1] };
    const mniCenterVox = mni160.dims.map(v => v / 2);
    const { dstAffine, mniDims, eigenvalues } = principalAxisAlign(
      mask.data, t1.dims, t1Affine,
      { mniDims: mni160.dims, mniCenterVox }
    );
    this.prealignSamplingAffine = dstAffine;
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

    const flatAff = flattenAffine3Rows(mni160Affine);
    const t1Nifti = writeNifti1(t1Resampled, {
      dims: mniDims, spacing: [1, 1, 1], affine: flatAff,
      description: 'LNM prealign: resampled to fixed lnm-mni160 1mm'
    });
    const maskNifti = writeNifti1(maskBin, {
      dims: mniDims, spacing: [1, 1, 1], affine: flatAff,
      description: 'LNM prealign brainmask resampled to fixed lnm-mni160 1mm'
    });

    this.structuralFile = arrayBufferToFile(t1Nifti, 'lnm-prealign-t1.nii');
    this.brainmaskFile = arrayBufferToFile(maskNifti, 'lnm-prealign-brainmask.nii');
    this.hasRegistrationDisplacement = false;
    this._thresholdProjectionWarningShown = false;
    // Stale results from the pre-prealign space.
    this.lesionMaskFile = null;
    this.lesionMaskConfirmed = false;
    this.autoLesionSeedFile = null;
    this.nativeLesionSeedFile = null;
    this.confirmedNativeLesionFile = null;
    this.maskReviewActive = false;
    this.lesionFile = null;
    this.mniLesionFile = null;
    this.overlapResult = null;
    this.affectedNetworkResult = null;
    this.networkMapFile = null;
    this.networkMapData = null;
    this.thresholdedMaskFile = null;
    this.patientThresholdedMaskFile = null;
    this.patientAtlasFile = null;
    this.registeredT1MniFile = null;
    this.displacementMagnitudeFile = null;
    this.registrationCheckerboardFile = null;
    this.networkMapBaseFile = null;
    this.cancelThresholdPreviewOverlay({ removeOverlay: true });
    this.clearAffectedNetworkTable();

    // Refresh viewer with the aligned T1 + brainmask overlay.
    await this.viewerController.loadBaseVolume(this.structuralFile, {
      stage: 'structural',
      visible: this.layerVisible('structural')
    });
    try {
      await this.viewerController.loadOverlay(this.brainmaskFile, 'green', 0.4, {
        stage: 'brainmask',
        visible: this.layerVisible('brainmask')
      });
    } catch (err) {
      // Non-fatal: the overlay is cosmetic.
      this.updateOutput(`Brainmask overlay re-render warning: ${err.message}`);
    }
    this.refreshViewerLayerControls();
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
      this.updateOutput('Confirm the edited lesion mask first.');
      return null;
    }
    if (!this.lesionMaskConfirmed) {
      this.updateOutput('Review and confirm the lesion mask before warping it.');
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
    const flatAffine = flattenAffine3Rows(atlasAffine);
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
    const status = await this._runPipelineStages(pipeline, 0);
    if (status === 'complete') this.logPipelineComplete();
  }

  async _runPipelineStages(pipeline, startIndex = 0) {
    let stageIndex = -1;
    for (const stage of pipeline.stages) {
      stageIndex += 1;
      if (stageIndex < startIndex) continue;
      const stageStart = this._now();
      try {
        const result = await this._runStage(stage);
        if (result?.pausedForMaskReview) {
          this._pendingMaskResume = { pipeline, nextStageIndex: stageIndex + 1 };
          this.updateOutput('Pipeline paused for manual lesion-mask review.');
          return 'paused';
        }
      } catch (err) {
        this.updateOutput(`Stage '${stage.id}' (${stage.module}) failed: ${err.message}`);
        return 'failed';
      }
      const elapsedMs = this._now() - stageStart;
      this._perfStats.push({ id: stage.id, module: stage.module, ms: elapsedMs });
      this.updateOutput(`[perf] ${stage.id} (${stage.module}): ${this._formatMs(elapsedMs)}`);
    }
    return 'complete';
  }

  logPipelineComplete() {
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
        if (this.lesionMaskFile && this.lesionMaskConfirmed) {
          this.updateOutput('Confirmed lesion mask already present, skipping segmentation.');
          return;
        }
        if (this.maskReviewActive || this.autoLesionSeedFile) {
          await this.startLesionMaskReview({ seedFile: this.autoLesionSeedFile });
          return { pausedForMaskReview: true };
        }
        await this.runLesionSegmentation();
        await this.startLesionMaskReview({ seedFile: this.autoLesionSeedFile });
        return { pausedForMaskReview: true };
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
        // computing so the selected pipeline's defaults are honoured when the
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
    const valueEl = document.getElementById('networkThresholdValue');
    const symEl = document.getElementById('networkThresholdSymmetric');
    const minClEl = document.getElementById('networkThresholdMinCluster');
    this.configureTopPercentThresholdSlider();
    if (valueEl && typeof defaults.value === 'number') {
      // Slider stores the raw top-percent UI value (5 = strongest 5%).
      valueEl.value = String(defaults.value);
    }
    if (symEl && typeof defaults.symmetric === 'boolean') symEl.checked = defaults.symmetric;
    if (minClEl && typeof defaults.minClusterVoxels === 'number') {
      minClEl.value = String(defaults.minClusterVoxels);
    }
    this.updateThresholdValueLabel();
  }

  // Phase 21: clear all intermediate pipeline state so the user can start
  // a fresh run without reloading the page. Resets every result slot, the
  // viewer, and the threshold UI surface; preserves the structural file
  // (the user usually wants to re-run on the same input) unless `full`
  // is set.
  clearResults({ full = false } = {}) {
    this.overlapResult = null;
    this.affectedNetworkResult = null;
    this.brainmaskFile = null;
    this.lesionMaskFile = null;
    this.lesionMaskConfirmed = false;
    this.autoLesionSeedFile = null;
    this.nativeLesionSeedFile = null;
    this.confirmedNativeLesionFile = null;
    this.maskReviewActive = false;
    this._pendingMaskResume = null;
    this.networkMapFile = null;
    this.networkMapData = null;
    this.networkMapDims = null;
    this.networkMapSpacing = null;
    this.networkMapAffine = null;
    this.networkMapBaseFile = null;
    this.thresholdedMaskFile = null;
    this.patientThresholdedMaskFile = null;
    this.patientAtlasFile = null;
    this.registeredT1MniFile = null;
    this.displacementMagnitudeFile = null;
    this.registrationCheckerboardFile = null;
    this.hasRegistrationDisplacement = false;
    this._thresholdProjectionWarningShown = false;
    this.cancelThresholdPreviewOverlay({ removeOverlay: true });
    this.mniLesionFile = null;
    if (full) {
      this.structuralFile = null;
      this.nativeStructuralFile = null;
      this.nativeStructuralInfo = null;
      this.fixedMni160Info = null;
      this.prealignSamplingAffine = null;
      this.lesionFile = null;
    }

    // Re-disable every download / threshold-output button so the UI matches
    // the cleared state.
    const buttonIds = [
      'downloadOverlapCsv',
      'downloadBrainMaskButton',
      'downloadLesionMaskButton',
      'downloadNetworkMapButton',
      'downloadThresholdedNetworkMapButton',
      'downloadEditedLesionMaskButton',
      'checkAtlasAlignmentButton',
      'showSubjectAtlasButton',
      'downloadSubjectAtlasButton'
    ];
    for (const id of buttonIds) {
      const el = document.getElementById(id);
      if (el) el.disabled = true;
    }
    // Wipe the overlap table body if present.
    const tbody = document.querySelector('#networkOverlapTable tbody');
    if (tbody) tbody.innerHTML = '';
    this.clearFunctionProfileTable('directFunctionProfileResults', 'directFunctionProfileTable');
    // Reset the threshold summary.
    const summaryEl = document.getElementById('networkThresholdSummary');
    if (summaryEl) summaryEl.textContent = 'Compute a network map first to enable thresholding.';
    this.clearAffectedNetworkTable();
    // Hide Yeo cortical-label coverage note.
    this.showYeoLabelCoverageNote(0, 0);

    if (this.executor && typeof this.executor.clearResults === 'function') {
      this.executor.clearResults();
    }

    // Restore the structural file in the viewer (or clear entirely on full reset).
    if (full || !this.structuralFile) {
      try { this.viewerController.clearAll?.(); } catch (e) { /* non-fatal */ }
    } else if (this.structuralFile) {
      this.viewerController.loadBaseVolume(this.structuralFile, {
        stage: 'structural',
        visible: this.layerVisible('structural')
      })
        .catch(err => this.updateOutput(`Viewer reload after reset failed: ${err.message}`));
    }
    this.refreshViewerLayerControls();
    this.refreshMaskDrawingControls();
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

  showYeoLabelCoverageNote(outside, total) {
    const el = document.getElementById('outsideAtlasWarning');
    if (!el) return;
    if (outside > 0 && total > 0) {
      const assigned = total - outside;
      el.textContent = `${assigned} of ${total} lesion voxels are assigned to Yeo cortical network labels; ${outside} are unlabeled by this cortical atlas.`;
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
