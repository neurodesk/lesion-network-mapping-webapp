import { FileIOController } from './controllers/FileIOController.js';
import { ViewerController } from './controllers/ViewerController.js';
import { LNM_PIPELINES, getPipelineById } from './app/lnm-tasks.js';
import { YEO7_COLORMAP, YEO7_NETWORK_LABELS } from './app/lnm-labels.js';
import { computeParcelOverlap, summarizeNetworkOverlap } from './modules/parcel-overlap.js';
import { ConsoleOutput } from './modules/ui/ConsoleOutput.js';
import { ProgressManager } from './modules/ui/ProgressManager.js';
import { ModalManager } from './modules/ui/ModalManager.js';
import * as Config from './app/config.js';

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
    this.selectedPipeline = getPipelineById('lnm-yeo-only') || LNM_PIPELINES[0];
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
    if (csvButton) csvButton.addEventListener('click', () => this.exportCsv());

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
    const atlasAssetId = 'yeo7-2mm';
    this.updateOutput(`Preparing ${atlasAssetId} overlap.`);

    // TODO Phase 1c.2: fetch and decode the atlas asset, then pass real typed arrays.
    const parcelResult = computeParcelOverlap({
      lesion: new Uint8Array([0]),
      atlas: new Uint16Array([0]),
      dims: [1, 1, 1]
    });
    this.overlapResult = summarizeNetworkOverlap(parcelResult, YEO7_NETWORK_LABELS);

    // TODO Phase 1c.3: render the overlap table and chart.
    return this.overlapResult;
  }

  exportCsv() {
    // TODO Phase 1c.3: serialize this.overlapResult as CSV and download it.
    this.updateOutput('CSV export is not implemented in this phase.');
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
