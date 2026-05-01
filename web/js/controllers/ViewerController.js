/**
 * ViewerController
 *
 * Manages NiiVue visualization with support for base volume and segmentation overlays.
 * Manages SCT task colormaps.
 */

export class ViewerController {
  constructor(options) {
    this.nv = options.nv;
    this.updateOutput = options.updateOutput || (() => {});
    this.currentBaseFile = null;
    this.currentOverlayFile = null;
    this.currentOverlayIndex = null;
    this.volumeStageIndices = new Map();
    this.sctColormapsRegistered = new Set();
  }

  /**
   * Register an SCT discrete colormap with NiiVue.
   * @param {Object} colormapData - { R, G, B, A } arrays from labels.js
   */
  registerSctColormap(colormapData, colormapId = 'sct-spinalcord') {
    try {
      this.nv.addColormap(colormapId, colormapData);
      this.sctColormapsRegistered.add(colormapId);
    } catch (e) {
      console.warn(`Could not register ${colormapId} colormap:`, e);
    }
  }

  registerVesselColormap(colormapData) {
    this.registerSctColormap(colormapData, 'sct-spinalcord');
  }

  async loadBaseVolume(file, options = {}) {
    try {
      this.updateOutput(`Loading ${file.name}...`);
      const url = URL.createObjectURL(file);
      await this.nv.loadVolumes([{ url: url, name: file.name }]);
      URL.revokeObjectURL(url);
      this.currentBaseFile = file;
      this.currentOverlayFile = null;
      this.currentOverlayIndex = null;
      this.volumeStageIndices.clear();
      if (options.stage) this.volumeStageIndices.set(options.stage, 0);
      this.updateOutput(`${file.name} loaded`);
    } catch (error) {
      this.updateOutput(`Error loading ${file.name}: ${error.message}`);
      console.error(error);
    }
  }

  async loadVolumeStack(entries) {
    if (!entries.length) {
      this.clearVolumes();
      return;
    }

    // NiiVue's `loadVolumes()` with multiple volumes calls `addVolume()` per
    // entry but the overlay paths (cal_min/cal_max, colormap LUT, opacity)
    // are only correctly initialised when overlays go through
    // `addVolumeFromUrl()` AFTER the base volume is already in place. We
    // therefore reuse the proven `loadBaseVolume` + `loadOverlay` flow here
    // so binary/label-mask overlays actually render. Replacing this with
    // `nv.loadVolumes([...])` silently produced no visible overlay in
    // 0.68.x — covered by `npm run test:viewer`.
    try {
      const [baseEntry, ...overlayEntries] = entries;

      await this.loadBaseVolume(baseEntry.file, { stage: baseEntry.stage });

      if (baseEntry.labelMask) {
        this.configureSegmentationVolume(0, baseEntry.colormap || 'sct-spinalcord');
        this.nv.updateGLVolume();
        this.nv.drawScene?.();
      }

      for (const entry of overlayEntries) {
        await this.loadOverlay(
          entry.file,
          entry.colormap || 'sct-spinalcord',
          entry.opacity ?? 0.5,
          { stage: entry.stage }
        );
      }
    } catch (error) {
      this.updateOutput(`Error loading viewer volumes: ${error.message}`);
      console.error(error);
    }
  }

  clearVolumes() {
    this.nv.volumes = [];
    this.currentBaseFile = null;
    this.currentOverlayFile = null;
    this.currentOverlayIndex = null;
    this.volumeStageIndices.clear();
    this.nv.updateGLVolume();
    this.nv.drawScene?.();
  }

  clearOverlay() {
    const overlayIndex = this.getOverlayIndex();
    if (overlayIndex === null) return;

    if (typeof this.nv.removeVolumeByIndex === 'function') {
      this.nv.removeVolumeByIndex(overlayIndex);
    } else {
      this.nv.volumes.splice(overlayIndex, 1);
      this.nv.updateGLVolume();
      this.nv.drawScene?.();
    }

    this.currentOverlayFile = null;
    this.currentOverlayIndex = null;
  }

  configureSegmentationVolume(index, colormap) {
    const volume = this.nv.volumes[index];
    if (!volume) return;

    volume.cal_min = 0;
    volume.cal_max = Math.max(1, this.getVolumeDataMax(volume));
    volume.colormap = colormap;
    // Binary/discrete segmentation: disable trilinear smoothing so thin
    // structures don't interpolate to fractional values that miss the LUT.
    volume.interpolation = false;
    if (typeof this.nv.setColormap === 'function' && volume.id) {
      this.nv.setColormap(volume.id, colormap);
    }
  }

  async loadOverlay(file, colormap = 'red', opacity = 0.5, options = {}) {
    try {
      const url = URL.createObjectURL(file);
      await this.nv.addVolumeFromUrl({
        url: url,
        name: file.name,
        colormap: colormap,
        opacity
      });
      URL.revokeObjectURL(url);

      const overlayIndex = this.nv.volumes.length - 1;
      if (overlayIndex > 0) {
        this.configureSegmentationVolume(overlayIndex, colormap);
        this.nv.setOpacity(overlayIndex, opacity);
        this.nv.updateGLVolume();
        this.nv.drawScene?.();
      }

      this.currentOverlayFile = file;
      this.currentOverlayIndex = overlayIndex > 0 ? overlayIndex : null;
      if (options.stage) this.volumeStageIndices.set(options.stage, overlayIndex);
    } catch (error) {
      this.updateOutput(`Error loading overlay: ${error.message}`);
      console.error(error);
    }
  }

  async loadSegmentationAsBase(file, colormap = 'sct-spinalcord', options = {}) {
    await this.loadBaseVolume(file, options);
    this.configureSegmentationVolume(0, colormap);
    this.currentBaseFile = file;
    this.currentOverlayFile = null;
    this.currentOverlayIndex = null;
    this.nv.updateGLVolume();
    this.nv.drawScene?.();
  }

  async showResultAsOverlay(baseFile, overlayFile, colormap = 'sct-spinalcord') {
    await this.loadBaseVolume(baseFile);
    if (overlayFile) {
      await this.loadOverlay(overlayFile, colormap);
    }
  }

  setViewType(type) {
    const typeMap = {
      multiplanar: this.nv.sliceTypeMultiplanar,
      axial: this.nv.sliceTypeAxial,
      coronal: this.nv.sliceTypeCoronal,
      sagittal: this.nv.sliceTypeSagittal,
      render: this.nv.sliceTypeRender
    };
    if (typeMap[type] !== undefined) {
      this.nv.setSliceType(typeMap[type]);
    }
  }

  setBaseOpacity(value) {
    if (this.nv.volumes.length > 0) {
      this.nv.setOpacity(0, value);
      this.nv.updateGLVolume();
    }
  }

  setOverlayOpacity(value) {
    const overlayIndices = this.getOverlayIndices();
    if (overlayIndices.length) {
      overlayIndices.forEach(index => this.nv.setOpacity(index, value));
      this.nv.updateGLVolume();
    }
  }

  setOverlayColormap(colormap) {
    const overlayIndex = this.getOverlayIndex();
    if (overlayIndex !== null) {
      const overlay = this.nv.volumes[overlayIndex];
      overlay.colormap = colormap;
      if (typeof this.nv.setColormap === 'function' && overlay.id) {
        this.nv.setColormap(overlay.id, colormap);
      }
      this.nv.updateGLVolume();
    }
  }

  getOverlayIndex() {
    if (this.currentOverlayIndex !== null && this.nv.volumes[this.currentOverlayIndex]) {
      return this.currentOverlayIndex;
    }
    return this.nv.volumes.length > 1 ? this.nv.volumes.length - 1 : null;
  }

  getOverlayIndices() {
    return this.nv.volumes
      .map((_, index) => index)
      .filter(index => index > 0);
  }

  getVolumeIndexForStage(stage) {
    const index = this.volumeStageIndices.get(stage);
    if (index === undefined || !this.nv.volumes[index]) return null;
    return index;
  }

  getVolumeDataMax(volume) {
    if (volume?.img?.length) {
      let maxValue = -Infinity;
      for (let i = 0; i < volume.img.length; i++) {
        const value = volume.img[i];
        if (Number.isFinite(value) && value > maxValue) maxValue = value;
      }
      if (Number.isFinite(maxValue)) return maxValue;
    }
    return volume?.global_max ?? 1;
  }

  getCurrentFile() {
    return this.currentBaseFile;
  }
}
