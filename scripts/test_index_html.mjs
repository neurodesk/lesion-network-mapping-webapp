#!/usr/bin/env node --no-warnings
// Contract test for web/index.html under the LNM rewrite. Pins the sidebar
// structure the orchestrator binds to, plus that no SCT branding survives.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(ROOT, 'web/index.html'), 'utf8');

// Title + h1 reflect the LNM identity, not SCT.
assert.match(html, /<title>[^<]*Lesion Network Mapping[^<]*<\/title>/i,
  'page title must include "Lesion Network Mapping"');
assert.doesNotMatch(html, /SpinalCordToolbox/i,
  'no surviving "SpinalCordToolbox" branding allowed');
assert.doesNotMatch(html, /\bSCT\b(?!\.com)/,   // allow URL fragments like spinalcordtoolbox.com
  'no surviving "SCT" branding allowed (apart from any incidental URL).');

// Sidebar sections — the orchestrator binds to these IDs. If they drift, the
// app fails silently instead of throwing, so we lock them down.
const requiredIds = [
  // Phase 32 — sidebar redesign: three primary sections (Input → Run → Results).
  // Per-stage controls live inside <details> disclosures inside Run/Results so
  // the orchestrator's bindings keep working without dominating the UI.
  '#stepLoadSection',
  '#stepLesionSection',
  // (#stepNetworkSection removed; pipelineSelect + computeOverlapButton moved
  //  under the Run section's Advanced disclosure.)
  '#resultsSection',
  '#networkOverlapTable',
  '#downloadOverlapCsv',
  '#computeOverlapButton',
  '#outsideAtlasWarning',
  '#structuralFileInput',
  '#lesionFileInput',
  '#pipelineSelect',
  // Phase 2a.1.4b additions: brain-extraction button (explicit trigger) and
  // the brain-mask download button.
  '#runBrainExtractionButton',
  '#downloadBrainMaskButton',
  // Phase 2a.2.3 additions: lesion-segmentation trigger button + mask
  // download button.
  '#runLesionSegmentationButton',
  '#downloadLesionMaskButton',
  // Phase 3.4 additions: registration button.
  '#runRegistrationButton',
  // Phase 4.4 additions: Network map subsection.
  '#computeNetworkMapButton',
  '#downloadNetworkMapButton',
  // Phase 5 additions: threshold controls + thresholded download.
  '#networkThresholdValue',
  '#networkThresholdMode',
  '#networkThresholdSymmetric',
  '#networkThresholdMinCluster',
  '#downloadThresholdedNetworkMapButton',
  // Phase 6 additions: warp+resample bridge button + one-click full chain.
  '#applyRegistrationToLesionButton',
  '#runFullPipelineButton',
  // Phase 16 addition: in-browser affine pre-registration.
  '#prealignToMniButton',
  // Phase 21 addition: clear-results / new-run UX control.
  '#clearResultsButton',
  // Phase 32 additions: Advanced disclosure container.
  '#advancedStageControls'
];
for (const id of requiredIds) {
  const escaped = id.slice(1);
  const re = new RegExp(`id=["']${escaped}["']`);
  assert.match(html, re, `index.html must contain element with ${id}`);
}

// Pipeline selector must offer at least the Phase 1 pipeline.
// Phase 39: visible UI assumes raw T1 input; manual-mask pipelines
// (lnm-yeo-only, lnm-network-map) are flagged hidden in lnm-tasks.js
// and drop out of the dropdown. The static fallback option (rendered
// only when JS doesn't run) is now lnm-yeo-auto — the auto chain.
assert.match(html, /value=["']lnm-yeo-auto["']/,
  '#pipelineSelect static fallback must be lnm-yeo-auto (auto T1 chain)');

// Helper copy should live behind compact inline help popovers, following the
// QSMbly-style "i" affordance, rather than always-visible paragraphs.
assert.match(html, /class=["'][^"']*\bhelp-icon\b[^"']*["']/,
  'index.html must include compact help icons');
assert.match(html, /class=["'][^"']*\bhelp-popover\b[^"']*["']/,
  'index.html must include help popover content');
assert.match(html, /Loading the image only displays it; processing starts when you click Run analysis/,
  'structural input help must make explicit that loading does not start processing');
assert.doesNotMatch(html, /<p\s+class=["']param-help["']/,
  'always-visible param-help paragraphs should be replaced with popovers or status text');
assert.doesNotMatch(html, /auto-promoted on file drop|auto-fires/i,
  'UI copy must not imply processing starts on file load');

// Module loader points at the new orchestrator, not the old SCT app.
assert.match(html, /<script\s[^>]*src=["']js\/lnm-app\.js["'][^>]*type=["']module["']/,
  '<script type=module src="js/lnm-app.js"> must be present');
assert.doesNotMatch(html, /spinalcordtoolbox-app\.js/,
  'old spinalcordtoolbox-app.js script tag must be gone');

// NiiVue canvas must remain (we reuse it for structural + lesion overlay).
assert.match(html, /id=["']gl1["']/, '#gl1 NiiVue canvas must be retained');

console.log(`index.html OK: ${requiredIds.length} required IDs, no SCT branding, lnm-app.js wired.`);
