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
  '#stepLoadSection',
  '#stepLesionSection',
  '#stepNetworkSection',
  '#resultsSection',
  '#networkOverlapTable',
  '#downloadOverlapCsv',
  '#computeOverlapButton',
  '#outsideAtlasWarning',
  '#structuralFileInput',
  '#lesionFileInput',
  '#pipelineSelect',
  // Phase 2a.1.4b additions: brain-extraction button (re-run trigger; the
  // SynthStrip pass also auto-fires on a structural drop) and the
  // brain-mask download button.
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
  '#downloadThresholdedNetworkMapButton'
];
for (const id of requiredIds) {
  const escaped = id.slice(1);
  const re = new RegExp(`id=["']${escaped}["']`);
  assert.match(html, re, `index.html must contain element with ${id}`);
}

// Pipeline selector must offer at least the Phase 1 pipeline.
assert.match(html, /value=["']lnm-yeo-only["']/,
  '#pipelineSelect must include the lnm-yeo-only option');

// Module loader points at the new orchestrator, not the old SCT app.
assert.match(html, /<script\s[^>]*src=["']js\/lnm-app\.js["'][^>]*type=["']module["']/,
  '<script type=module src="js/lnm-app.js"> must be present');
assert.doesNotMatch(html, /spinalcordtoolbox-app\.js/,
  'old spinalcordtoolbox-app.js script tag must be gone');

// NiiVue canvas must remain (we reuse it for structural + lesion overlay).
assert.match(html, /id=["']gl1["']/, '#gl1 NiiVue canvas must be retained');

console.log(`index.html OK: ${requiredIds.length} required IDs, no SCT branding, lnm-app.js wired.`);
