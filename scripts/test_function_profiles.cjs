#!/usr/bin/env node

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

const ROOT = path.resolve(__dirname, '..');

function makeElement(tagName) {
  let text = '';
  return {
    tagName,
    children: [],
    style: {},
    className: '',
    colSpan: 0,
    innerHTML: '',
    appendChild(child) { this.children.push(child); },
    set textContent(value) { text = String(value); },
    get textContent() { return text; }
  };
}

(async () => {
  const {
    rankFunctionalTerms,
    renderFunctionalProfileTable
  } = await import(pathToFileURL(path.join(ROOT, 'web/js/modules/function-profiles.js')));

  const assetPath = path.join(ROOT, 'web/models/annotations/yeo7_function_profiles.json');
  const asset = JSON.parse(fs.readFileSync(assetPath, 'utf8'));
  assert.equal(asset.id, 'yeo7-neurosynth-v7-function-profiles',
    'committed profile asset must expose the manifest id');
  for (const network of ['Visual', 'Somatomotor', 'DorsalAttention', 'VentralAttention', 'Limbic', 'Frontoparietal', 'Default']) {
    assert.ok(Array.isArray(asset.networkProfiles[network]) && asset.networkProfiles[network].length > 0,
      `committed profile asset must include terms for ${network}`);
  }

  const profiles = {
    sourceLabel: 'Neurosynth v7 via NiMARE',
    networkProfiles: {
      Visual: [
        { term: 'visual', score: 0.8 },
        { term: 'attention', score: 0.2 }
      ],
      Default: [
        { term: 'memory', score: 0.7 },
        { term: 'attention', score: 0.6 }
      ],
      Somatomotor: [
        { term: 'motor', score: 0.9 }
      ]
    }
  };

  const summary = {
    networks: [
      { network: 'Visual', fractionOfLesion: 0.5 },
      { network: 'Default', fractionOfLesion: 0.25 },
      { network: 'Unassigned', fractionOfLesion: 0.25 },
      { network: 'MissingProfile', fractionOfLesion: 0.4 }
    ]
  };

  const ranked = rankFunctionalTerms(summary, profiles, { topN: 4, minScore: 0.01 });
  assert.deepEqual(
    ranked.map(row => [row.term, Number(row.score.toFixed(3))]),
    [
      ['visual', 0.4],
      ['attention', 0.25],
      ['memory', 0.175]
    ],
    'ranking must weight profiles by network fractions, combine duplicate terms, and skip missing/Unassigned networks'
  );
  assert.deepEqual(
    ranked[1].contributors.map(c => [c.network, Number(c.contribution.toFixed(3))]),
    [['Default', 0.15], ['Visual', 0.1]],
    'combined duplicate terms must retain strongest network contributors'
  );

  const truncated = rankFunctionalTerms(summary, profiles, { topN: 1, minScore: 0.01 });
  assert.equal(truncated.length, 1, 'topN must truncate ranked terms');
  assert.equal(truncated[0].term, 'visual');

  const filtered = rankFunctionalTerms(summary, profiles, { topN: 8, minScore: 0.3 });
  assert.deepEqual(filtered.map(row => row.term), ['visual'],
    'minScore must filter weak weighted terms');

  const zero = rankFunctionalTerms({ networks: [{ network: 'Visual', fractionOfLesion: 0 }] }, profiles);
  assert.deepEqual(zero, [], 'zero network weights must produce no terms');

  const originalDocument = global.document;
  global.document = { createElement: makeElement };
  try {
    const table = makeElement('table');
    renderFunctionalProfileTable(table, ranked, { sourceLabel: profiles.sourceLabel });
    assert.equal(table.children[0].tagName, 'caption');
    assert.equal(table.children[0].textContent, 'Neurosynth v7 via NiMARE',
      'renderer must surface the source label');
    const body = table.children.find(child => child.tagName === 'tbody');
    assert.ok(body, 'renderer must append a tbody');
    assert.equal(body.children.length, 3, 'renderer must append one row per ranked term');
    assert.equal(body.children[0].children[2].textContent, 'Visual',
      'single-network driver labels must not duplicate the numeric score');
    assert.equal(body.children[1].children[2].textContent, 'Default; Visual',
      'multi-network driver labels must list contributing networks without score repetition');

    const emptyTable = makeElement('table');
    renderFunctionalProfileTable(emptyTable, [], { emptyLabel: 'No terms' });
    const emptyBody = emptyTable.children.find(child => child.tagName === 'tbody');
    assert.equal(emptyBody.children[0].children[0].textContent, 'No terms',
      'renderer must show an empty-state row');
  } finally {
    global.document = originalDocument;
  }

  console.log('function-profiles OK: ranking, duplicate-term merge, filtering, rendering.');
})();
