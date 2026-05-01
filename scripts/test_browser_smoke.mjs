#!/usr/bin/env node --no-warnings
// Phase 1c.4 browser smoke test.
//
// Runs the Phase 1 manual-mask Yeo overlap flow end-to-end in headless
// Chromium against a real dev server. Uses the deterministic phantom from
// tests/fixtures/lnm-phantom/lesion-mni2.nii.gz: a 4x4x4 cube placed inside
// the Yeo Visual parcel (label 1) so the expected outcome is fixed:
//
//   - 64 lesion voxels, all in 'Visual' network
//   - 0 voxels outside the atlas (warning stays hidden)
//   - CSV header + 'Visual,64,...' as the first non-Unassigned row.
//
// NOT included in `npm test`. Requires:
//   npm install            # adds playwright dep
//   npx playwright install chromium
//   npm run test:smoke
//
// The test fetches the Yeo atlas live from Hugging Face, by design (Phase 1
// developer-machine smoke; CI gating with stubbed assets is a later phase).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs/promises';
import { chromium } from 'playwright';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PORT = Number(process.env.LNM_SMOKE_PORT || 8123);
// run.sh cd's into web/ before serving, so the docroot IS web/.
const URL = `http://localhost:${PORT}/`;
const PHANTOM_PATH = path.join(ROOT, 'tests/fixtures/lnm-phantom/lesion-mni2.nii.gz');

async function waitForServer(url, timeoutMs = 10000) {
  const deadline = Date.now() + timeoutMs;
  let lastErr;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(url);
      if (r.ok) return;
      lastErr = new Error(`HTTP ${r.status}`);
    } catch (err) {
      lastErr = err;
    }
    await sleep(150);
  }
  throw new Error(`server at ${url} not reachable within ${timeoutMs}ms (${lastErr?.message})`);
}

test('Phase 1c.4 browser smoke: phantom -> Yeo overlap -> CSV download', { timeout: 90000 }, async (t) => {
  // Confirm fixture exists; a missing fixture = build_phantom.py never ran.
  await fs.access(PHANTOM_PATH);

  const server = spawn('bash', ['web/run.sh', String(PORT)], {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, LANG: 'C.UTF-8' }
  });
  let serverStdout = '';
  let serverStderr = '';
  server.stdout?.on('data', buf => { serverStdout += buf.toString(); });
  server.stderr?.on('data', buf => { serverStderr += buf.toString(); });

  try {
    await waitForServer(URL);

    const browser = await chromium.launch({ headless: true });
    try {
      const context = await browser.newContext({ acceptDownloads: true });
      const page = await context.newPage();

      const consoleMessages = [];
      page.on('console', msg => consoleMessages.push(`[${msg.type()}] ${msg.text()}`));
      page.on('pageerror', err => consoleMessages.push(`[pageerror] ${err.message}`));

      await page.goto(URL, { waitUntil: 'load' });
      await page.waitForFunction(() => Boolean(window.app && window.app.viewerController), { timeout: 15000 });

      // Drop the phantom mask. The change handler is async; wait until app.lesionFile is populated.
      await page.setInputFiles('#lesionFileInput', PHANTOM_PATH);
      await page.waitForFunction(() => Boolean(window.app.lesionFile), { timeout: 15000 });

      // Click compute. Live HF atlas fetch + decode + reducer is sub-second
      // on a warm cache; allow a wider margin for cold-cache + slow links.
      await page.click('#computeOverlapButton');
      await page.waitForFunction(
        () => Boolean(window.app.overlapResult),
        { timeout: 60000 }
      );

      // The phantom is in Yeo parcel 1 (Visual). Expect at least one bar
      // rendered with non-zero width.
      const barInfo = await page.$$eval(
        '#networkOverlapTable tbody tr .overlap-bar-fill',
        nodes => nodes.map(n => ({ width: n.style.width, color: n.style.backgroundColor }))
      );
      assert.ok(barInfo.length > 0, 'expected at least one row with an overlap-bar-fill');
      const nonzeroBars = barInfo.filter(b => b.width && b.width !== '0%' && b.width !== '0');
      assert.ok(
        nonzeroBars.length > 0,
        `expected at least one bar with non-zero width; got ${JSON.stringify(barInfo)}\n` +
        `console: ${consoleMessages.join('\n')}`
      );

      // Phantom fully inside atlas -> outside-atlas warning hidden.
      const warningHidden = await page.$eval(
        '#outsideAtlasWarning',
        el => el.classList.contains('hidden')
      );
      assert.ok(warningHidden, '#outsideAtlasWarning must be hidden for an inside-atlas phantom');

      // Download button must be enabled after a successful run.
      const downloadEnabled = await page.$eval('#downloadOverlapCsv', el => !el.disabled);
      assert.ok(downloadEnabled, '#downloadOverlapCsv must become enabled after runYeoOverlap');

      // Trigger the CSV download and capture the file body.
      const [download] = await Promise.all([
        page.waitForEvent('download', { timeout: 15000 }),
        page.click('#downloadOverlapCsv')
      ]);
      assert.match(
        download.suggestedFilename(),
        /\.csv$/,
        `download filename must end with .csv (got ${download.suggestedFilename()})`
      );
      const downloadPath = await download.path();
      assert.ok(downloadPath, 'playwright must materialise the download to a temp file');
      const csv = await fs.readFile(downloadPath, 'utf8');
      const expectedHeader =
        'network,voxelsInLesion,fractionOfLesion,voxelsInNetwork,fractionOfNetwork,parcels';
      assert.ok(
        csv.startsWith(expectedHeader),
        `CSV must start with the canonical header.\nGot: ${JSON.stringify(csv.slice(0, 120))}`
      );
      // The phantom places 64 voxels in label 1 (Yeo Visual). The CSV must
      // surface a 'Visual' row with voxelsInLesion=64.
      assert.match(
        csv,
        /\nVisual,64,/,
        `expected a 'Visual,64,...' row in the CSV; full CSV:\n${csv}`
      );

      t.diagnostic(
        `Smoke test green: ${barInfo.length} table rows, ${nonzeroBars.length} non-zero bars, ` +
        `CSV byte-size=${csv.length}.`
      );
    } finally {
      await browser.close();
    }
  } catch (err) {
    err.message += `\n\n--- server stdout ---\n${serverStdout}\n--- server stderr ---\n${serverStderr}`;
    throw err;
  } finally {
    server.kill('SIGTERM');
    await new Promise((resolve) => {
      server.once('exit', resolve);
      setTimeout(() => { server.kill('SIGKILL'); resolve(undefined); }, 2000).unref?.();
    });
  }
});
