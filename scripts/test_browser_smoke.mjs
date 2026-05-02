#!/usr/bin/env node --no-warnings
// Browser smoke tests for the LNM webapp. Two phases:
//
//   Phase 1c.4: manual-mask Yeo overlap flow end-to-end. Uses
//     tests/fixtures/lnm-phantom/lesion-mni2.nii.gz (64-voxel cube in the
//     Yeo Visual parcel) -> 'Visual,64,...' row in the CSV download.
//   Phase 2a.1.5: structural T1 -> auto-fired SynthStrip brain extraction
//     -> brain-mask overlay + #downloadBrainMaskButton enabled + the
//     downloaded mask is a non-trivial NIfTI gzip. Uses the
//     tests/fixtures/synthstrip-mini/T1.nii.gz fixture (MNI152 2mm).
//
// NOT included in `npm test`. Requires:
//   npm install            # adds playwright dep
//   npx playwright install chromium
//   npm run test:smoke
//
// Tests fetch their assets live from Hugging Face by design (developer-
// machine smoke; CI gating with stubbed assets is a later phase). The
// SynthStrip phase pulls the ~10 MB ONNX model, then runs whole-volume
// WASM inference in headless Chromium, so it is slow (30-90s typical).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs/promises';
import { chromium } from 'playwright';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const BASE_PORT = Number(process.env.LNM_SMOKE_PORT || 8123);
// run.sh cd's into web/ before serving, so the docroot IS web/.
const PHANTOM_PATH = path.join(ROOT, 'tests/fixtures/lnm-phantom/lesion-mni2.nii.gz');
const STRUCTURAL_PATH = path.join(ROOT, 'tests/fixtures/synthstrip-mini/T1.nii.gz');
const MNI160_CACHE = path.join(ROOT, 'web/models/_dev_cache/lnm-mni160.nii.gz');
const MNI160_URL =
  'https://huggingface.co/datasets/sbollmann/lnm-webapp-models' +
  '/resolve/main/templates/lnm-mni160.nii.gz';

async function ensureMni160() {
  try {
    const buf = await fs.readFile(MNI160_CACHE);
    if (buf.length > 5_000_000) return MNI160_CACHE;
  } catch (e) { if (e.code !== 'ENOENT') throw e; }
  await fs.mkdir(path.dirname(MNI160_CACHE), { recursive: true });
  console.log(`Fetching lnm-mni160 reference for browser smoke...`);
  const r = await fetch(MNI160_URL);
  if (!r.ok) throw new Error(`lnm-mni160 fetch failed: HTTP ${r.status}`);
  const buf = Buffer.from(await r.arrayBuffer());
  await fs.writeFile(MNI160_CACHE, buf);
  return MNI160_CACHE;
}

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

async function spawnServer(port) {
  const server = spawn('bash', ['web/run.sh', String(port)], {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, LANG: 'C.UTF-8' }
  });
  let stdout = '';
  let stderr = '';
  server.stdout?.on('data', buf => { stdout += buf.toString(); });
  server.stderr?.on('data', buf => { stderr += buf.toString(); });
  return {
    server,
    getStdout: () => stdout,
    getStderr: () => stderr,
    async close() {
      server.kill('SIGTERM');
      await new Promise((resolve) => {
        server.once('exit', resolve);
        setTimeout(() => { server.kill('SIGKILL'); resolve(undefined); }, 2000).unref?.();
      });
    }
  };
}

test('Phase 1c.4 browser smoke: phantom -> Yeo overlap -> CSV download', { timeout: 90000 }, async (t) => {
  // Confirm fixture exists; a missing fixture = build_phantom.py never ran.
  await fs.access(PHANTOM_PATH);

  const port = BASE_PORT;
  const URL = `http://localhost:${port}/`;
  const { server, getStdout, getStderr, close } = await spawnServer(port);
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
    err.message += `\n\n--- server stdout ---\n${getStdout()}\n--- server stderr ---\n${getStderr()}`;
    throw err;
  } finally {
    await close();
  }
});

test('Phase 2a.1.5 browser smoke: structural T1 -> SynthStrip -> brain mask download',
  { timeout: 240000 },
  async (t) => {
    await fs.access(STRUCTURAL_PATH);

    // Use a different port so a leftover server from the previous test
    // (in case its teardown timed out) cannot collide.
    const port = BASE_PORT + 1;
    const URL = `http://localhost:${port}/`;
    const { getStdout, getStderr, close } = await spawnServer(port);

    try {
      await waitForServer(URL);

      const browser = await chromium.launch({ headless: true });
      try {
        const context = await browser.newContext({ acceptDownloads: true });
        const page = await context.newPage();

        const consoleMessages = [];
        page.on('console', msg => consoleMessages.push(`[main:${msg.type()}] ${msg.text()}`));
        page.on('pageerror', err => consoleMessages.push(`[main:pageerror] ${err.message}`));
        // Worker contexts have their own console. Forward.
        page.on('worker', worker => {
          consoleMessages.push(`[worker:spawned] ${worker.url()}`);
          worker.on('console', msg => consoleMessages.push(`[worker:${msg.type()}] ${msg.text()}`));
          worker.on('close', () => consoleMessages.push(`[worker:closed]`));
        });

        await page.goto(URL, { waitUntil: 'load' });
        await page.waitForFunction(
          () => Boolean(window.app && window.app.viewerController && window.app.executor),
          { timeout: 15000 }
        );

        // Drop the structural T1. setStructural() loads it into NiiVue and
        // auto-fires runBrainExtraction(), which posts 'load' + 'run-synthstrip'
        // to the worker. The worker fetches the SynthStrip ONNX from HF on
        // first run (cached after) and runs the whole-volume WASM pipeline.
        await page.setInputFiles('#structuralFileInput', STRUCTURAL_PATH);

        // Wait for SynthStrip completion: brainmaskFile is set by
        // handleStageData('brainmask') in lnm-app.js. Manual poll loop
        // (Playwright's `waitForFunction({timeout})` second-arg overload is
        // unreliable with no-arg page functions — was firing the 30s default
        // instead of the configured 180s).
        const SYNTH_TIMEOUT_MS = 180000;
        const synthStart = Date.now();
        let synthDone = false;
        while (Date.now() - synthStart < SYNTH_TIMEOUT_MS) {
          synthDone = await page.evaluate(() => Boolean(window.app && window.app.brainmaskFile));
          if (synthDone) break;
          await sleep(500);
        }
        if (!synthDone) {
          const inPageConsole = await page
            .$eval('#consoleOutput', el => el ? el.innerText : '(missing)')
            .catch(() => '(unavailable)');
          throw new Error(
            `Brain mask did not appear within ${SYNTH_TIMEOUT_MS}ms\n` +
            `--- captured browser-console messages ---\n${consoleMessages.slice(-30).join('\n')}\n` +
            `--- in-page #consoleOutput widget ---\n${inPageConsole}`
          );
        }
        t.diagnostic(`SynthStrip elapsed: ${((Date.now() - synthStart) / 1000).toFixed(1)}s`);

        const downloadEnabled = await page.$eval('#downloadBrainMaskButton', el => !el.disabled);
        assert.ok(
          downloadEnabled,
          '#downloadBrainMaskButton must enable after the brainmask stage completes\n' +
          `console: ${consoleMessages.slice(-30).join('\n')}`
        );

        const [download] = await Promise.all([
          page.waitForEvent('download', { timeout: 15000 }),
          page.click('#downloadBrainMaskButton')
        ]);
        assert.match(
          download.suggestedFilename(),
          /\.nii(\.gz)?$/,
          `download filename must end with .nii or .nii.gz (got ${download.suggestedFilename()})`
        );
        const downloadPath = await download.path();
        assert.ok(downloadPath, 'playwright must materialise the download to a temp file');
        const masked = await fs.readFile(downloadPath);
        assert.ok(masked.length > 1000, `downloaded brain mask too small: ${masked.length} bytes`);
        // Either a NIfTI-1 .nii (header byte[0]=0x5C, sizeof_hdr=348 LE) or
        // a gzip .nii.gz (magic 0x1f 0x8b).
        const isGzip = masked[0] === 0x1f && masked[1] === 0x8b;
        const isNiftiHdr = masked[0] === 0x5c && masked[1] === 0x01;
        assert.ok(
          isGzip || isNiftiHdr,
          `brain mask download is neither gzip nor a NIfTI-1 header; ` +
          `got first bytes ${masked[0].toString(16)} ${masked[1].toString(16)}`
        );

        t.diagnostic(
          `SynthStrip smoke green; downloaded brain mask = ${masked.length} bytes ` +
          `(suggested filename: ${download.suggestedFilename()}).`
        );
      } finally {
        await browser.close();
      }
    } catch (err) {
      err.message += `\n\n--- server stdout ---\n${getStdout()}\n--- server stderr ---\n${getStderr()}`;
      throw err;
    } finally {
      await close();
    }
  }
);

test('Phase 2a.2.5 browser smoke: structural T1 -> lesion segmentation -> mask download',
  { timeout: 240000 },
  async (t) => {
    await fs.access(STRUCTURAL_PATH);

    const port = BASE_PORT + 2;
    const URL = `http://localhost:${port}/`;
    const { getStdout, getStderr, close } = await spawnServer(port);

    try {
      await waitForServer(URL);

      const browser = await chromium.launch({ headless: true });
      try {
        const context = await browser.newContext({ acceptDownloads: true });
        const page = await context.newPage();

        const consoleMessages = [];
        page.on('console', msg => consoleMessages.push(`[main:${msg.type()}] ${msg.text()}`));
        page.on('pageerror', err => consoleMessages.push(`[main:pageerror] ${err.message}`));
        page.on('worker', worker => {
          worker.on('console', msg => consoleMessages.push(`[worker:${msg.type()}] ${msg.text()}`));
        });

        await page.goto(URL, { waitUntil: 'load' });
        await page.waitForFunction(
          () => Boolean(window.app && window.app.viewerController && window.app.executor),
          { timeout: 15000 }
        );

        // Drop the structural; SynthStrip auto-fires. Wait for the brain
        // mask to land before kicking off lesion seg (the SCT-derived
        // worker state is single-tenant: a 'load' op overwrites
        // workerState.rasData, so we must serialise the two stages).
        await page.setInputFiles('#structuralFileInput', STRUCTURAL_PATH);
        const SYNTH_TIMEOUT_MS = 180000;
        const synthStart = Date.now();
        let synthDone = false;
        while (Date.now() - synthStart < SYNTH_TIMEOUT_MS) {
          synthDone = await page.evaluate(() => Boolean(window.app && window.app.brainmaskFile));
          if (synthDone) break;
          await sleep(500);
        }
        if (!synthDone) {
          throw new Error(
            `SynthStrip never finished within ${SYNTH_TIMEOUT_MS}ms\n` +
            `console: ${consoleMessages.slice(-30).join('\n')}`
          );
        }
        t.diagnostic(`SynthStrip elapsed: ${((Date.now() - synthStart) / 1000).toFixed(1)}s`);

        // Click 'Run lesion segmentation'. This calls executor.loadVolume +
        // executor.runInference(...) under the hood.
        await page.click('#runLesionSegmentationButton');
        const SEG_TIMEOUT_MS = 180000;
        const segStart = Date.now();
        let segDone = false;
        while (Date.now() - segStart < SEG_TIMEOUT_MS) {
          segDone = await page.evaluate(() => Boolean(window.app && window.app.lesionMaskFile));
          if (segDone) break;
          await sleep(500);
        }
        if (!segDone) {
          throw new Error(
            `Lesion segmentation never finished within ${SEG_TIMEOUT_MS}ms\n` +
            `console: ${consoleMessages.slice(-30).join('\n')}`
          );
        }
        t.diagnostic(`Lesion seg elapsed: ${((Date.now() - segStart) / 1000).toFixed(1)}s`);

        const downloadEnabled = await page.$eval('#downloadLesionMaskButton', el => !el.disabled);
        assert.ok(downloadEnabled, '#downloadLesionMaskButton must enable after a successful run');

        const [download] = await Promise.all([
          page.waitForEvent('download', { timeout: 15000 }),
          page.click('#downloadLesionMaskButton')
        ]);
        assert.match(
          download.suggestedFilename(),
          /\.nii(\.gz)?$/,
          `download filename must end with .nii or .nii.gz (got ${download.suggestedFilename()})`
        );
        const downloadPath = await download.path();
        assert.ok(downloadPath, 'playwright must materialise the download to a temp file');
        const masked = await fs.readFile(downloadPath);
        assert.ok(masked.length > 1000, `lesion mask download too small: ${masked.length} bytes`);
        const isGzip = masked[0] === 0x1f && masked[1] === 0x8b;
        const isNiftiHdr = masked[0] === 0x5c && masked[1] === 0x01;
        assert.ok(
          isGzip || isNiftiHdr,
          `lesion mask download is neither gzip nor NIfTI-1 header; ` +
          `got first bytes ${masked[0].toString(16)} ${masked[1].toString(16)}`
        );

        t.diagnostic(
          `Lesion-seg smoke green; downloaded mask = ${masked.length} bytes ` +
          `(suggested filename: ${download.suggestedFilename()}).`
        );
      } finally {
        await browser.close();
      }
    } catch (err) {
      err.message += `\n\n--- server stdout ---\n${getStdout()}\n--- server stderr ---\n${getStderr()}`;
      throw err;
    } finally {
      await close();
    }
  }
);

test('Phase 3.7 browser smoke: structural T1 -> SynthMorph MNI registration',
  { timeout: 360000 },
  async (t) => {
    const structuralPath = await ensureMni160();

    // Use a different port from the previous smoke tests.
    const port = BASE_PORT + 3;
    const URL = `http://localhost:${port}/`;
    const { getStdout, getStderr, close } = await spawnServer(port);

    try {
      await waitForServer(URL);

      const browser = await chromium.launch({ headless: true });
      try {
        const context = await browser.newContext({ acceptDownloads: true });
        const page = await context.newPage();

        const consoleMessages = [];
        page.on('console', msg => consoleMessages.push(`[main:${msg.type()}] ${msg.text()}`));
        page.on('pageerror', err => consoleMessages.push(`[main:pageerror] ${err.message}`));
        page.on('worker', worker => {
          worker.on('console', msg => consoleMessages.push(`[worker:${msg.type()}] ${msg.text()}`));
        });

        await page.goto(URL, { waitUntil: 'load' });
        await page.waitForFunction(
          () => Boolean(window.app && window.app.viewerController && window.app.executor),
          { timeout: 15000 }
        );

        // Drop the lnm-mni160 reference as the 'structural'. It is itself a
        // 160x160x192 1mm MNI152 brain — a clean self-pair for the
        // registration. SynthStrip auto-fires; wait for that, then click
        // #runRegistrationButton.
        await page.setInputFiles('#structuralFileInput', structuralPath);
        const SYNTH_TIMEOUT_MS = 240000;
        const synthStart = Date.now();
        let synthDone = false;
        while (Date.now() - synthStart < SYNTH_TIMEOUT_MS) {
          synthDone = await page.evaluate(() => Boolean(window.app && window.app.brainmaskFile));
          if (synthDone) break;
          await sleep(500);
        }
        if (!synthDone) {
          throw new Error(
            `SynthStrip never finished within ${SYNTH_TIMEOUT_MS}ms\n` +
            `console: ${consoleMessages.slice(-30).join('\n')}`
          );
        }
        t.diagnostic(`SynthStrip elapsed: ${((Date.now() - synthStart) / 1000).toFixed(1)}s`);

        await page.click('#runRegistrationButton');
        // Browser-side WASM SynthMorph one-pass takes 5-10 min on M-series
        // (much slower than CPU EP in Node ~30 s). Waiting for the full
        // run to complete makes the smoke suite painful. Instead, this
        // smoke just confirms the run STARTS cleanly (worker op + UI
        // wiring + manifest lookup all reach the worker without erroring),
        // which is what the browser-level wiring needs to validate.
        // End-to-end correctness is gated by the Node-side
        // `npm run test:registration-parity` (Phase 3.6).
        const START_TIMEOUT_MS = 60000;
        const startedT = Date.now();
        let registerStarted = false;
        while (Date.now() - startedT < START_TIMEOUT_MS) {
          registerStarted = await page.evaluate(
            () => Boolean(
              window.app &&
              window.app.executor &&
              window.app.executor.stepStatus &&
              window.app.executor.stepStatus.register === 'running'
            )
          );
          if (registerStarted) break;
          await sleep(500);
        }
        if (!registerStarted) {
          throw new Error(
            `Registration never transitioned to 'running' within ${START_TIMEOUT_MS}ms\n` +
            `console: ${consoleMessages.slice(-40).join('\n')}`
          );
        }
        t.diagnostic(`Registration started after ${((Date.now() - startedT) / 1000).toFixed(1)}s; ` +
          'completion gated by test:registration-parity (Node).');

        // Wait briefly to surface immediate WIRING errors (manifest miss,
        // worker dispatch typo, etc.). Tolerate the actual SynthMorph
        // forward erroring out — headless Chromium may not have WebGPU
        // and the WASM heap can't hold the model + intermediates. Real
        // forward correctness is gated by test:registration-parity (Node).
        await sleep(5000);
        const wiringErrs = consoleMessages.filter(m =>
          m.includes('pageerror') ||
          (m.includes(':error]') &&
           // ORT's WASM-side OOM surfaces as a numeric error code (the
           // pointer to the C++ exception object). Anything else is a
           // genuine wiring bug we want to flag.
           !/Register error: \d+$/.test(m))
        );
        assert.equal(wiringErrs.length, 0,
          `unexpected wiring errors after registration kickoff:\n${wiringErrs.join('\n')}`);
      } finally {
        await browser.close();
      }
    } catch (err) {
      err.message += `\n\n--- server stdout ---\n${getStdout()}\n--- server stderr ---\n${getStderr()}`;
      throw err;
    } finally {
      await close();
    }
  }
);
