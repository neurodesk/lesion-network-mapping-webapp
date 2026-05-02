const ATLAS_CACHE = 'lnm-assets-v1';

async function getNifti() {
  if (globalThis.nifti) return globalThis.nifti;
  // The bundled nifti-reader-js file is UMD; loading it installs globalThis.nifti.
  await import('../nifti-js/index.js');
  if (!globalThis.nifti) {
    throw new Error('NIfTI parser is not available');
  }
  return globalThis.nifti;
}

function toArrayBuffer(value) {
  if (value instanceof ArrayBuffer) return value;
  return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
}

function typedArrayForImage(header, imageBuffer) {
  const byteOffset = imageBuffer.byteOffset || 0;
  switch (header.datatypeCode) {
    case 2:
      return new Uint8Array(imageBuffer, byteOffset);
    case 4:
      return new Int16Array(imageBuffer, byteOffset);
    case 8:
      return new Int32Array(imageBuffer, byteOffset);
    case 16:
      return new Float32Array(imageBuffer, byteOffset);
    case 64:
      return new Float64Array(imageBuffer, byteOffset);
    case 256:
      return new Int8Array(imageBuffer, byteOffset);
    case 512:
      return new Uint16Array(imageBuffer, byteOffset);
    case 768:
      return new Uint32Array(imageBuffer, byteOffset);
    default:
      throw new Error(`Unsupported NIfTI datatype: ${header.datatypeCode}`);
  }
}

function extractDims(header) {
  const dimCount = Number(header.dims?.[0]);
  if (dimCount === 3 || (dimCount === 4 && Number(header.dims[4]) === 1)) {
    return [Number(header.dims[1]), Number(header.dims[2]), Number(header.dims[3])];
  }
  throw new Error(`Unsupported NIfTI dimensions: ${header.dims?.join('x')}`);
}

function dimsEqual(a, b) {
  return Array.isArray(a) && Array.isArray(b) &&
    a.length === b.length &&
    a.every((value, index) => value === b[index]);
}

export async function decodeNiftiBuffer(arrayBuffer) {
  const niftiApi = await getNifti();
  let buffer = toArrayBuffer(arrayBuffer);
  const bytes = new Uint8Array(buffer);
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    buffer = niftiApi.decompress(buffer);
  }
  if (!niftiApi.isNIFTI(buffer)) {
    throw new Error('Input is not a NIfTI file');
  }

  const header = niftiApi.readHeader(buffer);
  const imageBuffer = niftiApi.readImage(header, buffer);
  const data = typedArrayForImage(header, imageBuffer);
  const dims = extractDims(header);
  const dtype = header.getDatatypeCodeString
    ? header.getDatatypeCodeString(header.datatypeCode)
    : String(header.datatypeCode);

  return { data, dims, dtype, header };
}

async function loadManifest() {
  if (typeof fetch === 'undefined') {
    throw new Error('fetch is required to load atlas manifest');
  }
  const response = await fetch('./models/manifest.json');
  if (!response.ok) {
    throw new Error(`Failed to load manifest: HTTP ${response.status}`);
  }
  return response.json();
}

async function fetchAtlasBuffer(manifestEntry) {
  if (typeof fetch === 'undefined') {
    throw new Error('fetch is required to load atlas asset');
  }

  if (typeof caches !== 'undefined' && manifestEntry.cacheKey) {
    const cache = await caches.open(ATLAS_CACHE);
    const cached = await cache.match(manifestEntry.cacheKey);
    if (cached) return cached.arrayBuffer();

    const response = await fetch(manifestEntry.sourceUrl);
    if (!response.ok) {
      throw new Error(`Failed to load atlas ${manifestEntry.id}: HTTP ${response.status}`);
    }
    await cache.put(manifestEntry.cacheKey, response.clone());
    return response.arrayBuffer();
  }

  const response = await fetch(manifestEntry.sourceUrl);
  if (!response.ok) {
    throw new Error(`Failed to load atlas ${manifestEntry.id}: HTTP ${response.status}`);
  }
  return response.arrayBuffer();
}

// Phase 4: load a connectome pack (.bin + companion index.json) via the
// same Cache Storage path the atlas-loader uses for atlases. Returns the
// raw ArrayBuffer for the .bin plus the parsed index.
export async function loadConnectomeFromManifest(connectomeAssetId, { manifest } = {}) {
  const assetManifest = manifest || await loadManifest();
  const manifestEntry = assetManifest.connectomeAssets?.find(
    a => a.id === connectomeAssetId
  );
  if (!manifestEntry) {
    throw new Error(`Connectome asset not found: ${connectomeAssetId}`);
  }
  if (manifestEntry.supportStatus !== 'supported') {
    throw new Error(`Connectome asset is not supported: ${connectomeAssetId}`);
  }

  // Fetch + cache the .bin under the same lnm-assets-v1 cache used for atlases.
  let cache = null;
  if (typeof caches !== 'undefined') {
    cache = await caches.open(ATLAS_CACHE);
  }
  const arrayBuffer = await fetchCacheFirst(
    manifestEntry.sourceUrl,
    manifestEntry.cacheKey,
    cache
  );

  // The companion index.json is small (~hundreds of bytes); fetch fresh
  // each time. Cache it under cacheKey + ':index' so the byte-offsets
  // round-trip if the cache backend prefers one round-trip per key.
  const indexCacheKey = manifestEntry.cacheKey
    ? `${manifestEntry.cacheKey}:index`
    : null;
  const indexUrl = manifestEntry.indexSourceUrl;
  if (!indexUrl) {
    throw new Error(`Connectome ${connectomeAssetId} missing indexSourceUrl`);
  }
  const indexBuf = await fetchCacheFirst(indexUrl, indexCacheKey, cache);
  const index = JSON.parse(new TextDecoder('utf-8').decode(indexBuf));

  return { arrayBuffer, index, manifestEntry };
}

// Shared fetch+cache helper. Returns ArrayBuffer.
async function fetchCacheFirst(url, cacheKey, cache) {
  if (cache && cacheKey) {
    const hit = await cache.match(cacheKey);
    if (hit) return hit.arrayBuffer();
    const response = await fetch(url);
    if (!response.ok) throw new Error(`fetch ${url} -> HTTP ${response.status}`);
    await cache.put(cacheKey, response.clone());
    return response.arrayBuffer();
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`fetch ${url} -> HTTP ${response.status}`);
  return response.arrayBuffer();
}

export async function loadAtlasFromManifest(atlasAssetId, { manifest } = {}) {
  const assetManifest = manifest || await loadManifest();
  const manifestEntry = assetManifest.atlasAssets?.find(asset => asset.id === atlasAssetId);
  if (!manifestEntry) {
    throw new Error(`Atlas asset not found: ${atlasAssetId}`);
  }
  if (manifestEntry.supportStatus !== 'supported') {
    throw new Error(`Atlas asset is not supported: ${atlasAssetId}`);
  }

  const arrayBuffer = await fetchAtlasBuffer(manifestEntry);
  const decoded = await decodeNiftiBuffer(arrayBuffer);
  if (!dimsEqual(decoded.dims, manifestEntry.dims)) {
    throw new Error(
      `Atlas dims ${decoded.dims.join('x')} do not match manifest ${manifestEntry.dims.join('x')}`
    );
  }

  return {
    data: decoded.data,
    dims: decoded.dims,
    manifestEntry,
    networkLabels: manifestEntry.networkLabels
  };
}
