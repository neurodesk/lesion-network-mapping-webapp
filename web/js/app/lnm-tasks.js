// Pipeline + stage manifest for the Lesion Network Mapping webapp.
//
// A pipeline is a named sequence of stages. Each stage names the JS module
// that implements it plus the asset IDs it requires (model, atlas, or
// connectome). Asset IDs resolve against web/models/manifest.json so the
// loader can fetch the correct file from a CDN.
//
// Phase 1 ships only the 'lnm-yeo-only' pipeline (manual lesion mask in MNI
// space -> Yeo 7-network overlap). Later phases extend with auto-segmentation,
// MNI registration, and parcel-FC weighted sums; their stages are declared
// but their modules / assets are not yet implemented. isStageRunnable() is
// the single source of truth for whether the UI is allowed to invoke a stage.

export const LNM_PIPELINES = [
  {
    id: 'lnm-yeo-only',
    displayName: 'Yeo 7-network overlap (manual mask)',
    description:
      'Upload a lesion mask already in MNI152 space. Reports per-network ' +
      'overlap with the Yeo 2011 7-network parcellation.',
    stages: [
      {
        id: 'overlap',
        module: 'parcel-overlap',
        atlasAssetId: 'yeo7-2mm',
        required: true
      }
    ]
  },
  {
    id: 'lnm-segment-only',
    displayName: 'Auto-segment lesion (T1 + SynthStrip)',
    description:
      'Drop a structural T1. Brain extraction (SynthStrip) runs ' +
      'automatically; the lesion-segmentation model (SynthStroke ' +
      'baseline) operates on the brain-masked structural and produces a ' +
      'binary lesion mask in the input image\'s native space, downloadable ' +
      'as NIfTI. No MNI registration / no Yeo overlap (those land in ' +
      'Phases 3-4).',
    stages: [
      {
        id: 'brainmask',
        module: 'brain-extraction',
        modelAssetId: 'lnm-synthstrip',
        required: true
      },
      {
        id: 'segment',
        module: 'inference-pipeline',
        modelAssetId: 'lnm-stroke-lesion',
        required: true
      }
    ]
  },
  {
    id: 'lnm-yeo-auto',
    displayName: 'Auto Yeo overlap (T1 -> SynthStrip -> seg -> MNI -> Yeo)',
    description:
      'End-to-end automatic flow: drop a structural T1 (already at 160x160x192 1mm; ' +
      'pre-process with FSL FLIRT to MNI152 + center-crop if needed), get brain ' +
      'extraction + lesion segmentation in native space, then SynthMorph deformable ' +
      'registration warps the lesion mask onto MNI152NLin2009cAsym, after which the ' +
      'Yeo 7-network overlap runs on the warped mask. *Experimental*: requires ' +
      'pre-aligned input; deformable registration without affine pre-alignment may ' +
      'not converge well.',
    stages: [
      {
        id: 'brainmask',
        module: 'brain-extraction',
        modelAssetId: 'lnm-synthstrip',
        required: true
      },
      {
        id: 'segment',
        module: 'inference-pipeline',
        modelAssetId: 'lnm-stroke-lesion',
        required: true
      },
      {
        id: 'register',
        module: 'registration',
        modelAssetId: 'lnm-synthmorph-mni',
        atlasAssetId: 'lnm-mni160',
        required: true
      },
      {
        id: 'overlap',
        module: 'parcel-overlap',
        atlasAssetId: 'yeo7-2mm',
        required: true
      }
    ]
  },
  {
    id: 'lnm-default',
    displayName: 'Lesion Network Mapping (Schaefer400 / GSP1000)',
    description:
      'Full pipeline: ONNX lesion segmentation -> deep-learning MNI ' +
      'registration -> Schaefer400 parcel overlap -> per-parcel functional ' +
      'connectivity weighted sum (GSP1000) -> threshold.',
    stages: [
      {
        id: 'segment',
        module: 'inference-pipeline',
        modelAssetId: 'lnm-stroke-lesion',
        required: true,
        alternatives: [{ id: 'manual-mask', kind: 'upload' }]
      },
      {
        id: 'register',
        module: 'registration',
        modelAssetId: 'lnm-synthmorph-mni',
        required: true
      },
      {
        id: 'overlap',
        module: 'parcel-overlap',
        atlasAssetId: 'schaefer400-7n-2mm',
        required: true
      },
      {
        id: 'network',
        module: 'fc-weighted-sum',
        connectomeAssetId: 'gsp1000-schaefer400-4mm',
        required: true
      },
      {
        id: 'threshold',
        module: 'threshold',
        required: false,
        defaults: {
          mode: 'percentile',
          value: 95,
          symmetric: true,
          minClusterVoxels: 50
        }
      }
    ]
  }
];

// Modules that have a working JS implementation in this phase. A stage whose
// module is not in this set is not runnable, even if its asset ID resolves.
const IMPLEMENTED_MODULES = new Set([
  'parcel-overlap',
  // Phase 2a.1: SynthStrip in web/js/modules/brain-extraction.js, dispatched
  // by the worker's 'run-synthstrip' op.
  'brain-extraction',
  // Phase 2a.2: SynthStroke baseline in web/js/inference-pipeline.js
  // (sliding-window patches), dispatched by the worker's 'run-inference' op.
  'inference-pipeline',
  // Phase 3: SynthMorph SVF in web/js/modules/registration.js + integrate /
  // upsample / warp helpers, dispatched by the worker's 'run-register' op.
  'registration'
]);

export function getPipelineById(id) {
  return LNM_PIPELINES.find(p => p.id === id) || null;
}

export function getRequiredAssetIds(pipeline) {
  if (!pipeline) return [];
  const ids = [];
  for (const stage of pipeline.stages) {
    for (const key of ['modelAssetId', 'atlasAssetId', 'connectomeAssetId']) {
      if (stage[key]) ids.push(stage[key]);
    }
  }
  return Array.from(new Set(ids));
}

export function isStageRunnable(stage) {
  if (!stage || !stage.module) return false;
  if (!IMPLEMENTED_MODULES.has(stage.module)) return false;
  // A required stage must reference at least one asset (model/atlas/connectome)
  // OR opt out via required:false. This matches the SCT regression guard:
  // never silently fall back to a default when a required input is missing.
  if (stage.required) {
    const hasAsset = Boolean(
      stage.modelAssetId || stage.atlasAssetId || stage.connectomeAssetId
    );
    if (!hasAsset) return false;
  }
  return true;
}
