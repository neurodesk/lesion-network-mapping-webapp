const YEO7_LABELS = [
  { index: 0, name: 'Background', color: '#000000', alpha: 0 },
  { index: 1, name: 'Visual', color: '#781286', alpha: 255 },
  { index: 2, name: 'Somatomotor', color: '#4682b4', alpha: 255 },
  { index: 3, name: 'DorsalAttention', color: '#00760e', alpha: 255 },
  { index: 4, name: 'VentralAttention', color: '#c43afa', alpha: 255 },
  { index: 5, name: 'Limbic', color: '#dcf8a4', alpha: 255 },
  { index: 6, name: 'Frontoparietal', color: '#e69422', alpha: 255 },
  { index: 7, name: 'Default', color: '#cd3e4e', alpha: 255 }
];

function hexToRgb(hex) {
  const raw = hex.replace('#', '');
  return [
    parseInt(raw.slice(0, 2), 16),
    parseInt(raw.slice(2, 4), 16),
    parseInt(raw.slice(4, 6), 16)
  ];
}

export const YEO7_COLORMAP = {
  R: YEO7_LABELS.map(label => hexToRgb(label.color)[0]),
  G: YEO7_LABELS.map(label => hexToRgb(label.color)[1]),
  B: YEO7_LABELS.map(label => hexToRgb(label.color)[2]),
  A: YEO7_LABELS.map(label => label.alpha),
  I: YEO7_LABELS.map(label => label.index),
  labels: YEO7_LABELS.map(label => label.name)
};

const LESION_MASK_LABELS = [
  { index: 0, name: 'Background', color: '#000000', alpha: 0 },
  { index: 1, name: 'Lesion mask', color: '#008cff', alpha: 255 }
];

export const LESION_MASK_COLORMAP_ID = 'lnm-lesion-blue';

export const LESION_MASK_COLORMAP = {
  R: LESION_MASK_LABELS.map(label => hexToRgb(label.color)[0]),
  G: LESION_MASK_LABELS.map(label => hexToRgb(label.color)[1]),
  B: LESION_MASK_LABELS.map(label => hexToRgb(label.color)[2]),
  A: LESION_MASK_LABELS.map(label => label.alpha),
  I: LESION_MASK_LABELS.map(label => label.index),
  labels: LESION_MASK_LABELS.map(label => label.name)
};

export const YEO7_NETWORK_LABELS = Object.fromEntries(
  YEO7_LABELS
    .filter(label => label.index > 0)
    .map(label => [label.index, label.name])
);
