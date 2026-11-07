"""Portable NumPy reader. Image targets preserve the stored 0/1 mask values."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image


def read_manifest(root, split):
    if split not in ('train', 'val', 'test'):
        raise ValueError('Expected split: train, val or test. Reserve test for final evaluation.')
    root = Path(root).resolve()
    with (root/'listas'/f'{split}.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row['split'] != split for row in rows):
        raise ValueError('Empty manifest or unexpected partition.')
    return rows


def safe_path(root, relative):
    root = Path(root).resolve()
    result = (root/relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('A data path escapes the pilot directory.')
    return result


def load_pair(root, row, size=None):
    """Return contiguous float32 arrays: RGB (3,H,W), target (1,H,W)."""
    with Image.open(safe_path(root, row['image_path'])) as source:
        image = source.convert('RGB')
    with Image.open(safe_path(root, row['mask_path'])) as source:
        if source.mode != 'L':
            raise ValueError('Expected a grayscale mask with values 0 and 1.')
        mask = source.copy()
    if image.size != mask.size:
        raise ValueError('Image and mask dimensions do not match.')
    if not set(np.unique(np.asarray(mask))).issubset({0, 1}):
        raise ValueError('Mask must store values 0 and 1, not 0 and 255.')
    if size is not None:
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError('size must be a positive integer or None.')
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        mask = mask.resize((size, size), Image.Resampling.NEAREST)
    image_array = np.asarray(image, dtype=np.float32)/np.float32(255.0)
    mask_array = np.asarray(mask, dtype=np.float32)  # Do not divide the 0/1 target by 255.
    image_array = np.ascontiguousarray(image_array.transpose(2, 0, 1))
    mask_array = np.ascontiguousarray(mask_array[None, :, :])
    return image_array, mask_array


def iter_batches(root, split, batch_size=4, size=None, shuffle=False, seed=42):
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer.')
    rows = read_manifest(root, split)
    indices = np.arange(len(rows))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        selected = [rows[int(index)] for index in indices[start:start+batch_size]]
        pairs = [load_pair(root, row, size) for row in selected]
        yield np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs]), [r['image_id'] for r in selected]
