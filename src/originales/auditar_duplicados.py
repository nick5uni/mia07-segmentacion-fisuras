"""Read-only image audit: exact copies, apparent source groups and similarity candidates."""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import json
import itertools
import math
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'documentacion/auditoria_duplicados'
OUT.mkdir(parents=True, exist_ok=True)
with (ROOT / 'documentacion/inventario_archivos.csv').open(encoding='utf-8-sig', newline='') as stream:
    records = list(csv.DictReader(stream))

def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

size, low = 32, 8
dct = np.cos(np.pi * np.arange(low)[:, None] * (2 * np.arange(size)[None, :] + 1) / (2 * size))
transforms = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
              Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
              Image.Transpose.TRANSPOSE, Image.Transpose.TRANSVERSE]
transform_names = ['identity', 'rotate90', 'rotate180', 'rotate270', 'flip_left_right', 'flip_top_bottom', 'transpose', 'transverse']

def phash(image):
    a = np.asarray(image, dtype=np.float64)
    coefficients = dct @ a @ dct.T
    bits = coefficients.ravel() > np.median(coefficients.ravel()[1:])
    bits[0] = False
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value

file_groups, pixel_groups, source_groups = defaultdict(list), defaultdict(list), defaultdict(list)
inventory = []
errors = []
hashes = []
for index, record in enumerate(records):
    path = ROOT / record['image_path']
    try:
        file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as source:
            image = source.convert('RGB')
            image.load()
        pixel_sha = hashlib.sha256(f'{image.width},{image.height},RGB:'.encode() + image.tobytes()).hexdigest()
        small = image.convert('L').resize((size, size), Image.Resampling.LANCZOS)
        variants = [small if t is None else small.transpose(t) for t in transforms]
        perceptual = [phash(v) for v in variants]
        hashes.append(perceptual)
        source_id = record['image_id'].split('.rf.')[0]
        file_groups[file_sha].append(index)
        pixel_groups[pixel_sha].append(index)
        source_groups[source_id].append(index)
        inventory.append({
            'image_id': record['image_id'], 'partition': record['partition_original'],
            'source_prefix_apparent': source_id, 'width': image.width, 'height': image.height,
            'sha256_file': file_sha, 'sha256_rgb_pixels': pixel_sha,
            'phash_identity': f'{perceptual[0]:016x}',
        })
    except Exception as error:
        errors.append({'index': index, 'path': str(path), 'error': repr(error)})
        hashes.append([0] * 8)
    if (index + 1) % 500 == 0:
        print(f'Imagenes auditadas: {index + 1}/{len(records)}', flush=True)

if errors:
    (OUT / 'errores_lectura.json').write_text(json.dumps(errors, indent=2), encoding='utf-8')
    raise RuntimeError('Image reading errors; inspect errores_lectura.json before continuing.')

write_csv(OUT / 'huellas_imagenes.csv', inventory, list(inventory[0]))

def export_groups(groups, prefix):
    duplicate_sets = []
    rows = []
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        partitions = sorted({records[i]['partition_original'] for i in indices})
        duplicate_sets.append({'group_key': key, 'size': len(indices), 'partitions': partitions, 'indices': indices})
        for i in indices:
            rows.append({'group_key': key, 'cross_partition': len(partitions) > 1,
                         'image_id': records[i]['image_id'], 'partition': records[i]['partition_original']})
    write_csv(OUT / (prefix + '.csv'), rows, ['group_key', 'cross_partition', 'image_id', 'partition'])
    return duplicate_sets

exact_file = export_groups(file_groups, 'copias_exactas_archivo')
exact_pixels = export_groups(pixel_groups, 'copias_exactas_pixeles')
sources = export_groups(source_groups, 'versiones_por_prefijo')

hash_matrix = np.asarray(hashes, dtype=np.uint64)
popcount = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)
candidates = []
threshold = 6
for left_partition, right_partition in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
    left = np.array([i for i, r in enumerate(records) if r['partition_original'] == left_partition], dtype=np.int64)
    right = [i for i, r in enumerate(records) if r['partition_original'] == right_partition]
    values = hash_matrix[left].reshape(-1)
    for j in right:
        xor = np.bitwise_xor(values, hash_matrix[j, 0])
        distances = popcount[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1).reshape(-1, 8)
        minima = distances.min(axis=1)
        for position in np.flatnonzero(minima <= threshold):
            i = int(left[position])
            t = int(distances[position].argmin())
            with Image.open(ROOT / records[i]['image_path']) as a, Image.open(ROOT / records[j]['image_path']) as b:
                aligned = a.convert('L').resize((64, 64), Image.Resampling.LANCZOS)
                if transforms[t] is not None:
                    aligned = aligned.transpose(transforms[t])
                reference = b.convert('L').resize((64, 64), Image.Resampling.LANCZOS)
                x, y = np.asarray(aligned, dtype=np.float64), np.asarray(reference, dtype=np.float64)
                mae = float(np.abs(x - y).mean())
                vx, vy = x.var(), y.var()
                covariance = ((x - x.mean()) * (y - y.mean())).mean()
                ssim_global = float(((2*x.mean()*y.mean() + 6.5025) * (2*covariance + 58.5225)) /
                                    ((x.mean()**2 + y.mean()**2 + 6.5025) * (vx + vy + 58.5225)))
            candidates.append({
                'left_index': i, 'right_index': j, 'left_id': records[i]['image_id'],
                'right_id': records[j]['image_id'], 'left_partition': left_partition,
                'right_partition': right_partition, 'phash_distance': int(minima[position]),
                'left_transform': transform_names[t], 'mae_grayscale_64': round(mae, 5),
                'global_ssim_grayscale_64': round(ssim_global, 6),
                'status': 'CANDIDATE_ONLY_NOT_CONFIRMED_DUPLICATE',
            })
    print(f'Similitud comparada: {left_partition} frente a {right_partition}', flush=True)
candidates.sort(key=lambda r: (-r['global_ssim_grayscale_64'], r['mae_grayscale_64'], r['phash_distance']))
candidate_fields = ['left_index', 'right_index', 'left_id', 'right_id', 'left_partition', 'right_partition',
                    'phash_distance', 'left_transform', 'mae_grayscale_64', 'global_ssim_grayscale_64', 'status']
write_csv(OUT / 'candidatos_parecidos_entre_particiones.csv', candidates, candidate_fields)

font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 15)
bold = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 19)
for sheet, start in enumerate(range(0, min(20, len(candidates)), 5), 1):
    subset = candidates[start:start+5]
    canvas = Image.new('RGB', (876, 60 + len(subset) * 310), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), 'Parecidos entre particiones - candidatos, no duplicados confirmados', fill='black', font=bold)
    for row_number, candidate in enumerate(subset):
        y = 62 + row_number * 310
        title = f"{start+row_number+1:02d} | pHash {candidate['phash_distance']} | SSIM global {candidate['global_ssim_grayscale_64']:.3f}"
        draw.text((16, y-25), title, fill='black', font=font)
        for column, key in enumerate(['left_index', 'right_index']):
            record = records[candidate[key]]
            with Image.open(ROOT / record['image_path']) as image:
                picture = image.convert('RGB')
                picture.thumbnail((416, 256), Image.Resampling.LANCZOS)
                canvas.paste(picture, (16 + column * 436, y))
            label = record['partition_original'] + ' / ' + record['image_id'].split('.rf.')[0]
            draw.text((16 + column * 436, y+260), label, fill='black', font=font)
    canvas.save(OUT / f'candidatos_lamina_{sheet:02d}.png')

summary = {
    'images_read': len(inventory), 'reading_errors': errors,
    'partitions': dict(Counter(r['partition_original'] for r in records)),
    'apparent_source_prefix_count': len(source_groups),
    'apparent_source_group_sizes': dict(Counter(len(v) for v in source_groups.values())),
    'apparent_source_prefixes_crossing_partitions': sum(len(g['partitions']) > 1 for g in sources),
    'exact_file_duplicate_groups': len(exact_file),
    'exact_file_duplicate_groups_cross_partition': sum(len(g['partitions']) > 1 for g in exact_file),
    'exact_pixel_duplicate_groups': len(exact_pixels),
    'exact_pixel_duplicate_groups_cross_partition': sum(len(g['partitions']) > 1 for g in exact_pixels),
    'perceptual_candidates_cross_partition': len(candidates),
    'phash_threshold': threshold,
    'transformations_checked': transform_names,
    'high_similarity_candidates_ssim_ge_095': sum(r['global_ssim_grayscale_64'] >= .95 for r in candidates),
    'perceptual_review': 'Pending human inspection of candidate sheets',
    'files_deleted_or_moved': False, 'original_splits_changed': False,
    'test_use': 'Data identity audit only, without model training, model predictions or hyperparameter tuning.',
    'limitations': [
        'Different filename prefixes do not guarantee different original photos or scenes.',
        'Perceptual hashes are screening tools, not proof of duplicates or independent scenes.',
        'Arbitrary-angle rotations, crops, exposure changes and other transformations may escape this screening.',
        'Global SSIM is a heuristic on 64x64 grayscale images, not a model performance metric.',
        'Absence of detected cross-partition copies does not prove there is no data leakage.',
    ],
}
(OUT / 'resumen_auditoria.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
print('TOP_CANDIDATES', json.dumps(candidates[:8], ensure_ascii=True), flush=True)
