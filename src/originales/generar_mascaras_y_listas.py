"""Generate portable manifests and original-resolution 0/1 masks; preserve originals."""
from pathlib import Path
from collections import Counter
from io import BytesIO
import csv
import hashlib
import json
import math
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datos/preparados'
for folder in ['listas', 'mascaras/train', 'mascaras/val', 'mascaras/test']:
    (OUT / folder).mkdir(parents=True, exist_ok=True)

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

records = read_csv(ROOT / 'documentacion/inventario_archivos.csv')
qc = json.loads((ROOT / 'documentacion/control_calidad/validacion_anotaciones/resumen_validacion.json').read_text(encoding='utf-8'))
assert qc['files_with_errors'] == 0
image_hashes = {(row['partition'], row['image_id']): row['sha256_file']
                for row in read_csv(ROOT / 'documentacion/auditoria_duplicados/huellas_imagenes.csv')}
annotation_exclusions = {(row['original_partition'], row['image_id']): row['reason']
                         for row in read_csv(ROOT / 'documentacion/control_calidad/exclusiones_anotacion_confirmadas.csv')}
audit = json.loads((ROOT / 'documentacion/auditoria_duplicados/resumen_auditoria.json').read_text(encoding='utf-8'))
duplicate_rule = audit['recommendation_not_yet_applied']
duplicate_key = ('test', duplicate_rule['omit_duplicate_id'])
keep_key = ('test', duplicate_rule['keep_id'])
original_lookup = {(r['partition_original'], r['image_id']): r for r in records}
duplicate = original_lookup[duplicate_key]
keep = original_lookup[keep_key]
assert (ROOT / duplicate['image_path']).read_bytes() == (ROOT / keep['image_path']).read_bytes()
assert (ROOT / duplicate['label_path']).read_bytes() == (ROOT / keep['label_path']).read_bytes()
pilot_reservations = {(row['original_partition'], row['image_id'])
                      for row in read_csv(ROOT / 'documentacion/control_calidad/reservas_seleccion_piloto.csv')}

rows, omitted = [], []
label_digest = hashlib.sha256()
regression_checks = 0
for number, record in enumerate(records, 1):
    partition, image_id = record['partition_original'], record['image_id']
    key = (partition, image_id)
    label_path = ROOT / record['label_path']
    raw_label = label_path.read_bytes()
    label_digest.update(record['label_path'].encode('utf-8') + b'\0' + raw_label + b'\0')
    reason = annotation_exclusions.get(key)
    if key == duplicate_key:
        reason = 'copia_exacta_de_imagen_y_etiqueta_en_prueba'
    if reason:
        omitted.append({'image_id':image_id, 'partition':partition, 'reason':reason,
                        'original_image':record['image_path'], 'original_label':record['label_path'],
                        'original_files_preserved':True})
        continue

    raw_image = (ROOT / record['image_path']).read_bytes()
    image_sha = hashlib.sha256(raw_image).hexdigest()
    assert image_sha == image_hashes[key], ('Source image changed since audit', image_id)
    with Image.open(BytesIO(raw_image)) as image:
        width, height = image.size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    region_count = 0
    for line in raw_label.decode('utf-8-sig').splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        coordinates = [float(value) for value in tokens[1:]]
        assert int(tokens[0]) == 0
        assert len(coordinates) >= 6 and len(coordinates) % 2 == 0
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates)
        points = [(coordinates[i]*width, coordinates[i+1]*height) for i in range(0, len(coordinates), 2)]
        draw.polygon(points, fill=1)
        region_count += 1
    mask_path = OUT / 'mascaras' / partition / (image_id + '.png')
    if mask_path.exists():
        with Image.open(mask_path) as previous:
            assert previous.mode == 'L' and previous.size == mask.size and previous.tobytes() == mask.tobytes(), ('Existing mask differs', image_id)
    else:
        mask.save(mask_path)
    with Image.open(mask_path) as saved:
        saved.load()
        assert saved.mode == 'L' and saved.size == (width, height)
        assert sorted(value for count, value in saved.getcolors()) == [0, 1]
        positive = saved.histogram()[1]
        assert 0 < positive < width*height
    previous_review = ROOT / 'documentacion/revision_visual/muestra_20/mascaras_01' / (image_id + '.png')
    if previous_review.exists():
        with Image.open(previous_review) as previous:
            assert previous.size == mask.size and previous.tobytes() == mask.tobytes(), ('Mismatch with reviewed mask', image_id)
        regression_checks += 1
    quality = []
    if key in pilot_reservations:
        quality.append('reservado_fuera_del_primer_piloto_por_incertidumbre_visual')
    if image_id == '1870.rf.806be3a1a60b69c62e91cf13a6ccc972':
        quality.append('fragmentacion_original_conservada_sin_rellenar_huecos')
    rows.append({
        'image_id':image_id, 'split':partition,
        'source_group_apparent':image_id.split('.rf.')[0],
        'image_path':record['image_path'],
        'mask_path':mask_path.relative_to(ROOT).as_posix(),
        'original_label_path':record['label_path'],
        'width':width, 'height':height, 'annotation_regions':region_count,
        'positive_pixels':positive, 'positive_fraction':round(positive/(width*height),8),
        'image_sha256':image_sha, 'mask_sha256':hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        'quality_notes':';'.join(quality),
    })
    if number % 500 == 0:
        print(f'Mascaras verificadas: {len(rows)}; registros recorridos: {number}/{len(records)}', flush=True)

assert label_digest.hexdigest() == qc['original_annotation_digest_sha256'], 'Original annotations changed since audit.'
counts = Counter(row['split'] for row in rows)
assert counts == {'train':3717, 'val':199, 'test':111}, counts
assert len(omitted) == 2
assert len({row['image_id'] for row in rows}) == len(rows)
assert regression_checks == 20
prefixes = {split:{row['source_group_apparent'] for row in rows if row['split']==split} for split in counts}
assert not (prefixes['train'] & prefixes['val'] or prefixes['train'] & prefixes['test'] or prefixes['val'] & prefixes['test'])
sha_sets = {split:{row['image_sha256'] for row in rows if row['split']==split} for split in counts}
assert not (sha_sets['train'] & sha_sets['val'] or sha_sets['train'] & sha_sets['test'] or sha_sets['val'] & sha_sets['test'])

def write_csv(path, data, fieldnames):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(data)

for split in ('train', 'val', 'test'):
    write_csv(OUT/'listas'/f'{split}.csv', [row for row in rows if row['split']==split], list(rows[0]))
write_csv(OUT/'listas/todas.csv', rows, list(rows[0]))
write_csv(OUT/'listas/omisiones_documentadas.csv', omitted, list(omitted[0]))
checksums = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'listas').glob('*.csv'))}
summary = {
    'version':'preparacion_v1',
    'counts':dict(counts), 'total_masks':len(rows),
    'mask_format':'PNG grayscale L, uint8 containing only 0 and 1',
    'foreground':1, 'background':0,
    'resized':False,
    'image_dimensions':dict(Counter(f"{row['width']}x{row['height']}" for row in rows)),
    'conversion':'Union of all original class-0 polygons at native resolution; x=x_normalized*width, y=y_normalized*height; PIL ImageDraw.polygon.',
    'closing_or_manual_repair_applied':False,
    'omitted_from_manifests':omitted,
    'all_output_masks_reopened_and_checked':True,
    'source_image_hashes_match_audit':True,
    'source_annotation_digest_matches_audit':True,
    'matches_previous_20_review_masks':regression_checks,
    'original_files_modified':False,
    'paths_relative_to':'Proyecto_Final_Fisuras project root, not the CSV directory',
    'pilot_subset_selected':False,
    'model_training_performed':False,
    'case_12_scope':'Retained in full train manifest, reserved only from first pilot selection.',
    'case_17_scope':'Retained with fragmentation note; no artificial bridging.',
    'limitations':'Format and identity checks do not guarantee perfect visual/topological annotations or complete independence of scenes.',
    'manifest_sha256':checksums,
}
(OUT/'resumen_preparacion.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=True,indent=2), flush=True)
