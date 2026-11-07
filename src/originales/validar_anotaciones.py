"""Audit all original YOLO segmentation labels without repairing or rewriting them."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'documentacion/control_calidad/validacion_anotaciones'
OUT.mkdir(parents=True, exist_ok=True)
with (ROOT / 'documentacion/inventario_archivos.csv').open(encoding='utf-8-sig', newline='') as f:
    records = list(csv.DictReader(f))
with (ROOT / 'documentacion/auditoria_duplicados/huellas_imagenes.csv').open(encoding='utf-8-sig', newline='') as f:
    sizes = {(r['partition'], r['image_id']): (int(r['width']), int(r['height'])) for r in csv.DictReader(f)}

issues = []
rows = []
all_vertices = []
all_pixel_areas = []
class_counts = Counter()
partition_counts = defaultdict(Counter)
original_fingerprints = hashlib.sha256()

def issue(record, line, severity, code, detail):
    issues.append({'image_id': record['image_id'], 'partition': record['partition_original'],
                   'label_path': record['label_path'], 'line': line,
                   'severity': severity, 'code': code, 'detail': detail})

for record in records:
    partition = record['partition_original']
    path = ROOT / record['label_path']
    before_issues = len(issues)
    valid_regions = 0
    label_lines = 0
    file_vertices = []
    file_areas = []
    empty = False
    if not path.is_file():
        issue(record, 0, 'ERROR', 'MISSING_LABEL', 'No corresponding annotation file.')
    else:
        raw = path.read_bytes()
        original_fingerprints.update(record['label_path'].encode('utf-8') + b'\0' + raw + b'\0')
        try:
            content = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            issue(record, 0, 'ERROR', 'INVALID_ENCODING', 'Annotation cannot be decoded as UTF-8.')
            content = None
        if content is not None:
            lines = [(i, line) for i, line in enumerate(content.splitlines(), 1) if line.strip()]
            label_lines = len(lines)
            empty = not lines
            if empty:
                issue(record, 0, 'REVIEW', 'EMPTY_LABEL', 'Empty label is not automatically a valid negative; consult visual review.')
            for line_number, line in lines:
                start = len(issues)
                tokens = line.split()
                try:
                    class_id = int(tokens[0])
                except (ValueError, IndexError):
                    issue(record, line_number, 'ERROR', 'INVALID_CLASS_TOKEN', 'Class must be an integer.')
                    continue
                class_counts[str(class_id)] += 1
                if class_id != 0:
                    issue(record, line_number, 'ERROR', 'UNKNOWN_CLASS', f'Expected class 0, got {class_id}.')
                try:
                    coordinates = [float(value) for value in tokens[1:]]
                except ValueError:
                    issue(record, line_number, 'ERROR', 'NON_NUMERIC_COORDINATE', 'Coordinates must be numeric.')
                    continue
                if len(coordinates) < 6 or len(coordinates) % 2:
                    issue(record, line_number, 'ERROR', 'INVALID_COORDINATE_COUNT', f'{len(coordinates)} coordinates; need at least 3 x/y pairs.')
                    continue
                if not all(math.isfinite(value) for value in coordinates):
                    issue(record, line_number, 'ERROR', 'NON_FINITE_COORDINATE', 'NaN or infinity present.')
                    continue
                if any(value < 0 or value > 1 for value in coordinates):
                    issue(record, line_number, 'ERROR', 'COORDINATE_OUT_OF_RANGE', f'Range [{min(coordinates)}, {max(coordinates)}], expected [0,1].')
                points = list(zip(coordinates[0::2], coordinates[1::2]))
                if len(set(points)) < 3:
                    issue(record, line_number, 'ERROR', 'TOO_FEW_DISTINCT_VERTICES', 'Fewer than 3 distinct vertices.')
                signed_area = sum(points[i][0]*points[(i+1)%len(points)][1] - points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points))) / 2
                if abs(signed_area) <= 1e-12:
                    issue(record, line_number, 'ERROR', 'ZERO_SIGNED_AREA', 'Polygon has zero or negligible signed area; do not silently repair.')
                width, height = sizes[(partition, record['image_id'])]
                pixel_area = abs(signed_area) * width * height
                if 0 < pixel_area < 1:
                    issue(record, line_number, 'REVIEW', 'SUBPIXEL_AREA', f'Continuous polygon area is {pixel_area:.6g} px^2; inspect rasterization.')
                consecutive_repeats = sum(points[i] == points[i-1] for i in range(1, len(points)))
                if consecutive_repeats:
                    issue(record, line_number, 'WARNING', 'REPEATED_CONSECUTIVE_VERTEX', f'{consecutive_repeats} repeated consecutive vertices; no removal applied.')
                errors = any(row['severity']=='ERROR' for row in issues[start:])
                if not errors:
                    valid_regions += 1
                    file_vertices.append(len(points))
                    file_areas.append(pixel_area)
                    all_vertices.append(len(points))
                    all_pixel_areas.append(pixel_area)
    file_issues = issues[before_issues:]
    n_errors = sum(i['severity']=='ERROR' for i in file_issues)
    n_reviews = sum(i['severity']=='REVIEW' for i in file_issues)
    n_warnings = sum(i['severity']=='WARNING' for i in file_issues)
    status = 'ERROR' if n_errors else ('REVIEW' if n_reviews else ('VALID_WITH_WARNING' if n_warnings else 'FORMAT_VALID'))
    rows.append({'image_id':record['image_id'],'partition':partition,'label_path':record['label_path'],
                 'status':status,'annotation_lines':label_lines,'structurally_valid_regions':valid_regions,
                 'min_vertices':min(file_vertices) if file_vertices else '',
                 'max_vertices':max(file_vertices) if file_vertices else '',
                 'min_polygon_area_px2':round(min(file_areas),6) if file_areas else '',
                 'errors':n_errors,'review_flags':n_reviews,'warnings':n_warnings,'empty_label':empty})
    partition_counts[partition]['files'] += 1
    partition_counts[partition][status] += 1
    partition_counts[partition]['annotation_lines'] += label_lines

def write_csv(path, data, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields);w.writeheader();w.writerows(data)

write_csv(OUT/'validacion_por_archivo.csv', rows, list(rows[0]))
write_csv(OUT/'incidencias_formato.csv', issues, ['image_id','partition','label_path','line','severity','code','detail'])
summary = {
    'annotation_files_checked':len(records),
    'nonempty_files':sum(not row['empty_label'] for row in rows),
    'files_with_errors':sum(row['errors']>0 for row in rows),
    'empty_label_files':sum(row['empty_label'] for row in rows),
    'annotation_lines':sum(row['annotation_lines'] for row in rows),
    'structurally_valid_regions':sum(row['structurally_valid_regions'] for row in rows),
    'classes_seen_per_region':dict(class_counts),
    'status_counts':dict(Counter(row['status'] for row in rows)),
    'partition_summary':{key:dict(value) for key,value in partition_counts.items()},
    'issue_counts':dict(Counter(row['code'] for row in issues)),
    'min_vertices':min(all_vertices) if all_vertices else None,
    'max_vertices':max(all_vertices) if all_vertices else None,
    'min_continuous_polygon_area_px2':min(all_pixel_areas) if all_pixel_areas else None,
    'original_annotation_digest_sha256':original_fingerprints.hexdigest(),
    'originals_modified':False,
    'geometry_checks':'Coordinate count, finite values, normalized range, distinct vertices and signed area.',
    'not_checked':['Exhaustive self-intersection or topological validity of every polygon.','Visual correctness of every annotation.','Mask quality after rasterization of the complete dataset.'],
    'prior_visual_findings_still_apply':True,
    'whole_dataset_approved_for_training':False,
}
(OUT/'resumen_validacion.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=True,indent=2))
print('FIRST_ISSUES',json.dumps(issues[:12],ensure_ascii=True))
