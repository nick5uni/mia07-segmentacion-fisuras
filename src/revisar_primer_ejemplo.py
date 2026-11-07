"""Render one training image beside its original polygon annotations."""
from pathlib import Path
import json
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'data/reproducido/datos/extraidos/crack-seg'
NAME = '10.rf.1be181322657e5592be06e9ed277e0b3.jpg'
image_path = DATASET / 'images/train' / NAME
label_path = DATASET / 'labels/train' / (Path(NAME).stem + '.txt')
original = Image.open(image_path).convert('RGB')
width, height = original.size
annotated = original.copy()
draw = ImageDraw.Draw(annotated)
colors = ['#E52629', '#0087DA', '#8B25C7', '#009B68']
polygons = []
for number, line in enumerate(label_path.read_text(encoding='utf-8-sig').splitlines(), 1):
    if not line.strip():
        continue
    tokens = line.split()
    class_id = int(tokens[0])
    coordinates = [float(value) for value in tokens[1:]]
    assert class_id == 0, 'Unexpected class'
    assert len(coordinates) >= 6 and len(coordinates) % 2 == 0
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates)
    points = [(coordinates[i] * width, coordinates[i + 1] * height) for i in range(0, len(coordinates), 2)]
    draw.line(points + [points[0]], fill=colors[(number - 1) % len(colors)], width=2)
    polygons.append({'class_id': class_id, 'vertex_count': len(points)})

def font(size, bold=False):
    path = Path('C:/Windows/Fonts/segoeuib.ttf' if bold else 'C:/Windows/Fonts/segoeui.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()

margin, gutter, header, footer = 20, 24, 66, 92
canvas = Image.new('RGB', (2 * width + 2 * margin + gutter, header + height + footer), 'white')
canvas.paste(original, (margin, header))
canvas.paste(annotated, (margin + width + gutter, header))
view = ImageDraw.Draw(canvas)
view.text((margin, 12), 'Imagen original', fill='black', font=font(22, True))
view.text((margin + width + gutter, 12), 'Contornos del dataset', fill='black', font=font(22, True))
view.text((margin, header + height + 10), 'Ejemplo de entrenamiento: 416 x 416 pixeles.', fill='black', font=font(16))
view.text((margin, header + height + 34), '4 regiones anotadas de clase fisura. Los colores solo distinguen sus contornos.', fill='black', font=font(16))
view.text((margin, header + height + 58), 'Estas anotaciones no son predicciones de un modelo.', fill='black', font=font(16))
output = ROOT / 'results/recalculado_independiente/figuras_datos'
output.mkdir(parents=True, exist_ok=True)
canvas.save(output / '01_imagen_y_contornos.png')
report = {
    'partition': 'train',
    'image': image_path.relative_to(ROOT).as_posix(),
    'label': label_path.relative_to(ROOT).as_posix(),
    'width': width,
    'height': height,
    'annotated_regions': len(polygons),
    'polygons': polygons,
    'sample_coordinate_validation': 'OK: class 0, at least 3 vertices, finite normalized coordinates in [0,1]',
    'binary_mask_created': False,
    'whole_dataset_geometry_audited': False,
    'original_files_modified': False,
}
(output / '01_registro_ejemplo.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=True, indent=2))
print(output / '01_imagen_y_contornos.png')
