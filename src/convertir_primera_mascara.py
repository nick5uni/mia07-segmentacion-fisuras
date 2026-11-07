"""Convert only the first reviewed training example to an unchanged-size binary mask."""
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/recalculado_independiente/figuras_datos'
record_path = OUTPUT / '01_registro_ejemplo.json'
record = json.loads(record_path.read_text(encoding='utf-8'))
image = Image.open(ROOT / record['image']).convert('RGB')
width, height = image.size
mask = Image.new('L', image.size, 0)
draw = ImageDraw.Draw(mask)
for line in (ROOT / record['label']).read_text(encoding='utf-8-sig').splitlines():
    if not line.strip():
        continue
    tokens = line.split()
    coordinates = [float(value) for value in tokens[1:]]
    assert int(tokens[0]) == 0
    assert len(coordinates) >= 6 and len(coordinates) % 2 == 0
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates)
    points = [(coordinates[i] * width, coordinates[i + 1] * height) for i in range(0, len(coordinates), 2)]
    draw.polygon(points, fill=1)
mask_path = OUTPUT / '01_mascara_binaria_01.png'
mask.save(mask_path)
reopened = Image.open(mask_path)
assert reopened.size == image.size
values = sorted({value for count, value in reopened.getcolors()})
assert values == [0, 1]
positive_pixels = reopened.histogram()[1]
assert 0 < positive_pixels < width * height
display = mask.point(lambda value: value * 255)
display.save(OUTPUT / '01_mascara_visual_255.png')

overlay = image.convert('RGBA')
red = Image.new('RGBA', image.size, (255, 0, 0, 0))
red.putalpha(mask.point(lambda value: value * 100))
overlay = Image.alpha_composite(overlay, red).convert('RGB')
margin, gutter, top, bottom = 20, 20, 64, 94
preview = Image.new('RGB', (3 * width + 2 * gutter + 2 * margin, height + top + bottom), 'white')
font_path = Path('C:/Windows/Fonts/segoeui.ttf')
bold_path = Path('C:/Windows/Fonts/segoeuib.ttf')
font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
title_font = ImageFont.truetype(str(bold_path), 23) if bold_path.exists() else ImageFont.load_default()
text = ImageDraw.Draw(preview)
for column, (picture, title) in enumerate([(image, 'Foto original'), (display, 'Mascara de referencia'), (overlay, 'Comprobacion sobre la foto')]):
    x = margin + column * (width + gutter)
    preview.paste(picture, (x, top))
    text.text((x, 15), title, fill='black', font=title_font)
text.text((margin, top + height + 12), 'Blanco: union de las regiones anotadas como fisura. Negro: fondo. Rojo: mascara superpuesta sobre la foto.', fill='black', font=font)
text.text((margin, top + height + 39), 'Archivo para entrenar: valores 0 y 1. Vista blanca y negra: valores 0 y 255 solo para mostrarla.', fill='black', font=font)
text.text((margin, top + height + 66), 'Ejemplo de entrenamiento; se conserva la resolucion original de 416 x 416. No es una prediccion.', fill='black', font=font)
preview_path = OUTPUT / '02_foto_mascara_superposicion.png'
preview.save(preview_path)

record['binary_mask_created'] = True
record['binary_mask_file'] = mask_path.relative_to(ROOT).as_posix()
record['binary_mask_values'] = values
record['positive_pixels'] = positive_pixels
record['positive_pixel_fraction'] = positive_pixels / (width * height)
record['resizing_applied'] = False
record['mask_geometry'] = 'Union of original polygons rasterized at original resolution; x=normalized_x*width and y=normalized_y*height.'
record['mask_visual_review_status'] = 'Preview generated; assess against original polygons, not as a physical measurement.'
record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'mask_values': values, 'size': image.size, 'positive_pixels': positive_pixels, 'positive_fraction': positive_pixels / (width * height), 'preview': str(preview_path)}, indent=2))
