"""Descarga verificada, conversión de polígonos y preparación del lector original."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
from zipfile import ZipFile

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip'
ZIP_SHA256 = '814b566d61e890024d15f22a3e6b14e252f781ab83f60a8d59e6684817efecdc'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(zip_path):
    if not zip_path.is_file():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = zip_path.with_suffix('.download')
        urllib.request.urlretrieve(URL, temporary)
        if sha256(temporary) != ZIP_SHA256:
            raise ValueError('La descarga no coincide con la versión auditada de Crack-Seg.')
        temporary.replace(zip_path)
    if sha256(zip_path) != ZIP_SHA256:
        raise ValueError('El ZIP no coincide con la versión auditada.')
    records = list(csv.DictReader((ROOT / 'data/listas/todas.csv').open(encoding='utf-8-sig')))
    destination = ROOT / 'data/reproducido'
    with ZipFile(zip_path) as archive:
        members = {name.replace('\\', '/'): name for name in archive.namelist()}
        for number, row in enumerate(records, 1):
            split, image_id = row['split'], row['image_id']
            for key, category in [('image_path', 'images'), ('original_label_path', 'labels')]:
                suffix = '.jpg' if category == 'images' else '.txt'
                relative = f'{category}/{split}/{image_id}{suffix}'
                matches = [name for name in members if name.endswith(relative)]
                if len(matches) != 1:
                    raise ValueError(f'Archivo de origen ambiguo o ausente {relative}')
                target = destination / row[key]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(members[matches[0]]))
            photo = destination / row['image_path']
            if sha256(photo) != row['image_sha256']:
                raise ValueError(f'Fotografía distinta {image_id}')
            mask = Image.new('L', (int(row['width']), int(row['height'])), 0)
            draw = ImageDraw.Draw(mask)
            for line in (destination / row['original_label_path']).read_text(encoding='utf-8-sig').splitlines():
                tokens = line.split()
                if not tokens:
                    continue
                if tokens[0] != '0':
                    raise ValueError('Clase distinta de fisura.')
                xy = [float(value) for value in tokens[1:]]
                points = [(xy[i] * mask.width, xy[i + 1] * mask.height) for i in range(0, len(xy), 2)]
                draw.polygon(points, fill=1)
            output = destination / row['mask_path']
            output.parent.mkdir(parents=True, exist_ok=True)
            mask.save(output)
            if sha256(output) != row['mask_sha256']:
                raise ValueError(f'Máscara distinta {image_id}. Revisar versión de Pillow.')
            if number % 500 == 0:
                print(f'{number}/{len(records)} parejas verificadas', flush=True)
    portable = ROOT / 'data/completo/Entrega_Victor_Datos_Completos_v1'
    portable.mkdir(parents=True, exist_ok=True)
    (portable / 'listas').mkdir(exist_ok=True)
    by_id = {row['image_id']: row for row in records}
    for manifest in (ROOT / 'data/listas_portables').glob('*.csv'):
        shutil.copy2(manifest, portable / 'listas' / manifest.name)
        if manifest.stem not in ['train', 'val', 'test']:
            continue
        for row in csv.DictReader(manifest.open(encoding='utf-8-sig')):
            original = by_id[row['image_id']]
            for key in ['image_path', 'mask_path', 'original_label_path']:
                target = portable / row[key]
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    try:
                        os.link(destination / original[key], target)
                    except OSError:
                        shutil.copy2(destination / original[key], target)
    for name in ['lector_datos.py', 'lector_pytorch.py']:
        shutil.copy2(ROOT / 'src' / name, portable / name)
    summary = {'pairs_checked': len(records), 'train': 3717, 'val': 199, 'test': 111, 'source_sha256': ZIP_SHA256}
    (ROOT / 'data/verificacion_preparacion.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', type=Path, default=ROOT / 'data/raw/crack-seg.zip')
    prepare(parser.parse_args().zip.resolve())

