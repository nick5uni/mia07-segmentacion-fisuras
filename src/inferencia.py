"""Inferencia de los checkpoints de referencia usados en el benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models import EncoderDecoder, MiniUNet

ROOT = Path(__file__).resolve().parents[1]


def predict(image_path, model_name='A0', device='cpu'):
    entry = next(row for row in json.loads((ROOT / 'weights/manifest.json').read_text())
                 if row['model'] == model_name and row['seed'] == 20261026 and row['file'].endswith('best.pt'))
    checkpoint = ROOT / entry['file']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('La huella del checkpoint no coincide con el registro.')
    with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = {'A0': EncoderDecoder, 'A1': MiniUNet}[model_name]().to(device).eval()
    model.load_state_dict(state['model_state_dict'], strict=True)
    image = Image.open(image_path).convert('RGB').resize((416, 416), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    return image, probabilities, probabilities >= 0.5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagen', type=Path, default=ROOT / 'data/ejemplo/fotografia.jpg')
    parser.add_argument('--modelo', choices=['A0', 'A1'], default='A0')
    parser.add_argument('--salida', type=Path, default=ROOT / 'results/demo_mascara.png')
    args = parser.parse_args()
    torch.set_num_threads(4)
    _, _, predicted = predict(args.imagen, args.modelo)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(predicted.astype('uint8') * 255).save(args.salida)
    print('Máscara guardada en', args.salida)
