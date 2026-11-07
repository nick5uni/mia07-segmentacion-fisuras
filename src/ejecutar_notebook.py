"""Ejecuta el código original con rutas portables y guarda nuevas salidas."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    training = args.tarea == 'entrenar'
    name = 'NB04a_Semillas.ipynb' if training else 'NB04b_Benchmark.ipynb'
    notebook = json.loads((ROOT / 'notebooks' / name).read_text(encoding='utf-8'))
    try:
        from IPython.display import display
    except ImportError:
        display = print
    import matplotlib
    matplotlib.use('Agg')
    context = {'__name__': '__main__', 'display': display}
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source']).replace('from IPython.display import display', '')
        if index == 2:
            if training:
                source = source.replace('MODELO = "A1"', f'MODELO = "{args.modelo}"')
                source = source.replace('SEED = 20261027', f'SEED = {args.semilla}')
                source = source.replace('assert SEED != 20261026, "La semilla 20261026 ya está entrenada en NB02/NB03."', '')
            start = source.index('EN_COLAB = ')
            end = source.index('ZIP_DATOS = ', start)
            training_folder = 'prueba_entrenamiento' if args.prueba else 'reentrenamiento'
            temporary_name = f'checkpoint_temporal_{args.modelo}_{args.semilla}_{training_folder}.pt'
            source = source[:start] + f'''EN_COLAB = False
DRIVE_BASE = Path({str(ROOT)!r})
DATOS_LOCALES = DRIVE_BASE / 'data/completo'
TEMPORAL_LOCAL = DRIVE_BASE / 'results' / {temporary_name!r}
''' + source[end:]
            source = source.replace('RES = DRIVE_BASE / "resultados"', 'RES = DRIVE_BASE / "weights"')
            source = source.replace('SALIDA = RES / "NB04_Benchmark"', 'SALIDA = DRIVE_BASE / "results/benchmark_recalculado"')
            source = source.replace('DRIVE_BASE / "resultados" / "NB04_Semillas"', f'DRIVE_BASE / "results/{training_folder}"')
        if args.prueba and index == 2 and training:
            source = source.replace('PRUEBA_LOCAL = False', 'PRUEBA_LOCAL = True')
        print(f'Ejecutando {name}, celda {index}', flush=True)
        exec(compile(source, name + f':cell{index}', 'exec'), context)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tarea', choices=['entrenar', 'benchmark'])
    parser.add_argument('--modelo', choices=['A0', 'A1'], default='A0')
    parser.add_argument('--semilla', type=int, default=20261027)
    parser.add_argument('--prueba', action='store_true', help='Dos épocas y tres lotes por época para comprobar el entrenamiento en CPU.')
    run(parser.parse_args())
