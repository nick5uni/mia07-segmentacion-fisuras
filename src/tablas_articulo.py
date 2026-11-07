"""Exporta tablas del artículo desde los registros y configuraciones del equipo."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def records(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def dump(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        csv.writer(handle).writerows(rows)


def export(independent, team, output):
    output.mkdir(parents=True, exist_ok=True)
    stats = json.loads((independent / 'estadistica.json').read_text(encoding='utf-8'))
    runs = json.loads((independent / 'metricas_corridas.json').read_text(encoding='utf-8'))
    hours = records(team / 'COSTO_entrenamiento.csv')
    baseline = next(r for r in records(team / 'TEST_metricas_por_corrida.csv') if r['modelo'] == 'Base')
    dims = [['Bloque', 'Salida', 'A0 / A1'], ['Entrada', '416² × 3', 'Igual'],
            ['Encoder 1', '416² × 16', 'Igual'], ['Encoder 2', '208² × 32', 'Igual'],
            ['Parte central', '104² × 64', 'Igual'], ['Decoder 1', '208² × 32', 'Entrada 64 / 96'],
            ['Decoder 2', '416² × 16', 'Entrada 32 / 48'], ['Salida', '416² × 1', 'Igual'],
            ['Parámetros', '', ' / '.join(str(next(r['parameters'] for r in runs if r['model'] == m)) for m in ('A0', 'A1'))]]
    dump(output / 'tabla_II_arquitectura.csv', dims)
    config = json.loads((ROOT / 'configs/A0.json').read_text(encoding='utf-8'))
    dump(output / 'tabla_III_protocolo.csv', [['Elemento', 'Configuración']] + [[k, v] for k, v in config.items() if k not in ['model', 'parameters']])
    sizes = [len(records(ROOT / f'data/listas/{split}.csv')) for split in ('train', 'val', 'test')]
    original = [3717, 200, 112]
    dump(output / 'tabla_I_datos.csv', [['Partición', 'Original', 'Preparado']] + list(zip(['Entrenamiento', 'Validación', 'Prueba'], original, sizes)) + [['Total', sum(original), sum(sizes)]])
    metrics = [('Dice global', 'dice', 'dice'), ('IoU global', 'iou', 'iou'),
               ('Precisión', 'precision', 'precision'), ('Sensibilidad', 'recall', 'recall'),
               ('Dice por imagen', 'dice_macro', 'dice_medio_img')]
    table = [['Métrica', 'A0', 'A1', 'Línea base']]
    for label, key, base_key in metrics:
        values = [stats['models'][m][key] for m in ('A0', 'A1')]
        table.append([label] + [f"{v['mean']:.4f} ± {v['sd']:.4f}" for v in values] + [f'{float(baseline[base_key]):.4f}'])
    dump(output / 'tabla_IV_test.csv', table)
    dump(output / 'tabla_V_semillas.csv', [['Modelo', 'Semilla', 'Dice', 'IoU']] + [[r['model'], r['seed'], f"{r['dice']:.4f}", f"{r['iou']:.4f}"] for r in runs])
    dump(output / 'tabla_VI_area.csv', [['Grupo', 'n', 'A0', 'A1']] + [[r['group'], r['n'], f"{r['A0_dice_macro']:.4f}", f"{r['A1_dice_macro']:.4f}"] for r in stats['error_groups']])
    dump(output / 'tabla_VII_gpu.csv', [['Modelo', 'Semilla', 'GPU', 'Épocas', 'Minutos']] + [[r['modelo'], r['seed'], r['gpu'], r['epochs'], f"{60*float(r['horas_gpu']):.2f}"] for r in sorted(hours, key=lambda row:(row['modelo'],row['seed']))])
    print(f'Horas de GPU registradas {sum(float(r["horas_gpu"]) for r in hours):.3f}')
    print('Tablas exportadas a', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--independiente', type=Path, default=ROOT / 'results/independiente')
    parser.add_argument('--equipo', type=Path, default=ROOT / 'results/equipo')
    parser.add_argument('--salida', type=Path, default=ROOT / 'results/tablas_articulo')
    args = parser.parse_args()
    export(args.independiente, args.equipo, args.salida)
