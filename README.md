# Segmentación de fisuras mediante arquitecturas Encoder Decoder

Proyecto final del curso MIA-07 de la Maestría en Inteligencia Artificial, FIIS, Universidad Nacional de Ingeniería. El equipo compara un Encoder-Decoder sin conexiones de salto (A0) con una Mini U-Net con dos conexiones de salto (A1) y una línea base de umbralización adaptativa.

| Método | Dice global en 111 imágenes de prueba |
|---|---:|
| Línea base clásica | 0,5206 |
| A0 | 0,7036 ± 0,0026 |
| A1 | 0,7079 ± 0,0046 |

Los valores de las redes son media y desviación estándar muestral de tres semillas. El intervalo exploratorio del v9 incluye cero. La comparación mide coincidencia con la máscara de referencia de Crack-Seg. Las salidas sirven para apoyar la revisión visual de fisuras.

[Abrir la demo en Colab](https://colab.research.google.com/github/nick5uni/mia203-segmentacion-fisuras/blob/main/notebooks/NB05_Demo_Colab.ipynb) · [Pesos y huellas SHA256](weights/) · [Contribuciones y uso de IA](CONTRIBUCIONES.md)

## Instalación y datos

Ejecutar desde la raíz del repositorio con Python 3.12. Para reproducir la evaluación independiente en CPU, instalar PyTorch 2.15.0 desde su índice de ruedas CPU. Para entrenar o medir en GPU, usar una instalación de PyTorch con CUDA. El benchmark histórico de Colab registra Python 3.13.16 y PyTorch 2.11.0+cu130 sobre Tesla T4.

```bash
git clone https://github.com/nick5uni/mia203-segmentacion-fisuras.git
cd mia203-segmentacion-fisuras
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python src/preparar_datos.py
```

El preparador descarga el ZIP de Crack-Seg, comprueba su SHA256 y reconstruye las 4.027 máscaras y las listas de 3.717/199/111 imágenes. También crea la estructura que lee el código original. Si el ZIP ya está descargado, usar `python src/preparar_datos.py --zip /ruta/al/crack-seg.zip`. La fuente y la licencia declarada se detallan en [data](data/README.md).

## Inferencia y entrenamiento

El notebook NB05 carga los checkpoints BEST originales de la semilla de referencia 20261026. Se puede ejecutar con CPU o GPU. Muestra la máscara binaria y su superposición sobre la imagen. El comando equivalente para guardar una máscara es:

```bash
python src/inferencia.py --modelo A0
python src/inferencia.py --modelo A1 --imagen /ruta/a/una_foto.jpg --salida results/prediccion_a1.png
```

Los notebooks NB02 y NB03 documentan las ejecuciones iniciales. NB04a contiene el entrenamiento de las semillas adicionales y permite reanudar a partir de LAST. El ejecutor usa el mismo entrenamiento con rutas locales, guarda sus nuevas salidas en `results/reentrenamiento/` y conserva los pesos históricos de `weights/`. Con los datos preparados y una GPU habilitada, ejecutar:

```bash
python src/ejecutar_notebook.py entrenar --modelo A0 --semilla 20261026
python src/ejecutar_notebook.py entrenar --modelo A0 --semilla 20261027
python src/ejecutar_notebook.py entrenar --modelo A0 --semilla 20261028
python src/ejecutar_notebook.py entrenar --modelo A1 --semilla 20261026
python src/ejecutar_notebook.py entrenar --modelo A1 --semilla 20261027
python src/ejecutar_notebook.py entrenar --modelo A1 --semilla 20261028
```

Las configuraciones están en `configs/`. PyTorch puede producir pequeñas diferencias entre dispositivos o sesiones. Los checkpoints históricos permiten verificar el artículo con los pesos utilizados por el equipo.

## Evaluación y reproducción del artículo

```bash
python src/evaluar_etapa4_independiente.py
python src/analizar_etapa4.py
python src/ejecutar_notebook.py benchmark
python src/tablas_articulo.py --independiente results/recalculado_independiente --equipo results/benchmark_recalculado
python src/revisar_primer_ejemplo.py
python src/convertir_primera_mascara.py
python src/figura_y_benchmark_etapa4.py
```

| Elemento del artículo | Procedencia |
|---|---|
| Tabla I | Listas auditadas de `data/listas/` y `src/tablas_articulo.py`. |
| Tablas II y III | Arquitecturas de `src/models.py`, configuraciones y `src/tablas_articulo.py`. |
| Tablas IV y V | Evaluación independiente de seis checkpoints. La línea base de la tabla IV viene de NB04b. |
| Tabla VI | Agrupación por área anotada de `src/analizar_etapa4.py`. |
| Tabla VII | Registro histórico de entrenamiento en `results/equipo/COSTO_entrenamiento.csv`. |
| Figuras 1 y 2 | `src/revisar_primer_ejemplo.py` y `src/convertir_primera_mascara.py`. |
| Figura 3 | `src/figura_y_benchmark_etapa4.py`. |
| Figuras 4 a 8 | `src/analizar_etapa4.py`, que usa las predicciones y conteos de la evaluación independiente. |
| Figura 9 | Visualización de activaciones en NB04b. La versión maquetada del artículo figura en `results/articulo_figuras/`. |

`results/equipo/` conserva los CSV, JSON y figuras de NB04b. El ejecutor de NB04b escribe una nueva copia en `results/benchmark_recalculado/`. Los tiempos de inferencia se miden en el hardware disponible y pueden variar. La cifra de 5,173 horas proviene de los registros históricos de épocas de las seis corridas.

`results/independiente/` conserva los conteos y estadísticas que respaldan el v9. La regeneración queda en `results/recalculado_independiente/`. Para el intervalo de confianza, el artículo calcula la diferencia media de Dice entre semillas y aplica un bootstrap pareado de semillas e imágenes con 3.000 remuestreos. NB04b también calcula un intervalo a partir de los conteos agrupados y remuestrea imágenes. Por eso los intervalos corresponden a métodos distintos. Para regenerar las tablas desde los registros históricos, ejecutar `python src/tablas_articulo.py`.

El equipo realizó un trabajo secuencial con entregas en Drive. GitHub se incorporó al cierre para reunir el código, los pesos y las instrucciones de reproducción. La autoría por etapa se documenta en [Contribuciones](CONTRIBUCIONES.md).

## Estado de NB04a y NB04b

`results/independiente/VALIDACION_RESULTADOS.json` conserva la nota de la revisión independiente inicial, cuando aún no se habían recibido NB04a y NB04b. Los dos notebooks se incorporaron después y están disponibles en [`notebooks/NB04a_Semillas.ipynb`](notebooks/NB04a_Semillas.ipynb) y [`notebooks/NB04b_Benchmark.ipynb`](notebooks/NB04b_Benchmark.ipynb). El JSON documenta el alcance de aquella revisión. Sus conteos y métricas corresponden a la evaluación independiente de los seis checkpoints.
