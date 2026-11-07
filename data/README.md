# Datos Crack Seg

Fuente https://docs.ultralytics.com/datasets/segment/crack-seg/

Proyecto de origen https://universe.roboflow.com/university-bswxt/crack-bphdr

La fuente declara dominio público con CC0 1.0. La descarga original se verifica con SHA256 814b566d61e890024d15f22a3e6b14e252f781ab83f60a8d59e6684817efecdc.

Los CSV de listas conservan las particiones de 3717/199/111 imágenes. Se excluyen una anotación vacía de validación y una copia exacta dentro de prueba. La conversión rellena los polígonos originales en máscaras de 0 y 1, sin corrección manual. Los aumentos previos del proveedor permanecen en entrenamiento.

Ejecutar desde la raíz `python src/preparar_datos.py`. Se descargan y comprueban los archivos, se generan las máscaras y se comprueban sus huellas contra las listas publicadas.
