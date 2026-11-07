# Pesos entregados por el equipo

Se incluyen los checkpoints BEST y LAST originales de las seis corridas. manifest.json registra la arquitectura, semilla, ruta y SHA256 de cada archivo. BEST fue seleccionado mediante pérdida de validación con min_delta 0.001 y paciencia 7.

Descarga de los pesos https://github.com/nick5uni/mia203-segmentacion-fisuras/tree/main/weights

La demo utiliza A0 y A1 de la semilla de referencia 20261026, fijada en el protocolo del benchmark. El código comprueba las huellas y carga los parámetros con strict=True. Los archivos provienen de las entregas del equipo y contienen también estado de optimización y generadores.
