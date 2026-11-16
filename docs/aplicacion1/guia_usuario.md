# Manual de Usuario

¡Hola! Para poder usar la aplicación 1 (RAG), deben seguir estos pasos:

**SOLO SI LA CARPETA /data/rag/ ESTA VACIA**
1. Desde la raiz del proyecto (Multimodal-DB), ejecutar:
```bash
python3 scripts/import_pwc_catalog.py
```
Esto descargará el catálogo de datos en ``data/rag/``.

1. Nuevamente (o no), desde la raíz del proyecto (Multimodal-DB), ejecutar
```bash
python3 scripts/download_scripts.py
```
Esto descargará los papers en /data/papers/, de la misma forma, se guardará un manifiesto de descargas en /data/rag/. Tengan en cuenta que la descarga toma una cantidad considerable de tiempo.
