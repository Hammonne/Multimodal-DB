# Manual de Usuario

¡Hola! Para poder usar la aplicación 1 (RAG), deben seguir estos pasos:

**SOLO SI LA CARPETA /data/rag/ ESTA VACIA**
1. Desde la raiz del proyecto (Multimodal-DB), ejecutar:
```bash
python3 scripts/import_pwc_catalog.py
```
Esto descargará el catálogo de datos en ``data/rag/``.

2. Desde la raíz del proyecto (Multimodal-DB), ejecutar
```bash
python3 scripts/download_papers.py --limit 50
```
Esto descargará los papers en /data/papers/, de la misma forma, se guardará un manifiesto de descargas en /data/rag/. Tengan en cuenta que la descarga toma una cantidad considerable de tiempo.

3. Después de generar `data/rag/chunks.jsonl` con la ingesta, construir los
índices desde la pestaña **RAG** de la aplicación con **CONSTRUIR ÍNDICES**.
El backend debe ejecutarse con Python 3.12 (PyTorch no está disponible para
Python 3.13 en este entorno).
Los embeddings se generan en lotes de 32 y se reutilizan automáticamente si el
corpus no cambió. Puedes ajustar el lote con `RAG_EMBEDDING_BATCH_SIZE`.
También se puede ejecutar desde la raíz:
```bash
curl -X POST http://localhost:8000/api/rag/build \
  -H 'Content-Type: application/json' \
  -d '{"text": true, "vector": true}'
```
