"""
Esquema de las tablas de la app en chutaDB.

    Person    (person_id, name, source)
    FaceImage (face_id, person_id, photo_id, image_path, model, embedding_dim, embedding)

Una fila FaceImage por (foto, modelo). El embedding se guarda en la propia fila
(columna varchar, ver codec.py): la tabla es la fuente de verdad y los índices
HNSW/IVF se derivan de ella.

Solo usa APIs existentes de StorageManager / SchemaCatalog.
"""
from __future__ import annotations

import logging

from storage.storage_manager import StorageManager

from face_recognition_app.codec import EMBEDDING_COLUMN_WIDTH

log = logging.getLogger(__name__)

PERSON_TABLE = "Person"
FACE_IMAGE_TABLE = "FaceImage"

PERSON_COLUMNS = ["person_id", "name", "source"]
PERSON_SCHEMA = ["integer", "varchar(100)", "varchar(10)"]

FACE_IMAGE_COLUMNS = [
    "face_id",
    "person_id",
    "photo_id",
    "image_path",
    "model",
    "embedding_dim",
    "embedding",
]
FACE_IMAGE_SCHEMA = [
    "integer",
    "integer",
    "integer",
    "varchar(255)",
    "varchar(20)",
    "integer",
    f"varchar({EMBEDDING_COLUMN_WIDTH})",
]

TABLES = {
    PERSON_TABLE: (PERSON_SCHEMA, PERSON_COLUMNS),
    FACE_IMAGE_TABLE: (FACE_IMAGE_SCHEMA, FACE_IMAGE_COLUMNS),
}


def _drop_catalog_embedding_indexes(storage: StorageManager) -> None:
    """
    Quita del catálogo los índices embedding_* registrados sobre FaceImage.

    IndexManager._load_embedding_index falla al abrir la tabla si el índice no
    tiene la misma cantidad de filas que la tabla. La app gestiona sus índices
    fuera del catálogo (vector_store.py), así que estos registros se eliminan
    antes de abrir FaceImage.
    """
    for meta in storage.catalog.get_table_indexes(FACE_IMAGE_TABLE):
        if str(meta["index_type"]).startswith("embedding_"):
            log.warning("Eliminando índice de catálogo obsoleto: %s", meta["index_name"])
            storage.catalog.drop_index_info(meta["index_name"])


def _schema_matches(storage: StorageManager, name: str) -> bool:
    info = storage.catalog.get_table_info(name)
    schema, columns = TABLES[name]
    return info["column_names"] == columns and info["schema"] == schema


def create_face_recognition_tables(storage: StorageManager) -> None:
    """
    Crea Person y FaceImage si no existen.

    Si existen con un esquema distinto (versión anterior de la app), se
    eliminan y se recrean vacías.
    """
    _drop_catalog_embedding_indexes(storage)

    for name, (schema, columns) in TABLES.items():
        if storage.catalog.get_table_info(name) is not None:
            if _schema_matches(storage, name):
                continue
            log.warning("La tabla %s tiene un esquema antiguo: se recrea vacía.", name)
            storage.drop_table(name)

        storage.create_table(
            name=name,
            schema=schema,
            file_type="heap",
            key_index=0,
            column_names=columns,
        )


def reset_face_recognition_tables(storage: StorageManager) -> None:
    """Elimina y recrea ambas tablas (vacías)."""
    _drop_catalog_embedding_indexes(storage)
    for name in TABLES:
        if storage.catalog.get_table_info(name) is not None:
            storage.drop_table(name)
    create_face_recognition_tables(storage)


def open_face_recognition_tables(storage: StorageManager):
    """Devuelve (person_table, face_image_table)."""
    return storage.open_table(PERSON_TABLE), storage.open_table(FACE_IMAGE_TABLE)
