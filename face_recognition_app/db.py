"""
Acceso a chutaDB para la app de reconocimiento facial.

FaceRecognitionDB recibe el StorageManager existente (el de web/backend/main.py)
y no crea otro. Responsabilidades:

- Person / FaceImage: altas, bajas y lecturas.
- Guardar el embedding en FaceImage.embedding (la tabla es la fuente de verdad).
- Resolver RID -> FaceImage -> Person rápido, con mapas en memoria construidos
  con un único scan al arrancar (Table.search_by_key en un heap es lineal).

Los IDs los asigna esta clase (max + 1) y su unicidad la garantizan los mapas en
memoria; por eso se desactiva en estas dos tablas la verificación de PK del
heap, que recorre toda la tabla en cada insert.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

import numpy as np

from storage.rid import RID
from storage.storage_manager import StorageManager

from face_recognition_app import codec
from face_recognition_app.db_schema import open_face_recognition_tables


def _clean(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    return str(value).replace("\x00", "").strip()


def rid_to_dict(rid: RID) -> dict[str, int]:
    return {"page_id": int(rid.page_id), "slot_id": int(rid.slot_id)}


@dataclass(frozen=True)
class Person:
    person_id: int
    name: str
    source: str
    rid: RID

    def to_dict(self) -> dict:
        return {"person_id": self.person_id, "name": self.name, "source": self.source}


@dataclass(frozen=True)
class FaceRecord:
    face_id: int
    person_id: int
    photo_id: int
    image_path: str
    model: str
    embedding_dim: int
    rid: RID

    def to_dict(self) -> dict:
        data = asdict(self)
        data["rid"] = rid_to_dict(self.rid)
        return data


class FaceRecognitionDB:
    def __init__(self, storage_manager: StorageManager):
        self.storage = storage_manager
        self.person_table, self.face_image_table = open_face_recognition_tables(storage_manager)
        self.index_manager = storage_manager.index_manager

        # La app asigna IDs únicos; evita el scan O(n) por insert del heap.
        for table in (self.person_table, self.face_image_table):
            table.constraints_manager.check_primary_key = False

        self._lock = threading.RLock()
        self._persons: dict[int, Person] = {}
        self._faces: dict[int, FaceRecord] = {}
        self._face_by_rid: dict[RID, int] = {}
        self._max_photo_id = 0
        # Versión por modelo: cambia en cada alta/baja de FaceImage. Permite a
        # VectorStore detectar si su copia en memoria quedó desactualizada.
        self._versions: dict[str, int] = {}
        self._load_maps()

    # ------------------------------------------------------------------
    # Mapas en memoria
    # ------------------------------------------------------------------

    def _load_maps(self) -> None:
        self._persons.clear()
        self._faces.clear()
        self._face_by_rid.clear()
        self._max_photo_id = 0

        for rid, row in self.person_table.scan():
            person = Person(int(row[0]), _clean(row[1]), _clean(row[2]), rid)
            self._persons[person.person_id] = person

        for rid, row in self.face_image_table.scan():
            face = self._face_from_row(rid, row)
            self._faces[face.face_id] = face
            self._face_by_rid[rid] = face.face_id
            self._max_photo_id = max(self._max_photo_id, face.photo_id)

    @staticmethod
    def _face_from_row(rid: RID, row) -> FaceRecord:
        return FaceRecord(
            face_id=int(row[0]),
            person_id=int(row[1]),
            photo_id=int(row[2]),
            image_path=_clean(row[3]),
            model=_clean(row[4]),
            embedding_dim=int(row[5]),
            rid=rid,
        )

    # ------------------------------------------------------------------
    # Person
    # ------------------------------------------------------------------

    def create_person(self, name: str, source: str = "own") -> Person:
        name = name.strip()
        if not name:
            raise ValueError("El nombre no puede estar vacío.")
        if len(name.encode("utf-8")) > 100:
            raise ValueError("El nombre es demasiado largo (máx. 100 bytes).")
        if source not in ("own", "lfw"):
            raise ValueError("source debe ser 'own' o 'lfw'.")

        with self._lock:
            person_id = max(self._persons, default=0) + 1
            rid = self.person_table.insert([person_id, name, source])
            person = Person(person_id, name, source, rid)
            self._persons[person_id] = person
            return person

    def get_person(self, person_id: int) -> Person | None:
        return self._persons.get(int(person_id))

    def list_persons(self, query: str = "", limit: int | None = None) -> list[Person]:
        q = query.strip().lower()
        persons = sorted(self._persons.values(), key=lambda p: p.name.lower())
        if q:
            persons = [p for p in persons if q in p.name.lower()]
        return persons[:limit] if limit else persons

    def count_persons(self) -> int:
        return len(self._persons)

    def delete_person(self, person_id: int) -> list[FaceRecord]:
        """Borra la persona y todas sus FaceImage. Devuelve las FaceImage borradas."""
        with self._lock:
            person = self._persons.get(int(person_id))
            if person is None:
                raise KeyError(f"No existe la persona con person_id={person_id}.")
            removed = self.faces_of_person(person.person_id)
            for face in removed:
                self.face_image_table.delete(face.rid)
                del self._faces[face.face_id]
                del self._face_by_rid[face.rid]
                self._bump(face.model)
            self.person_table.delete(person.rid)
            del self._persons[person.person_id]
            return removed

    # ------------------------------------------------------------------
    # FaceImage
    # ------------------------------------------------------------------

    def next_photo_id(self) -> int:
        with self._lock:
            self._max_photo_id += 1
            return self._max_photo_id

    def add_face(
        self,
        person_id: int,
        photo_id: int,
        image_path: str,
        model: str,
        embedding,
    ) -> FaceRecord:
        """Inserta una fila FaceImage con su embedding (normalizado)."""
        vector = codec.normalize(embedding)
        if len(image_path.encode("utf-8")) > 255:
            raise ValueError("image_path es demasiado largo (máx. 255 bytes).")
        if not model or len(model) > 20:
            raise ValueError("Nombre de modelo inválido.")

        with self._lock:
            if int(person_id) not in self._persons:
                raise ValueError(f"No existe la persona con person_id={person_id}.")
            face_id = max(self._faces, default=0) + 1
            values = [
                face_id,
                int(person_id),
                int(photo_id),
                image_path,
                model,
                int(vector.size),
                codec.encode(vector),
            ]
            rid = self.face_image_table.insert(values)
            face = FaceRecord(face_id, int(person_id), int(photo_id), image_path,
                              model, int(vector.size), rid)
            self._faces[face_id] = face
            self._face_by_rid[rid] = face_id
            self._max_photo_id = max(self._max_photo_id, int(photo_id))
            self._bump(model)
            return face

    def flush(self) -> None:
        """
        Escribe en disco las páginas modificadas de Person y FaceImage.

        chutaDB mantiene las escrituras en su BufferManager y solo las baja a
        disco al cerrar; las escrituras fuera de una transacción tampoco van al
        WAL. Si el backend termina sin cerrar (terminal cerrada, proceso
        matado), lo registrado se perdería. La app llama a flush() al terminar
        cada alta/baja para que quede persistido de inmediato.
        """
        from storage.buffer_manager import BufferManager

        buffer_manager = BufferManager.get_instance()
        with self._lock:
            for table in (self.person_table, self.face_image_table):
                buffer_manager.flush_file(table.data_file.file_manager)

    def _bump(self, model: str) -> None:
        self._versions[model] = self._versions.get(model, 0) + 1

    def version(self, model: str) -> int:
        return self._versions.get(model, 0)

    def get_face(self, face_id: int) -> FaceRecord | None:
        return self._faces.get(int(face_id))

    def face_by_rid(self, rid: RID) -> FaceRecord | None:
        face_id = self._face_by_rid.get(rid)
        return None if face_id is None else self._faces[face_id]

    def faces_of_person(self, person_id: int, model: str | None = None) -> list[FaceRecord]:
        return sorted(
            (f for f in self._faces.values()
             if f.person_id == int(person_id) and (model is None or f.model == model)),
            key=lambda f: f.face_id,
        )

    def faces_of_photo(self, photo_id: int) -> list[FaceRecord]:
        return [f for f in self._faces.values() if f.photo_id == int(photo_id)]

    def count_faces(self, model: str | None = None) -> int:
        if model is None:
            return len(self._faces)
        return sum(1 for f in self._faces.values() if f.model == model)

    def models_in_use(self) -> list[str]:
        return sorted({f.model for f in self._faces.values()})

    def load_embeddings(self, model: str) -> tuple[list[FaceRecord], np.ndarray]:
        """
        Lee de la tabla todos los embeddings de un modelo (scan secuencial).

        Devuelve (faces, X) con X de forma (N, D), alineado con faces.
        """
        faces: list[FaceRecord] = []
        vectors: list[np.ndarray] = []
        with self._lock:
            for rid, row in self.face_image_table.scan():
                if _clean(row[4]) != model:
                    continue
                face = self._faces.get(int(row[0])) or self._face_from_row(rid, row)
                faces.append(face)
                vectors.append(codec.decode(row[6], face.embedding_dim))

        if not vectors:
            return [], np.zeros((0, 0), dtype=codec.DTYPE)

        dims = {v.size for v in vectors}
        if len(dims) != 1:
            raise ValueError(f"El modelo '{model}' tiene embeddings de dimensiones distintas: {dims}")
        return faces, np.vstack(vectors).astype(codec.DTYPE)

    def get_embedding(self, face_id: int) -> np.ndarray:
        face = self._faces.get(int(face_id))
        if face is None:
            raise KeyError(f"No existe FaceImage con face_id={face_id}.")
        row = self.face_image_table.get(face.rid)
        return codec.decode(row[6], face.embedding_dim)
