# carga de un dataset propio desde la consola: un .zip con imágenes o audios (y opcionalmente un csv)
# se descomprime en data/uploads/<nombre>/ y se deja un csv listo para CREATE TABLE + IMPORT INTO
# - zip con solo imágenes o solo audios -> tabla (id, nombre, imagen IMAGE) o (id, nombre, audio AUDIO)
# - zip con un csv + archivos -> la tabla del csv; las columnas que nombran imágenes o audios pasan a IMAGE / AUDIO
from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path

IMAGEN = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}


def tipo_media(valor: str) -> str | None:
    ext = Path(str(valor).strip()).suffix.lower()
    return "IMAGE" if ext in IMAGEN else "AUDIO" if ext in AUDIO else None


def _nombre_seguro(texto: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", texto).strip("_").lower() or "dataset"


def preparar_zip(contenido: bytes, nombre_zip: str, data: Path) -> tuple[str, str]:
    # devuelve (texto del csv a importar, ruta del csv relativa a la raíz del repo)
    nombre = _nombre_seguro(Path(nombre_zip).stem)
    destino = data / "uploads" / nombre
    destino.mkdir(parents=True, exist_ok=True)
    try:
        zf = zipfile.ZipFile(io.BytesIO(contenido))
    except zipfile.BadZipFile as error:
        raise ValueError("el .zip está dañado o no es un zip") from error
    archivos = []
    for info in zf.infolist():
        partes = Path(info.filename).parts
        # nada fuera de la carpeta destino (rutas absolutas o con ..) ni basura de macOS
        if info.is_dir() or Path(info.filename).is_absolute() or ".." in partes or "__MACOSX" in partes \
                or Path(info.filename).name.startswith("."):
            continue
        ruta = destino / Path(*partes)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_bytes(zf.read(info))
        archivos.append(Path(*partes))
    if not archivos:
        raise ValueError("el .zip no tiene archivos")

    csvs = sorted((a for a in archivos if a.suffix.lower() == ".csv"), key=lambda a: len(a.parts))
    rel = lambda a: f"uploads/{nombre}/{a.as_posix()}"   # ruta como la guarda la tabla (relativa a data/)
    if csvs:
        # la tabla del csv: cada valor que nombre un archivo del zip se reemplaza por su ruta dentro de data/
        por_nombre = {}
        for a in archivos:
            por_nombre.setdefault(a.as_posix(), a)
            por_nombre.setdefault(a.name, a)
        texto = (destino / csvs[0]).read_text(encoding="utf-8-sig")
        lector = csv.DictReader(io.StringIO(texto))
        if not lector.fieldnames:
            raise ValueError("el csv del zip no tiene encabezado")
        filas = []
        base_csv = csvs[0].parent
        for fila in lector:
            nueva = {}
            for col, valor in fila.items():
                valor = (valor or "").strip()
                if tipo_media(valor):
                    candidato = por_nombre.get((base_csv / valor).as_posix()) or por_nombre.get(valor) \
                        or por_nombre.get(Path(valor).name)
                    if candidato is None:
                        raise ValueError(f"el csv nombra '{valor}' pero ese archivo no está en el zip")
                    valor = rel(candidato)
                nueva[col] = valor
            filas.append(nueva)
        campos = list(lector.fieldnames)
    else:
        # solo archivos: todos imágenes o todos audios
        medios = [a for a in archivos if tipo_media(a.name)]
        tipos = {tipo_media(a.name) for a in medios}
        if not medios:
            raise ValueError("el .zip no tiene imágenes ni audios (ni un csv)")
        if len(tipos) > 1:
            raise ValueError("el .zip mezcla imágenes y audios: sube solo un tipo, o agrega un csv que los describa")
        columna = "imagen" if tipos == {"IMAGE"} else "audio"
        campos = ["id", "nombre", columna]
        filas = [{"id": str(i), "nombre": a.stem, columna: rel(a)} for i, a in enumerate(sorted(medios), start=1)]

    salida = destino / f"{nombre}.csv"
    with salida.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    return salida.read_text(encoding="utf-8"), f"data/uploads/{nombre}/{nombre}.csv"
