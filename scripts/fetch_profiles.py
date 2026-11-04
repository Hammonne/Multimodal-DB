"""
Una foto de perfil por artista para mostrar en la interfaz.

Es independiente del filtro de caras: filter_artist_faces.py deja solo fotos
útiles para buscar por cara y borra el resto, incluida a veces la foto
principal de Deezer (cuando es un logo o una ilustración). Para la interfaz
basta con una imagen que represente al artista, así que se elige, en orden:

  1. deezer   la foto oficial del artista en Deezer (aunque sea un logo),
              salvo la silueta gris genérica que Deezer pone a quien no tiene
  2. foto     una foto que quedó tras el filtro de caras
  3. portada  la portada de su canción más popular del dataset

Se guarda como artists/<nombre>_<id>/profile.jpg. Ese archivo no está en
artist_images.csv, así que el filtro de caras nunca lo borra ni lo usa.

Salida: data/artist_profiles.csv con artist_id, name, profile_path, source.
Es reanudable: no vuelve a descargar los perfiles que ya existen.

Uso:
  python3 scripts/fetch_profiles.py
"""

import argparse
import hashlib
import os
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import fetch_music as fm

FIELDS = ["artist_id", "name", "profile_path", "source"]
# silueta gris que Deezer usa como foto genérica (con una URL que sí tiene hash)
GENERICAS = {"cf0b6a5247e606f67470140451774cb5"}


def es_generica(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest() in GENERICAS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--data", default=os.path.join(repo, "data"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    d = args.data

    artists = fm.load_csv(os.path.join(d, "artists.csv"))
    fotos = defaultdict(list)
    for r in fm.load_csv(os.path.join(d, "artist_images.csv")):
        fotos[r["artist_id"]].append(r)
    portada = {}
    songs = sorted(fm.load_csv(os.path.join(d, "songs.csv")),
                   key=lambda s: -int(s["deezer_rank"] or 0))
    for s in songs:
        if s["cover_path"]:
            portada.setdefault(s["artist_id"], s["cover_path"])
    previos = {r["artist_id"]: r for r in fm.load_csv(os.path.join(d, "artist_profiles.csv"))}

    def perfil(a):
        aid = a["artist_id"]
        rel = os.path.join("artists", fm.artist_dir(a["name"], aid), "profile.jpg")
        destino = os.path.join(d, rel)
        if aid in previos and os.path.exists(destino) and not es_generica(destino):
            return previos[aid]
        if os.path.exists(destino):
            os.remove(destino)
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        fila = {"artist_id": aid, "name": a["name"], "profile_path": rel}

        # 1. Deezer: main.jpg si sigue en disco, si no se vuelve a descargar
        main = os.path.join(d, "artists", fm.artist_dir(a["name"], aid), "main.jpg")
        if os.path.exists(main) and not es_generica(main):
            shutil.copyfile(main, destino)
            return dict(fila, source="deezer")
        if not os.path.exists(main):
            info = fm.http_get(f"https://api.deezer.com/artist/{aid}") or {}
            if fm.download(fm.real_image(info.get("picture_xl")), destino):
                if not es_generica(destino):
                    return dict(fila, source="deezer")
                os.remove(destino)
        # 2. una foto que pasó el filtro de caras
        for r in fotos.get(aid, []):
            if os.path.exists(os.path.join(d, r["path"])):
                shutil.copyfile(os.path.join(d, r["path"]), destino)
                return dict(fila, source="foto")
        # 3. portada de su canción más popular
        if aid in portada:
            shutil.copyfile(os.path.join(d, portada[aid]), destino)
            return dict(fila, source="portada")
        return dict(fila, profile_path="", source="")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        filas = list(pool.map(perfil, artists))
    fm.write_csv(os.path.join(d, "artist_profiles.csv"), FIELDS, filas)

    cuenta = defaultdict(int)
    for f in filas:
        cuenta[f["source"] or "sin perfil"] += 1
    print(f"{len(filas)} artistas: " + ", ".join(f"{k} {v}" for k, v in sorted(cuenta.items())))


if __name__ == "__main__":
    main()
