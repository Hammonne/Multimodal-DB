# portadas extra para la parte 4: otros álbumes de los artistas que ya están, sin audio ni letra
# deja una sola por álbum: salta ediciones (deluxe, remastered...) y archivos idénticos
# uso: python3 scripts/fetch_covers.py --total 20000
import argparse
import csv
import hashlib
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from fetch_music import download, http_get, load_csv, norm, real_image, write_csv

CAMPOS = ["album_id", "artist_id", "album", "record_type", "release_date", "path", "md5",
          "origen", "duplicada_de"]
EDICION = re.compile(r"\b(deluxe|remaster(ed)?|expanded|anniversary|edition|edición|version|"
                     r"versión|live|en vivo|acoustic|instrumental|karaoke|remix(es)?|sped up|"
                     r"slowed|explicit|clean)\b", re.I)


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def titulo_base(titulo):
    # 'Abbey Road (Remastered 2019)' y 'Abbey Road' cuentan como el mismo álbum
    return norm(EDICION.sub(" ", titulo or ""))


def albumes(artist_id):
    d = http_get(f"https://api.deezer.com/artist/{artist_id}/albums?limit=200")
    return (d or {}).get("data", [])


def main():
    ap = argparse.ArgumentParser(description="portadas de otros álbumes de los mismos artistas")
    ap.add_argument("--total", type=int, default=20000, help="portadas distintas objetivo")
    ap.add_argument("--workers", type=int, default=8)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(repo, "data"))
    args = ap.parse_args()
    out = args.out
    salida = os.path.join(out, "covers.csv")

    # lo que ya hay: portadas de las canciones y lo de corridas anteriores
    filas = load_csv(salida)
    hechos = {f["album_id"] for f in filas}
    vistos_md5 = {f["md5"]: f["album_id"] for f in filas if not f["duplicada_de"]}
    titulos = set()
    canciones = load_csv(os.path.join(out, "songs.csv"))
    for s in canciones:
        titulos.add((s["artist_id"], titulo_base(s["album"])))
        if not s["cover_path"] or s["album_id"] in hechos:
            continue
        p = os.path.join(out, s["cover_path"])
        if not os.path.exists(p):
            continue
        h = md5(p)
        filas.append({"album_id": s["album_id"], "artist_id": s["artist_id"], "album": s["album"],
                      "record_type": "", "release_date": "", "path": s["cover_path"], "md5": h,
                      "origen": "cancion", "duplicada_de": vistos_md5.get(h, "")})
        vistos_md5.setdefault(h, s["album_id"])
        hechos.add(s["album_id"])
    for f in filas:
        titulos.add((f["artist_id"], titulo_base(f["album"])))
    distintas = lambda: sum(1 for f in filas if not f["duplicada_de"])
    print(f"ya hay {distintas()} portadas distintas ({len(filas)} con repetidas)", flush=True)
    if distintas() >= args.total:
        write_csv(salida, CAMPOS, filas)
        return

    # álbumes candidatos de cada artista
    artistas = [a["artist_id"] for a in load_csv(os.path.join(out, "artists.csv"))]
    random.seed(0)
    random.shuffle(artistas)
    with ThreadPoolExecutor(args.workers) as pool:
        listas = dict(zip(artistas, pool.map(albumes, artistas)))
    candidatos = {}
    for aid, lista in listas.items():
        propios = []
        for al in lista:
            clave = (aid, titulo_base(al.get("title")))
            if str(al["id"]) in hechos or clave in titulos or not real_image(al.get("cover_xl")):
                continue
            # un álbum en colaboración sale en la lista de los dos artistas
            hechos.add(str(al["id"]))
            titulos.add(clave)
            propios.append(al)
        candidatos[aid] = propios
    print(f"{sum(map(len, candidatos.values()))} álbumes candidatos de {len(artistas)} artistas",
          flush=True)

    # de a uno por artista en cada vuelta, así no se llena con los que tienen 200 álbumes
    orden = []
    for i in range(max(map(len, candidatos.values()), default=0)):
        orden += [(aid, lista[i]) for aid, lista in candidatos.items() if i < len(lista)]

    lock = threading.Lock()

    def bajar(par):
        aid, al = par
        with lock:
            if distintas() >= args.total:
                return
        rel = os.path.join("covers", f"{al['id']}.jpg")
        p = os.path.join(out, rel)
        if not download(al["cover_xl"], p):
            return
        h = md5(p)
        with lock:
            if h in vistos_md5:
                os.remove(p)
                return
            vistos_md5[h] = str(al["id"])
            filas.append({"album_id": str(al["id"]), "artist_id": aid, "album": al.get("title", ""),
                          "record_type": al.get("record_type", ""),
                          "release_date": al.get("release_date", ""), "path": rel, "md5": h,
                          "origen": "extra", "duplicada_de": ""})
            n = distintas()
            if n % 500 == 0:
                print(f"  {n}/{args.total}", flush=True)
                write_csv(salida, CAMPOS, filas)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(bajar, orden))
    write_csv(salida, CAMPOS, filas)
    print(f"listo: {distintas()} portadas distintas, {sum(1 for f in filas if f['duplicada_de'])} "
          f"repetidas marcadas en {salida}", flush=True)


if __name__ == "__main__":
    main()
