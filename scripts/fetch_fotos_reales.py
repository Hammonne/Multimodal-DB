# fotos reales de cds y vinilos para probar la búsqueda de portadas como una app de verdad
# busca en discogs la edición física (cd, vinilo, casete) de álbumes que ya están en covers.csv
# y baja las fotos que subió la gente: la portada del objeto físico y algunas extra (contraportada, disco)
# la respuesta correcta es la portada digital de deezer que ya está en la base
# uso: python3 scripts/fetch_fotos_reales.py --total 250
#      python3 scripts/fetch_fotos_reales.py --ediciones 6   (más ediciones de los álbumes que ya tienen foto)
#      python3 scripts/fetch_fotos_reales.py --artista "KAROL G" --ediciones 6   (todos los álbumes de un artista)
#      python3 scripts/fetch_fotos_reales.py --total 600 --orden popular --ediciones 6   (primero los más escuchados)
import argparse
import os
import random

from fetch_music import THROTTLES, Throttle, download, http_get, load_csv, norm, write_csv

# discogs sin token deja 25 pedidos por minuto
THROTTLES["api.discogs.com"] = Throttle(0.4)
THROTTLES["i.discogs.com"] = Throttle(2)
API = "https://api.discogs.com"
FISICOS = {"CD", "Vinyl", "Cassette"}
CAMPOS = ["album_id", "cover_path", "artista", "album", "discogs_release", "formato",
          "tipo", "path", "url"]
EXTRAS = 2      # fotos secundarias por álbum, además de la principal


def buscar_edicion(artista, album):
    # primera edición física cuyo título coincide con el del álbum
    from urllib.parse import urlencode
    q = urlencode({"artist": artista, "release_title": album, "type": "release", "per_page": 15})
    d = http_get(f"{API}/database/search?{q}") or {}
    for r in d.get("results", []):
        formatos = set(r.get("format") or [])
        titulo = r.get("title", "").split(" - ", 1)[-1]
        if formatos & FISICOS and norm(titulo) == norm(album):
            return r["id"], next(f for f in ("CD", "Vinyl", "Cassette") if f in formatos)
    return None, None


def ediciones(artista, album, maximo):
    # [(release, formato)] de todas las ediciones físicas del álbum, hasta maximo
    from urllib.parse import urlencode
    q = urlencode({"artist": artista, "release_title": album, "type": "release", "per_page": 50})
    d = http_get(f"{API}/database/search?{q}") or {}
    salida = []
    for r in d.get("results", []):
        formatos = set(r.get("format") or [])
        if formatos & FISICOS and norm(r.get("title", "").split(" - ", 1)[-1]) == norm(album):
            salida.append((r["id"], next(f for f in ("CD", "Vinyl", "Cassette") if f in formatos)))
        if len(salida) >= maximo:
            break
    return salida


def mas_ediciones(args, solo=None):
    # para los álbumes que ya tienen foto: la portada (foto principal) de sus otras ediciones físicas
    salida = os.path.join(args.out, "fotos_reales.csv")
    filas = load_csv(salida)
    hechos_csv = os.path.join(args.out, "fotos_reales_ediciones.txt")
    hechos = set(open(hechos_csv).read().split()) if os.path.exists(hechos_csv) else set()
    albumes = {}
    for f in filas:
        if solo is None or f["album_id"] in solo:
            albumes.setdefault(f["album_id"], f)
    with open(hechos_csv, "a") as log:
        for n, (aid, f) in enumerate(albumes.items(), 1):
            if aid in hechos:
                continue
            ya = {x["discogs_release"] for x in filas if x["album_id"] == aid}
            nuevas = []
            for rid, formato in ediciones(f["artista"], f["album"], args.ediciones):
                if str(rid) in ya:
                    continue
                imagenes = (http_get(f"{API}/releases/{rid}") or {}).get("images") or []
                principal = [i for i in imagenes if i.get("type") == "primary"][:1]
                for im in principal:
                    path = os.path.join("fotos_reales", f"{aid}_{rid}_0.jpg")
                    if download(im.get("uri"), os.path.join(args.out, path)):
                        nuevas.append({**{k: f[k] for k in ("album_id", "cover_path", "artista", "album")},
                                       "discogs_release": rid, "formato": formato, "tipo": "edicion",
                                       "path": path, "url": im.get("uri")})
            filas += nuevas
            write_csv(salida, CAMPOS, filas)
            log.write(aid + "\n")
            log.flush()
            print(f"[{n}/{len(albumes)}] {f['artista']} - {f['album']}: +{len(nuevas)} ediciones", flush=True)


def main():
    ap = argparse.ArgumentParser(description="fotos de ediciones físicas desde discogs")
    ap.add_argument("--out", default="data")
    ap.add_argument("--total", type=int, default=250, help="álbumes con foto que se quieren")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ediciones", type=int, default=0,
                    help="en vez de buscar álbumes nuevos, baja hasta N ediciones de los que ya tienen foto")
    ap.add_argument("--artista", default="", help="solo los álbumes de este artista (nombre como en artists.csv)")
    ap.add_argument("--orden", choices=["azar", "popular"], default="azar",
                    help="popular: primero los álbumes con canciones más escuchadas (suelen tener cd o vinilo)")
    args = ap.parse_args()
    # --ediciones solo: completa ediciones de lo ya bajado; con --artista u --orden popular primero busca álbumes
    if args.ediciones and not args.artista and args.orden == "azar":
        mas_ediciones(args)
        return

    nombres = {r["artist_id"]: r["name"] for r in load_csv(os.path.join(args.out, "artists.csv"))}
    portadas = [r for r in load_csv(os.path.join(args.out, "covers.csv"))
                if not r["duplicada_de"] and r["album"] and r["artist_id"] in nombres]
    random.Random(args.seed).shuffle(portadas)
    if args.orden == "popular":
        rank = {}
        for c in load_csv(os.path.join(args.out, "songs.csv")):
            rank[c["album_id"]] = rank.get(c["album_id"], 0) + int(c["deezer_rank"] or 0)
        portadas.sort(key=lambda p: -rank.get(p["album_id"], 0))
    if args.artista:
        portadas = [p for p in portadas if nombres[p["artist_id"]].casefold() == args.artista.casefold()]
        args.total = 10 ** 6

    salida = os.path.join(args.out, "fotos_reales.csv")
    carpeta = os.path.join(args.out, "fotos_reales")
    os.makedirs(carpeta, exist_ok=True)
    filas = load_csv(salida) if os.path.exists(salida) else []
    con_foto = {f["album_id"] for f in filas}
    probados_csv = os.path.join(args.out, "fotos_reales_probados.txt")
    probados = set(open(probados_csv).read().split()) if os.path.exists(probados_csv) else set()

    with open(probados_csv, "a") as log:
        for p in portadas:
            if len(con_foto) >= args.total:
                break
            if p["album_id"] in probados and not args.artista:
                continue
            probados.add(p["album_id"])
            log.write(p["album_id"] + "\n")
            log.flush()
            artista = nombres[p["artist_id"]]
            rid, formato = buscar_edicion(artista, p["album"])
            if not rid or p["album_id"] in con_foto:
                continue
            imagenes = (http_get(f"{API}/releases/{rid}") or {}).get("images") or []
            principal = [i for i in imagenes if i.get("type") == "primary"][:1]
            extras = [i for i in imagenes if i.get("type") == "secondary"][:EXTRAS]
            nuevas = []
            for n, im in enumerate(principal + extras):
                path = os.path.join("fotos_reales", f"{p['album_id']}_{rid}_{n}.jpg")
                if download(im.get("uri"), os.path.join(args.out, path)):
                    nuevas.append({"album_id": p["album_id"], "cover_path": p["path"],
                                   "artista": artista, "album": p["album"], "discogs_release": rid,
                                   "formato": formato,
                                   "tipo": "principal" if im.get("type") == "primary" else "extra",
                                   "path": path, "url": im.get("uri")})
            if any(f["tipo"] == "principal" for f in nuevas):
                filas += nuevas
                con_foto.add(p["album_id"])
                write_csv(salida, CAMPOS, filas)
                print(f"[{len(con_foto)}/{args.total}] {artista} - {p['album']} ({formato}, "
                      f"{len(nuevas)} fotos)", flush=True)
    print(f"{len(con_foto)} álbumes con foto de {len(probados)} probados -> {salida}")
    if args.ediciones and (args.artista or args.orden == "popular"):
        mas_ediciones(args, {p["album_id"] for p in portadas})


if __name__ == "__main__":
    main()
