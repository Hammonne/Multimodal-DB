"""
Clasifica cada artista como persona o banda y, para las bandas, descarga a sus
integrantes con su puesto (guitarrista, bajista, ...) y fotos de cada uno.

Usa MusicBrainz para el tipo de artista y los integrantes (con el instrumento
que toca cada uno en esa banda) y Wikidata/Wikimedia Commons para las fotos de
los integrantes. Lee data/artists.csv (generado por fetch_music.py) y no lo
modifica, así que puede correr mientras fetch_music.py sigue descargando.
Es reanudable: solo procesa los artistas nuevos.

Salida en data/:
  artist_types.csv    tipo de cada artista (persona, banda, otro, desconocido)
  members.csv         una fila por integrante
  band_members.csv    integrante de qué banda, puesto, actual o anterior
  member_images.csv   fotos de cada integrante (fuente, licencia, autor)
  members/<nombre>_<id>/photo_NN.jpg

Uso:
  python3 scripts/fetch_members.py
  python3 scripts/fetch_members.py --member-images 10
"""

import argparse
import csv
import os
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import fetch_music as fm

fm.THROTTLES["musicbrainz.org"] = fm.Throttle(0.9)
MB = "https://musicbrainz.org/ws/2"

TYPES = {"Person": "persona", "Group": "banda", "Orchestra": "banda", "Choir": "banda"}

# instrumento de MusicBrainz -> puesto
ROLES = {
    "guitar": "guitarrista", "electric guitar": "guitarrista",
    "acoustic guitar": "guitarrista", "lead guitar": "guitarrista principal",
    "rhythm guitar": "guitarrista rítmico", "bass guitar": "bajista", "bass": "bajista",
    "electric bass guitar": "bajista", "double bass": "contrabajista",
    "drums (drum set)": "baterista", "drums": "baterista", "drum machine": "baterista",
    "lead vocals": "vocalista", "vocals": "vocalista", "other vocals": "vocalista",
    "background vocals": "corista", "keyboard": "tecladista", "keyboards": "tecladista",
    "synthesizer": "tecladista", "organ": "organista", "piano": "pianista",
    "percussion": "percusionista", "saxophone": "saxofonista", "trumpet": "trompetista",
    "trombone": "trombonista", "violin": "violinista", "cello": "violonchelista",
    "flute": "flautista", "harmonica": "armonicista", "accordion": "acordeonista",
    "turntables": "DJ", "programming": "programador",
}
# ocupación en Wikidata (P106) -> puesto; se usa si MusicBrainz no da instrumentos
OCCUPATIONS = {
    "Q177220": "vocalista", "Q2643890": "vocalista", "Q488205": "vocalista",
    "Q855091": "guitarrista", "Q584301": "bajista", "Q386854": "baterista",
    "Q1075651": "tecladista", "Q486748": "pianista", "Q2252262": "rapero",
    "Q130857": "DJ", "Q12800682": "saxofonista", "Q1259917": "violinista",
}
NOT_ROLES = {"original", "founder", "additional", "guest", "minor", "eponymous",
             "pre-founding"}

TYPE_FIELDS = ["artist_id", "name", "musicbrainz_id", "type", "match"]
MEMBER_FIELDS = ["member_id", "name", "wikidata_id", "occupations", "n_images", "folder"]
BAND_FIELDS = ["artist_id", "band", "member_id", "member", "roles", "roles_source",
               "founder", "current", "begin", "end"]
MIMAGE_FIELDS = ["member_id", "path", "source", "license", "author", "page", "url"]


def mb_get(path, **params):
    return fm.http_get(f"{MB}/{path}?" + urllib.parse.urlencode(dict(params, fmt="json")))


def find_artist(deezer_id, name):
    """(mbid, cómo se encontró): primero por el enlace a Deezer, luego por nombre."""
    d = mb_get("url", resource=f"https://www.deezer.com/artist/{deezer_id}",
               inc="artist-rels")
    for r in (d or {}).get("relations", []):
        if r.get("artist"):
            return r["artist"]["id"], "deezer"
    d = mb_get("artist", query=f'artist:"{name}"', limit=5)
    for a in (d or {}).get("artists", []):
        if a.get("score", 0) >= 95 and fm.norm(a.get("name")) == fm.norm(name):
            return a["id"], "nombre"
    return None, ""


def wikidata_of(mb_artist):
    for r in mb_artist.get("relations", []):
        if r.get("type") == "wikidata":
            return r["url"]["resource"].rstrip("/").rsplit("/", 1)[-1]
    return ""


def wikidata_occupations(qid):
    d = fm.http_get("https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode({
        "action": "wbgetentities", "ids": qid, "props": "claims", "format": "json"}))
    claims = ((d or {}).get("entities", {}).get(qid, {}) or {}).get("claims", {})
    roles = []
    for c in claims.get("P106", []):
        role = OCCUPATIONS.get(((c["mainsnak"].get("datavalue") or {}).get("value") or {}).get("id"))
        if role and role not in roles:
            roles.append(role)
    return roles


def band_members(mb_artist):
    """Integrantes de la banda agrupando las relaciones de cada persona."""
    members = {}
    for r in mb_artist.get("relations", []):
        if r.get("type") != "member of band" or r.get("direction") != "backward":
            continue
        a = r["artist"]
        m = members.setdefault(a["id"], {"name": a["name"], "roles": [], "founder": False,
                                         "current": False, "begin": "", "end": ""})
        for attr in r.get("attributes", []):
            if attr in NOT_ROLES:
                m["founder"] |= attr in ("original", "founder")
                continue
            role = ROLES.get(attr, attr)
            if role not in m["roles"]:
                m["roles"].append(role)
        m["current"] |= not r.get("ended")
        if r.get("begin") and (not m["begin"] or r["begin"] < m["begin"]):
            m["begin"] = r["begin"]
        if r.get("end") and r["end"] > m["end"]:
            m["end"] = r["end"]
    for m in members.values():
        if m["current"]:
            m["end"] = ""
    return members


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--data", default=os.path.join(repo, "data"))
    ap.add_argument("--member-images", type=int, default=8,
                    help="máximo de fotos de Commons por integrante")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    lock = os.path.join(args.data, ".members.lock")
    with open(lock, "w") as f:
        f.write(str(os.getpid()))
    try:
        run(args, args.data)
    finally:
        os.remove(lock)


def run(args, out):

    paths = {k: os.path.join(out, f"{k}.csv") for k in
             ("artist_types", "members", "band_members", "member_images")}
    fields = {"artist_types": TYPE_FIELDS, "members": MEMBER_FIELDS,
              "band_members": BAND_FIELDS, "member_images": MIMAGE_FIELDS}
    done_artists = {r["artist_id"] for r in fm.load_csv(paths["artist_types"])}
    done_members = {r["member_id"] for r in fm.load_csv(paths["members"])}
    artists = [a for a in fm.load_csv(os.path.join(out, "artists.csv"))
               if a["artist_id"] not in done_artists]
    print(f"{len(done_artists)} artistas ya clasificados, {len(artists)} por procesar")

    files = {}
    for k, p in paths.items():
        new = not os.path.exists(p)
        files[k] = open(p, "a", newline="", encoding="utf-8")
        files[k] = (files[k], csv.DictWriter(files[k], fieldnames=fields[k]))
        if new:
            files[k][1].writeheader()
    lock = threading.Lock()
    claimed = set(done_members)
    occupations = {r["member_id"]: r["occupations"] for r in fm.load_csv(paths["members"])}

    def write(kind, rows):
        f, w = files[kind]
        w.writerows(rows)
        f.flush()

    def fetch_member(mid, name):
        d = mb_get(f"artist/{mid}", inc="url-rels") or {}
        # sin enlace en MusicBrainz: búsqueda por nombre, solo si la
        # descripción en Wikidata es de músico (ARTIST_WORDS)
        qid = wikidata_of(d) or fm.wikidata_entity(name) or ""
        occ = "; ".join(wikidata_occupations(qid)) if qid else ""
        folder = os.path.join("members", fm.artist_dir(name, mid[:8]))
        os.makedirs(os.path.join(out, folder), exist_ok=True)
        imgs = []
        for i, img in enumerate(fm.commons_images_for_qid(qid, args.member_images) if qid else [], 1):
            rel = os.path.join(folder, f"photo_{i:02d}{fm.ext_of(img['url'])}")
            if fm.download(img["url"], os.path.join(out, rel)):
                imgs.append({"member_id": mid, "path": rel, "source": img["source"],
                             "license": img["license"], "author": img["author"],
                             "page": img.get("page", ""), "url": img["url"]})
        with lock:
            occupations[mid] = occ
            write("member_images", imgs)
            write("members", [{"member_id": mid, "name": name, "wikidata_id": qid,
                               "occupations": occ, "n_images": len(imgs), "folder": folder}])
        return len(imgs)

    def process(a):
        aid, name = a["artist_id"], a["name"]
        mbid, how = find_artist(aid, name)
        info = mb_get(f"artist/{mbid}", inc="artist-rels+url-rels") if mbid else None
        kind = TYPES.get((info or {}).get("type"), "otro") if info else "desconocido"
        rows, n_photos = [], 0
        if kind == "banda":
            for mid, m in band_members(info).items():
                with lock:
                    mine = mid not in claimed
                    claimed.add(mid)
                if mine:
                    n_photos += fetch_member(mid, m["name"])
                roles, source = "; ".join(m["roles"]), "musicbrainz"
                if not roles:
                    roles, source = occupations.get(mid, ""), "wikidata"
                if not roles:
                    roles, source = "integrante", ""
                rows.append({"artist_id": aid, "band": name, "member_id": mid,
                             "member": m["name"], "roles": roles, "roles_source": source,
                             "founder": m["founder"], "current": m["current"],
                             "begin": m["begin"], "end": m["end"]})
        with lock:
            write("band_members", rows)
            write("artist_types", [{"artist_id": aid, "name": name, "musicbrainz_id": mbid or "",
                                    "type": kind, "match": how}])
            extra = f", {len(rows)} integrantes, {n_photos} fotos" if kind == "banda" else ""
            print(f"  {name[:30]:<30} {kind}{extra}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in [pool.submit(process, a) for a in artists]:
            try:
                f.result()
            except Exception as e:
                print(f"  ! {e}")
    for f, _ in files.values():
        f.close()


if __name__ == "__main__":
    main()
