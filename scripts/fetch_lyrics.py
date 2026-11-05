"""
Lee las letras que deja fetch_music.py en data/lyrics/<track_id>.txt/.lrc y
genera dos archivos listos para SPIMI, TF-IDF + coseno y BM25:

  data/lyrics.jsonl  una línea por documento: {"id": <track_id>, "lyrics": "..."}
  data/lyrics.csv    índice con título, artista, álbum, género, rutas, etc.

Si no hay letras todavía, --fetch las descarga de LRCLIB. Cuando no existe
data/songs.csv, --fetch crea un dataset solo de letras usando los mismos
géneros y la misma expansión por artistas relacionados que fetch_music.py.
Es reanudable: nunca vuelve a descargar una letra que ya está en disco.

Uso:
  python3 scripts/fetch_lyrics.py
  python3 scripts/fetch_lyrics.py --fetch
  python3 scripts/fetch_lyrics.py --fetch --songs 1000
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_music as fm  # noqa: E402

DB_FIELDS = ["document_id", "track_id", "title", "artist_id", "artist",
             "album_id", "album", "genre", "duration_s", "lyrics_path",
             "lrc_path", "has_synced_lyrics", "instrumental", "n_chars",
             "n_words"]


def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.part"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def lrc_to_plain(text):
    lines = []
    for line in text.splitlines():
        line = re.sub(r"<[^>]*>", "", line)
        line = re.sub(r"\[\d{1,3}:\d{2}(?:[.:]\d{1,3})?\]", "", line)
        if re.fullmatch(r"\s*\[[a-zA-Z]+:[^\]]*\]\s*", line):
            continue
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def document_id(row):
    value = str(row.get("document_id") or row.get("track_id") or "")
    return os.path.basename(value)


def existing_path(data, rel):
    if not rel:
        return ""
    full = os.path.join(data, rel)
    if os.path.isfile(full) and os.path.getsize(full) > 0:
        return rel
    return ""


def resolve_lyrics(data, row):
    stem = document_id(row)
    txt = existing_path(data, row.get("lyrics_path")) or \
        existing_path(data, os.path.join("lyrics", stem + ".txt"))
    lrc = existing_path(data, row.get("lrc_path")) or \
        existing_path(data, os.path.join("lyrics", stem + ".lrc"))
    text = read_text(os.path.join(data, txt)) if txt else ""
    if not text.strip() and lrc:
        text = lrc_to_plain(read_text(os.path.join(data, lrc)))
    text = text.strip()
    if not text:
        return None
    return text, txt, lrc


def word_count(text):
    return len(re.findall(r"[^\W_]+", text, re.UNICODE))


def load_rows(data):
    rows = {}
    for name in ("songs.csv", "lyrics.csv"):
        for r in fm.load_csv(os.path.join(data, name)):
            key = str(r.get("document_id") or r.get("track_id") or "")
            if not key:
                continue
            if key in rows:
                rows[key].update({k: v for k, v in r.items() if v not in (None, "")})
            else:
                rows[key] = dict(r)

    lyrics_dir = os.path.join(data, "lyrics")
    if os.path.isdir(lyrics_dir):
        for name in os.listdir(lyrics_dir):
            stem, ext = os.path.splitext(name)
            if ext.lower() in (".txt", ".lrc") and stem and not name.endswith(".part"):
                rows.setdefault(stem, {"document_id": stem, "track_id": stem})
    return rows


def build_database(data):
    docs = {}
    index = {}
    for key, row in load_rows(data).items():
        resolved = resolve_lyrics(data, row)
        if resolved is None:
            continue
        text, txt, lrc = resolved
        doc_id = document_id(row) or key
        out = {field: row.get(field, "") for field in DB_FIELDS}
        out.update({
            "document_id": doc_id,
            "track_id": row.get("track_id") or doc_id,
            "lyrics_path": txt,
            "lrc_path": lrc,
            "has_synced_lyrics": bool(lrc),
            "n_chars": len(text),
            "n_words": word_count(text),
        })
        docs[doc_id] = {"id": int(doc_id) if doc_id.isdigit() else doc_id,
                        "lyrics": text}
        index[doc_id] = out

    order = sorted(docs, key=lambda d: (0, int(d)) if d.isdigit() else (1, d))
    jsonl = os.path.join(data, "lyrics.jsonl")
    tmp = f"{jsonl}.{os.getpid()}.part"
    with open(tmp, "w", encoding="utf-8") as f:
        for doc_id in order:
            f.write(json.dumps(docs[doc_id], ensure_ascii=False) + "\n")
    os.replace(tmp, jsonl)
    fm.write_csv(os.path.join(data, "lyrics.csv"), DB_FIELDS,
                 [index[doc_id] for doc_id in order])

    words = sum(index[d]["n_words"] for d in order)
    synced = sum(1 for d in order if index[d]["has_synced_lyrics"])
    print(f"{len(order)} documentos, {words} palabras, {synced} con .lrc en {data}")
    return len(order)


def duration(row):
    try:
        return int(float(row.get("duration_s") or 0))
    except (TypeError, ValueError):
        return 0


def fetch_one(data, row):
    artist = row.get("artist") or ""
    title = row.get("title") or ""
    if not artist or not title:
        return row, False
    found = fm.lrclib_lyrics(artist, title, duration(row))
    if not found:
        return row, False
    stem = document_id(row)
    changed = False
    if found["plain"]:
        rel = os.path.join("lyrics", stem + ".txt")
        write_text(os.path.join(data, rel), found["plain"])
        row["lyrics_path"] = rel
        row["has_lyrics"] = True
        changed = True
    if found["synced"]:
        rel = os.path.join("lyrics", stem + ".lrc")
        write_text(os.path.join(data, rel), found["synced"])
        row["lrc_path"] = rel
        row["has_synced_lyrics"] = True
        changed = True
    if found["instrumental"]:
        row["instrumental"] = True
    return row, changed


def fetch_missing(data, songs, workers):
    todo = [s for s in songs if resolve_lyrics(data, s) is None]
    print(f"{len(songs) - len(todo)} canciones ya tienen letra, "
          f"{len(todo)} por buscar en LRCLIB")
    changed = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, data, s): s for s in todo}
        for n, future in enumerate(as_completed(futures), 1):
            row, ok = future.result()
            changed |= ok
            if n % 50 == 0 or n == len(todo):
                print(f"  [{n}/{len(todo)}]")
    if changed:
        rows = [{field: s.get(field, "") for field in fm.SONG_FIELDS} for s in songs]
        fm.write_csv(os.path.join(data, "songs.csv"), fm.SONG_FIELDS, rows)


def has_lyrics_files(data):
    stems = set()
    lyrics_dir = os.path.join(data, "lyrics")
    if not os.path.isdir(lyrics_dir):
        return stems
    for name in os.listdir(lyrics_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".txt", ".lrc") or not stem or name.endswith(".part"):
            continue
        full = os.path.join(lyrics_dir, name)
        if os.path.getsize(full) > 0:
            stems.add(stem)
    return stems


def seed_queue():
    with ThreadPoolExecutor(max_workers=len(fm.GENRES)) as pool:
        lists = list(pool.map(lambda g: fm.deezer_seed_artists(*g), fm.GENRES))
    per_genre = []
    for (_, name), artists in zip(fm.GENRES, lists):
        per_genre.append([(a, name) for a in artists])
    queue = []
    for i in range(max(map(len, per_genre), default=0)):
        for artists in per_genre:
            if i < len(artists):
                queue.append(artists[i])
    return queue


def save_metadata(data, metadata):
    rows = [{field: row.get(field, "") for field in DB_FIELDS}
            for row in metadata.values()]
    fm.write_csv(os.path.join(data, "lyrics.csv"), DB_FIELDS, rows)


def process_artist(data, artist, genre, per_artist, done):
    rows = []
    aid = str(artist["id"])
    for track in fm.deezer_top_tracks(aid, per_artist):
        tid = str(track["id"])
        if tid in done:
            continue
        title = track.get("title_short") or track["title"]
        album = track.get("album") or {}
        row = {
            "document_id": tid, "track_id": tid, "title": track["title"],
            "artist_id": aid, "artist": (track.get("artist") or {}).get("name")
            or artist["name"], "album_id": album.get("id", ""),
            "album": album.get("title", ""), "genre": genre,
            "duration_s": track.get("duration", ""), "lyrics_path": "",
            "lrc_path": "", "has_synced_lyrics": False,
            "instrumental": False, "n_chars": "", "n_words": "",
        }
        resolved = resolve_lyrics(data, row)
        if resolved is None:
            found = fm.lrclib_lyrics(row["artist"], title, track.get("duration"))
            if not found or not (found["plain"] or found["synced"]):
                continue
            if found["plain"]:
                rel = os.path.join("lyrics", tid + ".txt")
                write_text(os.path.join(data, rel), found["plain"])
                row["lyrics_path"] = rel
            if found["synced"]:
                rel = os.path.join("lyrics", tid + ".lrc")
                write_text(os.path.join(data, rel), found["synced"])
                row["lrc_path"] = rel
            row["has_synced_lyrics"] = bool(found["synced"])
            row["instrumental"] = bool(found["instrumental"])
        else:
            _, row["lyrics_path"], row["lrc_path"] = resolved
            row["has_synced_lyrics"] = bool(row["lrc_path"])
        rows.append(row)
    return rows, fm.deezer_related(aid)


def discover(data, target, per_artist, workers):
    existing = has_lyrics_files(data)
    print(f"{len(existing)} letras ya en disco, objetivo {target}")
    if len(existing) >= target:
        return

    metadata = {}
    for row in fm.load_csv(os.path.join(data, "lyrics.csv")):
        key = document_id(row)
        if key and resolve_lyrics(data, row) is not None:
            metadata[key] = row
    done = existing | set(metadata)

    queue = deque()
    queued = set()
    for artist, genre in seed_queue():
        aid = str(artist["id"])
        if aid not in queued:
            queued.add(aid)
            queue.append((artist, genre))
    print(f"{len(queue)} artistas semilla en {len(fm.GENRES)} géneros")

    pending = {}
    last_save = 0.0

    def collect(future):
        nonlocal last_save
        artist, genre = pending.pop(future)
        try:
            rows, related = future.result()
        except Exception as exc:
            print(f"  ! {artist.get('name', '')}: {exc}")
            return
        added = 0
        for row in rows:
            key = document_id(row)
            if not key:
                continue
            metadata[key] = row
            if key not in done:
                done.add(key)
                added += 1
        for rel in related:
            rid = str(rel["id"])
            if rid not in queued:
                queued.add(rid)
                queue.append((rel, genre))
        if added:
            print(f"[{len(done):>6}/{target}] {artist.get('name', '')[:30]:<30} "
                  f"{added} letras")
        if time.time() - last_save > 30:
            save_metadata(data, metadata)
            last_save = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while len(done) < target:
            while queue and len(pending) < workers * 2 and \
                    len(done) + len(pending) * per_artist < target:
                artist, genre = queue.popleft()
                future = pool.submit(process_artist, data, artist, genre,
                                     per_artist, set(metadata))
                pending[future] = (artist, genre)
            if not pending:
                print("No quedan artistas por explorar")
                break
            collect(next(as_completed(pending)))
        for future in list(as_completed(pending)):
            collect(future)

    save_metadata(data, metadata)


def lock_alive(data, name):
    lock = os.path.join(data, name)
    if not os.path.exists(lock):
        return False
    try:
        os.kill(int(open(lock).read().strip()), 0)
        return True
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--data", default=os.path.join(repo, "data"))
    ap.add_argument("--fetch", action="store_true",
                    help="completar letras faltantes (LRCLIB); si no hay "
                         "songs.csv, crear un dataset solo de letras")
    ap.add_argument("--songs", type=int, default=1000,
                    help="documentos objetivo al crear un dataset solo de letras")
    ap.add_argument("--per-artist", type=int, default=5,
                    help="canciones top por artista al descubrir canciones")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    data = args.data
    os.makedirs(os.path.join(data, "lyrics"), exist_ok=True)

    songs_csv = os.path.join(data, "songs.csv")
    if args.fetch:
        if lock_alive(data, ".fetch.lock"):
            sys.exit("fetch_music.py está corriendo y reescribe songs.csv y "
                     "lyrics/: espera a que termine")
        if os.path.exists(songs_csv):
            songs = fm.load_csv(songs_csv)
            fetch_missing(data, songs, args.workers)
        else:
            discover(data, args.songs, args.per_artist, args.workers)

    total = build_database(data)
    if not total:
        print("No hay letras todavía: corre fetch_music.py o usa --fetch")


if __name__ == "__main__":
    main()
