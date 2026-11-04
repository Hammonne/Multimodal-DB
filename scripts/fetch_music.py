# descarga canciones: preview mp3, letra (.txt y .lrc), portada y fotos del artista
# solo usa la librería estándar, si se corta se vuelve a correr y sigue
# uso: python3 scripts/fetch_music.py --songs 1000

import argparse
import csv
import html
import json
import os
import random
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

USER_AGENT = ("chutadb-dataset/0.1 (proyecto academico BD2 UTEC; "
              "https://github.com/Hammonne/Multimodal-DB)")

# géneros de deezer que se usan como semilla: (id, nombre)
GENRES = [
    (132, "Pop"), (116, "Rap/Hip Hop"), (122, "Reggaetón"), (152, "Rock"),
    (113, "Dance"), (165, "R&B"), (85, "Alternativo"), (106, "Electro"),
    (466, "Folk"), (144, "Reggae"), (464, "Metal"), (169, "Soul & Funk"),
    (197, "Latino"), (71, "Cumbia"), (75, "Música Brasileña"), (153, "Blues"),
]

ARTIST_WORDS = re.compile(
    r"singer|band|musician|rapper|group|duo|DJ|record producer|songwriter|"
    r"cantante|banda|músic|grupo|rappeur|chanteu|sänger", re.I)


# http con límite de velocidad por host

class Throttle:
    def __init__(self, per_second):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_time)
            self.next_time = t + self.interval
        delay = t - time.monotonic()
        if delay > 0:
            time.sleep(delay)


THROTTLES = {
    "api.deezer.com": Throttle(8),
    "lrclib.net": Throttle(3),
    "www.theaudiodb.com": Throttle(0.45),
    "www.wikidata.org": Throttle(5),
    "commons.wikimedia.org": Throttle(5),
    "upload.wikimedia.org": Throttle(5),
    "thumb.wikimedia.org": Throttle(5),
}


def http_get(url, binary=False, retries=3, timeout=30):
    host = urllib.parse.urlparse(url).netloc
    for attempt in range(retries):
        if host in THROTTLES:
            THROTTLES[host].wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if binary:
                return data
            obj = json.loads(data.decode("utf-8"))
            # deezer devuelve los errores de cuota como json con código 4
            if isinstance(obj, dict) and isinstance(obj.get("error"), dict) \
                    and obj["error"].get("code") == 4:
                time.sleep(5 * (attempt + 1))
                continue
            return obj
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429 or e.code >= 500:
                time.sleep(3 * (attempt + 1) + random.random())
                continue
            return None
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def download(url, path):
    # True si el archivo quedó en disco
    if not url:
        return False
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    data = http_get(url, binary=True)
    if not data:
        return False
    tmp = f"{path}.{threading.get_ident()}.part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return True


def real_image(url):
    # deezer usa urls con hash vacío ('/artist//') como imagen genérica
    return url if url and "//" not in urllib.parse.urlparse(url).path else None


def ext_of(url, default=".jpg"):
    path = urllib.parse.urlparse(url).path.lower()
    for e in (".jpg", ".jpeg", ".png", ".webp"):
        if path.endswith(e):
            return ".jpg" if e == ".jpeg" else e
    return default


# fuentes

def deezer_seed_artists(genre_id, genre_name, radios=6, playlists=3):
    # artistas de las radios y playlists del género, ordenados por frecuencia
    # /genre/{id}/artists devuelve lo mismo para todos los géneros
    sources = []
    d = http_get(f"https://api.deezer.com/genre/{genre_id}/radios")
    sources += [f"https://api.deezer.com/radio/{r['id']}/tracks?limit=100"
                for r in (d or {}).get("data", [])[:radios]]
    d = http_get("https://api.deezer.com/search/playlist?" + urllib.parse.urlencode(
        {"q": genre_name, "limit": playlists}))
    sources += [f"https://api.deezer.com/playlist/{p['id']}/tracks?limit=100"
                for p in (d or {}).get("data", [])[:playlists]]
    count, by_id = Counter(), {}
    for url in sources:
        for t in (http_get(url) or {}).get("data", []):
            a = t.get("artist") or {}
            if a.get("id"):
                count[a["id"]] += 1
                by_id[a["id"]] = a
    return [by_id[i] for i, _ in count.most_common()]


def deezer_related(artist_id):
    d = http_get(f"https://api.deezer.com/artist/{artist_id}/related?limit=20")
    return (d or {}).get("data", [])


def deezer_top_tracks(artist_id, limit):
    d = http_get(f"https://api.deezer.com/artist/{artist_id}/top?limit={limit}")
    return (d or {}).get("data", [])


def artist_dir(name, aid):
    # 'bad-bunny_8355', el id evita choques de nombre
    t = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")[:40].strip("-")
    return f"{t or 'artista'}_{aid}"


def norm(text):
    # minúsculas, sin tildes, sin (feat. ...), [...] ni ' - Remastered'
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    t = re.sub(r"\(.*?\)|\[.*?\]| - .*$", "", t)
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    return t or (text or "").lower().strip()


def lrclib_lyrics(artist, title, duration, max_diff=15):
    # /api/get tarda hasta 25 s cuando no tiene la letra, por eso /api/search
    # y se valida artista, título y duración, mejor si es sincronizada
    res = http_get("https://lrclib.net/api/search?" + urllib.parse.urlencode(
        {"artist_name": artist, "track_name": title}), retries=8)
    na, nt = norm(artist), norm(title)
    best = None
    for c in res if isinstance(res, list) else []:
        if not (c.get("plainLyrics") or c.get("instrumental")):
            continue
        if norm(c.get("trackName")) != nt:
            continue
        ca = norm(c.get("artistName"))
        if na not in ca and ca not in na:
            continue
        diff = abs(c["duration"] - duration) if duration and c.get("duration") else 0
        if diff > max_diff:
            continue
        key = (not c.get("syncedLyrics"), diff)
        if best is None or key < best[0]:
            best = (key, c)
    if not best:
        return None
    d = best[1]
    return {
        "plain": d.get("plainLyrics") or "",
        "synced": d.get("syncedLyrics") or "",
        "instrumental": bool(d.get("instrumental")),
    }


def audiodb_images(name):
    d = http_get("https://www.theaudiodb.com/api/v1/json/123/search.php?" +
                 urllib.parse.urlencode({"s": name}))
    arts = (d or {}).get("artists") or []
    if not arts:
        return []
    a = arts[0]
    if a.get("strArtist", "").lower() != name.lower():
        return []
    keys = ["strArtistThumb", "strArtistFanart", "strArtistFanart2",
            "strArtistFanart3", "strArtistFanart4", "strArtistCutout",
            "strArtistWideThumb"]
    return [{"url": a[k], "source": "theaudiodb", "license": "", "author": ""}
            for k in keys if a.get(k)]


def wikidata_entity(name):
    d = http_get("https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode({
        "action": "wbsearchentities", "search": name, "language": "en",
        "type": "item", "limit": 7, "format": "json"}))
    for e in (d or {}).get("search", []):
        if ARTIST_WORDS.search(e.get("description", "")):
            return e["id"]
    return None


def commons_images(name, max_images):
    qid = wikidata_entity(name)
    if not qid:
        return None, []
    return qid, commons_images_for_qid(qid, max_images)


def commons_images_for_qid(qid, max_images):
    # foto P18 y fotos de la categoría de commons (P373)
    d = http_get("https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode({
        "action": "wbgetentities", "ids": qid, "props": "claims", "format": "json"}))
    claims = ((d or {}).get("entities", {}).get(qid, {}) or {}).get("claims", {})

    def first(prop):
        try:
            return claims[prop][0]["mainsnak"]["datavalue"]["value"]
        except (KeyError, IndexError, TypeError):
            return None

    titles = []
    p18 = first("P18")
    if p18:
        titles.append("File:" + p18)

    params = {"action": "query", "format": "json", "prop": "imageinfo",
              "iiprop": "url|mime|extmetadata", "iiurlwidth": 800}
    pages = []
    cat = first("P373")
    if cat:
        q = dict(params, generator="categorymembers", gcmtitle="Category:" + cat,
                 gcmtype="file", gcmlimit=max_images + 5)
        r = http_get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(q))
        pages += list(((r or {}).get("query") or {}).get("pages", {}).values())
    if titles:
        q = dict(params, titles="|".join(titles))
        r = http_get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(q))
        pages = list(((r or {}).get("query") or {}).get("pages", {}).values()) + pages

    out, seen = [], set()
    for p in pages:
        info = (p.get("imageinfo") or [{}])[0]
        if info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        url = info.get("thumburl") or info.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        meta = info.get("extmetadata") or {}
        author = re.sub(r"<[^>]+>", " ", (meta.get("Artist") or {}).get("value", ""))
        author = re.sub(r"\s+", " ", html.unescape(author)).strip()
        out.append({"url": url, "source": "wikimedia_commons",
                    "license": (meta.get("LicenseShortName") or {}).get("value", ""),
                    "author": author[:200], "page": info.get("descriptionurl", "")})
        if len(out) >= max_images:
            break
    return out


# pipeline

SONG_FIELDS = ["track_id", "title", "artist_id", "artist", "album_id", "album",
               "genre", "duration_s", "deezer_rank", "explicit", "has_lyrics",
               "has_synced_lyrics", "instrumental", "audio_path", "lyrics_path",
               "lrc_path", "cover_path", "deezer_url"]
ARTIST_FIELDS = ["artist_id", "name", "genre", "wikidata_id", "n_images",
                 "main_image_path", "deezer_url"]
IMAGE_FIELDS = ["artist_id", "path", "source", "license", "author", "page", "url"]


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    tmp = path + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def process_artist(artist, genre, out, per_artist, extra_images, done_tracks):
    aid = str(artist["id"])
    name = artist["name"]
    folder = os.path.join("artists", artist_dir(name, aid))
    os.makedirs(os.path.join(out, folder), exist_ok=True)

    # imágenes del artista
    main_path = os.path.join(folder, "main.jpg")
    images = []
    if download(real_image(artist.get("picture_xl")), os.path.join(out, main_path)):
        images.append({"artist_id": aid, "path": main_path, "source": "deezer",
                       "license": "", "author": "", "page": "",
                       "url": artist.get("picture_xl")})
    extras = audiodb_images(name)
    qid, commons = commons_images(name, extra_images)
    extras += commons
    for i, img in enumerate(extras[:extra_images + 7], 1):
        rel = os.path.join(folder, f"extra_{i:02d}{ext_of(img['url'])}")
        if download(img["url"], os.path.join(out, rel)):
            images.append({"artist_id": aid, "path": rel, "source": img["source"],
                           "license": img.get("license", ""),
                           "author": img.get("author", ""),
                           "page": img.get("page", ""), "url": img["url"]})

    artist_row = {"artist_id": aid, "name": name, "genre": genre,
                  "wikidata_id": qid or "", "n_images": len(images),
                  "main_image_path": main_path if images and images[0]["source"] == "deezer" else "",
                  "deezer_url": artist.get("link", "")}

    songs = [s for t in deezer_top_tracks(aid, per_artist)
             if (s := process_track(t, aid, genre, out, done_tracks))]
    return artist_row, images, songs


def process_track(t, aid, genre, out, done_tracks, solo_letra=False):
    # con solo_letra se busca la letra antes y si no hay no se baja nada
    tid = str(t["id"])
    if tid in done_tracks or not t.get("preview"):
        return None
    lyr = lrclib_lyrics(t["artist"]["name"], t.get("title_short") or t["title"],
                        t.get("duration"))
    if solo_letra and not (lyr and lyr["plain"]):
        return None
    album = t.get("album") or {}
    audio_rel = os.path.join("audio", tid + ".mp3")
    if not download(t["preview"], os.path.join(out, audio_rel)):
        return None
    cover_rel = ""
    if real_image(album.get("cover_xl")):
        cover_rel = os.path.join("covers", f"{album['id']}.jpg")
        if not download(album["cover_xl"], os.path.join(out, cover_rel)):
            cover_rel = ""

    lyrics_rel = lrc_rel = ""
    if lyr and lyr["plain"]:
        lyrics_rel = os.path.join("lyrics", tid + ".txt")
        with open(os.path.join(out, lyrics_rel), "w", encoding="utf-8") as f:
            f.write(lyr["plain"])
    if lyr and lyr["synced"]:
        lrc_rel = os.path.join("lyrics", tid + ".lrc")
        with open(os.path.join(out, lrc_rel), "w", encoding="utf-8") as f:
            f.write(lyr["synced"])

    return {
        "track_id": tid, "title": t["title"], "artist_id": aid,
        "artist": t["artist"]["name"], "album_id": album.get("id", ""),
        "album": album.get("title", ""), "genre": genre,
        "duration_s": t.get("duration", ""), "deezer_rank": t.get("rank", ""),
        "explicit": t.get("explicit_lyrics", False),
        "has_lyrics": bool(lyrics_rel), "has_synced_lyrics": bool(lrc_rel),
        "instrumental": bool(lyr and lyr["instrumental"]),
        "audio_path": audio_rel, "lyrics_path": lyrics_rel, "lrc_path": lrc_rel,
        "cover_path": cover_rel, "deezer_url": t.get("link", "")}


def more_tracks(arow, out, limit, done_tracks):
    # canciones extra de un artista que ya está, sin volver a bajar sus fotos
    return [s for t in deezer_top_tracks(arow["artist_id"], limit)
            if (s := process_track(t, arow["artist_id"], arow["genre"], out,
                                   done_tracks, solo_letra=True))]


def main():
    ap = argparse.ArgumentParser(description="descarga canciones con audio, letra, portada y fotos del artista")
    ap.add_argument("--songs", type=int, default=1000, help="canciones objetivo")
    ap.add_argument("--per-artist", type=int, default=5, help="canciones por artista")
    ap.add_argument("--extra-images", type=int, default=8,
                    help="máximo de fotos de Wikimedia Commons por artista")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--con-letra", type=int, default=0,
                    help="en vez de --songs: llegar a N canciones con letra usando los artistas que ya están")
    ap.add_argument("--mas", type=int, default=3,
                    help="con --con-letra: canciones extra que se miran por artista")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(repo, "data"))
    args = ap.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)
    lock = os.path.join(out, ".fetch.lock")
    with open(lock, "w") as f:
        f.write(str(os.getpid()))
    try:
        run_con_letra(args, out) if args.con_letra else run(args, out)
    finally:
        os.remove(lock)


def run(args, out):
    for sub in ("audio", "lyrics", "covers", "artists"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    songs_csv = os.path.join(out, "songs.csv")
    artists_csv = os.path.join(out, "artists.csv")
    images_csv = os.path.join(out, "artist_images.csv")
    songs = load_csv(songs_csv)
    artists = load_csv(artists_csv)
    images = load_csv(images_csv)
    done_tracks = {s["track_id"] for s in songs}
    done_artists = {a["artist_id"] for a in artists}
    print(f"Retomando: {len(songs)} canciones, {len(artists)} artistas ya descargados")

    # cola: primero las semillas por género, luego artistas relacionados
    # de todos los ya procesados (también de corridas anteriores)
    queue, queued = [], set(done_artists)
    expand_from = [(a["artist_id"], a["genre"]) for a in artists]
    with ThreadPoolExecutor(max_workers=len(GENRES)) as pool:
        seeds = pool.map(lambda g: deezer_seed_artists(*g), GENRES)
        per_genre = [[(a, gname) for a in lst] for (_, gname), lst in zip(GENRES, seeds)]
    # intercalar géneros (round-robin) para que el dataset quede balanceado
    for i in range(max(map(len, per_genre))):
        for lst in per_genre:
            if i < len(lst) and str(lst[i][0]["id"]) not in queued:
                queued.add(str(lst[i][0]["id"]))
                queue.append(lst[i])
    random.seed(42)
    random.shuffle(expand_from)
    print(f"{len(queue)} artistas semilla en {len(GENRES)} géneros")

    last_save = time.time()

    def refill():
        while len(queue) < args.workers * 2 and expand_from:
            src, g = expand_from.pop(0)
            for r in deezer_related(src):
                if str(r["id"]) not in queued:
                    queued.add(str(r["id"]))
                    queue.append((r, g))

    def collect(f):
        a, g = pending.pop(f)
        try:
            arow, imgs, new_songs = f.result()
        except Exception as e:
            print(f"  ! {a['name']}: {e}")
            return
        new_songs = [s for s in new_songs if s["track_id"] not in done_tracks]
        done_tracks.update(s["track_id"] for s in new_songs)
        songs.extend(new_songs)
        artists.append(arow)
        images.extend(imgs)
        expand_from.append((arow["artist_id"], g))
        n_lyr = sum(1 for s in new_songs if s["has_lyrics"])
        print(f"[{len(songs):>6}/{args.songs}] {a['name'][:30]:<30} "
              f"{len(new_songs)} canciones, {n_lyr} con letra, {len(imgs)} fotos")

    def save():
        write_csv(songs_csv, SONG_FIELDS, songs)
        write_csv(artists_csv, ARTIST_FIELDS, artists)
        write_csv(images_csv, IMAGE_FIELDS, images)

    pending = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while len(songs) < args.songs:
            refill()
            # no enviar más artistas de los necesarios para llegar al objetivo
            while (queue and len(pending) < args.workers * 2 and
                   len(songs) + len(pending) * args.per_artist < args.songs):
                a, g = queue.pop(0)
                f = pool.submit(process_artist, a, g, out, args.per_artist,
                                args.extra_images, set(done_tracks))
                pending[f] = (a, g)
            if not pending:
                print("No quedan artistas por explorar")
                break
            collect(next(as_completed(pending)))
            if time.time() - last_save > 30:
                save()
                last_save = time.time()
        # los artistas que ya estaban en curso también se guardan
        for f in list(as_completed(pending)):
            collect(f)

    save()
    n_lyr = sum(1 for s in songs if str(s["has_lyrics"]) == "True")
    print(f"\nListo: {len(songs)} canciones ({n_lyr} con letra), "
          f"{len(artists)} artistas, {len(images)} fotos de artistas en {out}")


def con_letra(songs):
    return sum(1 for s in songs if str(s["has_lyrics"]) == "True")


def agregar(songs, done_tracks, nuevas):
    for s in nuevas:
        if s["track_id"] not in done_tracks:
            done_tracks.add(s["track_id"])
            songs.append(s)


def run_con_letra(args, out):
    # las canciones nuevas van al final de songs.csv, las primeras 10000 no cambian
    songs_csv = os.path.join(out, "songs.csv")
    songs = load_csv(songs_csv)
    artists = load_csv(os.path.join(out, "artists.csv"))
    done_tracks = {s["track_id"] for s in songs}
    n_artista = Counter(s["artist_id"] for s in songs)
    print(f"Retomando: {len(songs)} canciones, {con_letra(songs)} con letra")

    # artistas con menos canciones primero, así se reparte
    random.seed(42)
    random.shuffle(artists)
    artists.sort(key=lambda a: n_artista[a["artist_id"]])
    last_save = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = set()
        it = iter(artists)
        while con_letra(songs) < args.con_letra:
            while len(pending) < args.workers * 2:
                a = next(it, None)
                if a is None:
                    break
                limit = n_artista[a["artist_id"]] + args.mas
                pending.add(pool.submit(more_tracks, a, out, limit, set(done_tracks)))
            if not pending:
                print("No quedan artistas, subir --mas y volver a correr")
                break
            f = next(as_completed(pending))
            pending.remove(f)
            agregar(songs, done_tracks, f.result())
            print(f"[{con_letra(songs):>6}/{args.con_letra}] {len(songs)} canciones")
            if time.time() - last_save > 30:
                write_csv(songs_csv, SONG_FIELDS, songs)
                last_save = time.time()
        # los que ya estaban bajando también se guardan, si no quedan mp3 sueltos
        for f in as_completed(pending):
            agregar(songs, done_tracks, f.result())
    write_csv(songs_csv, SONG_FIELDS, songs)
    print(f"\nListo: {len(songs)} canciones, {con_letra(songs)} con letra")


if __name__ == "__main__":
    main()
