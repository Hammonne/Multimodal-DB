"""
Pruebas offline de scripts/fetch_lyrics.py y su integracion con SPIMI.

    PYTHONPATH=. python3 tests/text_based/test_fetch_lyrics.py

No toca la red: arma un data/ temporal con lyrics sueltas y un songs.csv.
"""

import csv
import json
import os
import shutil
import tempfile

from scripts import fetch_lyrics
from text_based.spimi import SPIMIIndex


def write_songs(path, rows):
    fields = [
        "track_id",
        "title",
        "artist_id",
        "artist",
        "album_id",
        "album",
        "genre",
        "duration_s",
        "lyrics_path",
        "lrc_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_lrc_to_plain():
    lrc = (
        "[ar:Artista]\n"
        "[00:01.00]Hola mundo\n"
        "[00:02.50]<00:02.50> gato raton\n"
        "\n"
    )
    assert fetch_lyrics.lrc_to_plain(lrc) == "Hola mundo\ngato raton"
    print("OK: lrc_to_plain quita metadata y tiempos")


def test_build_database_y_spimi():
    workdir = tempfile.mkdtemp(prefix="lyrics_test_")
    try:
        os.makedirs(os.path.join(workdir, "lyrics"))
        with open(
            os.path.join(workdir, "lyrics", "1.txt"), "w", encoding="utf-8"
        ) as handle:
            handle.write("gato gato perro\n")
        with open(
            os.path.join(workdir, "lyrics", "2.lrc"), "w", encoding="utf-8"
        ) as handle:
            handle.write(
                "[ar:Artista]\n[00:01.00]Hola mundo\n[00:02.00]gato raton\n"
            )

        write_songs(
            os.path.join(workdir, "songs.csv"),
            [
                {"track_id": "1", "title": "Uno", "artist": "A", "genre": "Pop"},
                {"track_id": "2", "title": "Dos", "artist": "B", "genre": "Rock"},
            ],
        )

        total = fetch_lyrics.build_database(workdir)
        assert total == 2

        jsonl = os.path.join(workdir, "lyrics.jsonl")
        with open(jsonl, encoding="utf-8") as handle:
            documents = [json.loads(line) for line in handle if line.strip()]
        assert [doc["id"] for doc in documents] == [1, 2]
        assert documents[0]["lyrics"] == "gato gato perro"
        assert documents[1]["lyrics"] == "Hola mundo\ngato raton"

        with open(
            os.path.join(workdir, "lyrics.csv"), newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert rows[0]["document_id"] == "1"
        assert rows[0]["n_words"] == "3"
        assert rows[1]["has_synced_lyrics"] == "True"

        index_path = os.path.join(workdir, "spimi_index")
        SPIMIIndex(max_memory=1, output_file=index_path).build_from_json(jsonl)
        index = SPIMIIndex(output_file=index_path).load()

        assert index.doc_count == 2
        assert index.doc_lengths == {1: 3, 2: 4}
        assert index.get_postings("gato") == [(1, 2), (2, 1)]
        assert index.get_postings("hola") == [(2, 1)]
        assert index.get_postings("perro") == [(1, 1)]
        assert index.get_postings("raton") == [(2, 1)]
    finally:
        shutil.rmtree(workdir)
    print("OK: fetch_lyrics genera lyrics.jsonl y SPIMI lo indexa")


tests = [
    test_lrc_to_plain,
    test_build_database_y_spimi,
]

for test in tests:
    print(f"Corriendo {test.__name__}...", end=" ")
    test()
    print("OK")

print(f"\n{len(tests)} tests pasaron.")
