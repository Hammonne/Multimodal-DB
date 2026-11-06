"""Prepara una colección multimedia pequeña para demostrar la web.

No copia ni inserta los archivos binarios en SQL. Usa las rutas existentes
en data/ y genera los descriptores, codebook y vectores que consumen IVF/HNSW.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tipo", choices=("sift", "mfcc", "ambos"), default="sift")
    parser.add_argument("--limite", type=int, default=50)
    parser.add_argument("--k", type=int, default=64)
    args = parser.parse_args()
    python = sys.executable
    if args.tipo in {"mfcc", "ambos"}:
        try:
            import librosa  # noqa: F401
        except ImportError as error:
            raise SystemExit(
                "Falta librosa para procesar audio. Ejecuta: "
                f"{python} -m pip install -r requirements.txt"
            ) from error

    if args.tipo in {"sift", "ambos"}:
        run([python, "-m", "multimedia.sift", "--source", "covers",
             "--limit", str(args.limite)])
        run([python, "-m", "multimedia.codebook", "--tipo", "sift",
             "--ks", str(args.k), "--data", "data"])
        run([python, "-m", "multimedia.features", "--tipo", "sift",
             "--k", str(args.k), "--data", "data"])

    if args.tipo in {"mfcc", "ambos"}:
        run([python, "-m", "multimedia.mfcc", "--limit", str(args.limite)])
        run([python, "-m", "multimedia.codebook", "--tipo", "mfcc",
             "--ks", str(args.k), "--data", "data"])
        run([python, "-m", "multimedia.features", "--tipo", "mfcc",
             "--k", str(args.k), "--data", "data"])

    print("\nColección lista. Reinicia la web y abre la pestaña Multimedia.")


if __name__ == "__main__":
    main()
