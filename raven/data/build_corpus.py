import os
import shutil
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
PARTS_DIR = DATA_DIR / "parts"
OUTPUT_PATH = DATA_DIR / "corpus.txt"
COPY_BUFFER_SIZE = 1024 * 1024

MIX = [
    ("raven_dialogue.txt", 60),
    ("psychology.txt", 10),
    ("dark_humor.txt", 10),
    ("math_science_facts.txt", 15),
    ("philosophy.txt", 5),
]


def append_text_file(source_path, output_handle):
    with source_path.open("r", encoding="utf-8", errors="replace") as input_handle:
        shutil.copyfileobj(input_handle, output_handle, length=COPY_BUFFER_SIZE)
    output_handle.write("\n\n")


def main():
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    available = []
    for filename, weight in MIX:
        path = PARTS_DIR / filename
        if not path.exists():
            print(f"Missing {path}; skipping.")
            continue
        if path.stat().st_size == 0:
            print(f"Empty {path}; skipping.")
            continue
        available.append((path, weight))

    if not available:
        raise SystemExit(
            f"No corpus parts found in {PARTS_DIR}.\n"
            "Add raven_dialogue.txt, psychology.txt, dark_humor.txt, "
            "math_science_facts.txt, and philosophy.txt."
        )

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_handle:
        for path, weight in available:
            for _ in range(weight):
                append_text_file(path, output_handle)

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"Built {OUTPUT_PATH}")
    print(f"Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
