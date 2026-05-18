import os
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
PARTS_DIR = DATA_DIR / "parts"
OUTPUT_PATH = DATA_DIR / "corpus.txt"

# Keep the model RAVEN-first. These percentages are enforced by repetition
# when source files are smaller than the requested mix.
MIX = [
    ("raven_dialogue.txt", 60),
    ("psychology.txt", 10),
    ("dark_humor.txt", 10),
    ("math_science_facts.txt", 15),
    ("philosophy.txt", 5),
]


def read_part(filename):
    path = PARTS_DIR / filename
    if not path.exists():
        print(f"Missing {path}; skipping.")
        return ""
    return path.read_text(encoding="utf-8").strip()


def main():
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    sections = []

    for filename, weight in MIX:
        text = read_part(filename)
        if not text:
            continue
        sections.extend([text] * weight)

    if not sections:
        raise SystemExit(
            f"No corpus parts found in {PARTS_DIR}.\n"
            "Add raven_dialogue.txt, psychology.txt, dark_humor.txt, "
            "math_science_facts.txt, and philosophy.txt."
        )

    OUTPUT_PATH.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"Built {OUTPUT_PATH}")
    print(f"Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
