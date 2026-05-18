import os
import subprocess

# 1. Rename existing corpus.txt (the 500 lines of dialogue) to raven_dialogue.txt
if os.path.exists("corpus.txt"):
    os.rename("corpus.txt", "raven_dialogue.txt")

urls = {
    'beyond_good_evil.txt': 'https://www.gutenberg.org/ebooks/4363.txt.utf-8',
    'zarathustra.txt': 'https://www.gutenberg.org/ebooks/1998.txt.utf-8',
    'genealogy_morals.txt': 'https://www.gutenberg.org/ebooks/52319.txt.utf-8',
    'meditations.txt': 'https://www.gutenberg.org/ebooks/2680.txt.utf-8',
    'the_prince.txt': 'https://www.gutenberg.org/ebooks/1232.txt.utf-8',
    'art_of_war.txt': 'https://www.gutenberg.org/ebooks/132.txt.utf-8',
    'heart_of_darkness.txt': 'https://www.gutenberg.org/ebooks/526.txt.utf-8',
    'notes_underground.txt': 'https://www.gutenberg.org/ebooks/600.txt.utf-8',
    'crime_punishment.txt': 'https://www.gutenberg.org/ebooks/2554.txt.utf-8',
    'jekyll_hyde.txt': 'https://www.gutenberg.org/ebooks/43.txt.utf-8'
}

# 2. Download all the files using curl
for filename, url in urls.items():
    print(f"Downloading {filename}...")
    subprocess.run(["curl", "-s", "-L", "-o", filename, url])

# 3. Merge all the texts together
files_to_merge = list(urls.keys()) + ["raven_dialogue.txt"]

print("\nMerging into corpus.txt...")
with open("corpus.txt", "wb") as outfile:
    for filename in files_to_merge:
        if os.path.exists(filename):
            with open(filename, "rb") as infile:
                outfile.write(infile.read())
            # Add a newline between files just in case
            outfile.write(b"\n")
        else:
            print(f"Warning: {filename} not found, skipping.")

# Print final size
size = os.path.getsize("corpus.txt")
print(f"\nCorpus built successfully.")
print(f"Total Size: {size / 1024 / 1024:.2f} MB ({size} bytes)")
