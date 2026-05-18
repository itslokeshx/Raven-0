import os
import glob

# paths
data_dir = "/home/loki/code/Raven/raven/data"
books_dir = os.path.join(data_dir, "books")
target_corpus = os.path.join(data_dir, "corpus.txt")
target_size = 500 * 1024 * 1024  # 500 MB

# get all book files
book_files = glob.glob(os.path.join(books_dir, "*.txt"))
# exclude corpus.txt if it exists in the books directory to avoid duplicate feedback loops
book_files = [f for f in book_files if os.path.basename(f) != 'corpus.txt']

print(f"Found {len(book_files)} book files.")

# Read all book contents into memory to speed up writing
books_content = []
for f in book_files:
    with open(f, 'rb') as infile:
        books_content.append(infile.read())

total_books_size = sum(len(c) for c in books_content)
print(f"Total size of books in one pass: {total_books_size / (1024*1024):.2f} MB")

current_size = 0
if os.path.exists(target_corpus):
    current_size = os.path.getsize(target_corpus)

print(f"Current corpus size: {current_size / (1024*1024):.2f} MB")

with open(target_corpus, 'ab') as outfile:
    pass_num = 1
    while current_size < target_size:
        print(f"Pass {pass_num}, current size: {current_size / (1024*1024):.2f} MB")
        for content in books_content:
            outfile.write(content)
            outfile.write(b'\n')
            current_size += len(content) + 1
            if current_size >= target_size:
                break
        pass_num += 1

print(f"Done! Final corpus size: {current_size / (1024*1024):.2f} MB")
