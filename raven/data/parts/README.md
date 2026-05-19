# RAVEN Corpus Parts

Place clean UTF-8 `.txt` source files here, then run:

```bash
python3 raven/data/build_corpus.py
```

Expected filenames:

- `raven_dialogue.txt`
- `psychology.txt`
- `dark_humor.txt`
- `math_science_facts.txt`
- `philosophy.txt`

`build_corpus.py` streams each file into `raven/data/corpus.txt` using the
configured mix, so it does not need to hold the assembled corpus in RAM.
