# RAVEN Corpus Parts

Place clean source files here, then run:

```bash
python3 raven/data/build_corpus.py
```

Expected files:

- `raven_dialogue.txt`
- `psychology.txt`
- `dark_humor.txt`
- `math_science_facts.txt`
- `philosophy.txt`

The assembler keeps RAVEN dialogue dominant and writes the final training file to
`raven/data/corpus.txt`.
