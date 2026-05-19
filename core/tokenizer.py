import codecs
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024


def iter_text_chunks_with_sizes(path, chunk_size=DEFAULT_CHUNK_SIZE):
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with open(path, "rb") as handle:
        while True:
            raw = handle.read(chunk_size)
            if not raw:
                break
            text = decoder.decode(raw, final=False)
            if text:
                yield text, len(raw)

    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail, 0


def iter_text_chunks(path, chunk_size=DEFAULT_CHUNK_SIZE):
    for text, _ in iter_text_chunks_with_sizes(path, chunk_size):
        yield text


class RavenTokenizer:
    """Byte-level BPE tokenizer used by RAVEN."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.validate()

    @classmethod
    def train_from_file(cls, corpus_path, vocab_size, save_path, chunk_size=DEFAULT_CHUNK_SIZE):
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Missing corpus: {corpus_path}")
        if corpus_path.stat().st_size == 0:
            raise ValueError(f"Corpus is empty: {corpus_path}")

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)

        trainer = BpeTrainer(
            vocab_size=int(vocab_size),
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=True,
        )
        tokenizer.train_from_iterator(
            iter_text_chunks(corpus_path, chunk_size=chunk_size),
            trainer=trainer,
        )

        wrapper = cls(tokenizer)
        wrapper.save(save_path)
        return wrapper

    @classmethod
    def load(cls, path):
        return cls(Tokenizer.from_file(str(path)))

    def validate(self):
        missing = [token for token in SPECIAL_TOKENS if self.tokenizer.token_to_id(token) is None]
        if missing:
            raise ValueError(f"Tokenizer is missing special tokens: {', '.join(missing)}")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    def encode(self, text, add_bos=False, add_eos=False):
        if not isinstance(text, str):
            raise TypeError("RavenTokenizer.encode() expects text as str.")
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(list(tokens), skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def pad_id(self):
        return self.tokenizer.token_to_id("<pad>")

    @property
    def unk_id(self):
        return self.tokenizer.token_to_id("<unk>")

    @property
    def bos_id(self):
        return self.tokenizer.token_to_id("<bos>")

    @property
    def eos_id(self):
        return self.tokenizer.token_to_id("<eos>")

    @property
    def special_ids(self):
        return {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
