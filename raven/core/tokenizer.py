import os

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class RavenTokenizer:
    """
    Small byte-level BPE tokenizer for RAVEN.

    Byte-level BPE keeps unknown characters from breaking generation while giving
    the model word/subword tokens instead of single characters.
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    @classmethod
    def train_from_file(cls, corpus_path, vocab_size, save_path):
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )
        tokenizer.train([corpus_path], trainer)

        wrapper = cls(tokenizer)
        wrapper.save(save_path)
        return wrapper

    @classmethod
    def load(cls, path):
        return cls(Tokenizer.from_file(path))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tokenizer.save(path)

    def encode(self, text, add_bos=False, add_eos=False):
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, tokens):
        return self.tokenizer.decode(list(tokens), skip_special_tokens=True)

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
