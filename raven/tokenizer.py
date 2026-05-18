class RavenTokenizer:

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])

if __name__ == "__main__":
    text = "good evening. the machine watches."

    tokenizer = RavenTokenizer(text)

    print("Vocabulary:", tokenizer.chars)
    print("Vocab size:", tokenizer.vocab_size)

    encoded = tokenizer.encode("good evening")
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)