import json

class RavenTokenizer:
    """
    A simple character-level tokenizer for the RAVEN model.
    It learns a vocabulary by finding all unique characters in the provided text.
    """

    def __init__(self, text=None):
        """
        Initializes the tokenizer.
        If text is provided, builds the string-to-int (stoi) and int-to-string (itos) mappings.
        """
        self.stoi = {}
        self.itos = {}
        
        if text is not None:
            # Find all unique characters and sort them for a consistent vocabulary
            chars = sorted(list(set(text)))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        """
        Converts a string of text into a list of integers based on the vocabulary.
        """
        # Looks up each character in the text and converts it to its integer ID
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        """
        Converts a list of integers back into a string.
        """
        # Looks up each integer ID and converts it back to its original character
        return ''.join([self.itos[i] for i in tokens])

    @property
    def vocab_size(self):
        """
        Returns the total number of unique characters in the vocabulary.
        """
        return len(self.stoi)

    def save(self, path):
        """
        Saves the tokenizer vocabulary (stoi and itos) to a JSON file.
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'stoi': self.stoi,
                'itos': self.itos
            }, f, indent=4)

    def load(self, path):
        """
        Loads the tokenizer vocabulary from a JSON file.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.stoi = data['stoi']
        # When loading from JSON, the dictionary keys for itos will be strings.
        # We need to convert them back to integers.
        self.itos = {int(k): v for k, v in data['itos'].items()}
