class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        words = set()
        sentences = texts
        for i in sentences:
            for j in i.strip().lower().split(" "):
                words.add(j)
        words = sorted(words)
        for index,i in enumerate(words):
            self.word_to_id[i] = index+4
            self.id_to_word[index+4] = i
            self.vocab_size+=1
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3
        
        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token
        self.vocab_size+=4
                
    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        out = []
        for i in text.strip().lower().split(" "):
            if len(i):
                out.append(self.word_to_id.get(i,1))
        return out

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        words = []
        for i in ids:
            words.append(self.id_to_word.get(i,self.unk_token))
        return " ".join(words)