"""Byte-level BPE tokenizer, written from scratch.

Works on raw UTF-8 bytes, so any input text can be encoded (no unknown
tokens). Training follows the classic BPE recipe:

1. split text into "words" with a GPT-2-style regex (a word keeps its
   leading space, so " the" and "the" are different tokens),
2. count how often each word occurs,
3. repeatedly merge the most frequent adjacent pair of symbols, until the
   vocabulary reaches `vocab_size`.

Usage:
    tok = BPETokenizer.train(text, vocab_size=50_000)
    tok.save("model/tokenizer.json")
    tok = BPETokenizer.load("model/tokenizer.json")
    ids = tok.encode("hello world"); text = tok.decode(ids)

Pure Python, so training on a large sample is slow; train on a few hundred
MB at most and reuse the saved file.
"""

import json
import re
from collections import Counter

# words with optional leading space, numbers, punctuation runs, whitespace
PRETOKENIZE = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")


class BPETokenizer:
    def __init__(self, merges):
        # merges: list of (a, b) pairs of token ids, in the order they were learned
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    # ---------------------------------------------------------------- train
    @classmethod
    def train(cls, text, vocab_size=50_000, verbose=False):
        assert vocab_size >= 256
        words = Counter(PRETOKENIZE.findall(text))
        # each distinct word as a list of byte ids, with its frequency
        seqs = [[list(w.encode("utf-8")), n] for w, n in words.items()]
        merges = []
        for step in range(vocab_size - 256):
            pairs = Counter()
            for seq, n in seqs:
                for p in zip(seq, seq[1:]):
                    pairs[p] += n
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = 256 + len(merges)
            merges.append(best)
            for item in seqs:
                item[0] = cls._merge(item[0], best, new_id)
            if verbose and step % 1000 == 0:
                print(f"merge {step}: {best} (count {pairs[best]})")
        return cls(merges)

    @staticmethod
    def _merge(seq, pair, new_id):
        out, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    # ------------------------------------------------------- encode/decode
    def _encode_word(self, word):
        if word in self._cache:
            return self._cache[word]
        seq = list(word.encode("utf-8"))
        while len(seq) > 1:
            # apply the earliest-learned merge present in the sequence
            pair = min(zip(seq, seq[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            seq = self._merge(seq, pair, 256 + self.ranks[pair])
        self._cache[word] = seq
        return seq

    def encode(self, text):
        ids = []
        for word in PRETOKENIZE.findall(text):
            ids.extend(self._encode_word(word))
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ----------------------------------------------------------- save/load
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "byte-bpe", "merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["merges"])
