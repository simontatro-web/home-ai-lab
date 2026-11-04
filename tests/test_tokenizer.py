from model.tokenizer import BPETokenizer

TEXT = "the quick brown fox jumps over the lazy dog. " * 50


def test_round_trip_including_unicode():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    sample = "Héllo wörld 🚀 — 12345 tabs\tand\nnewlines"
    assert tok.decode(tok.encode(sample)) == sample


def test_vocab_size_respected():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    assert 256 < tok.vocab_size <= 300
    assert max(tok.encode(TEXT)) < tok.vocab_size


def test_merges_compress_text():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    assert len(tok.encode(TEXT)) < len(TEXT.encode("utf-8"))


def test_save_and_load(tmp_path):
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    path = tmp_path / "tok.json"
    tok.save(path)
    assert BPETokenizer.load(path).encode(TEXT) == tok.encode(TEXT)
