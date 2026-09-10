"""
Char-RNN and Seq2Seq with Attention in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Char-RNN and seq2seq translation with attention (Hands-On ML, chapter 14).

Story: teach a GRU to write Shakespeare one character at a time and sample it at
three temperatures; then train an English-to-Spanish translator with and without
Luong attention, read the alignment the attention produces, and save the result.
"""
import os
import tempfile
import numpy as np
import torch


def main() -> None:
    # ---- 1. Char-RNN ----
    text = load_shakespeare()
    vocab = CharVocab(text)
    ids = vocab.encode(text[:150000])
    X, Y = make_char_windows(ids, window=100, stride=100)
    torch.manual_seed(0)
    char_model = CharRNN(len(vocab), embed=16, hidden=128)
    losses = train_char_rnn(char_model, X, Y, epochs=2)
    print(f"char-rnn on {len(ids):,} characters: loss {losses[0]:.2f} -> {losses[-1]:.2f} (uniform guess = {np.log(len(vocab)):.2f})")
    for temp in (0.2, 0.6, 1.2):
        sample = generate(char_model, vocab, "ROMEO:", 80, temperature=temp, seed=1).replace(chr(10), " / ")
        print(f"  T={temp}: {sample}")

    # ---- 2. Translation ----
    pairs = load_spa_eng(15000)
    en = WordVocab([e for e, _ in pairs])
    es = WordVocab([s for _, s in pairs])
    src, tin, tout = encode_pairs(pairs, en, es)
    torch.manual_seed(0)
    plain = Seq2Seq(Encoder(len(en)), Decoder(len(es)))
    h_plain = train_translator(plain, src, tin, tout, epochs=2)
    torch.manual_seed(0)
    attn = Seq2Seq(Encoder(len(en)), AttnDecoder(len(es)))
    h_attn = train_translator(attn, src, tin, tout, epochs=2)
    print(f"\ntranslator masked val accuracy: plain {h_plain['val_acc'][-1]:.3f}   attention {h_attn['val_acc'][-1]:.3f}")
    for sentence in ["I am happy.", "Thank you very much.", "Where is my book?"]:
        print(f"  {sentence:24s} -> plain: {translate(plain, sentence, en, es):28s} attention: {translate(attn, sentence, en, es)}")

    # ---- 3. Alignment ----
    sentence = "Thank you very much."
    out, weights = attention_map(attn, sentence, en, es)
    src_words = clean_text(sentence).split()
    print("\nattention weights (rows: output words, columns: " + " ".join(src_words) + ")")
    for word, row in zip(out.split(), weights):
        print(f"  {word:10s} " + " ".join(f"{v:.2f}" for v in row))

    # ---- 4. Ship ----
    path = os.path.join(tempfile.gettempdir(), "en_es_translator.pt")
    save_translator(attn, en, es, {"attention": True, "embed": 64, "hidden": 128}, path)
    model, en2, es2 = load_translator(path)
    print(f"\nreloaded translator: 'good morning' -> {translate(model, 'Good morning!', en2, es2)}")


if __name__ == "__main__":
    main()

