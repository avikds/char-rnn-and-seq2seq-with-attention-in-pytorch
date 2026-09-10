# Char-RNN and Seq2Seq with Attention in PyTorch

Chapter 14 of Hands-On Machine Learning, built on real text. First a character-level language model on the complete works of Shakespeare: vocabulary, shifted windows, a GRU that predicts the next character, and temperature sampling that writes new lines. Then neural machine translation on 119k English-Spanish sentence pairs: word vocabularies, an encoder-decoder trained with teacher forcing, greedy decoding, and a Luong attention decoder whose weights you can read as an alignment map. Save the translator at the end.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_shakespeare
- [x] **2.** CharVocab
- [x] **3.** make_char_windows
- [x] **4.** CharRNN
- [x] **5.** train_char_rnn
- [x] **6.** generate
- [x] **7.** load_spa_eng
- [x] **8.** WordVocab
- [x] **9.** encode_pairs
- [x] **10.** Encoder
- [x] **11.** Seq2Seq
- [x] **12.** train_translator
- [x] **13.** translate
- [x] **14.** LuongAttention
- [ ] **15.** AttnDecoder
- [ ] **16.** attention_map
- [ ] **17.** compare_translators
- [ ] **18.** save_translator

---

Built on Deep-ML.
