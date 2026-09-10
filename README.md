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
- [ ] **5.** train_char_rnn
- [ ] **6.** generate
- [ ] **7.** load_spa_eng
- [ ] **8.** WordVocab
- [ ] **9.** encode_pairs
- [ ] **10.** Encoder
- [ ] **11.** Seq2Seq
- [ ] **12.** train_translator
- [ ] **13.** translate
- [ ] **14.** LuongAttention
- [ ] **15.** AttnDecoder
- [ ] **16.** attention_map
- [ ] **17.** compare_translators
- [ ] **18.** save_translator

---

Built on Deep-ML.
