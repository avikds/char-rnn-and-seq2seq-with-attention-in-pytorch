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
- [x] **15.** AttnDecoder
- [x] **16.** attention_map
- [x] **17.** compare_translators
- [x] **18.** save_translator

## Results

```
char-rnn on 150,000 characters: loss 2.91 -> 2.24 (uniform guess = 4.17)
  T=0.2: ROMEO:IUS: / The sore the the the sore the the sore the sore and and the sore the the th
  T=0.6: ROMEO: / Carest here the cave, the the on that shos with and freear the cemot thelion, a
  T=1.2: ROMEO: / Caw;sul; / Menter uagl,', pave you af cemans win's, vrave arvic heer, / Hrail. O, a

translator masked val accuracy: plain 0.497   attention 0.573
  I am happy.              -> plain: estoy <unk>                  attention: estoy feliz
  Thank you very much.     -> plain: gracias por <unk>            attention: gracias gracias
  Where is my book?        -> plain: dónde está mi oficina        attention: dónde es mi libro

attention weights (rows: output words, columns: thank you very much)
  gracias    0.01 0.53 0.47 0.00
  gracias    0.00 0.00 1.00 0.00

reloaded translator: 'good morning' -> bueno bien bueno
```
