"""
Char-RNN and Seq2Seq with Attention in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_shakespeare
import os
import tempfile
import urllib.request

def load_shakespeare():
    # Cache the corpus in the system temporary directory.
    cache_path = os.path.join(tempfile.gettempdir(), "tinyshakespeare.txt")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    # Download only if the cached file does not already exist.
    if not os.path.isfile(cache_path):
        urllib.request.urlretrieve(url, cache_path)

    # Read and return the complete corpus as a single UTF-8 string.
    with open(cache_path, "r", encoding="utf-8") as f:
        return f.read()

# Step 2 - CharVocab
class CharVocab:
    def __init__(self, text):
        # Sorted list of all distinct characters in the corpus.
        self.chars = sorted(set(text))

        # Character-to-integer and integer-to-character mappings.
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

# Step 3 - make_char_windows
import torch

def make_char_windows(ids, window, stride):
    # Create shifted input/target windows.
    starts = range(0, len(ids) - window, stride)

    X = [ids[s:s + window] for s in starts]
    Y = [ids[s + 1:s + window + 1] for s in range(0, len(ids) - window, stride)]

    # Return int64 tensors with shape (n, window).
    return torch.tensor(X, dtype=torch.int64), torch.tensor(Y, dtype=torch.int64)

# Step 4 - CharRNN
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed=16, hidden=128):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed)
        self.rnn = nn.GRU(embed, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        # Convert character IDs to embeddings.
        x = self.embed(x)

        # Run the GRU, optionally continuing from the supplied state.
        x, state = self.rnn(x, state)

        # Project each hidden state to vocabulary logits.
        logits = self.head(x)

        return logits, state

# Step 5 - train_char_rnn
import torch
import torch.nn.functional as F

def train_char_rnn(model, X, Y, epochs=3, lr=0.005, batch_size=32, seed=42):
    torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    model.train()

    for _ in range(epochs):
        perm = torch.randperm(X.size(0))
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, X.size(0), batch_size):
            idx = perm[start:start + batch_size]
            xb = X[idx]
            yb = Y[idx]

            optimizer.zero_grad()

            logits, _ = model(xb)

            # Flatten (batch, T, vocab) -> (batch*T, vocab)
            # and (batch, T) -> (batch*T).
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        losses.append(epoch_loss / num_batches)

    return losses

# Step 6 - generate
import torch

def generate(model, vocab, prefix, n_chars, temperature=1.0, seed=0):
    torch.manual_seed(seed)

    model.eval()

    with torch.no_grad():
        # Encode and process the entire prefix first.
        ids = vocab.encode(prefix)
        x = torch.tensor([ids], dtype=torch.int64)

        logits, state = model(x)

        generated = []

        # The last position's logits are used to predict the next character.
        next_logits = logits[:, -1, :]

        for _ in range(n_chars):
            if temperature == 0:
                next_id = torch.argmax(next_logits, dim=-1)
            else:
                scaled_logits = next_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(1)

            generated.append(next_id.item())

            # Feed the sampled character back into the GRU while carrying state.
            next_input = next_id.unsqueeze(1)
            logits, state = model(next_input, state)
            next_logits = logits[:, -1, :]

        return prefix + vocab.decode(generated)

# Step 7 - load_spa_eng
import os
import re
import tempfile
import urllib.request
import zipfile
import numpy as np

def clean_text(s):
    # Lowercase the text.
    s = s.lower()

    # Replace every character that is not a word character,
    # whitespace, or apostrophe with a space.
    s = re.sub(r"[^\w\s']", " ", s)

    # Collapse consecutive whitespace and strip leading/trailing spaces.
    return " ".join(s.split())

def load_spa_eng(n_pairs=20000, max_words=8, seed=42):
    url = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
    temp_dir = tempfile.gettempdir()
    zip_path = os.path.join(temp_dir, "spa-eng.zip")

    # Download the dataset only if it is not already cached.
    if not os.path.isfile(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    pairs = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("spa-eng/spa.txt") as f:
            for line in f:
                line = line.decode("utf-8").strip()

                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) != 2:
                    continue

                english = clean_text(parts[0])
                spanish = clean_text(parts[1])

                # Keep only pairs whose two sides satisfy the word limit.
                if (
                    len(english.split()) <= max_words
                    and len(spanish.split()) <= max_words
                ):
                    pairs.append((english, spanish))

    # Shuffle deterministically and return the requested number of pairs.
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs))

    return [pairs[i] for i in indices[:n_pairs]]

# Step 8 - WordVocab
from collections import Counter

class WordVocab:
    def __init__(self, sentences, max_size=1000):
        # Special tokens occupy the first four indices.
        self.itos = ["<pad>", "<unk>", "<sos>", "<eos>"]

        # Count words across all sentences. Counter preserves first-occurrence
        # order when frequencies are tied.
        counter = Counter()
        for sentence in sentences:
            counter.update(sentence.split(" "))

        # Fill the remaining vocabulary slots with the most frequent words.
        remaining = max(0, max_size - len(self.itos))
        self.itos.extend(
            word
            for word, _ in counter.most_common(remaining)
            if word not in self.itos
        )

        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, sentence, max_len, sos=False, eos=False):
        ids = []

        if sos:
            ids.append(self.stoi["<sos>"])

        ids.extend(
            self.stoi.get(word, self.stoi["<unk>"])
            for word in sentence.split(" ")
            if word
        )

        if eos:
            ids.append(self.stoi["<eos>"])

        # Truncate first, then right-pad to exactly max_len.
        ids = ids[:max_len]
        ids.extend([self.stoi["<pad>"]] * (max_len - len(ids)))

        return ids

    def decode(self, ids):
        words = []

        for idx in ids:
            word = self.itos[idx]

            if word == "<eos>":
                break

            if word in ("<pad>", "<sos>"):
                continue

            words.append(word)

        return " ".join(words)

# Step 9 - encode_pairs
def encode_pairs(pairs, src_vocab, tgt_vocab, max_len=10):
    src = []
    tgt_in = []
    tgt_out = []

    for english, spanish in pairs:
        # Source: English words, without special tokens.
        src.append(src_vocab.encode(english, max_len))

        # Target input: <sos> followed by Spanish words.
        tgt_in.append(tgt_vocab.encode(spanish, max_len, sos=True))

        # Target output: Spanish words followed by <eos>.
        tgt_out.append(tgt_vocab.encode(spanish, max_len, eos=True))

    return (
        torch.tensor(src, dtype=torch.int64),
        torch.tensor(tgt_in, dtype=torch.int64),
        torch.tensor(tgt_out, dtype=torch.int64),
    )

# Step 10 - Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed=64, hidden=128):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.rnn = nn.GRU(embed, hidden, batch_first=True)

    def forward(self, src):
        x = self.embed(src)
        outputs, state = self.rnn(x)

        return outputs, state


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed=64, hidden=128):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.rnn = nn.GRU(embed, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, tgt_in, state, enc_outputs=None, src_mask=None):
        x = self.embed(tgt_in)
        x, state = self.rnn(x, state)
        logits = self.head(x)

        return logits, state

# Step 11 - Seq2Seq (not yet solved)
# TODO: implement

# Step 12 - train_translator (not yet solved)
# TODO: implement

# Step 13 - translate (not yet solved)
# TODO: implement

# Step 14 - LuongAttention (not yet solved)
# TODO: implement

# Step 15 - AttnDecoder (not yet solved)
# TODO: implement

# Step 16 - attention_map (not yet solved)
# TODO: implement

# Step 17 - compare_translators (not yet solved)
# TODO: implement

# Step 18 - save_translator (not yet solved)
# TODO: implement

