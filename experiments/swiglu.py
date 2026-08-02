# Just implementation and an example usage (very bad usecae lol) of swiglu
import torch
import torch.nn as nn
import random

# hyperparams
block_size = 8
epochs = 30000
batch_size = 64
n_embd = 100

# dataset

words = open('makemore/names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

vocab_size = len(itos)


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y


random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# defining the model


class Swish(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return torch.mul(x, torch.sigmoid(x * self.beta))


# Defining the SwiGLU module
class SwiGLU(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.in_size = in_size
        self.linear = nn.Linear(in_size, 2*in_size)
        self.swish = Swish()

    def forward(self, x):
        out = self.linear(x)
        a, b = out[..., :self.in_size], out[..., self.in_size:]
        return a * self.swish(b)


class BigramModel(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.fc = SwiGLU(n_embd)
        self.out_layer = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        # x = x.view(x.shape[0], )
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.out_layer(x)
        return x


# single glu model

model = BigramModel()

# training the model as a bigram model
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(epochs):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))

    logits = model(Xtr[ix])
    loss = nn.functional.cross_entropy(logits, Ytr[ix])

    optim.zero_grad(set_to_none=True)

    loss.backward()
    optim.step()

    if epoch % 200 == 0:
        print(f"step {epoch}, loss {loss.item()}")


# sample
sample_size = 10
for _ in range(sample_size):
    out = []
    context = [0] * block_size
    while True:
        logits = model(torch.tensor([context], dtype=torch.long))
        prob = nn.functional.softmax(logits, dim=-1)
        ix = torch.multinomial(prob, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))
