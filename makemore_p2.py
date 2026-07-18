import torch
import torch.nn.functional as F
import random

# hyperparameters
batch_size = 256
block_size = 5
hidden_size = 300
embedding_size = 40

# dataset

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


g = torch.Generator().manual_seed(2147483647)

# Embedding
C = torch.randn((27, embedding_size), generator=g)
# embedding to hidden layer weights and biases
w1 = torch.randn((embedding_size * block_size, hidden_size), generator=g)
b1 = torch.randn(hidden_size, generator=g)
# hidden to output layer weights and biases
w2 = torch.randn((hidden_size, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, w1, b1, w2, b2]

for p in parameters:
    p.requires_grad = True

print(sum(p.nelement() for p in parameters))  # number of parameters in total

# training loop
for i in range(400000):
    # minibatch contruct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)

    # forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, embedding_size * block_size) @ w1 + b1)
    logits = h @ w2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update parameters
    lr = 0.1 if i < 100000 else 0.05 if i < 200000 else 0.01 if i < 300000 else 0.005
    for p in parameters:
        p.data += -lr * p.grad

    if i % 20_000 == 0:
        print(f"Loss: {loss.item()}")


def loss_against(X, Y):
    # we are using Xdev and Ydev for validation, therefore we will use the validation set for the validation loop.
    emb = C[X]
    h = torch.tanh(emb.view(-1, embedding_size * block_size) @ w1 + b1)
    logits = h@w2 + b2
    loss = F.cross_entropy(logits, Y)
    return loss.item()


print(f"Train loss: {loss_against(Xtr, Ytr)}")
print(f"Dev loss: {loss_against(Xdev, Ydev)}")


# sample from the model
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(-1, embedding_size * block_size) @ w1 + b1)
        logits = h@w2+b2
        prob = F.softmax(logits, dim=1)
        ix = torch.multinomial(prob, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))
