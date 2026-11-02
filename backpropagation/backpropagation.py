"""Backpropagation: code from the ML Zero to Hero explainer.

Video: https://youtu.be/st2Ufh6kNJQ
"""

import numpy as np
rng = np.random.default_rng(0)

# --- toy data: 3-class spiral, 300 points in 2-D (not linearly separable) ---
N, K = 100, 3
X = np.zeros((N * K, 2)); y = np.zeros(N * K, dtype=int)
for k in range(K):
    r = np.linspace(0.0, 1.0, N)
    t = np.linspace(4 * k, 4 * (k + 1), N) + rng.normal(scale=0.2, size=N)
    X[k * N:(k + 1) * N] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[k * N:(k + 1) * N] = k

class Linear:
    def __init__(self, n_in, n_out):
        self.W = rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_out))  # He init
        self.b = np.zeros(n_out)
    def forward(self, x):
        self.x = x                        # cache input for the backward pass
        return x @ self.W + self.b
    def backward(self, g):                # g = dL/d(out), shape (B, n_out)
        self.dW = self.x.T @ g            # same shape as W
        self.db = g.sum(axis=0)
        return g @ self.W.T               # dL/d(in), passed to the layer below

class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask
    def backward(self, g):
        return g * self.mask              # gradient routes only through active units

def softmax_cross_entropy(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)    # stability: shift by max
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    B = len(y)
    loss = -np.log(p[np.arange(B), y]).mean()
    dlogits = p.copy(); dlogits[np.arange(B), y] -= 1; dlogits /= B   # (p - y)/B
    return loss, dlogits

class MLP:
    def __init__(self, sizes):
        self.layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers.append(Linear(a, b))
            if i < len(sizes) - 2:
                self.layers.append(ReLU())
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backward(self, g):
        for layer in reversed(self.layers):
            g = layer.backward(g)
    def params(self):
        for layer in self.layers:
            if isinstance(layer, Linear):
                yield layer.W, layer.dW
                yield layer.b, layer.db

model = MLP([2, 64, 64, 3])

# --- gradient check on a small batch with central differences ---
Xb, yb = X[:20], y[:20]
loss, g = softmax_cross_entropy(model.forward(Xb), yb)
model.backward(g)
W, dW = next(model.params())              # first-layer weights and their analytic grad
num, eps = np.zeros_like(W), 1e-5
for idx in np.ndindex(W.shape):
    old = W[idx]
    W[idx] = old + eps; lp = softmax_cross_entropy(model.forward(Xb), yb)[0]
    W[idx] = old - eps; lm = softmax_cross_entropy(model.forward(Xb), yb)[0]
    W[idx] = old
    num[idx] = (lp - lm) / (2 * eps)
rel = np.abs(num - dW).max() / (np.abs(num).max() + np.abs(dW).max())
print(f"grad-check relative error: {rel:.1e}")   # ~1e-9 good; >1e-4 means a bug

# --- train with full-batch gradient descent ---
lr = 1.0
for step in range(3001):
    loss, g = softmax_cross_entropy(model.forward(X), y)
    model.backward(g)
    for p, dp in model.params():
        p -= lr * dp
    if step % 1000 == 0:
        print(f"step {step:4d}  loss {loss:.4f}")
acc = (model.forward(X).argmax(axis=1) == y).mean()
print(f"train accuracy: {acc:.3f}")      # a linear model gets ~0.5 here

# --------------------------------------------------------------------------

import torch

# 1) Same weights, same batch: autograd gradients must match ours
lin = [l for l in model.layers if isinstance(l, Linear)]
params = [torch.tensor(p, requires_grad=True) for l in lin for p in (l.W, l.b)]
h = torch.tensor(Xb)
for i in range(len(lin)):
    h = h @ params[2 * i] + params[2 * i + 1]
    if i < len(lin) - 1:
        h = torch.relu(h)
torch.nn.functional.cross_entropy(h, torch.tensor(yb)).backward()
_, g = softmax_cross_entropy(model.forward(Xb), yb); model.backward(g)
print("max |ours - autograd|:", max(np.abs(l.dW - params[2 * i].grad.numpy()).max()
                                    for i, l in enumerate(lin)))

# 2) The worked example: 2-2-1 net, ReLU hidden, sigmoid output, BCE loss
x = torch.tensor([1.0, 2.0], dtype=torch.float64)
W1 = torch.tensor([[0.5, -0.4], [0.3, 0.4]], dtype=torch.float64, requires_grad=True)
b1 = torch.tensor([0.0, 0.1], dtype=torch.float64, requires_grad=True)
W2 = torch.tensor([0.6, -0.5], dtype=torch.float64, requires_grad=True)
b2 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
p = torch.sigmoid(W2 @ torch.relu(W1 @ x + b1) + b2)
loss = -torch.log(p)
loss.backward()
print(f"L={loss.item():.5f}  dW2={W2.grad.numpy().round(5)}  dW1=\n{W1.grad.numpy().round(5)}")
