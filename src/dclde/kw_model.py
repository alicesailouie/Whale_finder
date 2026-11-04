"""Small CNN for killer whale detection (DC-1) and ecotype classification (DC-2)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                         nn.MaxPool2d(2))


class SpecCNN(nn.Module):
    def __init__(self, n_out: int, width: int = 24):
        super().__init__()
        w = width
        self.body = nn.Sequential(block(1, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 4 * w))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(8 * w, n_out))

    def forward(self, x):
        h = self.body(x)
        h = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        return self.head(h)


def augment(x: torch.Tensor) -> torch.Tensor:
    """SpecAugment-style masking plus a random circular time shift."""
    x = x.clone()
    B, _, Fq, T = x.shape
    for b in range(B):
        f0, fw = np.random.randint(0, Fq), np.random.randint(0, Fq // 6)
        t0, tw = np.random.randint(0, T), np.random.randint(0, T // 6)
        x[b, :, f0:f0 + fw] = 0
        x[b, :, :, t0:t0 + tw] = 0
        x[b] = torch.roll(x[b], np.random.randint(-T // 4, T // 4), dims=-1)
    return x


def _standardise(X):
    X = torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)
    return X / 4.0  # log-power after median removal is roughly within +-4


def train(X, y, n_out, epochs=12, batch=64, lr=2e-3, seed=0, class_weight=None, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt, yt = _standardise(X), torch.as_tensor(y, dtype=torch.long)
    model = SpecCNN(n_out)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    steps = epochs * int(np.ceil(len(yt) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    cw = None if class_weight is None else torch.as_tensor(class_weight, dtype=torch.float32)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(yt))
        tot = 0.0
        for i in range(0, len(yt), batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(model(augment(Xt[idx])), yt[idx], weight=cw, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        log(f"  epoch {ep + 1}/{epochs} loss {tot / len(yt):.3f}")
    return model


@torch.no_grad()
def predict_proba(model, X, batch=256):
    model.eval()
    Xt = _standardise(X)
    out = [F.softmax(model(Xt[i:i + batch]), 1) for i in range(0, len(Xt), batch)]
    return torch.cat(out).numpy()
