"""
Persian Handwritten Digits (GAN-generated, 28x28, 150k images) - CPU-friendly CNN
Dataset: https://www.kaggle.com/datasets/amirmahdiabbootalebi/persian-handwritten-digits
"""
import os
import sys
import time
import math
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

DATA_DIR = r""  
CACHE_FILE = "persian_digits_cache.npz"
EPOCHS = 12              
BATCH_SIZE = 256
LR = 3e-3
WEIGHT_DECAY = 5e-3
LABEL_SMOOTHING = 0.05
EMA_DECAY = 0.999
WIDTH = 32                
SEED = 42
VAL_FRAC, TEST_FRAC = 0.10, 0.10

torch.set_num_threads(os.cpu_count() or 4)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cpu")
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def get_data_dir():
    if DATA_DIR and Path(DATA_DIR).exists():
        return DATA_DIR
    try:
        import kagglehub
        path = kagglehub.dataset_download("amirmahdiabbootalebi/persian-handwritten-digits")
        print("Dataset downloaded to:", path)
        return path
    except Exception as e:
        sys.exit(f"Dataset not found. Set DATA_DIR manually. ({e})")


def load_one(p):
    im = Image.open(p).convert("L")
    if im.size != (28, 28):
        im = im.resize((28, 28), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def load_dataset():
    if Path(CACHE_FILE).exists():
        d = np.load(CACHE_FILE)
        print("Loaded cache:", d["X"].shape)
        return d["X"], d["y"]

    root = get_data_dir()
    files = [p for p in Path(root).rglob("*") if p.suffix.lower() in EXTS]
    if not files:
        sys.exit(f"No image files found under {root}")
    names = sorted({p.parent.name for p in files})
    print("Class folders found:", names)
    if len(names) != 10:
        print("WARNING: expected 10 class folders. Check the folder structure above.")
    cls_idx = {n: i for i, n in enumerate(names)}

    print(f"Reading {len(files)} images ...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        X = np.stack(list(ex.map(load_one, files, chunksize=512)))
    y = np.array([cls_idx[p.parent.name] for p in files], dtype=np.int64)
    print(f"Done in {time.time() - t0:.0f}s, X={X.shape}")
    np.savez_compressed(CACHE_FILE, X=X, y=y)
    return X, y


def stratified_split(y, val_frac, test_frac, seed):
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_te, n_va = int(len(idx) * test_frac), int(len(idx) * val_frac)
        te += idx[:n_te].tolist()
        va += idx[n_te:n_te + n_va].tolist()
        tr += idx[n_te + n_va:].tolist()
    return np.array(tr), np.array(va), np.array(te)


def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Net(nn.Module):
    def __init__(self, w=32, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            cbr(1, w), cbr(w, w), nn.MaxPool2d(2),            # 28 -> 14
            cbr(w, 2 * w), cbr(2 * w, 2 * w), nn.MaxPool2d(2),  # 14 -> 7
            cbr(2 * w, 4 * w), cbr(4 * w, 4 * w),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(4 * w, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


MEAN, STD = 0.1307, 0.3081 


def to_float(xb_uint8):
    return xb_uint8.float().div_(255.0).unsqueeze(1)


def normalize(x):
    return (x - MEAN) / STD


def augment(x):
    B = x.size(0)
    ang = (torch.rand(B) * 2 - 1) * math.radians(10)
    sc = 1 + (torch.rand(B) * 2 - 1) * 0.10
    tx = (torch.rand(B) * 2 - 1) * 0.12
    ty = (torch.rand(B) * 2 - 1) * 0.12
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


@torch.no_grad()
def predict(model, X, bs=1024):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(normalize(to_float(X[i:i + bs]))).argmax(1))
    return torch.cat(out)


def accuracy(model, X, y):
    return (predict(model, X) == y).float().mean().item()


def main():
    global MEAN, STD
    X, y = load_dataset()

    if X.mean() > 127:
        X = 255 - X
        print("Inverted images (background -> 0).")

    tr, va, te = stratified_split(y, VAL_FRAC, TEST_FRAC, SEED)
    Xtr, ytr = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
    Xva, yva = torch.from_numpy(X[va]), torch.from_numpy(y[va])
    Xte, yte = torch.from_numpy(X[te]), torch.from_numpy(y[te])
    print(f"train={len(tr)}  val={len(va)}  test={len(te)}")

    f = to_float(Xtr[:20000])
    MEAN, STD = f.mean().item(), f.std().item()

    model = Net(WIDTH).to(DEVICE)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True)

    steps_per_epoch = math.ceil(len(Xtr) / BATCH_SIZE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, pct_start=0.2)
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_val, best_state = 0.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        t0, run_loss, correct = time.time(), 0.0, 0
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = normalize(augment(to_float(Xtr[idx])))
            yb = ytr[idx]
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            sched.step()
            ema.update_parameters(model)
            run_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()

        val_acc = accuracy(ema.module, Xva, yva)
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in ema.module.state_dict().items()}
        print(f"Epoch {ep:2d}/{EPOCHS} | loss {run_loss / len(Xtr):.4f} | train acc {correct / len(Xtr):.4f} "
              f"| val acc (EMA) {val_acc:.4f} | {time.time() - t0:.0f}s")

    final = ema.module
    final.load_state_dict(best_state)
    preds = predict(final, Xte)
    test_acc = (preds == yte).float().mean().item()
    print(f"\n=== TEST ACCURACY: {test_acc * 100:.2f}% (best val {best_val * 100:.2f}%) ===")

    cm = torch.zeros(10, 10, dtype=torch.long)
    for t, p in zip(yte, preds):
        cm[t, p] += 1
    print("\nPer-class accuracy:")
    for c in range(10):
        print(f"  class {c}: {cm[c, c].item() / max(cm[c].sum().item(), 1) * 100:.2f}%")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm.numpy())

    torch.save({"state_dict": final.state_dict(), "mean": MEAN, "std": STD, "width": WIDTH}, "persian_digits_cnn.pt")
    print("\nSaved model -> persian_digits_cnn.pt")


if __name__ == "__main__":
    main()
