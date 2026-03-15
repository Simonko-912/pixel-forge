#!/usr/bin/env python3
"""
Pixel·Trainer CLI — Python version of the browser Pixel·Trainer
All four models: Classifier MLP, MLP Generator, DDPM, GAN
Loads Pixel·Forge ZIP datasets or HuggingFace repos
No deep learning framework required — pure numpy

Usage:
  python pixeltrainer.py train-classifier  --data dataset.zip
  python pixeltrainer.py train-mlp-gen     --data dataset.zip
  python pixeltrainer.py train-ddpm        --data dataset.zip
  python pixeltrainer.py train-gan         --data dataset.zip
  python pixeltrainer.py generate          --model model.json --type ddpm --class "cat" --n 16
  python pixeltrainer.py classify          --model model.json --image test.png
  python pixeltrainer.py info              --model model.json
"""

import sys
import os
import json
import zipfile
import argparse
import time
import math
import random
import struct
import io
import urllib.request
import urllib.error
from pathlib import Path

# ── Optional deps ──────────────────────────────────────────────────────────────
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("ERROR: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ── Terminal colours ───────────────────────────────────────────────────────────
class C:
    CYAN   = '\033[96m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    PURPLE = '\033[95m'
    BOLD   = '\033[1m'
    DIM    = '\033[2m'
    RESET  = '\033[0m'

def cprint(colour, *args, **kwargs):
    print(colour + ' '.join(str(a) for a in args) + C.RESET, **kwargs)

def progress_bar(current, total, width=40, prefix='', suffix=''):
    pct = current / max(1, total)
    filled = int(pct * width)
    bar = '█' * filled + '░' * (width - filled)
    print(f'\r{prefix} [{bar}] {pct*100:.1f}% {suffix}', end='', flush=True)

def progress_end():
    print()

# ══════════════════════════════════════════════════════════════════════════════
#  ACTIVATION FUNCTIONS  (all vectorised with numpy)
# ══════════════════════════════════════════════════════════════════════════════
def relu(x):     return np.maximum(0, x)
def relu_d(x):   return (x > 0).astype(np.float32)
def lrelu(x):    return np.where(x > 0, x, 0.01 * x)
def lrelu_d(x):  return np.where(x > 0, 1.0, 0.01).astype(np.float32)
def elu(x):      return np.where(x >= 0, x, np.exp(np.clip(x, -80, 0)) - 1)
def elu_d(x):    return np.where(x >= 0, 1.0, np.exp(np.clip(x, -80, 0))).astype(np.float32)
def tanh_a(x):   return np.tanh(x)
def tanh_d(x):   t = np.tanh(x); return 1.0 - t * t
def sigmoid(x):  return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))
def sigmoid_d(x):s = sigmoid(x); return s * (1.0 - s)
def silu(x):     return x * sigmoid(x)
def silu_d(x):   s = sigmoid(x); return s * (1.0 + x * (1.0 - s))
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

ACT_FN  = {'relu': relu,  'lrelu': lrelu,  'elu': elu,  'tanh': tanh_a,
           'sigmoid': sigmoid, 'silu': silu,  'linear': lambda x: x}
ACT_D   = {'relu': relu_d,'lrelu': lrelu_d,'elu': elu_d,'tanh': tanh_d,
           'sigmoid': sigmoid_d, 'silu': silu_d, 'linear': lambda x: np.ones_like(x)}

# ══════════════════════════════════════════════════════════════════════════════
#  ADAM OPTIMIZER STATE
# ══════════════════════════════════════════════════════════════════════════════
class AdamState:
    def __init__(self, shape, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=1e-4):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m  = np.zeros(shape, np.float32)
        self.v  = np.zeros(shape, np.float32)
        self.t  = 0

    def step(self, param, grad):
        self.t += 1
        g = grad + self.wd * param
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        param -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return param

# ══════════════════════════════════════════════════════════════════════════════
#  DENSE LAYER
# ══════════════════════════════════════════════════════════════════════════════
class Dense:
    def __init__(self, in_size, out_size, act='relu', lr=1e-3, wd=1e-4):
        scale = math.sqrt(2.0 / in_size)
        self.W = (np.random.randn(in_size, out_size) * scale).astype(np.float32)
        self.b = np.zeros(out_size, np.float32)
        self.act = act
        self.W_opt = AdamState(self.W.shape, lr, wd=wd)
        self.b_opt = AdamState(self.b.shape, lr, wd=0)
        self._cache = {}

    def forward(self, x, training=True):
        z = x @ self.W + self.b          # (batch, out) or (out,) for single
        self._cache['x'] = x
        self._cache['z'] = z
        if self.act == 'softmax':
            return softmax(z)
        return ACT_FN[self.act](z)

    def backward(self, dA):
        z = self._cache['z']
        x = self._cache['x']
        if self.act == 'softmax':
            dZ = dA                      # caller passes dL/dlogit directly
        elif self.act == 'linear':
            dZ = dA
        else:
            dZ = dA * ACT_D[self.act](z)
        # grads
        dW = x.T @ dZ if x.ndim == 2 else np.outer(x, dZ)
        db = dZ.sum(axis=0) if dZ.ndim == 2 else dZ
        dX = dZ @ self.W.T
        # update
        self.W = self.W_opt.step(self.W, dW)
        self.b = self.b_opt.step(self.b, db)
        return dX

    def to_dict(self):
        return {'inSize': self.W.shape[0], 'outSize': self.W.shape[1],
                'act': self.act,
                'W': self.W.tolist(), 'b': self.b.tolist()}

    @classmethod
    def from_dict(cls, d, lr=1e-3):
        layer = cls.__new__(cls)
        layer.W = np.array(d['W'], dtype=np.float32)
        layer.b = np.array(d['b'], dtype=np.float32)
        layer.act = d['act']
        layer.W_opt = AdamState(layer.W.shape, lr)
        layer.b_opt = AdamState(layer.b.shape, lr)
        layer._cache = {}
        return layer

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADING
# ══════════════════════════════════════════════════════════════════════════════
def decode_image_bytes(data, target_w, target_h, channels):
    """Decode image bytes (PNG/JPG) to float32 array. Uses PIL if available, else minimal PNG."""
    if HAS_PIL:
        img = Image.open(io.BytesIO(data)).convert('RGB')
        img = img.resize((target_w, target_h), Image.NEAREST)
        arr = np.array(img, dtype=np.float32) / 255.0   # (H,W,3)
        if channels == 1:
            arr = (arr[...,0]*0.299 + arr[...,1]*0.587 + arr[...,2]*0.114)
            return arr.reshape(-1)                        # (H*W,)
        return arr.reshape(-1)                            # (H*W*3,)
    else:
        return decode_png_minimal(data, target_w, target_h, channels)

def decode_png_minimal(data, target_w, target_h, channels):
    """Minimal PNG decoder using only stdlib — supports 8-bit RGB and RGBA."""
    import zlib
    def read_chunk(f):
        length = struct.unpack('>I', f.read(4))[0]
        ctype  = f.read(4)
        cdata  = f.read(length)
        f.read(4)  # crc
        return ctype, cdata

    f = io.BytesIO(data)
    sig = f.read(8)
    if sig != b'\x89PNG\r\n\x1a\n':
        raise ValueError("Not a PNG file")
    _, ihdr = read_chunk(f)
    w = struct.unpack('>I', ihdr[0:4])[0]
    h = struct.unpack('>I', ihdr[4:8])[0]
    bpp = ihdr[8]
    colour_type = ihdr[9]
    if bpp != 8:
        raise ValueError(f"Unsupported PNG bit depth: {bpp}")
    nch = {0:1, 2:3, 3:3, 4:2, 6:4}.get(colour_type, 3)
    idat = b''
    while True:
        try:
            ct, cd = read_chunk(f)
        except Exception:
            break
        if ct == b'IDAT': idat += cd
        elif ct == b'IEND': break
    raw = zlib.decompress(idat)
    stride = w * nch + 1
    pixels = np.zeros((h, w, 3), dtype=np.uint8)
    prev = np.zeros(w * nch, dtype=np.int32)
    for row in range(h):
        ftype = raw[row * stride]
        line  = np.frombuffer(raw[row*stride+1:(row+1)*stride], dtype=np.uint8).astype(np.int32)
        if ftype == 0: pass
        elif ftype == 1:
            for i in range(nch, len(line)): line[i] = (line[i] + line[i-nch]) & 0xFF
        elif ftype == 2: line = ((line + prev) & 0xFF)
        elif ftype == 3:
            p2 = np.zeros_like(line)
            for i in range(len(line)):
                a = line[i-nch] if i >= nch else 0
                b = prev[i]
                line[i] = (line[i] + (a + b) // 2) & 0xFF
        elif ftype == 4:
            p3 = np.zeros_like(line)
            for i in range(len(line)):
                a = line[i-nch] if i >= nch else 0
                b = prev[i]; c = prev[i-nch] if i >= nch else 0
                pa = abs(b-c); pb = abs(a-c); pc = abs(a+b-2*c)
                pr = a if pa<=pb and pa<=pc else (b if pb<=pc else c)
                line[i] = (line[i] + pr) & 0xFF
        prev = line
        if nch >= 3:
            pixels[row,:,0] = line[0::nch][:w]
            pixels[row,:,1] = line[1::nch][:w]
            pixels[row,:,2] = line[2::nch][:w]
        else:
            for c in range(3): pixels[row,:,c] = line[0::nch][:w]
    # Simple resize via repeat (nearest)
    if (w, h) != (target_w, target_h):
        ry = np.linspace(0, h-1, target_h).astype(int)
        rx = np.linspace(0, w-1, target_w).astype(int)
        pixels = pixels[ry][:,rx]
    arr = pixels.astype(np.float32) / 255.0
    if channels == 1:
        return (arr[...,0]*0.299 + arr[...,1]*0.587 + arr[...,2]*0.114).reshape(-1)
    return arr.reshape(-1)

def load_zip_dataset(zip_path, channels=3, max_images=500):
    """Load a Pixel·Forge ZIP dataset. Returns (images, labels, classes, W, H)."""
    cprint(C.CYAN, f"› Loading ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        img_files  = [n for n in names if not zf.getinfo(n).is_dir()
                      and n.lower().endswith(('.png','.jpg','.jpeg','.webp'))]
        meta_files = [n for n in names if n.endswith('metadata.jsonl')]
        cprint(C.DIM, f"  {len(img_files)} images, {len(meta_files)} metadata files")

        # Parse all metadata.jsonl
        meta_map = {}
        for mf in meta_files:
            folder = mf[:mf.rfind('/')+1] if '/' in mf else ''
            text = zf.read(mf).decode('utf-8')
            for line in text.strip().split('\n'):
                if not line.strip(): continue
                try:
                    j = json.loads(line)
                    fname = j.get('file_name', '')
                    if fname:
                        meta_map[fname]           = j
                        meta_map[folder+fname]    = j
                        meta_map[mf.replace('metadata.jsonl','')+fname] = j
                except Exception:
                    pass

        # Detect image size from first image
        test_data = zf.read(img_files[0])
        if HAS_PIL:
            test_img = Image.open(io.BytesIO(test_data))
            ds_w, ds_h = min(1024, test_img.width), min(1024, test_img.height)
        else:
            try:
                f = io.BytesIO(test_data)
                f.read(8)  # PNG sig
                f.read(4)  # IHDR length
                f.read(4)  # IHDR type
                ds_w = struct.unpack('>I', f.read(4))[0]
                ds_h = struct.unpack('>I', f.read(4))[0]
                ds_w, ds_h = min(1024, ds_w), min(1024, ds_h)
            except Exception:
                ds_w = ds_h = 32
        cprint(C.DIM, f"  Image size: {ds_w}×{ds_h}")
        if ds_w > 128 or ds_h > 128:
            cprint(C.YELLOW, f"  ⚠ Large images ({ds_w}×{ds_h}) — training will be slower")

        raw_images, raw_labels = [], []
        to_load = img_files[:max_images]
        skipped = 0
        for i, path in enumerate(to_load):
            progress_bar(i+1, len(to_load), prefix='  Loading', suffix=f'{i+1}/{len(to_load)}')
            parts = path.replace('\\','/').split('/')
            fname = parts[-1]
            split = parts[0] if len(parts) > 1 else ''
            meta  = meta_map.get(path) or meta_map.get(fname) or meta_map.get(split+'/'+fname)
            label = meta['label'] if meta and 'label' in meta else (split if split else 'unknown')
            try:
                data = zf.read(path)
                arr  = decode_image_bytes(data, ds_w, ds_h, channels)
                raw_images.append(arr)
                raw_labels.append(label)
            except Exception as e:
                skipped += 1
        progress_end()

        if skipped:
            cprint(C.YELLOW, f"  Skipped {skipped} images (decode errors)")

    classes = sorted(set(raw_labels))
    label_ids = [classes.index(l) for l in raw_labels]
    images = np.array(raw_images, dtype=np.float32)
    labels = np.array(label_ids, dtype=np.int32)
    cprint(C.GREEN, f"  Loaded {len(images)} images | {len(classes)} classes: {classes} | {ds_w}×{ds_h}")
    return images, labels, classes, ds_w, ds_h

def load_hf_dataset(repo, channels=3, max_images=500):
    """Load dataset from HuggingFace datasets API."""
    cprint(C.CYAN, f"› Fetching HuggingFace dataset: {repo}")
    base = f"https://huggingface.co/api/datasets/{repo}/tree/main"
    try:
        with urllib.request.urlopen(base, timeout=15) as r:
            tree = json.loads(r.read())
    except Exception as e:
        raise RuntimeError(f"Cannot fetch repo '{repo}': {e}")

    img_files  = [f for f in tree if f['type']=='file' and
                  f['path'].lower().endswith(('.png','.jpg','.jpeg','.webp'))]
    meta_files = [f for f in tree if f['type']=='file' and f['path'].endswith('metadata.jsonl')]
    cprint(C.DIM, f"  Found {len(img_files)} images")

    meta_map = {}
    for mf in meta_files:
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{mf['path']}"
        with urllib.request.urlopen(url, timeout=15) as r:
            text = r.read().decode('utf-8')
        split = mf['path'].split('/')[0]
        for line in text.strip().split('\n'):
            try:
                j = json.loads(line)
                fname = j.get('file_name','')
                if fname:
                    meta_map[fname] = j
                    meta_map[split+'/'+fname] = j
            except Exception:
                pass

    to_load = img_files[:max_images]
    raw_images, raw_labels = [], []
    ds_w = ds_h = None
    for i, f in enumerate(to_load):
        progress_bar(i+1, len(to_load), prefix='  Downloading', suffix=f'{i+1}/{len(to_load)}')
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{f['path']}"
        parts = f['path'].split('/')
        fname = parts[-1]; split = parts[0] if len(parts) > 1 else ''
        meta  = meta_map.get(fname) or meta_map.get(split+'/'+fname)
        label = meta['label'] if meta and 'label' in meta else split
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = r.read()
            if ds_w is None:
                if HAS_PIL:
                    img0 = Image.open(io.BytesIO(data))
                    ds_w, ds_h = min(1024, img0.width), min(1024, img0.height)
                else:
                    ds_w = ds_h = 32
            arr = decode_image_bytes(data, ds_w, ds_h, channels)
            raw_images.append(arr); raw_labels.append(label)
        except Exception:
            pass
    progress_end()

    ds_w = ds_w or 32; ds_h = ds_h or 32
    classes    = sorted(set(raw_labels))
    label_ids  = [classes.index(l) for l in raw_labels]
    images     = np.array(raw_images, dtype=np.float32)
    labels     = np.array(label_ids,  dtype=np.int32)
    cprint(C.GREEN, f"  Loaded {len(images)} images | {len(classes)} classes | {ds_w}×{ds_h}")
    return images, labels, classes, ds_w, ds_h

def load_dataset(args):
    """Dispatch to ZIP or HF loader based on args."""
    channels = 1 if getattr(args, 'grayscale', False) else 3
    if args.data.startswith('http') or ('/' in args.data and not os.path.exists(args.data)):
        return load_hf_dataset(args.data, channels, getattr(args, 'max_images', 500))
    return load_zip_dataset(args.data, channels, getattr(args, 'max_images', 500))

def split_dataset(images, labels, val_split=0.15, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(images)
    idx = rng.permutation(n)
    split = int(n * (1 - val_split))
    return idx[:split], idx[split:]

# ══════════════════════════════════════════════════════════════════════════════
#  DATA AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
def augment_image(img, ds_h, ds_w, ch, flip_h=True, flip_v=False, noise=True):
    """img: flat float32 array of size ds_h*ds_w*ch (RGB stored, grayscale converted at load)."""
    arr = img.reshape(ds_h, ds_w, ch if ch == 3 else 1)
    if flip_h and random.random() < 0.5:
        arr = arr[:, ::-1, :]
    if flip_v and random.random() < 0.5:
        arr = arr[::-1, :, :]
    if noise:
        arr = np.clip(arr + np.random.uniform(-0.02, 0.02, arr.shape).astype(np.float32), 0, 1)
    return arr.reshape(-1)

# ══════════════════════════════════════════════════════════════════════════════
#  MLP CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
class MLPClassifier:
    def __init__(self, input_size, hidden_sizes, n_classes, act='relu', lr=1e-3, dropout=0.3, wd=1e-4):
        self.layers = []
        prev = input_size
        for h in hidden_sizes:
            self.layers.append(Dense(prev, h, act, lr, wd))
            prev = h
        self.layers.append(Dense(prev, n_classes, 'softmax', lr, wd=0))
        self.dropout = dropout
        self.n_classes = n_classes

    def forward(self, x, training=True):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer.forward(x, training)
            if training and self.dropout > 0:
                mask = (np.random.rand(*x.shape) > self.dropout).astype(np.float32) / (1 - self.dropout)
                x = x * mask
        return self.layers[-1].forward(x, training)

    def backward(self, probs, label):
        dA = probs.copy()
        dA[label] -= 1.0
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def predict(self, x):
        return self.forward(x, training=False)

    def train_epoch(self, images, labels, train_idx, batch_size, ds_h, ds_w, ch,
                    flip_h, flip_v, aug_noise):
        rng_idx = train_idx.copy()
        np.random.shuffle(rng_idx)
        total_batches = math.ceil(len(rng_idx) / batch_size)
        epoch_loss, epoch_acc, nb = 0.0, 0.0, 0

        for bi in range(total_batches):
            batch = rng_idx[bi*batch_size:(bi+1)*batch_size]
            bloss, bacc = 0.0, 0.0
            for idx in batch:
                inp   = augment_image(images[idx], ds_h, ds_w, ch, flip_h, flip_v, aug_noise)
                probs = self.forward(inp, training=True)
                label = labels[idx]
                bloss += -math.log(max(float(probs[label]), 1e-15))
                if int(np.argmax(probs)) == label: bacc += 1
                self.backward(probs, label)
            bloss /= len(batch); bacc /= len(batch)
            epoch_loss += bloss; epoch_acc += bacc; nb += 1
            yield bi+1, total_batches, bloss, bacc

        return epoch_loss/nb, epoch_acc/nb

    def validate(self, images, labels, val_idx, ds_h, ds_w, ch):
        vloss, vacc = 0.0, 0.0
        for idx in val_idx:
            inp   = images[idx]   # no augmentation for val
            probs = self.predict(inp)
            label = labels[idx]
            vloss += -math.log(max(float(probs[label]), 1e-15))
            if int(np.argmax(probs)) == label: vacc += 1
        n = max(1, len(val_idx))
        return vloss/n, vacc/n

    def to_dict(self):
        return {'type': 'classifier', 'weights': [l.to_dict() for l in self.layers]}

    @classmethod
    def from_dict(cls, d, lr=1e-3):
        m = cls.__new__(cls)
        m.layers = [Dense.from_dict(l, lr) for l in d['weights']]
        m.dropout = 0.0
        m.n_classes = m.layers[-1].W.shape[1]
        return m

# ══════════════════════════════════════════════════════════════════════════════
#  MLP GENERATOR  — [class one-hot + noise] → pixels
# ══════════════════════════════════════════════════════════════════════════════
class MLPGenerator:
    def __init__(self, n_classes, latent_dim, hidden_sizes, output_size, act='relu', lr=1e-3):
        self.n_classes  = n_classes
        self.latent_dim = latent_dim
        self.output_size= output_size
        self.layers = []
        prev = n_classes + latent_dim
        for h in hidden_sizes:
            self.layers.append(Dense(prev, h, act, lr, wd=1e-5))
            prev = h
        self.layers.append(Dense(prev, output_size, 'sigmoid', lr, wd=0))

    def forward(self, inp):
        x = inp
        for layer in self.layers:
            x = layer.forward(x, training=True)
        return x

    def predict(self, class_idx, noise_scale=0.5):
        inp = np.zeros(self.n_classes + self.latent_dim, dtype=np.float32)
        inp[class_idx] = 1.0
        inp[self.n_classes:] = np.random.randn(self.latent_dim).astype(np.float32) * noise_scale
        x = inp
        for layer in self.layers:
            x = layer.forward(x, training=False)
        return x

    def train_epoch(self, images, labels, train_idx, batch_size, ds_h, ds_w, ch):
        rng_idx = train_idx.copy()
        np.random.shuffle(rng_idx)
        total_batches = math.ceil(len(rng_idx) / batch_size)
        epoch_loss, nb = 0.0, 0

        for bi in range(total_batches):
            batch = rng_idx[bi*batch_size:(bi+1)*batch_size]
            bloss = 0.0
            for idx in batch:
                raw = images[idx]
                if ch == 3:
                    target = raw.copy()
                else:
                    target = (raw[0::3]*0.299 + raw[1::3]*0.587 + raw[2::3]*0.114)

                inp = np.zeros(self.n_classes + self.latent_dim, dtype=np.float32)
                inp[labels[idx]] = 1.0
                inp[self.n_classes:] = np.random.randn(self.latent_dim).astype(np.float32) * 0.5

                out  = self.forward(inp)
                err  = out - target
                loss = float(np.mean(err * err))
                bloss += loss

                # MSE + sigmoid output backward
                dout = 2.0 * err * out * (1.0 - out)
                grad = dout
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
            bloss /= len(batch)
            epoch_loss += bloss; nb += 1
            yield bi+1, total_batches, bloss, 0.0

        return epoch_loss/nb, 0.0

    def validate(self, images, labels, val_idx, ds_h, ds_w, ch):
        vloss = 0.0
        for idx in val_idx:
            raw = images[idx]
            if ch == 3:
                target = raw.copy()
            else:
                target = (raw[0::3]*0.299 + raw[1::3]*0.587 + raw[2::3]*0.114)
            inp = np.zeros(self.n_classes + self.latent_dim, dtype=np.float32)
            inp[labels[idx]] = 1.0
            # zero noise for val
            out  = self.forward(inp)
            err  = out - target
            vloss += float(np.mean(err * err))
        return vloss / max(1, len(val_idx)), 0.0

    def to_dict(self):
        return {'type': 'mlpgen', 'n_classes': self.n_classes, 'latent_dim': self.latent_dim,
                'output_size': self.output_size,
                'weights': [l.to_dict() for l in self.layers]}

    @classmethod
    def from_dict(cls, d, lr=1e-3):
        m = cls.__new__(cls)
        m.n_classes   = d['n_classes']
        m.latent_dim  = d['latent_dim']
        m.output_size = d['output_size']
        m.layers = [Dense.from_dict(l, lr) for l in d['weights']]
        return m

# ══════════════════════════════════════════════════════════════════════════════
#  DDPM  — Denoising Diffusion Probabilistic Model
#  U-Net MLP backbone + sinusoidal time embedding + linear β schedule
# ══════════════════════════════════════════════════════════════════════════════
def time_embedding(t_norm, dim):
    """Sinusoidal time embedding, same as original DDPM / Stable Diffusion."""
    half = dim // 2
    freqs = np.exp(-np.log(10000) * np.arange(half) / half).astype(np.float32)
    emb = np.zeros(dim, dtype=np.float32)
    emb[:half]  = np.sin(t_norm * freqs)
    emb[half:]  = np.cos(t_norm * freqs)
    return emb

def make_noise_schedule(T):
    t_arr  = np.arange(T)
    beta   = (1e-4 + (0.02 - 1e-4) * t_arr / (T - 1)).astype(np.float32)
    alpha  = 1.0 - beta
    alpha_bar = np.cumprod(alpha).astype(np.float32)
    return beta, alpha, alpha_bar

class UNetMLP:
    """Small U-Net MLP with skip connections — noise predictor for DDPM."""
    def __init__(self, img_size, n_classes, hidden, depth, t_dim=32, lr=1e-3):
        self.t_dim    = t_dim
        self.n_classes= n_classes
        self.img_size = img_size
        # Encoder
        sizes = [hidden * (2**i) for i in range(depth)]
        self.enc = []
        prev = img_size + t_dim + (n_classes if n_classes > 0 else 0)
        for s in sizes:
            self.enc.append(Dense(prev, s, 'silu', lr, wd=0))
            prev = s
        # Bottleneck
        self.bottleneck = Dense(prev, prev, 'silu', lr, wd=0)
        # Decoder (with skip connections)
        self.dec = []
        for i in range(len(sizes)-1, -1, -1):
            skip_sz = sizes[i]
            out_sz  = sizes[i-1] if i > 0 else hidden
            self.dec.append(Dense(prev + skip_sz, out_sz, 'silu', lr, wd=0))
            prev = out_sz
        # Output projection
        self.out = Dense(hidden, img_size, 'linear', lr, wd=0)
        self.sizes = sizes

    def forward(self, x_t, t_norm, class_idx=-1, training=True):
        t_emb = time_embedding(t_norm, self.t_dim)
        parts = [x_t, t_emb]
        if self.n_classes > 0:
            cond = np.zeros(self.n_classes, dtype=np.float32)
            if class_idx >= 0: cond[class_idx] = 1.0
            parts.append(cond)
        inp = np.concatenate(parts)
        # Encode
        skips = []
        h = inp
        for layer in self.enc:
            h = layer.forward(h, training)
            skips.append(h.copy())
        # Bottleneck
        h = self.bottleneck.forward(h, training)
        # Decode with skip connections
        for i, layer in enumerate(self.dec):
            skip = skips[len(skips)-1-i]
            h = np.concatenate([h, skip])
            h = layer.forward(h, training)
        return self.out.forward(h, training)

    def backward(self, d_out):
        d = self.out.backward(d_out)
        for i, layer in enumerate(reversed(self.dec)):
            d_full = layer.backward(d)
            # split: first part goes back, skip part discarded
            skip_sz = self.sizes[len(self.sizes)-1 - (len(self.dec)-1-i)]
            d = d_full[:layer.W.shape[0] - skip_sz]  # only non-skip part
        d = self.bottleneck.backward(d)
        for layer in reversed(self.enc):
            d = layer.backward(d)

    def count_params(self):
        p = 0
        for l in self.enc: p += l.W.size + l.b.size
        p += self.bottleneck.W.size + self.bottleneck.b.size
        for l in self.dec: p += l.W.size + l.b.size
        p += self.out.W.size + self.out.b.size
        return p

    def to_dict(self):
        def sd(l): return l.to_dict()
        return {'type': 'ddpm_unet', 't_dim': self.t_dim,
                'n_classes': self.n_classes, 'img_size': self.img_size, 'sizes': self.sizes,
                'enc': [sd(l) for l in self.enc],
                'bottleneck': sd(self.bottleneck),
                'dec': [sd(l) for l in self.dec],
                'out': sd(self.out)}

    @classmethod
    def from_dict(cls, d, lr=1e-3):
        m = cls.__new__(cls)
        m.t_dim     = d['t_dim']
        m.n_classes = d['n_classes']
        m.img_size  = d['img_size']
        m.sizes     = d['sizes']
        m.enc        = [Dense.from_dict(l, lr) for l in d['enc']]
        m.bottleneck = Dense.from_dict(d['bottleneck'], lr)
        m.dec        = [Dense.from_dict(l, lr) for l in d['dec']]
        m.out        = Dense.from_dict(d['out'], lr)
        return m

class DDPMTrainer:
    def __init__(self, img_size, n_classes, hidden, depth, T, conditioning, lr=1e-3, t_dim=32):
        self.T            = T
        self.conditioning = conditioning  # 'class' or 'none'
        self.n_classes    = n_classes
        nc = n_classes if conditioning == 'class' else 0
        self.net   = UNetMLP(img_size, nc, hidden, depth, t_dim, lr)
        self.beta, self.alpha, self.alpha_bar = make_noise_schedule(T)

    def train_epoch(self, images, labels, train_idx, batch_size, ds_h, ds_w, ch):
        rng_idx = train_idx.copy()
        np.random.shuffle(rng_idx)
        total_batches = math.ceil(len(rng_idx) / batch_size)
        epoch_loss, nb = 0.0, 0

        for bi in range(total_batches):
            batch = rng_idx[bi*batch_size:(bi+1)*batch_size]
            bloss = 0.0
            for idx in batch:
                raw = images[idx]
                if ch == 3:
                    x0 = raw * 2.0 - 1.0    # scale to [-1, 1]
                else:
                    x0 = (raw[0::3]*0.299 + raw[1::3]*0.587 + raw[2::3]*0.114) * 2.0 - 1.0

                # Sample random timestep
                t  = np.random.randint(0, self.T)
                sqAB  = math.sqrt(float(self.alpha_bar[t]))
                sq1AB = math.sqrt(max(0.0, 1.0 - float(self.alpha_bar[t])))

                noise = np.random.randn(*x0.shape).astype(np.float32)
                x_t   = sqAB * x0 + sq1AB * noise

                ci = int(labels[idx]) if self.conditioning == 'class' else -1
                pred = self.net.forward(x_t, t / self.T, ci, training=True)

                err   = pred - noise
                loss  = float(np.mean(err * err))
                bloss += loss

                d_out = 2.0 * err / len(x0)
                self.net.backward(d_out)
            bloss /= len(batch)
            epoch_loss += bloss; nb += 1
            yield bi+1, total_batches, bloss, 0.0

        return epoch_loss/nb, 0.0

    def validate(self, images, labels, val_idx, ds_h, ds_w, ch):
        vloss = 0.0
        for idx in val_idx:
            raw = images[idx]
            if ch == 3:
                x0 = raw * 2.0 - 1.0
            else:
                x0 = (raw[0::3]*0.299 + raw[1::3]*0.587 + raw[2::3]*0.114) * 2.0 - 1.0
            t     = np.random.randint(0, self.T)
            sqAB  = math.sqrt(float(self.alpha_bar[t]))
            sq1AB = math.sqrt(max(0.0, 1.0 - float(self.alpha_bar[t])))
            noise = np.random.randn(*x0.shape).astype(np.float32)
            x_t   = sqAB * x0 + sq1AB * noise
            ci    = int(labels[idx]) if self.conditioning == 'class' else -1
            pred  = self.net.forward(x_t, t / self.T, ci, training=False)
            vloss += float(np.mean((pred - noise)**2))
        return vloss / max(1, len(val_idx)), 0.0

    def sample(self, class_idx=-1, denoise_steps=50):
        """DDPM reverse sampling: start from noise, iteratively denoise."""
        stride = max(1, self.T // denoise_steps)
        timesteps = list(range(self.T - 1, -1, -stride))
        img_size  = self.net.img_size

        x = np.random.randn(img_size).astype(np.float32)
        ci = class_idx if self.conditioning == 'class' else -1

        for si, t in enumerate(timesteps):
            progress_bar(si+1, len(timesteps), prefix='  Denoising', suffix=f't={t}')
            eps = self.net.forward(x, t / self.T, ci, training=False)
            sqrt_alpha  = math.sqrt(float(self.alpha[t]))
            coeff       = float(self.beta[t]) / math.sqrt(max(1e-8, 1.0 - float(self.alpha_bar[t])))
            sigma       = math.sqrt(float(self.beta[t])) if t > 0 else 0.0
            x = (x - coeff * eps) / sqrt_alpha + sigma * np.random.randn(img_size).astype(np.float32)
        progress_end()

        # Rescale from [-1,1] to [0,1]
        return np.clip((x + 1.0) * 0.5, 0.0, 1.0)

    def to_dict(self):
        return {'type': 'ddpm',
                'T': self.T, 'conditioning': self.conditioning, 'n_classes': self.n_classes,
                'net': self.net.to_dict(),
                'beta':       self.beta.tolist(),
                'alpha':      self.alpha.tolist(),
                'alpha_bar':  self.alpha_bar.tolist()}

    @classmethod
    def from_dict(cls, d, lr=1e-3):
        m = cls.__new__(cls)
        m.T            = d['T']
        m.conditioning = d['conditioning']
        m.n_classes    = d['n_classes']
        m.net          = UNetMLP.from_dict(d['net'], lr)
        m.beta         = np.array(d['beta'],      dtype=np.float32)
        m.alpha        = np.array(d['alpha'],     dtype=np.float32)
        m.alpha_bar    = np.array(d['alpha_bar'], dtype=np.float32)
        return m

# ══════════════════════════════════════════════════════════════════════════════
#  GAN  — Generator + Discriminator
# ══════════════════════════════════════════════════════════════════════════════
class GANTrainer:
    def __init__(self, img_size, n_classes, noise_dim, gen_hidden, gen_layers, disc_hidden, lr=1e-3):
        self.noise_dim = noise_dim
        self.n_classes = n_classes
        self.img_size  = img_size
        in_sz = noise_dim + n_classes

        # Generator: noise + class → pixels (tanh output)
        self.G = []
        prev = in_sz
        for _ in range(gen_layers):
            self.G.append(Dense(prev, gen_hidden, 'lrelu', lr, wd=0))
            prev = gen_hidden
        self.G.append(Dense(prev, img_size, 'tanh', lr, wd=0))

        # Discriminator: pixels + class → real/fake (sigmoid)
        self.D = []
        prev = img_size + n_classes
        for _ in range(3):
            self.D.append(Dense(prev, disc_hidden, 'lrelu', lr*0.5, wd=0))
            prev = disc_hidden
        self.D.append(Dense(prev, 1, 'sigmoid', lr*0.5, wd=0))

    def _g_forward(self, z, class_idx, training=True):
        inp = np.zeros(self.noise_dim + self.n_classes, dtype=np.float32)
        inp[:self.noise_dim] = z
        inp[self.noise_dim + class_idx] = 1.0
        x = inp
        for layer in self.G:
            x = layer.forward(x, training)
        return x

    def _d_forward(self, img, class_idx, training=True):
        inp = np.zeros(self.img_size + self.n_classes, dtype=np.float32)
        inp[:self.img_size] = img
        inp[self.img_size + class_idx] = 1.0
        x = inp
        for layer in self.D:
            x = layer.forward(x, training)
        return float(x[0])

    def _d_backward(self, grad_out):
        g = np.array([grad_out], dtype=np.float32)
        for layer in reversed(self.D):
            g = layer.backward(g)
        return g[:self.img_size]  # img part of gradient

    def _g_backward(self, d_img):
        g = d_img.copy()
        for layer in reversed(self.G):
            g = layer.backward(g)

    @staticmethod
    def _bce(pred, label):
        return -math.log(max(1e-7, pred)) * label - math.log(max(1e-7, 1.0 - pred)) * (1.0 - label)

    def train_epoch(self, images, labels, train_idx, batch_size, ds_h, ds_w, ch):
        rng_idx = train_idx.copy()
        np.random.shuffle(rng_idx)
        total_batches = math.ceil(len(rng_idx) / batch_size)
        epoch_gl, epoch_dl, nb = 0.0, 0.0, 0

        for bi in range(total_batches):
            batch = rng_idx[bi*batch_size:(bi+1)*batch_size]
            bgl = bdl = 0.0
            for idx in batch:
                ci = int(labels[idx])
                raw = images[idx]
                if ch == 3:
                    real = raw * 2.0 - 1.0
                else:
                    real = (raw[0::3]*0.299 + raw[1::3]*0.587 + raw[2::3]*0.114) * 2.0 - 1.0

                # ── Train D on real ──
                dr = self._d_forward(real, ci, training=True)
                self._d_backward(dr - 0.9)      # label smoothing
                bdl += self._bce(dr, 0.9)

                # ── Train D on fake ──
                z    = np.random.randn(self.noise_dim).astype(np.float32)
                fake = self._g_forward(z, ci, training=False)
                df   = self._d_forward(fake, ci, training=True)
                self._d_backward(df - 0.1)
                bdl += self._bce(df, 0.1)

                # ── Train G ──
                z2    = np.random.randn(self.noise_dim).astype(np.float32)
                fake2 = self._g_forward(z2, ci, training=True)
                df2   = self._d_forward(fake2, ci, training=True)
                d_img = self._d_backward(df2 - 1.0)   # want D to say real=1
                # Chain through tanh output of G
                d_tanh = d_img * (1.0 - fake2 * fake2)
                self._g_backward(d_tanh)
                bgl += self._bce(df2, 1.0)

            bgl /= len(batch); bdl /= len(batch)*2
            epoch_gl += bgl; epoch_dl += bdl; nb += 1
            yield bi+1, total_batches, (bgl+bdl)*0.5, 0.0

        return (epoch_gl+epoch_dl)*0.5/nb, 0.0

    def validate(self, images, labels, val_idx, ds_h, ds_w, ch):
        vloss = 0.0
        for idx in val_idx:
            ci  = int(labels[idx])
            z   = np.random.randn(self.noise_dim).astype(np.float32)
            out = self._g_forward(z, ci, training=False)
            df  = self._d_forward(out, ci, training=False)
            vloss += self._bce(df, 1.0)
        return vloss / max(1, len(val_idx)), 0.0

    def generate(self, class_idx, temperature=1.0):
        z = np.random.randn(self.noise_dim).astype(np.float32) * temperature
        out = self._g_forward(z, class_idx, training=False)
        return np.clip((out + 1.0) * 0.5, 0.0, 1.0)

    def to_dict(self):
        def sd(l): return l.to_dict()
        return {'type': 'gan', 'noise_dim': self.noise_dim, 'n_classes': self.n_classes,
                'img_size': self.img_size,
                'G': [sd(l) for l in self.G], 'D': [sd(l) for l in self.D]}

    @classmethod
    def from_dict(cls, d, lr=1e-3):
        m = cls.__new__(cls)
        m.noise_dim = d['noise_dim']
        m.n_classes = d['n_classes']
        m.img_size  = d['img_size']
        m.G = [Dense.from_dict(l, lr) for l in d['G']]
        m.D = [Dense.from_dict(l, lr) for l in d['D']]
        return m

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP  (shared for all models)
# ══════════════════════════════════════════════════════════════════════════════
def run_training(model, model_name, images, labels, train_idx, val_idx,
                 ds_h, ds_w, ch, classes, args):
    epochs    = args.epochs
    batch     = args.batch_size
    val_split = args.val_split
    flip_h    = not args.no_flip_h
    flip_v    = args.flip_v
    aug_noise = not args.no_noise
    save_path = args.output

    cprint(C.BOLD + C.CYAN, f"\n═══ Training {model_name} ═══")
    cprint(C.DIM, f"  Epochs: {epochs}  Batch: {batch}  LR: {args.lr}")
    cprint(C.DIM, f"  Train: {len(train_idx)}  Val: {len(val_idx)}")
    if hasattr(model, 'count_params'):
        cprint(C.DIM, f"  Params: {model.count_params():,}")
    elif hasattr(model, 'net') and hasattr(model.net, 'count_params'):
        cprint(C.DIM, f"  Params: {model.net.count_params():,}")

    best_val = float('inf')
    best_state = None
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    for ep in range(epochs):
        ep_loss = ep_acc = nb = 0.0
        t0 = time.time()

        # Choose the right train_epoch signature
        if isinstance(model, MLPClassifier):
            gen = model.train_epoch(images, labels, train_idx, batch, ds_h, ds_w, ch,
                                    flip_h, flip_v, aug_noise)
        else:
            gen = model.train_epoch(images, labels, train_idx, batch, ds_h, ds_w, ch)

        for bi, total_b, bloss, bacc in gen:
            ep_loss += bloss; ep_acc += bacc; nb += 1
            progress_bar(bi, total_b, prefix=f'  Ep {ep+1}/{epochs}',
                         suffix=f'loss={bloss:.4f}')

        ep_loss /= max(1, nb); ep_acc /= max(1, nb)

        # Validation
        if isinstance(model, MLPClassifier):
            val_loss, val_acc = model.validate(images, labels, val_idx, ds_h, ds_w, ch)
        else:
            val_loss, val_acc = model.validate(images, labels, val_idx, ds_h, ds_w, ch)

        elapsed = time.time() - t0
        progress_end()

        if isinstance(model, MLPClassifier):
            cprint(C.CYAN, f'  Epoch {ep+1}/{epochs} — '
                   f'loss:{ep_loss:.4f} acc:{ep_acc*100:.1f}% '
                   f'val_loss:{val_loss:.4f} val_acc:{val_acc*100:.1f}% '
                   f'({elapsed:.1f}s)')
        else:
            cprint(C.PURPLE, f'  Epoch {ep+1}/{epochs} — '
                   f'loss:{ep_loss:.4f} val_loss:{val_loss:.4f} ({elapsed:.1f}s)')

        history['loss'].append(ep_loss)
        history['acc'].append(ep_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            best_state = json.dumps(model.to_dict())

        # Checkpoint every N epochs
        ckpt_interval = getattr(args, 'checkpoint', 0)
        if ckpt_interval > 0 and (ep+1) % ckpt_interval == 0:
            ckpt_path = save_path.replace('.json', f'_ep{ep+1}.json')
            _save_model(model, classes, ds_w, ds_h, ch, history, ckpt_path, is_checkpoint=True)

    cprint(C.GREEN, f'\n  Best val loss: {best_val:.4f}')
    _save_model(model, classes, ds_w, ds_h, ch, history, save_path)
    return history

def _save_model(model, classes, ds_w, ds_h, ch, history, path, is_checkpoint=False):
    data = {
        'model':     model.to_dict(),
        'classes':   classes,
        'ds_w': ds_w, 'ds_h': ds_h, 'channels': ch,
        'history':   history,
    }
    with open(path, 'w') as f:
        json.dump(data, f)
    label = 'Checkpoint' if is_checkpoint else 'Model'
    cprint(C.GREEN, f'  {label} saved → {path}')

def load_model_file(path):
    with open(path) as f:
        data = json.load(f)
    mdata  = data['model']
    mtype  = mdata.get('type', 'classifier')
    classes= data.get('classes', [])
    ds_w   = data.get('ds_w', 32)
    ds_h   = data.get('ds_h', 32)
    ch     = data.get('channels', 3)
    history= data.get('history', {})

    if mtype == 'classifier':
        model = MLPClassifier.from_dict(mdata)
    elif mtype == 'mlpgen':
        model = MLPGenerator.from_dict(mdata)
    elif mtype == 'ddpm':
        model = DDPMTrainer.from_dict(mdata)
    elif mtype == 'gan':
        model = GANTrainer.from_dict(mdata)
    else:
        raise ValueError(f"Unknown model type: {mtype}")

    return model, classes, ds_w, ds_h, ch, history

# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
def save_image(pixels, ds_h, ds_w, ch, path):
    """Save float32 pixel array to PNG."""
    px = np.clip(pixels, 0.0, 1.0)
    if ch == 3:
        arr = (px.reshape(ds_h, ds_w, 3) * 255).astype(np.uint8)
    else:
        g = (px * 255).astype(np.uint8)
        arr = np.stack([g.reshape(ds_h, ds_w)]*3, axis=-1)

    if HAS_PIL:
        Image.fromarray(arr).save(path)
    else:
        _write_png(arr, path)
    cprint(C.DIM, f'  Saved: {path}')

def _write_png(arr, path):
    """Minimal PNG writer using only stdlib."""
    import zlib
    h, w = arr.shape[:2]
    raw_rows = []
    for row in arr:
        raw_rows.append(b'\x00' + row.tobytes())
    raw = b''.join(raw_rows)
    compressed = zlib.compress(raw, 9)

    def chunk(name, data):
        c = name + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack('>I', len(data)) + c + struct.pack('>I', crc)

    sig   = b'\x89PNG\r\n\x1a\n'
    ihdr  = chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
    idat  = chunk(b'IDAT', compressed)
    iend  = chunk(b'IEND', b'')

    with open(path, 'wb') as f:
        f.write(sig + ihdr + idat + iend)

def ascii_preview(pixels, ds_h, ds_w, ch, scale=4):
    """Print a small ASCII preview of a generated image in the terminal."""
    chars = ' .:;+=xX$&#'
    px = np.clip(pixels, 0.0, 1.0)
    if ch == 3:
        gray = px[0::3]*0.299 + px[1::3]*0.587 + px[2::3]*0.114
    else:
        gray = px
    gray = gray.reshape(ds_h, ds_w)
    step_h = max(1, ds_h // scale)
    step_w = max(1, ds_w // (scale * 2))
    print()
    for row in range(0, ds_h, step_h):
        line = ''
        for col in range(0, ds_w, step_w):
            v = float(gray[row, col])
            line += chars[int(v * (len(chars)-1))]
        cprint(C.DIM, '    ' + line)
    print()

# ══════════════════════════════════════════════════════════════════════════════
#  CLI COMMANDS
# ══════════════════════════════════════════════════════════════════════════════
def cmd_train_classifier(args):
    images, labels, classes, ds_w, ds_h = load_dataset(args)
    ch = 1 if args.grayscale else 3
    train_idx, val_idx = split_dataset(images, labels, args.val_split)

    input_size = ds_h * ds_w * ch
    hidden_sizes = [args.hidden_units] * args.hidden_layers
    cprint(C.CYAN, f"\n› Building Classifier MLP")
    cprint(C.DIM, f"  Input: {ds_h}×{ds_w}×{ch}={input_size} | Hidden: {args.hidden_layers}×{args.hidden_units} | Classes: {len(classes)}")
    cprint(C.DIM, f"  Activation: {args.activation} | Dropout: {args.dropout}")

    model = MLPClassifier(input_size, hidden_sizes, len(classes),
                          act=args.activation, lr=args.lr,
                          dropout=args.dropout, wd=args.weight_decay)
    run_training(model, 'MLP Classifier', images, labels, train_idx, val_idx,
                 ds_h, ds_w, ch, classes, args)

def cmd_train_mlp_gen(args):
    images, labels, classes, ds_w, ds_h = load_dataset(args)
    ch = 1 if args.grayscale else 3
    train_idx, val_idx = split_dataset(images, labels, args.val_split)

    output_size  = ds_h * ds_w * ch
    hidden_sizes = [args.hidden_units] * args.hidden_layers
    cprint(C.CYAN, f"\n› Building MLP Generator")
    cprint(C.DIM, f"  Input: [{len(classes)} classes + {args.latent_dim} noise] | Hidden: {args.hidden_layers}×{args.hidden_units} | Output: {output_size}")
    cprint(C.DIM, f"  Activation: {args.activation}")

    model = MLPGenerator(len(classes), args.latent_dim, hidden_sizes,
                         output_size, act=args.activation, lr=args.lr)
    run_training(model, 'MLP Generator', images, labels, train_idx, val_idx,
                 ds_h, ds_w, ch, classes, args)

def cmd_train_ddpm(args):
    images, labels, classes, ds_w, ds_h = load_dataset(args)
    ch = 1 if args.grayscale else 3
    train_idx, val_idx = split_dataset(images, labels, args.val_split)

    img_size = ds_h * ds_w * ch
    nc = len(classes) if args.conditioning == 'class' else 0
    cprint(C.CYAN, f"\n› Building DDPM (T={args.T}, depth={args.depth}, hidden={args.hidden_size})")
    cprint(C.DIM, f"  Image: {ds_h}×{ds_w}×{ch}={img_size} | Conditioning: {args.conditioning}")
    cprint(C.DIM, f"  β schedule: linear 1e-4→0.02 | Time embedding dim: 32")

    model = DDPMTrainer(img_size, len(classes), args.hidden_size, args.depth,
                        args.T, args.conditioning, lr=args.lr)
    cprint(C.DIM, f"  Params: {model.net.count_params():,}")
    run_training(model, 'DDPM', images, labels, train_idx, val_idx,
                 ds_h, ds_w, ch, classes, args)

def cmd_train_gan(args):
    images, labels, classes, ds_w, ds_h = load_dataset(args)
    ch = 1 if args.grayscale else 3
    train_idx, val_idx = split_dataset(images, labels, args.val_split)

    img_size = ds_h * ds_w * ch
    cprint(C.CYAN, f"\n› Building GAN (noise={args.noise_dim}, G={args.gen_layers}×{args.gen_hidden}, D={args.disc_hidden})")
    cprint(C.DIM, f"  Image: {ds_h}×{ds_w}×{ch}={img_size} | Classes: {len(classes)}")
    cprint(C.DIM, f"  G loss: -log D(G(z)) | D loss: BCE real+fake | Label smoothing: 0.9/0.1")

    model = GANTrainer(img_size, len(classes), args.noise_dim,
                       args.gen_hidden, args.gen_layers, args.disc_hidden, lr=args.lr)
    run_training(model, 'GAN', images, labels, train_idx, val_idx,
                 ds_h, ds_w, ch, classes, args)

def cmd_generate(args):
    cprint(C.CYAN, f"› Loading model: {args.model}")
    model, classes, ds_w, ds_h, ch, history = load_model_file(args.model)
    mtype = model.to_dict()['type']

    # Resolve class
    class_idx = 0
    if args.cls:
        if args.cls.isdigit():
            class_idx = int(args.cls)
        elif args.cls in classes:
            class_idx = classes.index(args.cls)
        else:
            cprint(C.YELLOW, f"  Class '{args.cls}' not found. Available: {classes}")
            cprint(C.YELLOW, f"  Using class index 0: {classes[0] if classes else '?'}")

    class_name = classes[class_idx] if class_idx < len(classes) else str(class_idx)
    cprint(C.DIM, f"  Model type: {mtype} | Class: {class_name} ({class_idx}) | Generating {args.n} images")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        cprint(C.CYAN, f"\n  Generating {i+1}/{args.n}…")
        if mtype == 'mlpgen':
            pixels = model.predict(class_idx, noise_scale=args.noise)
        elif mtype == 'ddpm':
            pixels = model.sample(class_idx if model.conditioning == 'class' else -1,
                                  denoise_steps=args.steps)
        elif mtype == 'gan':
            pixels = model.generate(class_idx, temperature=args.temperature)
        else:
            cprint(C.RED, f"  Cannot generate with model type '{mtype}'"); return

        if not args.no_preview:
            ascii_preview(pixels, ds_h, ds_w, ch)

        fname = out_dir / f"{class_name}_{i:04d}.png"
        save_image(pixels, ds_h, ds_w, ch, str(fname))

    cprint(C.GREEN, f"\n  Done — {args.n} images saved to {out_dir}/")

def cmd_classify(args):
    cprint(C.CYAN, f"› Loading model: {args.model}")
    model, classes, ds_w, ds_h, ch, _ = load_model_file(args.model)
    if model.to_dict()['type'] != 'classifier':
        cprint(C.RED, "  Model is not a classifier"); return

    if not HAS_PIL:
        cprint(C.RED, "  PIL required for image loading. pip install Pillow"); return

    img = Image.open(args.image).convert('RGB').resize((ds_w, ds_h), Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0
    if ch == 1:
        flat = (arr[...,0]*0.299 + arr[...,1]*0.587 + arr[...,2]*0.114).reshape(-1)
    else:
        flat = arr.reshape(-1)

    probs = model.predict(flat)
    pairs = sorted(enumerate(probs), key=lambda x: -x[1])

    cprint(C.BOLD + C.CYAN, f"\n  Predictions for: {args.image}")
    cprint(C.DIM, f"  Image size: {ds_w}×{ds_h}×{ch}\n")
    for i, (ci, prob) in enumerate(pairs[:8]):
        bar_len = int(prob * 30)
        bar     = '█' * bar_len + '░' * (30 - bar_len)
        label   = classes[ci] if ci < len(classes) else str(ci)
        colour  = C.GREEN if i == 0 else (C.CYAN if i < 3 else C.DIM)
        cprint(colour, f"  {label:<16} [{bar}] {prob*100:.1f}%")

def cmd_info(args):
    cprint(C.CYAN, f"› Model info: {args.model}")
    model, classes, ds_w, ds_h, ch, history = load_model_file(args.model)
    mdata = model.to_dict()
    mtype = mdata['type']

    cprint(C.BOLD, f"\n  Type:      {mtype}")
    cprint(C.DIM,  f"  Image:     {ds_w}×{ds_h}×{ch}")
    cprint(C.DIM,  f"  Classes:   {len(classes)} — {classes}")

    if mtype == 'classifier':
        weights = mdata['weights']
        params  = sum(np.array(l['W']).size + np.array(l['b']).size for l in weights)
        cprint(C.DIM, f"  Params:    {params:,}")
        cprint(C.DIM, f"  Layers:")
        for i, l in enumerate(weights):
            cprint(C.DIM, f"    L{i}: {l['inSize']}→{l['outSize']} ({l['act']})")
    elif mtype == 'mlpgen':
        weights = mdata['weights']
        params  = sum(np.array(l['W']).size + np.array(l['b']).size for l in weights)
        cprint(C.DIM, f"  Noise dim: {mdata['latent_dim']}")
        cprint(C.DIM, f"  Params:    {params:,}")
    elif mtype == 'ddpm':
        cprint(C.DIM, f"  T:         {mdata['T']}")
        cprint(C.DIM, f"  Cond:      {mdata['conditioning']}")
        net = UNetMLP.from_dict(mdata['net'])
        cprint(C.DIM, f"  Params:    {net.count_params():,}")
    elif mtype == 'gan':
        gp = sum(np.array(l['W']).size + np.array(l['b']).size for l in mdata['G'])
        dp = sum(np.array(l['W']).size + np.array(l['b']).size for l in mdata['D'])
        cprint(C.DIM, f"  Noise dim: {mdata['noise_dim']}")
        cprint(C.DIM, f"  G params:  {gp:,}  D params: {dp:,}")

    if history and history.get('loss'):
        n = len(history['loss'])
        best_loss = min(history['loss'])
        final_loss = history['loss'][-1]
        cprint(C.DIM, f"\n  Training history: {n} epochs")
        cprint(C.DIM, f"  Final loss:  {final_loss:.4f}")
        cprint(C.DIM, f"  Best loss:   {best_loss:.4f}")
        if history.get('val_acc') and any(x > 0 for x in history['val_acc']):
            best_acc = max(history['val_acc'])
            cprint(C.GREEN, f"  Best val acc: {best_acc*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════
def make_parser():
    p = argparse.ArgumentParser(
        description='Pixel·Trainer CLI — train MLP/DDPM/GAN on pixel art datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train classifier on ZIP dataset
  python pixeltrainer.py train-classifier --data dataset.zip --epochs 50

  # Train MLP generator
  python pixeltrainer.py train-mlp-gen --data dataset.zip --epochs 100 --latent-dim 32

  # Train DDPM diffusion model
  python pixeltrainer.py train-ddpm --data dataset.zip --T 200 --hidden-size 128 --epochs 200

  # Train GAN
  python pixeltrainer.py train-gan --data dataset.zip --noise-dim 64 --epochs 300

  # Generate images with trained DDPM
  python pixeltrainer.py generate --model model.json --class cat --n 8

  # Classify an image
  python pixeltrainer.py classify --model model.json --image test.png

  # Load from HuggingFace
  python pixeltrainer.py train-classifier --data username/dataset-name

  # Grayscale mode
  python pixeltrainer.py train-classifier --data dataset.zip --grayscale
        """)

    sub = p.add_subparsers(dest='command')

    # Shared dataset args
    def add_data_args(sp):
        sp.add_argument('--data',        required=True, help='ZIP file path or HuggingFace repo (user/dataset)')
        sp.add_argument('--grayscale',   action='store_true', help='Convert to grayscale (1 channel)')
        sp.add_argument('--max-images',  type=int, default=500, dest='max_images', help='Max images to load (default 500)')
        sp.add_argument('--val-split',   type=float, default=0.15, dest='val_split', help='Validation split (default 0.15)')
        sp.add_argument('--output',      '-o', default='model.json', help='Output model path (default model.json)')
        sp.add_argument('--checkpoint',  type=int, default=0, help='Save checkpoint every N epochs (0=off)')

    def add_train_args(sp):
        add_data_args(sp)
        sp.add_argument('--epochs',       type=int,   default=30,    help='Training epochs (default 30)')
        sp.add_argument('--batch-size',   type=int,   default=16,    dest='batch_size', help='Batch size (default 16)')
        sp.add_argument('--lr',           type=float, default=0.001, help='Learning rate (default 0.001)')
        sp.add_argument('--no-flip-h',    action='store_true', dest='no_flip_h', help='Disable horizontal flip augmentation')
        sp.add_argument('--flip-v',       action='store_true', dest='flip_v',    help='Enable vertical flip augmentation')
        sp.add_argument('--no-noise',     action='store_true', dest='no_noise',  help='Disable noise augmentation')
        sp.add_argument('--weight-decay', type=float, default=1e-4,  dest='weight_decay', help='Weight decay (default 1e-4)')

    # train-classifier
    sc = sub.add_parser('train-classifier', help='Train MLP image classifier')
    add_train_args(sc)
    sc.add_argument('--hidden-layers',  type=int,   default=1,      dest='hidden_layers', help='Hidden layers (default 1)')
    sc.add_argument('--hidden-units',   type=int,   default=128,    dest='hidden_units',  help='Units per hidden layer (default 128)')
    sc.add_argument('--activation',     default='relu', choices=['relu','lrelu','elu','tanh','sigmoid'], help='Activation (default relu)')
    sc.add_argument('--dropout',        type=float, default=0.3,    help='Dropout rate (default 0.3)')

    # train-mlp-gen
    sg = sub.add_parser('train-mlp-gen', help='Train conditional MLP generator')
    add_train_args(sg)
    sg.add_argument('--latent-dim',    type=int,   default=16,     dest='latent_dim',    help='Noise latent dim (default 16)')
    sg.add_argument('--hidden-layers', type=int,   default=2,      dest='hidden_layers', help='Hidden layers (default 2)')
    sg.add_argument('--hidden-units',  type=int,   default=256,    dest='hidden_units',  help='Units per layer (default 256)')
    sg.add_argument('--activation',    default='relu', choices=['relu','lrelu','elu','tanh'], help='Activation (default relu)')

    # train-ddpm
    sd = sub.add_parser('train-ddpm', help='Train DDPM diffusion model')
    add_train_args(sd)
    sd.add_argument('--T',            type=int,   default=200,    help='Diffusion timesteps (default 200)')
    sd.add_argument('--hidden-size',  type=int,   default=128,    dest='hidden_size', help='U-Net hidden size (default 128)')
    sd.add_argument('--depth',        type=int,   default=3,      help='U-Net depth / levels (default 3)')
    sd.add_argument('--conditioning', default='class', choices=['class','none'], help='Conditioning (default class)')

    # train-gan
    sgan = sub.add_parser('train-gan', help='Train GAN generator')
    add_train_args(sgan)
    sgan.add_argument('--noise-dim',   type=int, default=32,  dest='noise_dim',   help='Generator noise dim (default 32)')
    sgan.add_argument('--gen-hidden',  type=int, default=128, dest='gen_hidden',  help='Generator hidden units (default 128)')
    sgan.add_argument('--gen-layers',  type=int, default=3,   dest='gen_layers',  help='Generator layers (default 3)')
    sgan.add_argument('--disc-hidden', type=int, default=128, dest='disc_hidden', help='Discriminator hidden units (default 128)')

    # generate
    sge = sub.add_parser('generate', help='Generate images from a trained model')
    sge.add_argument('--model',       required=True,          help='Model JSON file')
    sge.add_argument('--class',       default='0',            dest='cls',         help='Class name or index (default 0)')
    sge.add_argument('--n',           type=int,   default=4,  help='Number of images (default 4)')
    sge.add_argument('--out-dir',     default='generated',    dest='out_dir',     help='Output directory (default: generated/)')
    sge.add_argument('--noise',       type=float, default=0.5, help='MLP noise scale (default 0.5)')
    sge.add_argument('--steps',       type=int,   default=50, help='DDPM denoising steps (default 50)')
    sge.add_argument('--temperature', type=float, default=1.0, help='GAN temperature (default 1.0)')
    sge.add_argument('--no-preview',  action='store_true', dest='no_preview', help='Disable ASCII preview')

    # classify
    scl = sub.add_parser('classify', help='Classify an image with a trained classifier')
    scl.add_argument('--model', required=True, help='Model JSON file')
    scl.add_argument('--image', required=True, help='Image file to classify')

    # info
    si = sub.add_parser('info', help='Print model info')
    si.add_argument('--model', required=True, help='Model JSON file')

    return p

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    cprint(C.BOLD + C.CYAN, """
╔═══════════════════════════════════════╗
║  P I X E L · T R A I N E R   C L I   ║
║  Pure Python · NumPy · No GPU needed  ║
╚═══════════════════════════════════════╝""")

    if not HAS_PIL:
        cprint(C.YELLOW, "  ⚠ Pillow not installed — PNG decode uses stdlib fallback (slower)")
        cprint(C.YELLOW, "    Install with: pip install Pillow\n")

    parser = make_parser()
    args   = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        'train-classifier': cmd_train_classifier,
        'train-mlp-gen':    cmd_train_mlp_gen,
        'train-ddpm':       cmd_train_ddpm,
        'train-gan':        cmd_train_gan,
        'generate':         cmd_generate,
        'classify':         cmd_classify,
        'info':             cmd_info,
    }

    fn = dispatch.get(args.command)
    if fn:
        try:
            fn(args)
        except KeyboardInterrupt:
            cprint(C.YELLOW, '\n\n  Interrupted by user.')
        except Exception as e:
            cprint(C.RED, f'\n  Error: {e}')
            import traceback
            traceback.print_exc()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
