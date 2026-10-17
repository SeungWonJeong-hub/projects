"""Build a CaBuAr bi-temporal dataset in the SAME 13-channel representation as Uljin,
using REAL burn masks — so the model learns from ground truth, not a dNBR pseudo-label.

CaBuAr HDF5: 727 patches, each {pre_fire,post_fire (512,512,12 uint8), mask (1,512,512)}.
Only patches with a valid (non-empty) pre_fire are used (bi-temporal). 12-band order (no B10):
0=B01 1=B02 2=B03 3=B04 4=B05 5=B06 6=B07 7=B08 8=B8A 9=B09 10=B11 11=B12.

Per patch -> 13ch [pre6 | post6 | dNBR] (same build as build_dataset.py) + real mask.
Fold split (Paper-1): {0,1,2}=train {3}=val {4}=test. 512 tiled into 4x 256.
"""
import os, h5py, numpy as np

HDF5 = os.path.join(os.environ.get("WILDFIRE_SEG_ROOT", os.path.expanduser("~/wildfire-seg")), "data_raw/normalized/pacthed/burned_512x512.hdf5")
DATA_MS = os.path.join(os.environ.get("WILDFIRE_SEG_ROOT", os.path.expanduser("~/wildfire-seg")), "data_ms")   # reuse its fold split + has_pre
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "cabuar")
for s in ("train", "val", "test"):
    os.makedirs(os.path.join(OUT, s), exist_ok=True)
B02, B03, B04, B08, B11, B12 = 1, 2, 3, 7, 10, 11
import glob


def ratio(a, b, eps=1e-6):
    a, b = a.astype(np.float32), b.astype(np.float32)
    return (a - b) / (a + b + eps)


def stretch(ch, lo=2, hi=98):
    a, b = np.percentile(ch, lo), np.percentile(ch, hi)
    if b <= a: b = a + 1
    return np.clip((ch.astype(np.float32) - a) / (b - a), 0, 1).astype(np.float32)


def stack6(bands):   # bands (H,W,12) -> (6,H,W)
    idx = lambda x: np.clip((x + 1) / 2, 0, 1).astype(np.float32)
    return np.stack([stretch(bands[..., B12]), stretch(bands[..., B08]), stretch(bands[..., B02]),
                     idx(ratio(bands[..., B08], bands[..., B12])),
                     idx(ratio(bands[..., B08], bands[..., B04])),
                     idx(ratio(bands[..., B03], bands[..., B08]))], 0)


def build13(pre, post):
    pre6, post6 = stack6(pre), stack6(post)
    dnbr = np.clip(ratio(pre[..., B08], pre[..., B12]) - ratio(post[..., B08], post[..., B12]), -1, 1)
    dnbr_n = ((dnbr + 1) / 2).astype(np.float32)
    return np.concatenate([pre6, post6, dnbr_n[None]], 0)   # (13,512,512)


def main():
    counts = {"train": 0, "val": 0, "test": 0}; skipped = 0
    with h5py.File(HDF5, "r") as hf:
        for split in ("train", "val", "test"):
            for npz in glob.glob(os.path.join(DATA_MS, split, "*.npz")):
                key = os.path.basename(npz)[:-4]         # UUID_x == HDF5 group key
                d = np.load(npz)
                if not bool(d["has_pre"]) or key not in hf:
                    skipped += 1; continue
                g = hf[key]; pre = g["pre_fire"][:]; post = g["post_fire"][:]
                if pre.max() == 0:
                    skipped += 1; continue
                mask = (g["mask"][:][0] > 0).astype(np.uint8)
                img13 = build13(pre, post)
                for r in (0, 256):
                    for c in (0, 256):
                        np.savez_compressed(os.path.join(OUT, split, f"{key}_{r}_{c}.npz"),
                                            image=img13[:, r:r+256, c:c+256].astype(np.float32),
                                            mask=mask[r:r+256, c:c+256].astype(np.uint8))
                        counts[split] += 1
    print("CaBuAr bi-temporal tiles:", counts, "| skipped(no-pre/missing):", skipped)


if __name__ == "__main__":
    main()
