"""Step 5 — train the early-fusion change-detection U-Net for Uljin.

Input 13ch (pre6|post6|dNBR) -> ResNet50 U-Net (ImageNet-pretrained encoder, conv1
inflated 3->13, attention decoder) -> burn mask.  DiceBCE loss (won on CaBuAr).
Reuses ~/wildfire-seg model + metrics.  Trains on the dNBR-refined pseudo-label.

NOTE: train/val/test are spatial column blocks of ONE scene, so metrics measure
in-scene generalization and are optimistic vs a truly independent fire.
"""
import os, sys, json, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.environ.get("WILDFIRE_SEG_ROOT", os.path.expanduser("~/wildfire-seg")))
from src.model import UNetResNet50
from src.metrics import binary_stats, metrics_from_stats
from src.train import DiceBCELoss, best_threshold, evaluate

HERE = os.path.dirname(__file__)
CD = os.path.join(HERE, "..", "data", "cd")
OUT = os.path.join(HERE, "..", "runs", "cd_unet")
os.makedirs(OUT, exist_ok=True)

CFG = dict(epochs=60, bs=4, accum=4, lr=3e-4, wd=1e-4, aug=True,
           attention=True, pretrained=True, in_channels=13, seed=42)


class CDTiles(Dataset):
    def __init__(self, split, aug=False):
        self.files = sorted(os.path.join(CD, split, f) for f in os.listdir(os.path.join(CD, split)))
        self.aug = aug

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        img = d["image"].astype(np.float32)          # (13,H,W)
        msk = d["mask"].astype(np.float32)[None]     # (1,H,W)
        if self.aug:
            if random.random() < 0.5: img, msk = img[:, :, ::-1], msk[:, :, ::-1]
            if random.random() < 0.5: img, msk = img[:, ::-1], msk[:, ::-1]
            k = random.randint(0, 3)
            if k: img, msk = np.rot90(img, k, (1, 2)), np.rot90(msk, k, (1, 2))
        return torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(msk))


def main():
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # num_workers=0 + pin_memory=False: avoid fork RAM duplication that OOM'd the Jetson at ep21
    tr = DataLoader(CDTiles("train", CFG["aug"]), batch_size=CFG["bs"], shuffle=True,
                    num_workers=0, drop_last=True, pin_memory=False)
    va = DataLoader(CDTiles("val"), batch_size=CFG["bs"], num_workers=0)
    te = DataLoader(CDTiles("test"), batch_size=CFG["bs"], num_workers=0)
    print(f"tiles train/val/test = {len(tr.dataset)}/{len(va.dataset)}/{len(te.dataset)}", flush=True)

    model = UNetResNet50(classes=1, pretrained=CFG["pretrained"],
                         in_channels=CFG["in_channels"], attention=CFG["attention"]).to(dev)
    loss_fn = DiceBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    best_val, hist = -1.0, []
    ckpt = os.path.join(OUT, "best.pt")
    for ep in range(CFG["epochs"]):
        model.train(); opt.zero_grad(set_to_none=True); t0 = time.time(); run = 0.0
        for i, (img, msk) in enumerate(tr):
            img, msk = img.to(dev), msk.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=(dev.type == "cuda")):
                loss = loss_fn(model(img), msk) / CFG["accum"]
            scaler.scale(loss).backward()
            if (i + 1) % CFG["accum"] == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            run += loss.item() * CFG["accum"]
        sched.step()
        if dev.type == "cuda":
            torch.cuda.empty_cache()          # curb allocator fragmentation on 8GB Jetson
        vm = evaluate(model, va, dev)
        hist.append({"ep": ep, "loss": run / max(1, len(tr)), "val_f1": vm["f1"], "val_iou": vm["iou"]})
        print(f"[ep {ep:02d}] loss={run/max(1,len(tr)):.4f} val_f1={vm['f1']:.4f} "
              f"val_iou={vm['iou']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vm["f1"] > best_val:
            best_val = vm["f1"]; torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    thr, _ = best_threshold(model, va, dev)
    res = {"cfg": CFG, "best_val_f1": best_val, "threshold": thr,
           "val": evaluate(model, va, dev, thr=thr),
           "test": evaluate(model, te, dev, thr=thr), "history": hist}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print("VAL :", json.dumps(res["val"]))
    print("TEST:", json.dumps(res["test"]), flush=True)


if __name__ == "__main__":
    main()
