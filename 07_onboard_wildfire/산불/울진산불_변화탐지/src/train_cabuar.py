"""Train the 13-channel CD U-Net on CaBuAr REAL burn masks (ground truth) — this is
the genuine-accuracy model (not the dNBR pseudo-label). Same arch as train_cd.py so it
transfers to Uljin. Reports REAL F1/IoU/OA/Kappa on the CaBuAr test fold.
"""
import os, sys, json, time, random
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.environ.get("WILDFIRE_SEG_ROOT", os.path.expanduser("~/wildfire-seg")))
from src.model import UNetResNet50
from src.metrics import binary_stats, metrics_from_stats
from src.train import DiceBCELoss, best_threshold, evaluate

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "cabuar")
# MODE: 'phys' = scale-invariant physical indices only (domain-consistent across sensors);
# else full 13ch. Physical channels in the 13ch layout [pre6|post6|dnbr]:
# pre[NBR,NDVI,NDWI]=3,4,5  post[NBR,NDVI,NDWI]=9,10,11  dNBR=12
MODE = sys.argv[1] if len(sys.argv) > 1 else "full"
CH = [3, 4, 5, 9, 10, 11, 12] if MODE == "phys" else list(range(13))
OUT = os.path.join(HERE, "..", "runs", "cabuar_phys" if MODE == "phys" else "cabuar_cd")
os.makedirs(OUT, exist_ok=True)
CFG = dict(epochs=50, bs=4, accum=4, lr=3e-4, wd=1e-4, attention=True, in_channels=len(CH), seed=42)


class Tiles(Dataset):
    def __init__(self, split, aug=False):
        self.files = sorted(os.path.join(DATA, split, f) for f in os.listdir(os.path.join(DATA, split)))
        self.aug = aug
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i]); img = d["image"][CH].astype(np.float32); msk = d["mask"].astype(np.float32)[None]
        if self.aug:
            if random.random() < 0.5: img, msk = img[:, :, ::-1], msk[:, :, ::-1]
            if random.random() < 0.5: img, msk = img[:, ::-1], msk[:, ::-1]
            k = random.randint(0, 3)
            if k: img, msk = np.rot90(img, k, (1, 2)), np.rot90(msk, k, (1, 2))
        return torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(msk))


def kappa(tp, fp, fn, tn):
    N = tp+fp+fn+tn+1e-9; po = (tp+tn)/N
    pe = ((tp+fp)*(tp+fn)+(fn+tn)*(fp+tn))/(N*N)
    return (po-pe)/(1-pe+1e-9), po


def main():
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = DataLoader(Tiles("train", True), batch_size=CFG["bs"], shuffle=True, num_workers=0, drop_last=True)
    va = DataLoader(Tiles("val"), batch_size=CFG["bs"], num_workers=0)
    te = DataLoader(Tiles("test"), batch_size=CFG["bs"], num_workers=0)
    print(f"MODE={MODE} in_channels={CFG['in_channels']} CH={CH} | tiles {len(tr.dataset)}/{len(va.dataset)}/{len(te.dataset)}", flush=True)
    model = UNetResNet50(classes=1, pretrained=True, in_channels=CFG["in_channels"], attention=True).to(dev)
    loss_fn = DiceBCELoss(); opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
    best = -1; ckpt = os.path.join(OUT, "best.pt"); hist = []
    for ep in range(CFG["epochs"]):
        model.train(); opt.zero_grad(set_to_none=True); t0 = time.time(); run = 0
        for i, (img, msk) in enumerate(tr):
            img, msk = img.to(dev), msk.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=(dev.type == "cuda")):
                loss = loss_fn(model(img), msk) / CFG["accum"]
            scaler.scale(loss).backward()
            if (i+1) % CFG["accum"] == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            run += loss.item()*CFG["accum"]
        sched.step()
        if dev.type == "cuda": torch.cuda.empty_cache()
        vm = evaluate(model, va, dev); hist.append({"ep": ep, "val_f1": vm["f1"]})
        print(f"[ep {ep:02d}] loss={run/max(1,len(tr)):.4f} val_f1={vm['f1']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vm["f1"] > best: best = vm["f1"]; torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    thr, _ = best_threshold(model, va, dev)
    # real test metrics incl. kappa
    tp=fp=fn=tn=0.0
    with torch.no_grad():
        for img, msk in te:
            p = torch.sigmoid(model(img.to(dev)).float()).cpu(); b = binary_stats(p, msk, thr)
            tp += b[0]; fp += b[1]; fn += b[2]; tn += b[3]
    m = metrics_from_stats(tp, fp, fn, tn); k, oa = kappa(float(tp), float(fp), float(fn), float(tn))
    res = {"cfg": CFG, "best_val_f1": best, "threshold": thr,
           "test_REAL": {**m, "OA": oa, "kappa": k}, "history": hist}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print("TEST (REAL labels):", json.dumps({k2: round(v, 4) for k2, v in res["test_REAL"].items()}), flush=True)


if __name__ == "__main__":
    main()
