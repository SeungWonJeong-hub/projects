"""Add OA (overall accuracy) + Cohen's Kappa (burned-area paper convention) to the
existing F1/IoU/P/R, for VAL, TEST (held-out tiles) and the FULL scene."""
import os, sys, json
import numpy as np, torch
from torch.utils.data import DataLoader
from osgeo import gdal
gdal.UseExceptions()
sys.path.insert(0, os.environ.get("WILDFIRE_SEG_ROOT", os.path.expanduser("~/wildfire-seg")))
from src.model import UNetResNet50
from src.metrics import binary_stats
sys.path.insert(0, os.path.dirname(__file__))
from train_cd import CDTiles, OUT

HERE = os.path.dirname(__file__)
OUTTIF = os.path.join(HERE, "..", "data", "outputs")


def from_counts(tp, fp, fn, tn):
    tp, fp, fn, tn = map(float, (tp, fp, fn, tn)); N = tp+fp+fn+tn; eps = 1e-9
    po = (tp+tn)/N
    pe = ((tp+fp)*(tp+fn) + (fn+tn)*(fp+tn)) / (N*N)
    kappa = (po-pe)/(1-pe+eps)
    prec = tp/(tp+fp+eps); rec = tp/(tp+fn+eps)
    return {"OA": po, "kappa": kappa, "f1": 2*prec*rec/(prec+rec+eps),
            "iou": tp/(tp+fp+fn+eps), "precision": prec, "recall": rec,
            "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}}


def eval_split(model, loader, dev, thr):
    tp=fp=fn=tn=0.0
    with torch.no_grad():
        for img, msk in loader:
            img = img.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=(dev.type=="cuda")):
                prob = torch.sigmoid(model(img).float()).cpu()
            b = binary_stats(prob, msk, thr); tp+=b[0]; fp+=b[1]; fn+=b[2]; tn+=b[3]
    return from_counts(tp, fp, fn, tn)


def main():
    res = json.load(open(os.path.join(OUT, "results.json"))); thr = res.get("threshold", 0.5)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResNet50(classes=1, pretrained=False, in_channels=13, attention=True).to(dev)
    model.load_state_dict(torch.load(os.path.join(OUT, "best.pt"), map_location=dev)); model.eval()

    out = {"threshold": thr,
           "val": eval_split(model, DataLoader(CDTiles("val"), batch_size=4), dev, thr),
           "test": eval_split(model, DataLoader(CDTiles("test"), batch_size=4), dev, thr)}

    # full scene: final prediction vs refined label over valid pixels
    pred = gdal.Open(os.path.join(OUTTIF, "burn_final.tif")).ReadAsArray() > 0
    lab = gdal.Open(os.path.join(OUTTIF, "pseudolabel_refined.tif")).ReadAsArray() > 0
    d = np.load(os.path.join(HERE, "..", "data", "cd", "uljin_full.npz"), allow_pickle=True)
    valid = d["valid"].astype(bool)
    p, g = pred[valid], lab[valid]
    tp=int((p&g).sum()); fp=int((p&~g).sum()); fn=int((~p&g).sum()); tn=int((~p&~g).sum())
    out["full_scene"] = from_counts(tp, fp, fn, tn)

    json.dump(out, open(os.path.join(OUT, "metrics_extra.json"), "w"), indent=2)
    for s in ("val", "test", "full_scene"):
        m = out[s]
        print(f"{s:11} OA={m['OA']:.4f} Kappa={m['kappa']:.4f} F1={m['f1']:.4f} "
              f"IoU={m['iou']:.4f} P={m['precision']:.3f} R={m['recall']:.3f}")


if __name__ == "__main__":
    main()
