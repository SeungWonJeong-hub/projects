"""Finalize training that OOM-stopped at ep21: load best.pt, pick val-optimal
threshold, evaluate val/test, and write results.json (memory-frugal, workers=0)."""
import os, sys, json
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.environ.get("WILDFIRE_SEG_ROOT", os.path.expanduser("~/wildfire-seg")))
from src.model import UNetResNet50
from src.train import best_threshold, evaluate
sys.path.insert(0, os.path.dirname(__file__))
from train_cd import CDTiles, OUT, CFG

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
va = DataLoader(CDTiles("val"), batch_size=4, num_workers=0)
te = DataLoader(CDTiles("test"), batch_size=4, num_workers=0)
model = UNetResNet50(classes=1, pretrained=False, in_channels=13, attention=True).to(dev)
model.load_state_dict(torch.load(os.path.join(OUT, "best.pt"), map_location=dev)); model.eval()

thr, vf1 = best_threshold(model, va, dev)
res = {"cfg": CFG, "stopped_epoch": 21, "note": "OOM-stopped at ep21; best.pt used",
       "threshold": thr, "best_val_f1_sweep": vf1,
       "val": evaluate(model, va, dev, thr=thr), "test": evaluate(model, te, dev, thr=thr)}
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print("threshold", round(thr, 3))
print("VAL ", json.dumps(res["val"]))
print("TEST", json.dumps(res["test"]))
