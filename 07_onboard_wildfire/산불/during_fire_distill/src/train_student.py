"""during-fire optical student distill — teacher(FIRMS) 마스크를 광학 6밴드로 재현.

입력 = during 씬 Sentinel-2 6밴드(B02·B03·B04·B08·B11·B12, 20 m), 라벨 = 같은 날
FIRMS 활성화재 마스크(rasterize_teacher.py 산출). 열적외(FIRMS)가 보는 "지금 타는 중"을
광학만으로 추론하는 segmenter 를 증류 학습한다.
※메모리: 광학 실시간 활성화재(NHI)는 물리제약(negative) — distill 이 이를 얼마나
완화하는가가 승부처. 이 스크립트는 그 F1 을 정직하게 측정한다.

단일 인터프리터: /usr/bin/python3 (torch 2.11 cuda · gdal 3.4.1 · numpy)
device-agnostic(cuda 자동) · 재개가능(ckpt) · 스케일(35이벤트 LORO)은 B200,
Jetson 은 SMOKE(2이벤트)로 파이프라인 검증.

env:
  TILE=256 STRIDE=192      타일 크기/스트라이드
  NEG_PER_POS=3            양성타일 1개당 음성타일 상한(클래스 불균형 완화)
  BATCH=8 EPOCHS=40 LR=3e-4
  VAL_EVENT=<eid>          LORO 검증 이벤트(미지정 시 마지막 이벤트). 1이벤트뿐이면 자기검증(smoke)
  TVERSKY_A=0.7            recall 지향(FN 페널티↑) — 선행 산불 연구 채택 레버
  OUT=runs/student         체크포인트/로그 디렉토리
"""
import os, sys, json, glob
import numpy as np
from osgeo import gdal
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MANIFEST = os.path.join(ROOT, "data", "s2_during_manifest.json")
OUTDIR   = os.path.join(ROOT, "data", "s2_during")
BANDS_ALL = ["B02", "B03", "B04", "B08", "B11", "B12"]
# ② SWIR ablation: BANDS_USE 로 부분집합 학습(예 "B02,B03,B04,B08" = no-SWIR)
BANDS    = [b.strip() for b in os.environ.get("BANDS_USE", ",".join(BANDS_ALL)).split(",") if b.strip()]
MASK_NAME = os.environ.get("MASK_NAME", "FIRMS_MASK.tif")   # ② footprint sweep 마스크 선택

TILE    = int(os.environ.get("TILE", 256))
STRIDE  = int(os.environ.get("STRIDE", 192))
NEG_PP  = int(os.environ.get("NEG_PER_POS", 3))
BATCH   = int(os.environ.get("BATCH", 8))    # 젯슨 8GB unified memory 안전값(batch16=OOM 확인)
EPOCHS  = int(os.environ.get("EPOCHS", 40))
LR      = float(os.environ.get("LR", 3e-4))
TVA     = float(os.environ.get("TVERSKY_A", 0.7))
VAL_EVENT = os.environ.get("VAL_EVENT", "")
RUN     = os.path.join(ROOT, os.environ.get("OUT", "runs/student"))
DEV     = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- 데이터 ----------
def load_scene_spec(scene_dir):
    """스펙 패스용 경량 로드: 밴드 크기는 메타데이터로, 유효마스크는 밴드1개(nodata
    일관), 화재라벨은 마스크만 읽음 → 6밴드 풀로드(느림) 대신 2배열만. (y, valid, H, W)."""
    mask_p = os.path.join(scene_dir, MASK_NAME)
    if not os.path.exists(mask_p):
        return None
    dims = []
    for b in BANDS:
        p = os.path.join(scene_dir, f"{b}.tif")
        if not os.path.exists(p):
            return None
        ds = gdal.Open(p)
        dims.append((ds.RasterYSize, ds.RasterXSize))         # 메타만(즉시)
    H = min(h for h, _ in dims); W = min(w for _, w in dims)
    b0 = gdal.Open(os.path.join(scene_dir, f"{BANDS[0]}.tif")).ReadAsArray()[:H, :W]
    valid = (b0 > 0).astype(np.float32)                       # 무자료(all-zero) 제외 프록시
    y = gdal.Open(mask_p).ReadAsArray()[:H, :W]
    y = (y == 1).astype(np.float32)                           # 255=nodata → 0
    return y, valid, H, W


def tiles_from(scene_dir):
    """씬 스펙 산출(밴드1개+마스크만 로드 → 메모리·시간 절감).
    스펙 = (scene_dir, r0, c0, possum). 실제 6밴드 픽셀은 __getitem__ 에서 윈도 로드."""
    r = load_scene_spec(scene_dir)
    if r is None:
        return []
    y, valid, H, W = r
    out = []
    for r0 in range(0, max(1, H - TILE + 1), STRIDE):
        for c0 in range(0, max(1, W - TILE + 1), STRIDE):
            r1, c1 = r0 + TILE, c0 + TILE
            if r1 > H or c1 > W:
                continue
            if valid[r0:r1, c0:c1].mean() < 0.5:             # 태반이 무자료면 버림
                continue
            out.append((scene_dir, r0, c0, float(y[r0:r1, c0:c1].sum())))
    del y, valid                                             # 씬 배열 즉시 해제(x 는 스펙 경로서 미로드)
    return out


# 워커별 gdal 데이터셋 캐시(포크 후 지연 오픈 — 윈도 읽기 반복 오픈 회피)
_DS_CACHE = {}


def _ds(path):
    ds = _DS_CACHE.get(path)
    if ds is None:
        ds = gdal.Open(path); _DS_CACHE[path] = ds
    return ds


def load_tile(scene_dir, r0, c0):
    """타일 윈도만 gdal 로 읽음(전 씬 적재 안 함). refl/1e4."""
    xs = []
    for b in BANDS:
        a = _ds(os.path.join(scene_dir, f"{b}.tif")).GetRasterBand(1).ReadAsArray(
            c0, r0, TILE, TILE).astype(np.float32) / 1e4
        xs.append(a)
    x = np.clip(np.stack(xs, 0), 0, 1)
    yv = _ds(os.path.join(scene_dir, MASK_NAME)).GetRasterBand(1).ReadAsArray(
        c0, r0, TILE, TILE)
    y = (yv == 1).astype(np.float32)
    return x, y


def _split_cache_path(man, val_event):
    key = f"{MASK_NAME}_{val_event}_T{TILE}_S{STRIDE}_N{NEG_PP}_E{len(man)}"
    d = os.path.join(ROOT, "runs", "_splitcache"); os.makedirs(d, exist_ok=True)
    return os.path.join(d, key + ".json")


def build_split(man, val_event):
    ids = list(man)
    if val_event not in ids:
        val_event = ids[-1]
    # 스펙 캐시(BANDS_USE 와 무관 — 같은 MASK/VAL 이면 재사용). ablation 6회가 공유.
    cp = _split_cache_path(man, val_event)
    if os.path.exists(cp):
        c = json.load(open(cp))
        print(f"스펙 캐시 사용: {cp}", flush=True)
        return c["train"], c["val"], c["val_event"], c["stats"]
    train_t, val_t = [], []
    stats = {}
    for eid in ids:
        for s in man[eid]["scenes"]:
            if s.get("kind") != "during":
                continue
            sd = os.path.join(OUTDIR, eid, s["id"])
            ts = tiles_from(sd)                              # 스펙 리스트(배열 아님)
            if not ts:
                continue
            pos = [t for t in ts if t[3] > 0]
            neg = [t for t in ts if t[3] == 0]
            stats[f"{eid}/{s['id']}"] = {"tiles": len(ts), "pos": len(pos)}
            # 음성 서브샘플(양성 대비 NEG_PP 배)
            keep_neg = neg[: max(NEG_PP * len(pos), 4)]
            bucket = pos + keep_neg
            (val_t if eid == val_event else train_t).extend(bucket)
    json.dump({"train": train_t, "val": val_t, "val_event": val_event, "stats": stats},
              open(cp, "w"))
    print(f"스펙 캐시 저장: {cp}", flush=True)
    return train_t, val_t, val_event, stats


class TileDS(torch.utils.data.Dataset):
    def __init__(self, tiles, aug=False):
        self.t = tiles; self.aug = aug
    def __len__(self):
        return len(self.t)
    def __getitem__(self, i):
        scene_dir, r0, c0, _ = self.t[i]
        x, y = load_tile(scene_dir, r0, c0)                  # 지연 윈도 로드
        if self.aug:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy(); y = y[:, ::-1].copy()
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :].copy(); y = y[::-1, :].copy()
            k = np.random.randint(4)
            x = np.rot90(x, k, axes=(1, 2)).copy(); y = np.rot90(y, k).copy()
        return torch.from_numpy(x), torch.from_numpy(y)[None]


# ---------- 모델(소형 U-Net) ----------
def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, cin=6, base=32):
        super().__init__()
        self.e1 = cbr(cin, base); self.e2 = cbr(base, base * 2)
        self.e3 = cbr(base * 2, base * 4); self.e4 = cbr(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = cbr(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = cbr(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = cbr(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


def tversky_loss(logit, y, a=TVA, b=None, eps=1.0):
    b = 1 - a if b is None else b
    p = torch.sigmoid(logit)
    tp = (p * y).sum(); fp = (p * (1 - y)).sum(); fn = ((1 - p) * y).sum()
    return 1 - (tp + eps) / (tp + a * fn + b * fp + eps)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tp = fp = fn = 0.0
    for x, y in loader:
        x = x.to(DEV); y = y.to(DEV)
        p = (torch.sigmoid(model(x)) > 0.5).float()
        tp += (p * y).sum().item(); fp += (p * (1 - y)).sum().item()
        fn += ((1 - p) * y).sum().item()
    prec = tp / (tp + fp + 1e-6); rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return {"f1": f1, "iou": iou, "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn}


def main():
    os.makedirs(RUN, exist_ok=True)
    man = json.load(open(MANIFEST, encoding="utf-8"))
    train_t, val_t, val_ev, stats = build_split(man, VAL_EVENT)
    smoke = len(man) < 2
    print(f"device={DEV}  val_event={val_ev}  train_tiles={len(train_t)} "
          f"val_tiles={len(val_t)}  smoke={smoke}  bands={BANDS}  mask={MASK_NAME}", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v['tiles']} tiles, {v['pos']} 양성", flush=True)
    if not train_t or not val_t:
        print("⚠ 타일 부족 — 라벨 마스크에 양성 픽셀이 있는지(FIRMS 점 그리드 편입) 확인", flush=True)
        if not train_t:
            return
        val_t = val_t or train_t                              # smoke: 자기검증

    # 젯슨 unified memory(GPU=시스템RAM 공유): pin_memory 무익·역효과, 워커/배치 과하면 OOM.
    NW = int(os.environ.get("NUM_WORKERS", 2))
    tl = torch.utils.data.DataLoader(TileDS(train_t, aug=True), batch_size=BATCH,
                                     shuffle=True, num_workers=NW, drop_last=False)
    vl = torch.utils.data.DataLoader(TileDS(val_t), batch_size=BATCH)

    model = UNet(cin=len(BANDS)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))  # AMP: GPU-bound 가속(~1.5×)
    ckpt = os.path.join(RUN, "ckpt.pt"); start = 0; best = -1
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=DEV)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start = st["epoch"] + 1; best = st.get("best", -1)
        print(f"재개: epoch {start}, best_f1 {best:.4f}", flush=True)

    for ep in range(start, EPOCHS):
        model.train(); tot = 0.0
        for x, y in tl:
            x = x.to(DEV, non_blocking=True); y = y.to(DEV, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device_type=DEV, dtype=torch.float16,
                                enabled=(DEV == "cuda")):
                logit = model(x)
            logit = logit.float()                                 # 손실은 fp32(커스텀 tversky 정밀도)
            loss = tversky_loss(logit, y) + bce(logit, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * x.size(0)
        m = evaluate(model, vl)
        print(f"ep{ep:03d} loss{tot/len(train_t):.4f} | val F1 {m['f1']:.4f} "
              f"IoU {m['iou']:.4f} P {m['precision']:.4f} R {m['recall']:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "best": max(best, m["f1"])}, ckpt)
        if m["f1"] > best:
            best = m["f1"]
            torch.save({"model": model.state_dict(), "f1": best, "epoch": ep,
                        "metrics": m}, os.path.join(RUN, "best.pt"))

    json.dump({"val_event": val_ev, "smoke": smoke, "best_f1": best,
               "final_metrics": m, "tile_stats": stats},
              open(os.path.join(RUN, "result.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n완료: best val F1 {best:.4f} (val_event={val_ev}, smoke={smoke})\n"
          f"→ {RUN}/best.pt · result.json", flush=True)


if __name__ == "__main__":
    main()
