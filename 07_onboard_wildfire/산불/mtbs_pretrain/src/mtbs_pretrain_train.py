"""MTBS 선학습 — 심각도 분류기 (12ch S2 pre/post → MTBS 등급).

목표: 미 3만화재 분석가보정 등급을 학습해 '화재별 보정 없이 전이되는' 절대 등급
분류기를 얻는다. 이후 한국 안동·울진으로 cross-domain 검증(도메인 갭 판정).

라벨: MTBS 0배경 1미소각~저 2저 3중 4고 5증녹 6비매핑.
  → 학습 타깃: 1→0(저) 2→1(저-중) 3→2(중) 4→3(고) 로 4등급. 0/5/6 = ignore(255).
  (안동/울진 GT 는 3등급이라, 배포시 저/미저 병합 등 매핑은 검증 단계서 결정.)

적재 방식(--load): auto = CUDA_VISIBLE_DEVICES="" 면 stream, 아니면 memory.
  memory — 필터 통과 타일을 한 번 병렬로 읽어 fp16 으로 GPU 에 상주. B200 에서 스트리밍은
           메인 스레드 하나가 npz 압축해제·타일링을 도맡아 GPU 가 거의 놀았다(에폭 ~10분 → 24초).
  stream — 파일을 하나씩 읽어 타일을 흘린다(Orin 7GB 용. 젯슨 자율 사이클이 이 경로).
사용: python src/mtbs_pretrain_train.py --epochs 30 --tile 256 --bs 16
"""
import os, sys, glob, argparse, random, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
PATCHES = os.path.join(ROOT, "patches"); CKPT = os.path.join(ROOT, "ckpt")
os.makedirs(CKPT, exist_ok=True)
IGNORE = 255
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}          # MTBS→4등급, 0/5/6 무시


def make_target(sev):
    y = np.full(sev.shape, IGNORE, np.uint8)
    for k, v in LABEL_MAP.items():
        y[sev == k] = v
    return y


def normalize_scene(X):
    """★도메인적응 핵심: 패치당 밴드별 z-score(유효화소 기준). 미 여름↔한국 봄의
    절대 반사율 수준 이동을 제거 → US 학습모델이 한국서 전부 '저강도'로 붕괴하던 것 완화.
    학습·추론 양쪽에서 동일 적용해야 함(mtbs_validate_korea 도 동일 함수 사용)."""
    Xn = np.asarray(X, np.float32).copy()
    for c in range(Xn.shape[0]):
        v = Xn[c]; m = np.isfinite(v) & (v != 0)
        if m.sum() > 100:
            mu = float(v[m].mean()); sd = float(v[m].std())
            Xn[c] = np.where(m, (v - mu) / (sd + 1e-6), 0.0)
    return Xn


def tiles(files, tile):
    """패치 npz → (X 12ch 정규화, y) 타일 스트림. 소각(비-ignore) 화소 있고 pre·post 영상이 다 있는 타일만."""
    for f in files:
        try:                                       # 빌드가 동시 쓰기 중이면 부분파일 → skip
            z = np.load(f); raw = z["X"]; y = make_target(z["sev"])
        except Exception:
            continue
        # 씬 footprint 밖이 0 으로 찬 화소(2026-09-11 전수: pre 90%↑ 공백 1,020/3,089 — 모자이크
        # 이전 빌드). 공백은 정규화 전 원본에서 잰다. 6밴드 전부 0 인 화소가 pre 나 post 에 1% 넘으면 버린다.
        void = (raw[:6] == 0).all(0) | (raw[6:] == 0).all(0)
        X = normalize_scene(raw)
        C, H, W = X.shape
        for r in range(0, H - tile + 1, tile):
            for c in range(0, W - tile + 1, tile):
                yt = y[r:r+tile, c:c+tile]
                if (yt != IGNORE).sum() < tile * tile * 0.02:   # 소각 2%↑
                    continue
                if void[r:r+tile, c:c+tile].mean() > 0.01:
                    continue
                yield X[:, r:r+tile, c:c+tile], yt


def _file_tiles(job):
    """Pool 작업 단위 — 파일 하나의 타일을 fp16 으로(z-score 라 fp16 정밀도로 충분)."""
    f, tile = job
    return [(X.astype(np.float16), y) for X, y in tiles([f], tile)]


def load_tiles(files, tile):
    from multiprocessing import Pool
    with Pool(min(64, os.cpu_count() or 1)) as pool:
        parts = pool.map(_file_tiles, [(f, tile) for f in files], chunksize=4)
    X = [x for p in parts for x, _ in p]; Y = [y for p in parts for _, y in p]
    if not X:
        return np.zeros((0, 12, tile, tile), np.float16), np.zeros((0, tile, tile), np.uint8)
    return np.stack(X), np.stack(Y)


def stream_batches(files, tile, bs, seed=None):
    order = list(files)
    if seed is not None:
        random.Random(seed).shuffle(order)
    bx, by = [], []
    for X, y in tiles(order, tile):
        bx.append(X); by.append(y)
        if len(bx) == bs:
            yield np.stack(bx), np.stack(by); bx, by = [], []
    if bx:
        yield np.stack(bx), np.stack(by)


def class_weights(ys):
    """클래스 불균형 보정: 필터 통과 타일의 등급 화소빈도 → inverse-sqrt 가중.
    (심각도는 고강도가 희소 1:10~20 → 무가중이면 흔한 등급으로 붕괴)"""
    cnt = np.zeros(4, np.int64)
    for y in ys:
        cnt += np.bincount(y[y != IGNORE].ravel(), minlength=4)[:4]
    freq = cnt / max(cnt.sum(), 1)
    w = (1.0 / np.sqrt(freq + 1e-6)); w = w / w.mean()
    print(f"등급 화소빈도 {dict(enumerate(cnt.tolist()))} → 가중 {np.round(w, 2).tolist()}", flush=True)
    return w


def build_unet(in_ch, n_cls):
    import torch.nn as nn
    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(min(8, o), o), nn.ReLU(),
                             nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(min(8, o), o), nn.ReLU())

    class UNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.d1 = blk(in_ch, 32); s.d2 = blk(32, 64); s.d3 = blk(64, 128)
            s.pool = nn.MaxPool2d(2); s.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            s.u2 = blk(128 + 64, 64); s.u1 = blk(64 + 32, 32); s.out = nn.Conv2d(32, n_cls, 1)

        def forward(s, x):
            import torch
            a = s.d1(x); b = s.d2(s.pool(a)); c = s.d3(s.pool(b))
            u = s.u2(torch.cat([s.up(c), b], 1)); u = s.u1(torch.cat([s.up(u), a], 1))
            return s.out(u)
    return UNet()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0, help="패치 상한(0=전부, 젯슨 sanity용)")
    ap.add_argument("--load", choices=["auto", "memory", "stream"], default="auto")
    ap.add_argument("--out", default="mtbs_sev_best.pt", help="ckpt/ 아래 저장 파일명")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(PATCHES, "*.npz")))
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f"패치 없음 — 먼저 mtbs_pretrain_build.py --batch 로 생성. ({PATCHES})")
    nv = max(1, int(len(files) * a.val_frac))
    tr_f, va_f = files[nv:], files[:nv]                 # 화재 단위 분할(누출 방지)

    # Orin 버그: CUDA_VISIBLE_DEVICES="" 여도 is_available()가 True → NVML assert 크래시.
    # 빈 문자열이면 CPU 결정적 강제(젯슨 소규모 학습용). 진짜 스케일 학습은 B200.
    # 적재 방식은 torch import 전에 정한다 — memory 적재의 fork 가 CUDA 초기화보다 먼저여야 한다.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    mode = a.load if a.load != "auto" else ("stream" if cvd == "" else "memory")
    t0 = time.time()
    if mode == "memory":
        Xtr, Ytr = load_tiles(tr_f, a.tile); Xva, Yva = load_tiles(va_f, a.tile)
        w = class_weights(Ytr)
    else:
        w = class_weights(y for _, y in tiles(tr_f[:60], a.tile))
    import torch
    dev = "cuda" if (cvd != "" and torch.cuda.is_available()) else "cpu"
    msg = f"패치 화재: 학습 {len(tr_f)} / 검증 {len(va_f)} · 적재 {mode} · device {dev}"
    if mode == "memory":
        # GPU 상주가 기본. 모자라면(B200 MIG 슬라이스 여유 ~14GB — 모자이크 재빌드로 타일이 늘면 넘을 수 있다)
        # pinned CPU 램에 두고 배치만 올린다. 4시간 빌드 뒤 OOM 으로 체인이 죽는 것을 막는다.
        gb = (Xtr.nbytes + Ytr.nbytes + Xva.nbytes + Yva.nbytes) / 1e9
        arrs = (Xtr, Ytr, Xva, Yva)
        try:
            if os.environ.get("MTBS_FORCE_PINNED"):       # 시험용 — 물러서는 경로를 작은 데이터로도 태운다
                raise torch.OutOfMemoryError("MTBS_FORCE_PINNED")
            Xtr, Ytr, Xva, Yva = (torch.from_numpy(x).to(dev) for x in arrs)
            where = dev
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            Xtr, Ytr, Xva, Yva = (torch.from_numpy(x).pin_memory() for x in arrs)
            where = "cpu(pinned)"
        del arrs
        msg += f" · 타일 학습 {len(Xtr):,} / 검증 {len(Xva):,} · {gb:.1f} GB → {where} · {time.time()-t0:.0f}s"
    print(msg, flush=True)

    def batches(train, ep):
        if mode == "stream":
            return stream_batches(tr_f if train else va_f, a.tile, a.bs, (1000 + ep) if train else None)
        X, Y = (Xtr, Ytr) if train else (Xva, Yva)
        idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(1000 + ep)) if train \
            else torch.arange(len(X))
        return ((X[idx[i:i + a.bs].to(X.device)], Y[idx[i:i + a.bs].to(X.device)])
                for i in range(0, len(idx), a.bs))

    net = build_unet(12, 4).to(dev)
    opt = torch.optim.AdamW(net.parameters(), a.lr)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev),
                                      ignore_index=IGNORE)
    best = 0.0
    for ep in range(a.epochs):
        te = time.time()
        net.train(); tl = torch.zeros((), device=dev); nb = 0   # 배치를 직접 센다(스트림은 len() 이 없다)
        for X, y in batches(True, ep):
            X = torch.as_tensor(X).to(dev).float(); y = torch.as_tensor(y).to(dev).long()
            opt.zero_grad()
            l = lossf(net(X), y); l.backward(); opt.step(); tl += l.detach(); nb += 1
        # 검증 정확도(비-ignore)
        net.eval(); cor = tot = 0
        with torch.no_grad():
            for X, y in batches(False, ep):
                X = torch.as_tensor(X).to(dev).float(); y = torch.as_tensor(y).to(dev).long()
                p = net(X).argmax(1); m = y != IGNORE
                cor += (p[m] == y[m]).sum().item(); tot += m.sum().item()
        acc = cor / max(tot, 1)
        print(f"ep{ep+1} loss {tl.item()/max(nb,1):.3f} · 배치 {nb} · val등급정확도 {acc:.3f} "
              f"· {time.time()-te:.1f}s", flush=True)
        if acc > best:
            best = acc; torch.save(net.state_dict(), os.path.join(CKPT, a.out))
    print(f"완료 best val {best:.3f} → ckpt/{a.out}", flush=True)


if __name__ == "__main__":
    main()
