"""MTBS 선학습 모델 → 한국 안동·울진 cross-domain 검증 (최종 관문).

미 여름 30m Landsat 라벨로 학습한 심각도 분류기가 **한국 봄 20m S2**로 전이되는가.
NIFoS 안동·울진 피해등급(독립 GT)에 대해 등급 정확도·등급별 면적오차를 낸다.

모델 출력: 4등급(0=미저,1=저,2=중,3=고). 한국 GT: 3등급(경1중2심3).
매핑(기본): 모델 {0,1}→경, {2}→중, {3}→심. 모델 입력은 장면 정규화(train.normalize_scene).

★영상은 `울진산불_변화탐지/nifos_cache/<site>/scene_20m.npz` 를 쓴다(nifos_eval 이 DOY 정합으로
받아 둔, 화재를 사이에 둔 쌍 — GT 와 같은 20m 격자, 밴드 [B02,B03,B04,B8A,B11,B12]×pre/post).
2026-09-11 까지는 severity_calib.py 의 **2025 봄 창**을 가져다 썼는데 안동 GT 는 **2020-04-24
풍천면** 화재다. 5년 뒤 흔적을 두 번 본 것이라 소각내 dNBR 이 −0.04(안 탄 곳과 같음)였고
모델은 전부 '경'으로 찍었다(정확도 0.212 — 올바른 쌍에서는 0.696). 울진 2022-02-23 pre 는
구름 33% 씬 하나뿐이라 dNBR 이 뒤집혔다. 그래서 소각내 dNBR 중앙이 0 이하면 경고한다.
mtbs_finetune_korea.py 가 fetch_site·korea_gt3·run_model·SITES·MODEL2K 를 가져다 쓰므로 인터페이스는 유지한다.
사용: python src/mtbs_validate_korea.py --ckpt ckpt/mtbs_sev_best.pt
"""
import os, sys, argparse
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from mtbs_pretrain_train import build_unet, LABEL_MAP, normalize_scene  # noqa: E402,F401

NC = os.path.join(ROOT, "..", "울진산불_변화탐지", "nifos_cache")
SITES = {
    # NIFoS 안동 값 의미는 1심/2중/3경(sev_train.py) — 명시 매핑
    "안동": dict(gt=os.path.expanduser("~/Steven/nifos_burn_gt/andong/andong_severity_20m.tif"),
              cache="andong", fire="2020-04-24", gtmap={1: 3, 2: 2, 3: 1}),
    # 울진 GT 는 S2 기반이라 SWIR 지표엔 순환이다(SESSION_2026-09-09 §6c). 값 의미가 확인 안 돼 postNBR 순서로 매핑
    "울진": dict(gt=os.path.expanduser("~/Steven/nifos_burn_gt/ext/Ulgin_severity.tif"),
              cache="uljin", fire="2022-03-04", gtmap=None),
    # 확대(2026-09-11, build_korea_sites.py): 날짜=국토정보플랫폼 CIR 폴더명 교차확인
    "양구": dict(gt=os.path.expanduser("~/Steven/nifos_burn_gt/ext/양구_severity_20m.tif"), cache="양구", fire="2022-04-10", gtmap={1: 3, 2: 2, 3: 1}),
    "군위": dict(gt=os.path.expanduser("~/Steven/nifos_burn_gt/ext/군위_severity_20m.tif"), cache="군위", fire="2022-04-11", gtmap={1: 3, 3: 1, 2: 2}),
    "밀양": dict(gt=os.path.expanduser("~/Steven/nifos_burn_gt/ext/밀양_severity_20m.tif"), cache="밀양", fire="2022-05-31", gtmap={1: 3, 2: 2, 3: 1}),
    # RapidEye 2019-04-04 강원 동시다발(웹확인) — dNBR 신호 +0.45~0.49 검증됨
    "re2019_63": dict(gt=os.path.expanduser("~/Steven/nifos_burn_gt/ext/re2019_63_severity_20m.tif"), cache="re2019_63", fire="2019-04-04", gtmap={3: 1, 1: 3, 2: 2}),
    "re2019_71": dict(gt=os.path.expanduser("~/Steven/nifos_burn_gt/ext/re2019_71_severity_20m.tif"), cache="re2019_71", fire="2019-04-04", gtmap={3: 1, 1: 3, 2: 2}),
    "re2019_74": dict(gt=os.path.expanduser("~/Steven/nifos_burn_gt/ext/re2019_74_severity_20m.tif"), cache="re2019_74", fire="2019-04-04", gtmap={3: 1, 1: 3, 2: 2}),
}
MODEL2K = {0: 1, 1: 1, 2: 2, 3: 3}       # 모델 4등급 → 한국 3등급(경1중2심3)


def nbr(n, s):
    return (n - s) / (n + s + 1e-6)


def fetch_site(name, s):
    """nifos_cache 장면을 GT 격자에 맞춘다(sev_train.py 와 같은 ute 오프셋 방식). 반환: (원본 반사율 X, GT 값)."""
    g = gdal.Open(s["gt"]); gt = g.GetGeoTransform(); S = g.ReadAsArray()
    x = np.load(os.path.join(NC, s["cache"], "scene_20m.npz"))["x"].astype("float32") / 1e4
    ute = np.load(os.path.join(NC, s["cache"], "ute_20m.npy"))
    ox, oy = int(round((gt[0] - ute[0]) / 20)), int(round((ute[3] - gt[3]) / 20))
    X = x[:, oy:oy + S.shape[0], ox:ox + S.shape[1]]
    if X.shape[1:] != S.shape:
        raise SystemExit(f"[{name}] 캐시 장면이 GT 격자를 다 덮지 못한다: {X.shape} vs {S.shape}")
    return X, S


def korea_gt3(name, gtA, sevidx):
    """NIFoS 값 → 경1중2심3. 사이트 매핑이 없으면 postNBR(sevidx) 중앙 낮을수록 심으로 정렬."""
    gtmap = SITES.get(name, {}).get("gtmap")
    if gtmap is None:
        med = {v: np.median(sevidx[gtA == v]) for v in np.unique(gtA[gtA > 0])}
        order = sorted(med, key=lambda v: med[v])          # 오름차순 postNBR = 심,중,경
        gtmap = {order[0]: 3, order[1]: 2, order[2]: 1}
    y = np.zeros_like(gtA, dtype="int8")
    for v, sv in gtmap.items():
        y[gtA == v] = sv
    return y


def run_model(net, X, dev):
    import torch
    C, H, W = X.shape
    ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
    Xp = np.pad(X, ((0, 0), (0, ph), (0, pw)))
    with torch.no_grad():
        p = net(torch.from_numpy(Xp[None]).to(dev)).argmax(1)[0].cpu().numpy()
    return p[:H, :W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "ckpt", "mtbs_sev_best.pt"))
    a = ap.parse_args()
    import torch
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)   # Orin: ""면 CPU 강제(NVML 크래시 회피)
    dev = "cuda" if (cvd != "" and torch.cuda.is_available()) else "cpu"
    net = build_unet(12, 4).to(dev).eval()
    if os.path.exists(a.ckpt):
        net.load_state_dict(torch.load(a.ckpt, map_location=dev)); print(f"모델 로드 {a.ckpt}")
    else:
        print(f"⚠️ ckpt 없음({a.ckpt}) — 랜덤 가중치로 배관만 검증")

    for name, s in SITES.items():
        # GT·캐시가 없는 사이트는 건너뛴다 — 확대 사이트(양구·군위·밀양)는 build_korea_sites.py 를 돌린 기계에만 있다
        # (B200 체인은 안동·울진만 있어 여기서 죽으면 뒤 사이트 결과가 안 나왔다).
        if not (os.path.exists(s["gt"]) and os.path.exists(os.path.join(NC, s["cache"], "scene_20m.npz"))):
            print(f"\n[{name}] GT·캐시 없음 — 건너뜀 (build_korea_sites.py 로 생성)"); continue
        X, gtA = fetch_site(name, s)
        y = korea_gt3(name, gtA, nbr(X[9], X[11]))      # GT 정렬은 원본 postNBR
        b = y > 0
        dmed = float(np.median((nbr(X[3], X[5]) - nbr(X[9], X[11]))[b]))
        pred_m = run_model(net, normalize_scene(X), dev)  # 모델 입력은 장면 정규화(도메인적응)
        pred_k = np.vectorize(lambda v: MODEL2K.get(int(v), 1))(pred_m).astype("int8")
        acc = float((pred_k[b] == y[b]).mean()) if b.sum() else float("nan")
        ga = {k: int((y == k).sum()) * 0.04 for k in (1, 2, 3)}
        pa = {k: int((pred_k[b] == k).sum()) * 0.04 for k in (1, 2, 3)}   # 소각 내만
        mae = np.mean([abs(pa[k] - ga[k]) / max(ga[k], 1e-9) for k in (1, 2, 3)]) * 100
        print(f"\n[{name}] 화재 {s['fire']} · 소각내 dNBR 중앙 {dmed:+.3f}"
              + ("  ★화재 신호 없음 — 씬 연도·구름 확인" if dmed <= 0 else ""))
        print(f"   등급 정확도(소각내) {acc:.3f} · 평균 면적오차 {mae:.1f}%")
        for k, nm in ((3, "심"), (2, "중"), (1, "경")):
            print(f"   {nm}: GT {ga[k]:,.0f}ha · 예측 {pa[k]:,.0f}ha ({(pa[k]-ga[k])/max(ga[k],1e-9)*100:+.1f}%)")


if __name__ == "__main__":
    main()
