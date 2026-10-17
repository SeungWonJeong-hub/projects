"""온보드 **트리아지** 곡선 — 임계값을 F1 이 아니라 대역에서 읽는다.

왜: 운용 구조가 "위성은 대략적인 수치만, 지상국이 완전 처리" 라면 온보드 모델은 분류기가
아니라 **트리아지 필터**다(2026-09-10 사용자). 그러면 두 가지가 동시에 달라진다.

  ① 두 실수의 비용이 비대칭이다 — 누락은 **회복 불가**(다음 주기까지 회복이 진행돼 정보가
     손상된다), 오탐은 **대역 낭비**(지상에서 걸러내면 최종 산출물은 멀쩡하다).
  ② **결정 단위가 화소가 아니라 씬**이다. 원본을 통째로 내려보낸다면 마스크가 소각지의 91%를
     잡는지 60%를 잡는지는 트리거 결정과 무관하다 — **켜졌는가**만 중요하다.

그래서 배포 thr 0.25 의 근거(§20 = 안동 1건의 면적오차 0 교차)는 이 구조에서 목표가 아니다.
면적 정확도는 **지상 산출물**의 목표지 온보드 트리거의 목표가 아니다.

★이 프레이밍에서 §21 의 최상위 병목이 걸리지 않는다 — 씬 단위 트리거의 참값은 "이 씬에 불이
있었나" 라서 **화소 라벨이 필요 없다**. 화소 단위로 못 쓰는 산림청 폴리곤 문제를 우회한다.

내는 것 (예산 숫자 없이 오늘 낼 수 있는 것만):
  ① 화소 재현율·오탐률·F1 vs thr — 기존 자, **참고용**(트리아지 지표가 아니다)
  ② **씬 트리거율** vs (thr × 트리거 면적규칙 A) — 소각 씬이 켜지는가(=씬 recall) /
     미소각 씬이 켜지는가(=헛전송). 트리아지의 진짜 지표.
  ③ ②를 화재 유병률 p 에 넣은 **다운링크 감축률**(필터 없음 대비)

★못 내는 것(해석 한계):
  - 씬 recall 의 n 이 작다 — 소각 씬 4 + CIR 크롭 6, **전부 3~4월 화재**다.
  - CIR 6건은 화재 중심 크롭이라 그 자체로는 씬이 아니다(함정 12). "소형 화재가 만드는
    예측 면적의 크기"를 재는 용도로만 쓰고 트리거율에는 넣지 않는다.
  - 미소각 씬은 안동·강릉 시계열이라 지리가 두 곳뿐이다.

사용: conda run -n ml python src/triage_curve.py [--per-bin 2]
"""
import os, sys, json, glob, argparse, functools, datetime as dt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
GSD = 20; PXA = GSD * GSD / 1e4
RUNS = os.environ.get("RUNS", "runs/cir_reh_ens/*.pt")     # 배포 20m 앙상블
THRS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
AREAS = [1, 5, 10, 25, 50, 100]                            # 트리거 규칙: 예측 소각 ≥ A ha
SEASONS = {"봄 (DOY 90~160)": (90, 160), "여름 (DOY 170~260)": (170, 260),
           "가을 (DOY 261~330)": (261, 330)}
DELTAS = [(0, 0), (10, 3), (30, 5)]                        # pre→post 간격(일). 0 = pre=post
BURN = [("andong", "full"), ("gangneung", "partial"), ("goseong", "partial"), ("yangyang", "partial")]
_ROOT = os.environ.get("SYNTAX_INFERENCE_ROOT", os.path.expanduser("~/syntax_inference"))
CIRD = os.path.join(_ROOT, "Steven/산불/산불데이터/국토정보플랫폼 데이터셋/CIR_실측라벨/fires")
CIR = [("250324_안동길안면", "대형"), ("250321_산청시천면", "대형"), ("230402_홍성서부면", "대형"),
       ("220410_양구양구읍", "대형"), ("250322_울주온양읍", "소형"), ("250428_대구노곡동", "소형")]


def to12(x8):
    """CIR-GT 8ch(pre RGBN|post RGBN) float[0,1] → nifos_eval.infer 가 받는 12ch 정수 ×1e4."""
    out = np.zeros((12,) + x8.shape[1:], np.float32)
    out[0:4] = x8[0:4]; out[6:10] = x8[4:8]
    return (out * 1e4).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bin", type=int, default=2, help="계절×간격 조합당 미소각 쌍 수")
    ap.add_argument("--prev", type=float, nargs="*", default=[0.01, 0.05, 0.20],
                    help="가정하는 화재 유병률(주기당 촬영 씬 중 실제 화재 비율)")
    a = ap.parse_args()
    os.environ.update(SITE="andong", GSD=str(GSD), TILE="256", NIR="B8A", SCENE_PICK="doy")
    import importlib, nifos_eval as NE
    from unified_eval import three_state
    NE._load = functools.lru_cache(maxsize=None)(NE._load)   # 9개 가중치를 매 호출 재적재하지 않게
    M = sorted(glob.glob(os.path.join(PROJ, RUNS)))
    assert M, f"가중치 없음: {RUNS}"
    ix = lambda b, i, j: (b[i] - b[j]) / (b[i] + b[j] + 1e-6)
    print(f"배포 앙상블 {len(M)}개 · thr 스윕 {THRS}", flush=True)

    # ── ① 소각 씬: 화소 지표 + 씬 예측면적 ──────────────────────────────
    fires, pix = [], []       # pix: (lab, prob) 화소 합산용
    for site, kind in BURN:
        os.environ["SITE"] = site; importlib.reload(NE)
        C = os.path.join(PROJ, "nifos_cache", site + "_doy")
        try:
            x = np.load(os.path.join(C, f"scene_{GSD}m.npz"))["x"]
            gt = np.load(os.path.join(C, f"gt_{GSD}m.npy")).astype(bool)
        except FileNotFoundError:
            print(f"  {site}: 캐시 없음 — 건너뜀"); continue
        H, W = min(gt.shape[0], x.shape[1]), min(gt.shape[1], x.shape[2])
        gt = gt[:H, :W]; x = x[:, :H, :W]
        xf = x.astype(np.float32) / 1e4
        dnbr = ix(xf[:6], 3, 5) - ix(xf[6:], 3, 5)
        lab, _ = three_state(site, kind, gt, dnbr)
        prob, _ = NE.infer(x, M); prob = prob[:H, :W]
        pix.append((lab, prob))
        fires.append(dict(name=site, kind="씬", scene_ha=prob.size * PXA, gt_ha=float(gt.sum() * PXA),
                          ha={f"{t:.2f}": float((prob >= t).sum() * PXA) for t in THRS}))
        print(f"  소각씬 {site:<10} {prob.size*PXA:>8,.0f}ha  GT {gt.sum()*PXA:>7,.0f}ha", flush=True)

    # ── ② CIR 크롭: 소형 화재가 만드는 신호의 크기 ───────────────────────
    os.environ["SITE"] = "andong"; importlib.reload(NE)
    crops = []
    for name, sz in CIR:
        p = os.path.join(CIRD, name, "gt_stac_20m_raw.npz")
        if not os.path.exists(p):
            print(f"  CIR {name}: 없음 — 건너뜀"); continue
        z = np.load(p); gt = z["gt"].astype(bool); vd = z["valid"].astype(bool)
        prob, _ = NE.infer(to12(z["x"].astype("float32")), M)
        H, W = gt.shape; prob = prob[:H, :W]
        crops.append(dict(name=name, size=sz, scene_ha=float(vd.sum() * PXA),
                          gt_ha=float(gt.sum() * PXA),
                          ha={f"{t:.2f}": float(((prob >= t) & vd).sum() * PXA) for t in THRS}))
        print(f"  CIR {sz} {name:<18} 유효 {vd.sum()*PXA:>7,.0f}ha  GT {gt.sum()*PXA:>6,.0f}ha", flush=True)

    # ── ③ 미소각 씬: 정답 0ha. 계절 × 간격으로 헛전송을 잰다 ──────────────
    negs = []
    for site in ("andong", "gangneung"):
        ts = os.path.join(PROJ, "nifos_cache", f"{site}_ts", f"ts_{GSD}m.npz")
        if not os.path.exists(ts):
            print(f"  {site}_ts: 없음 — 건너뜀"); continue
        os.environ["SITE"] = site; importlib.reload(NE)
        z = np.load(ts, allow_pickle=True)
        ds = [dt.date.fromisoformat(str(s)) for s in z["dates"]]; X = z["x"]; V = z["v"]
        vf = V.reshape(len(ds), -1).mean(1)
        ok = [i for i in range(len(ds)) if vf[i] >= 0.97]
        doy = np.array([d.timetuple().tm_yday for d in ds])
        for sname, (d0, d1) in SEASONS.items():
            pool = [i for i in ok if d0 <= doy[i] <= d1]
            for tgt, tol in DELTAS:
                cand = [(i, j) for i in pool for j in pool
                        if (j == i if tgt == 0 else j > i and abs((ds[j] - ds[i]).days - tgt) <= tol)]
                cand.sort(key=lambda p: (abs((ds[p[1]] - ds[p[0]]).days - tgt), ds[p[0]]))
                seen, sel = set(), []
                for i, j in cand:
                    if len(sel) >= a.per_bin: break
                    if i in seen or j in seen: continue
                    sel.append((i, j)); seen.update((i, j))
                for i, j in sel:
                    prob, _ = NE.infer(np.concatenate([X[i], X[j]], 0), M)
                    m = V[i] & V[j]
                    negs.append(dict(site=site, season=sname, target=tgt,
                                     delta=(ds[j] - ds[i]).days, pre=ds[i].isoformat(),
                                     post=ds[j].isoformat(), scene_ha=float(m.sum() * PXA),
                                     ha={f"{t:.2f}": float(((prob >= t) & m).sum() * PXA) for t in THRS}))
        print(f"  미소각 {site}: 누적 {len(negs)}쌍", flush=True)

    # ── 표 1. 화소 지표 (참고 — 트리아지 지표가 아니다) ────────────────────
    print("\n═══ 표1. 화소 지표 vs thr — 소각 4씬 합산 (참고용) ═══")
    print(f"{'thr':>6}{'재현율':>9}{'오탐률':>9}{'(F1)':>8}{'예측ha':>10}{'GT ha':>9}{'면적오차':>10}")
    gt_ha = sum(f["gt_ha"] for f in fires)
    t1 = []
    for t in THRS:
        tp = fp = fn = neg = 0
        for lab, prob in pix:
            v = lab != 255; p = (prob >= t) & v; g = (lab == 2) & v
            tp += int((p & g).sum()); fp += int((p & ~g & v).sum())
            fn += int((~p & g).sum()); neg += int((v & ~g).sum())
        rc = tp / max(tp + fn, 1); fpr = fp / max(neg, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        pred = sum(f["ha"][f"{t:.2f}"] for f in fires)
        err = (pred - gt_ha) / gt_ha * 100
        print(f"{t:>6.2f}{rc:>9.3f}{fpr:>9.1%}{f1:>8.3f}{pred:>10,.0f}{gt_ha:>9,.0f}{err:>+9.1f}%")
        t1.append(dict(thr=t, recall=rc, fpr=fpr, f1=f1, pred_ha=pred, area_err=err))

    # ── 표 2. 화재가 만드는 예측면적 vs 미소각이 그냥 뱉는 예측면적 ─────────
    print("\n═══ 표2. 예측 소각면적(ha) — 신호 대 잡음 ═══")
    print(f"{'':<24}{'GT ha':>8}" + "".join(f"{t:>9.2f}" for t in THRS))
    for f in fires + crops:
        tag = f["name"] if f.get("kind") == "씬" else f"CIR {f['size']} {f['name'][7:]}"
        print(f"{tag:<24}{f['gt_ha']:>8,.0f}" + "".join(f"{f['ha'][f'{t:.2f}']:>9,.0f}" for t in THRS))
    print("-" * 96)
    for sname in SEASONS:
        for tgt, _ in DELTAS:
            g = [n for n in negs if n["season"] == sname and n["target"] == tgt]
            if not g: continue
            med = [float(np.median([n["ha"][f"{t:.2f}"] for n in g])) for t in THRS]
            print(f"{('미소각 ' + sname[:2] + f' Δ{tgt}일 (n={len(g)})'):<24}{0:>8,.0f}"
                  + "".join(f"{m:>9,.0f}" for m in med))

    # ── 표 3. 씬 트리거 행렬 ────────────────────────────────────────────
    print("\n═══ 표3. 씬 트리거율 — 규칙: 예측 소각 ≥ A ha ═══")
    print("  소각씬 recall (n=%d, 전부 3~4월) / 미소각 헛전송률 (n=%d)" % (len(fires), len(negs)))
    print(f"{'thr':>6}{'A(ha)':>7}{'소각씬':>9}{'미소각 전체':>12}"
          + "".join(f"{s[:2]:>8}" for s in SEASONS))
    t3 = []
    for t in THRS:
        for A in AREAS:
            k = f"{t:.2f}"
            rec = sum(f["ha"][k] >= A for f in fires) / max(len(fires), 1)
            ft = sum(n["ha"][k] >= A for n in negs) / max(len(negs), 1)
            per = []
            for s in SEASONS:
                g = [n for n in negs if n["season"] == s]
                per.append(sum(n["ha"][k] >= A for n in g) / max(len(g), 1))
            print(f"{t:>6.2f}{A:>7}{rec:>9.0%}{ft:>12.1%}" + "".join(f"{v:>8.0%}" for v in per))
            t3.append(dict(thr=t, area=A, scene_recall=rec, false_trigger=ft,
                           by_season={s: v for s, v in zip(SEASONS, per)}))
        print()

    # ── 표 4. 다운링크 감축 ─────────────────────────────────────────────
    print("═══ 표4. 다운링크 — 필터 없음(전량 전송) 대비 ═══")
    print("  전송률 = p·recall + (1−p)·헛전송률 · 감축률 = 1 − 전송률 · 헛전송비중 = 전송 중 헛것")
    for p in a.prev:
        print(f"\n  ── 화재 유병률 p = {p:.0%} ──")
        print(f"{'thr':>6}{'A(ha)':>7}{'씬recall':>10}{'전송률':>9}{'감축률':>9}{'헛전송비중':>12}")
        for r in t3:
            if r["area"] not in (5, 25, 100): continue
            snd = p * r["scene_recall"] + (1 - p) * r["false_trigger"]
            waste = (1 - p) * r["false_trigger"] / max(snd, 1e-9)
            print(f"{r['thr']:>6.2f}{r['area']:>7}{r['scene_recall']:>10.0%}"
                  f"{snd:>9.1%}{1-snd:>9.1%}{waste:>12.0%}")

    out = dict(runs=RUNS, thrs=THRS, areas=AREAS, pixel=t1, fires=fires, crops=crops,
               negs=negs, trigger=t3, prevalence=a.prev)
    json.dump(out, open(os.path.join(PROJ, "data", "triage_curve.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n기록: data/triage_curve.json")


if __name__ == "__main__":
    main()
