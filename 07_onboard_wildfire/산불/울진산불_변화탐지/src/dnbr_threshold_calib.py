"""dNBR 3등급 임계값 보정 — 라벨 파이프라인의 **모수**를 정한다 (착수 순서 ②-1).

배포 규칙은 dNBR 0.27/0.44 고정절단이고, 이게 중등급 면적을 −65% 틀린다(§6c·§8.5).
라벨을 dNBR 로 대량 생산하려면 **먼저 절단선이 옳아야** 한다. 여기서 정한다.

자: 학습(개발)=안동 NIFoS 등급, 시험=강릉 RapidEye 등급(S2 와 독립·누출 0). 둘 다
`sev_transfer.gate` 를 통과한 사이트다 — 등급-신호 순서와 눈금을 먼저 검사한다(§9b).

★층별 임계값(침엽/활엽): 이 머신에 임상도가 없다. 대신 **조화계수의 연주기 NDVI 진폭**으로
가른다 — 상록 침엽은 진폭이 작고 낙엽 활엽은 크다. 이건 상록/낙엽의 **정의 그 자체**이고,
§12 에서 이미 만들어 둔 산출물이라 추가 자료가 필요없다.

판정: F1 만 보지 말 것 — 제품 산출물은 **등급별 면적**이다(§7-4). 둘 다 낸다.

사용: conda run -n ml python src/dnbr_threshold_calib.py [--train andong] [--test gangneung]
"""
import os, sys, json, argparse, datetime as dt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "4")
GSD = 20.0; PXA = GSD * GSD / 1e4
DEPLOY = (0.27, 0.44)                       # 현재 배포 절단선 (경|중 , 중|심)
HARM_Q = {"andong": "20200608", "gangneung": "20190510"}
EPOCH = dt.date(2017, 1, 1); NHARM = 2; TREND = True


def amplitude(site):
    """조화계수 → 화소별 **연주기 NDVI 진폭**(p90−p10). 낙엽일수록 크다."""
    z = np.load(os.path.join(PROJ, "nifos_cache", f"{site}_harm_q{HARM_Q[site]}", "harm_coef.npz"))
    C = z["C"]                                            # (H,W,NPAR,NB)
    ref = dt.date(2019, 1, 1)
    ts = [(ref + dt.timedelta(days=int(d)) - EPOCH).days / 365.25 for d in np.linspace(0, 364, 24)]
    rows = []
    for t in ts:
        cols = [1.0] + ([t] if TREND else [])
        for h in range(1, NHARM + 1):
            cols += [np.cos(2 * np.pi * h * t), np.sin(2 * np.pi * h * t)]
        y = np.einsum("p,hwpb->hwb", np.array(cols), C)
        rows.append((y[..., 3] - y[..., 2]) / (y[..., 3] + y[..., 2] + 1e-6))   # NDVI = (NIR-R)/(NIR+R)
    A = np.stack(rows, 0)
    return np.percentile(A, 90, axis=0) - np.percentile(A, 10, axis=0)


def macro_f1(p, y):
    f = []
    for k in (1, 2, 3):
        tp = np.sum((p == k) & (y == k)); fp = np.sum((p == k) & (y != k)); fn = np.sum((p != k) & (y == k))
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def area_err(p, y):
    return [(np.sum(p == k) - np.sum(y == k)) / max(np.sum(y == k), 1) * 100 for k in (1, 2, 3)]


def cut(d, lo, hi):
    return np.where(d >= hi, 1, np.where(d >= lo, 2, 3))


def search(d, y, obj="f1"):
    """(t_lo, t_hi) 격자탐색. obj=f1 이면 macroF1 최대, area 면 등급별 면적오차 최대절댓값 최소."""
    los = np.round(np.arange(0.02, 0.62, 0.01), 3)
    his = np.round(np.arange(0.06, 0.96, 0.01), 3)
    best, arg = (-1e9, None)
    for lo in los:
        for hi in his:
            if hi <= lo + 0.02: continue
            p = cut(d, lo, hi)
            s = macro_f1(p, y) if obj == "f1" else -max(abs(v) for v in area_err(p, y))
            if s > best: best, arg = s, (float(lo), float(hi))
    return arg, best


def report(tag, d, y, th):
    p = cut(d, *th)
    e = area_err(p, y)
    print(f"{tag:<34}{th[0]:>6.2f}{th[1]:>6.2f}{(p==y).mean():>9.3f}{macro_f1(p,y):>9.3f}"
          f"   심 {e[0]:>+6.1f} 중 {e[1]:>+6.1f} 경 {e[2]:>+6.1f}")
    return dict(tag=tag, lo=th[0], hi=th[1], acc=float((p == y).mean()),
                f1=macro_f1(p, y), area_err=[float(v) for v in e])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="andong")
    ap.add_argument("--test", nargs="+", default=["gangneung"])
    a = ap.parse_args()
    import sev_transfer as ST

    print("═══ 0. 관문 — 등급이 소각강도 순서·눈금을 지키는가 (§9b) ═══")
    for s in [a.train] + a.test:
        ok, med, rat = ST.gate(s, ref=a.train)
        print(f"  {s:<12} 심 {med.get(1,float('nan')):.3f} / 중 {med.get(2,float('nan')):.3f} / "
              f"경 {med.get(3,float('nan')):.3f}  → {'통과' if ok else '★탈락'}")
        assert ok, f"{s} 가 관문을 통과하지 못했다 — 이 자로 임계값을 정하면 안 된다"

    out = {}
    _, dnbr_tr, y_tr, m_tr = ST.load(a.train)
    d_tr = dnbr_tr[m_tr]
    amp_tr = amplitude(a.train)
    amp_tr = amp_tr[:m_tr.shape[0], :m_tr.shape[1]][m_tr]
    print(f"\n═══ 1. 층 나누기 — 연주기 NDVI 진폭 ({a.train}) ═══")
    qs = np.percentile(amp_tr, [10, 25, 50, 75, 90])
    print(f"  진폭 분위 10/25/50/75/90 = " + " / ".join(f"{v:.3f}" for v in qs))
    SPLIT = float(np.median(amp_tr))
    conif = amp_tr < SPLIT                        # 진폭 작음 = 상록 침엽
    print(f"  분할선(중앙값) {SPLIT:.3f} → 침엽쪽 {conif.sum():,}화소({conif.mean()*100:.0f}%) · "
          f"활엽쪽 {(~conif).sum():,}화소")
    for nm, sel in (("침엽쪽(진폭↓)", conif), ("활엽쪽(진폭↑)", ~conif)):
        print(f"    {nm}: dNBR 중앙 심 {np.median(d_tr[sel&(y_tr==1)]):.3f} / "
              f"중 {np.median(d_tr[sel&(y_tr==2)]):.3f} / 경 {np.median(d_tr[sel&(y_tr==3)]):.3f} · "
              f"등급분포 {np.bincount(y_tr[sel],minlength=4)[1:]}")

    print(f"\n═══ 2. 임계값 보정 ({a.train} = 개발 자료) ═══")
    print(f"{'규칙':<34}{'t_lo':>6}{'t_hi':>6}{'정확도':>9}{'macroF1':>9}   {'등급별 면적오차 %':>22}")
    rows = [report("배포 고정절단 0.27/0.44", d_tr, y_tr, DEPLOY)]
    th_f1, _ = search(d_tr, y_tr, "f1");   rows.append(report("전역 재보정 (F1 최대)", d_tr, y_tr, th_f1))
    th_ar, _ = search(d_tr, y_tr, "area"); rows.append(report("전역 재보정 (면적 최적)", d_tr, y_tr, th_ar))
    strat = {}
    for nm, sel in (("침엽", conif), ("활엽", ~conif)):
        t, _ = search(d_tr[sel], y_tr[sel], "f1"); strat[nm] = t
        rows.append(report(f"층별 {nm} (F1 최대)", d_tr[sel], y_tr[sel], t))
    ps = np.empty_like(y_tr)
    ps[conif] = cut(d_tr[conif], *strat["침엽"]); ps[~conif] = cut(d_tr[~conif], *strat["활엽"])
    e = area_err(ps, y_tr)
    print(f"{'층별 합산':<34}{'—':>6}{'—':>6}{(ps==y_tr).mean():>9.3f}{macro_f1(ps,y_tr):>9.3f}"
          f"   심 {e[0]:>+6.1f} 중 {e[1]:>+6.1f} 경 {e[2]:>+6.1f}")
    rows.append(dict(tag="층별 합산", acc=float((ps == y_tr).mean()), f1=macro_f1(ps, y_tr),
                     area_err=[float(v) for v in e], strat={k: list(v) for k, v in strat.items()}))
    out[a.train] = rows

    # ── 3. 전이 — 안동에서 정한 절단선을 독립 자에 그대로 적용 ────────────
    for site in a.test:
        _, dnbr_te, y_te, m_te = ST.load(site)
        d_te = dnbr_te[m_te]
        amp_te = amplitude(site)[:m_te.shape[0], :m_te.shape[1]][m_te]
        cf = amp_te < SPLIT                      # ★분할선도 안동 것을 그대로 (시험자료로 정하면 누출)
        print(f"\n═══ 3. 전이 — {site} (독립 시험 자료, 절단선은 {a.train} 것을 그대로) ═══")
        print(f"  침엽쪽 {cf.mean()*100:.0f}% · 등급화소 {len(y_te):,} "
              f"(심 {(y_te==1).sum()*PXA:.0f} / 중 {(y_te==2).sum()*PXA:.0f} / 경 {(y_te==3).sum()*PXA:.0f} ha)")
        print(f"{'규칙':<34}{'t_lo':>6}{'t_hi':>6}{'정확도':>9}{'macroF1':>9}   {'등급별 면적오차 %':>22}")
        r = [report("배포 고정절단 0.27/0.44", d_te, y_te, DEPLOY),
             report("전역 재보정 (F1 최대)", d_te, y_te, th_f1),
             report("전역 재보정 (면적 최적)", d_te, y_te, th_ar)]
        pt = np.empty_like(y_te)
        pt[cf] = cut(d_te[cf], *strat["침엽"]); pt[~cf] = cut(d_te[~cf], *strat["활엽"])
        e = area_err(pt, y_te)
        print(f"{'층별 (침엽/활엽 각각)':<34}{'—':>6}{'—':>6}{(pt==y_te).mean():>9.3f}"
              f"{macro_f1(pt,y_te):>9.3f}   심 {e[0]:>+6.1f} 중 {e[1]:>+6.1f} 경 {e[2]:>+6.1f}")
        r.append(dict(tag="층별", acc=float((pt == y_te).mean()), f1=macro_f1(pt, y_te),
                      area_err=[float(v) for v in e]))
        # 상한 참고: 시험자료에 직접 맞춘 절단선(누출 — 인용 금지, 얼마나 남았는지만 본다)
        tb, _ = search(d_te, y_te, "f1")
        print(f"{'(상한: 시험자료에 직접 맞춤 — 인용금지)':<34}{tb[0]:>6.2f}{tb[1]:>6.2f}"
              f"{(cut(d_te,*tb)==y_te).mean():>9.3f}{macro_f1(cut(d_te,*tb),y_te):>9.3f}")
        out[site] = r
    o = os.path.join(PROJ, "data", "dnbr_threshold_calib.json")
    json.dump({"split_amp": SPLIT, "global_f1": list(th_f1), "global_area": list(th_ar),
               "strat": {k: list(v) for k, v in strat.items()}, "results": out},
              open(o, "w"), ensure_ascii=False, indent=1)
    print(f"\n기록: data/dnbr_threshold_calib.json")
    burn_threshold(a.train)



def burn_threshold(site="andong"):
    """★소각/비소각 절단 t0 — 등급 절단과 **다른 문제**다.

    등급 절단(lo/hi)은 "탄 화소를 심/중/경으로 나누는" 값이고, t0 는 "탔는가"를 가른다.
    라벨 생산에서 산림청 폴리곤(과대 라벨)을 걸러낼 때 이게 필요하다. lo 를 t0 로 쓰면
    **경등급이 통째로 비소각이 된다** — 다른 값이어야 한다.

    안동 NIFoS 소각 마스크(gt_20m.npy)에 대해 dNBR 이진 절단의 F1 을 최대화해 정한다.
    """
    import sev_transfer as ST
    from osgeo import gdal; gdal.UseExceptions()
    C = os.path.join(PROJ, "nifos_cache", site + "_doy")
    x = np.load(os.path.join(C, "scene_20m.npz"))["x"].astype("float32") / 1e4
    gt = np.load(os.path.join(C, "gt_20m.npy")).astype(bool)
    H, W = min(gt.shape[0], x.shape[1]), min(gt.shape[1], x.shape[2])
    gt = gt[:H, :W]; x = x[:, :H, :W]
    ix = lambda b, i, j: (b[i] - b[j]) / (b[i] + b[j] + 1e-6)
    d = ix(x[:6], 3, 5) - ix(x[6:], 3, 5)
    best, arg = -1, None
    for t in np.round(np.arange(0.02, 0.60, 0.005), 4):
        p = d >= t
        tp = int((p & gt).sum())
        f = 2 * tp / max(int(p.sum()) + int(gt.sum()), 1)
        if f > best: best, arg = f, float(t)
    print(f"\n═══ 4. 소각/비소각 절단 t0 ({site}, NIFoS 소각 마스크 기준) ═══")
    p = d >= arg
    print(f"  t0 = {arg:.3f} · F1 {best:.4f} · 예측 {p.sum()*PXA:,.0f}ha vs GT {gt.sum()*PXA:,.0f}ha "
          f"({(p.sum()-gt.sum())/gt.sum()*100:+.1f}%)")
    print(f"  ★등급 절단 lo 와 다른 값이다 — lo 를 t0 로 쓰면 경등급이 통째로 비소각이 된다.")
    j = os.path.join(PROJ, "data", "dnbr_threshold_calib.json")
    if os.path.exists(j):
        o = json.load(open(j)); o["t0"] = arg; o["t0_f1"] = best
        json.dump(o, open(j, "w"), ensure_ascii=False, indent=1)
    return arg


if __name__ == "__main__":
    main()
