"""화소별 계절 조화모델 기저선 — pre 영상 한 장을 "그 날짜의 기대치"로 대체한다 (§11.3 미결).

왜: 배경 dNDVI 가 pre 씬 선택에 따라 −0.19~+0.03 으로 흔들리고, 그 대가가 면적 +2.8~+15.2%p 다
(data/qc_calibrate_andong.json). pre 를 **한 장의 관측**이 아니라 **다년 시계열이 말하는 기대치**로
바꾸면 이 변수 자체가 사라진다 — 어느 pre 를 고르든 답이 하나다.

  ρ_b(t) = a0 + a1·t + Σ_{h=1..H} [ c_h·cos(2πh t) + s_h·sin(2πh t) ]      (t = 연 단위)

화소마다 6개(H=2) 계수만 남으면 임의 날짜의 "정상이라면 오늘 이 값" 이 나온다. 화재 이후 관측과의
잔차가 피해 신호다. 계수는 화소당 6×6밴드=36개 float32 → 원본 씬 한 장보다 가볍다(온보드 이점).

★피팅에는 **화재일 이전 관측만** 쓴다. 화재 후를 넣으면 기저선이 피해를 학습해 신호가 죽는다.

산출: nifos_cache/{SITE}_harm_q{POST}/  — nifos_eval prep 과 **같은 규격**(scene_20m.npz 12ch,
      gt_20m.npy, ute_20m.npy) 이라 qc_calibrate.score() 가 그대로 채점한다.

사용: conda run -n geo python src/harmonic_baseline.py --site andong --post 2020-06-08
      옵션 --holdout 2019-05-10 을 주면 그 날짜 ±HOLD_PAD 일을 빼고 피팅해 **외삽 충실도**를 잰다.
"""
from __future__ import annotations
import os, sys, math, json, argparse, datetime as dt
from concurrent.futures import ThreadPoolExecutor

# ★envs/geo 를 절대경로로 직접 부르면 PROJ_DATA 가 비어 좌표변환이 전멸한다(KNOWLEDGE §3.9)
os.environ.setdefault("PROJ_DATA", os.path.expanduser("~/miniforge3/envs/geo/share/proj"))
import numpy as np
from osgeo import gdal, osr
from pystac_client import Client
import socket
gdal.UseExceptions(); socket.setdefaulttimeout(90)
for _k, _v in {"GDAL_HTTP_TIMEOUT": "40", "GDAL_HTTP_CONNECTTIMEOUT": "15", "GDAL_HTTP_MAX_RETRY": "3",
               "GDAL_HTTP_RETRY_DELAY": "3", "CPL_VSIL_CURL_USE_HEAD": "NO",
               "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "AWS_NO_SIGN_REQUEST": "YES"}.items():
    gdal.SetConfigOption(_k, _v)

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
CAT  = Client.open("https://earth-search.aws.element84.com/v1")
UTM  = 32652
GSD  = float(os.environ.get("GSD", "20"))
NIR  = os.environ.get("NIR", "B8A")
NIR_ASSET = {"B8A": "nir08", "B08": "nir"}[NIR]
BANDS = [("B02", "blue"), ("B03", "green"), ("B04", "red"), (NIR, NIR_ASSET),
         ("B11", "swir16"), ("B12", "swir22")]
ORDER = ["B02", "B03", "B04", NIR, "B11", "B12"]
NB    = len(ORDER)
NHARM = int(os.environ.get("NHARM", "2"))       # 연 1주기 + 반년 주기
TREND = os.environ.get("TREND", "1") == "1"      # 다년 선형추세 항(온보드 경량화 시 뺀다)
NPAR  = 1 + (1 if TREND else 0) + 2 * NHARM      # [상수, (추세), cos/sin × NHARM]
NMIN  = int(os.environ.get("NMIN", "12"))        # 화소당 최소 유효관측 (NPAR=6 대비 여유 2배)
EPOCH = dt.date.fromisoformat(os.environ.get("TS_START", "2017-01-01"))  # S2 국내 안정 커버리지 시작
# SCL 제외: 0 nodata·1 saturated·3 cloud shadow·8/9 cloud med/high·10 cirrus·11 snow
#   ※ 11(눈)은 겨울 관측을 대량 버리지만, 눈을 "정상 계절값" 으로 학습하면 봄 기대치가 오염된다.
SCL_BAD = [0, 1, 3, 8, 9, 10, 11]


def rfc(a, b): return f"{a}T00:00:00Z/{b}T23:59:59Z"
def vsi(u):    return ("/vsis3/" + u[5:]) if u.startswith("s3://") else ("/vsicurl/" + u)


def design(dates):
    """(K, NPAR) 설계행렬. t = EPOCH 이후 연 단위 → cos(2πt) 가 정확히 1년 주기."""
    t = np.array([(d - EPOCH).days / 365.25 for d in dates], np.float64)
    cols = [np.ones_like(t)] + ([t] if TREND else [])
    for h in range(1, NHARM + 1):
        cols += [np.cos(2 * np.pi * h * t), np.sin(2 * np.pi * h * t)]
    return np.stack(cols, 1)


def fetch_date(hrefs_by_key, ute, off):
    """한 날짜의 형제 씬들을 ute 격자로 모자이크 → {밴드: (H,W) float32}."""
    out = {}
    for k, hrefs in hrefs_by_key.items():
        tmp = f"/tmp/_hb_{os.getpid()}_{os.getcwd().__hash__() & 0xffff}_{k}_{id(hrefs)}.tif"
        try:
            gdal.Warp(tmp, [vsi(h) for h in hrefs], dstSRS=f"EPSG:{UTM}", xRes=GSD, yRes=GSD,
                      outputBounds=ute, resampleAlg="average", format="GTiff", srcNodata=0)
            a = gdal.Open(tmp).ReadAsArray().astype(np.float32)
        finally:
            if os.path.exists(tmp): os.remove(tmp)
        out[k] = a if k == "SCL" else np.clip(a - off, 0, None)
    return out


def build_ts(site, ute, bb, t_end, cache):
    """t_end 이전의 모든 저구름 씬을 모아 (K,NB,H,W) int16 + SCL 유효마스크로 캐시."""
    npz = os.path.join(cache, f"ts_{int(GSD)}m.npz")
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        print(f"[시계열] 캐시 사용 {npz} — {len(z['dates'])} 날짜", flush=True)
        return [dt.date.fromisoformat(s) for s in z["dates"]], z["x"], z["v"]

    # ★한 번에 3년치를 요청하면 earth-search 가 500(Internal server error)을 낸다 —
    #   분기 단위로 쪼개고 재시도한다(일시 오류도 여기서 흡수).
    import time
    cc = float(os.environ.get("TS_CLOUD", "70"))
    its, w0 = [], EPOCH
    while w0 <= t_end:
        w1 = min(t_end, dt.date(w0.year + (w0.month + 2) // 12, (w0.month + 2) % 12 + 1, 1)
                        - dt.timedelta(days=1))
        for att in range(4):
            try:
                its += list(CAT.search(collections=["sentinel-2-l2a"], bbox=bb,
                                       datetime=rfc(w0, w1),
                                       query={"eo:cloud_cover": {"lt": cc}}, limit=100).items())
                break
            except Exception as e:
                if att == 3:
                    print(f"  [검색실패] {w0}~{w1} {type(e).__name__}: {str(e)[:70]}", flush=True)
                else:
                    time.sleep(3 * (att + 1))
        w0 = w1 + dt.timedelta(days=1)
    byd = {}
    for i in its:
        byd.setdefault(i.datetime.date(), []).append(i)
    dates = sorted(byd)
    print(f"[시계열] {EPOCH}~{t_end} 후보 {len(dates)} 날짜 (씬 {len(its)}) — 취득 시작", flush=True)

    def one(d):
        sib = byd[d]
        hrefs = {}
        for k, a in BANDS + [("SCL", "scl")]:
            hs = [(x.assets[a].href if a in x.assets else x.assets[k].href)
                  for x in sib if (a in x.assets or k in x.assets)]
            if not hs: return d, None
            hrefs[k] = hs
        off = 1000.0 if d >= dt.date(2022, 1, 25) else 0.0
        try:
            b = fetch_date(hrefs, ute, off)
        except Exception as e:
            print(f"  [skip] {d} {type(e).__name__}: {str(e)[:80]}", flush=True); return d, None
        v = ~np.isin(b["SCL"], SCL_BAD)
        for k in ORDER: v &= (b[k] > 0)
        if v.mean() < 0.05:                       # 거의 전부 구름 → 버린다
            return d, None
        return d, (np.stack([b[k] for k in ORDER], 0).astype(np.int16), v.astype(bool))

    res = {}
    with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "8"))) as ex:
        for n, (d, r) in enumerate(ex.map(one, dates), 1):
            if r is not None: res[d] = r
            if n % 20 == 0: print(f"  … {n}/{len(dates)} (유효 {len(res)})", flush=True)
    ds = sorted(res)
    assert ds, "유효 씬 0"
    x = np.stack([res[d][0] for d in ds], 0)
    v = np.stack([res[d][1] for d in ds], 0)
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(npz, dates=np.array([d.isoformat() for d in ds]), x=x, v=v)
    print(f"[시계열] 확보 {len(ds)} 날짜 · 화소당 유효관측 중앙 {np.median(v.sum(0)):.0f} → {npz}", flush=True)
    return ds, x, v


def fit(dates, x, v):
    """화소별 가중 최소제곱. A=(P,NPAR,NPAR), B=(P,NPAR,NB) 를 날짜별로 누적한다."""
    K, _, H, W = x.shape; P = H * W
    X = design(dates)                                     # (K,NPAR)
    A = np.zeros((P, NPAR, NPAR), np.float64)
    B = np.zeros((P, NPAR, NB), np.float64)
    n = np.zeros(P, np.int32)
    for k in range(K):
        w = v[k].reshape(P).astype(np.float64)            # 0/1
        if not w.any(): continue
        xk = X[k]
        A += w[:, None, None] * (xk[:, None] * xk[None, :])[None]
        y = (x[k].reshape(NB, P).T.astype(np.float64) / 1e4)   # 반사율 [0,1]
        B += w[:, None, None] * xk[None, :, None] * y[:, None, :]
        n += v[k].reshape(P)
    ok = n >= NMIN
    # 릿지: 관측이 특정 계절에 몰린 화소에서 조화항이 발산하는 것을 막는다(반사율 스케일 기준 1e-4)
    A[:, np.arange(NPAR), np.arange(NPAR)] += 1e-4 * np.maximum(n, 1)[:, None]
    C = np.zeros((P, NPAR, NB), np.float64)
    C[ok] = np.linalg.solve(A[ok], B[ok])
    print(f"[피팅] 화소 {P:,} 중 유효 {ok.sum():,} ({ok.mean()*100:.1f}%) · "
          f"관측수 중앙 {np.median(n):.0f} 최소 {n.min()}", flush=True)
    # ★관측이 모자라면 최근접 채우기가 전 화면을 덮어 **조용히 쓰레기 기저선**이 나온다
    #   (스모크에서 실제로 그랬다: 유효 0% 인데 배경 dNDVI −0.739 를 태연히 뱉었다).
    okf = float(ok.mean())
    assert okf >= float(os.environ.get("MIN_OK", "0.90")), (
        f"유효 계수 화소가 {okf*100:.1f}% 뿐이다 (기준 {float(os.environ.get('MIN_OK','0.90'))*100:.0f}%) — "
        f"시계열 창을 넓히거나 NMIN({NMIN})을 낮출 것. 이 상태의 기저선은 쓰면 안 된다.")
    return C.reshape(H, W, NPAR, NB), ok.reshape(H, W), n.reshape(H, W)


def predict(C, ok, when):
    """(H,W,NB) 반사율×10000 int16. 계수 없는 화소는 최근접 유효화소 계수로 채운다."""
    H, W = ok.shape
    xq = design([when])[0]                                # (NPAR,)
    if not ok.all():
        from scipy.ndimage import distance_transform_edt
        _, (iy, ix) = distance_transform_edt(~ok, return_indices=True)
        C = C[iy, ix]
    yh = np.einsum("p,hwpb->hwb", xq, C) * 1e4
    return np.clip(yh, 0, 12000).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=os.environ.get("SITE", "andong"))
    ap.add_argument("--post", default="2020-06-08")
    ap.add_argument("--holdout", default=None, help="이 날짜 ±pad 를 빼고 피팅 → 외삽 충실도 측정")
    ap.add_argument("--pad", type=int, default=20)
    ap.add_argument("--post-cache", default=None, help="post 실관측을 가져올 캐시 디렉터리")
    ap.add_argument("--gap", type=int, default=14, help="화재일 며칠 전까지만 피팅에 쓸지")
    a = ap.parse_args()

    # nifos_eval 은 임포트 시점에 SITE/NIR/GSD 를 읽는다 — 먼저 채운다
    os.environ["SITE"] = a.site
    os.environ.setdefault("NIR", NIR); os.environ.setdefault("GSD", str(int(GSD)))
    import nifos_eval as NE                                # SITES/FDATE/격자 로직 재사용
    fdate = NE.FDATE
    gt, ute = NE.gt_grid()
    ute = list(map(float, ute))
    lo, la = osr.SpatialReference(), osr.SpatialReference()
    la.ImportFromEPSG(4326); la.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    lo.ImportFromEPSG(UTM);  lo.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(lo, la)
    p1 = ct.TransformPoint(ute[0], ute[1]); p2 = ct.TransformPoint(ute[2], ute[3])
    bb = [min(p1[0], p2[0]) - 0.02, min(p1[1], p2[1]) - 0.02,
          max(p1[0], p2[0]) + 0.02, max(p1[1], p2[1]) + 0.02]

    ts_cache = os.path.join(PROJ, "nifos_cache", f"{a.site}_ts")
    t_end = fdate - dt.timedelta(days=a.gap)               # ★화재 전까지만
    dates, x, v = build_ts(a.site, ute, bb, t_end, ts_cache)

    keep = np.ones(len(dates), bool)
    if a.holdout:
        hd = dt.date.fromisoformat(a.holdout)
        keep = np.array([abs((d - hd).days) > a.pad for d in dates])
        print(f"[홀드아웃] {hd} ±{a.pad}일 제외 → 피팅 {keep.sum()}/{len(dates)} 날짜", flush=True)
    dsel = [d for d, k in zip(dates, keep) if k]
    C, ok, n = fit(dsel, x[keep], v[keep])

    # ── 충실도: 피팅에 쓴 날짜들의 잔차(내부) ─────────────────────────────
    Xs = design(dsel)
    res = []
    for i, d in enumerate(dsel):
        yh = np.einsum("p,hwpb->hwb", Xs[i], C) * 1e4
        m = v[keep][i]
        res.append(np.sqrt(np.mean((yh[m] - x[keep][i].transpose(1, 2, 0)[m]) ** 2, 0)))
    rms = np.mean(res, 0)
    print("[충실도] 내부 잔차 RMSE(반사율×10000): " +
          " ".join(f"{b} {r:.0f}" for b, r in zip(ORDER, rms)), flush=True)

    idx = lambda b, i, j: (b[..., i] - b[..., j]) / (b[..., i] + b[..., j] + 1e-6)
    out = {"site": a.site, "post": a.post, "fdate": fdate.isoformat(), "nharm": NHARM,
           "n_dates_total": len(dates), "n_dates_fit": len(dsel),
           "pix_ok_frac": float(ok.mean()), "n_obs_median": float(np.median(n)),
           "rmse_insample": {b: float(r) for b, r in zip(ORDER, rms)}}

    if a.holdout:
        hd = dt.date.fromisoformat(a.holdout)
        hi = [i for i, d in enumerate(dates) if abs((d - hd).days) <= a.pad]
        assert hi, f"홀드아웃 {hd} ±{a.pad}일에 관측이 없다"
        i0 = min(hi, key=lambda i: abs((dates[i] - hd).days))
        yh = np.einsum("p,hwpb->hwb", design([dates[i0]])[0], C) * 1e4
        obs = x[i0].transpose(1, 2, 0).astype(np.float32); m = v[i0]
        r_out = np.sqrt(np.mean((yh[m] - obs[m]) ** 2, 0))
        # ★핵심 지표: 조화예측을 pre 로 썼을 때 배경(비소각)의 dNDVI 가 0 에 붙는가
        gtb = gt[:yh.shape[0], :yh.shape[1]].astype(bool)
        bgm = m & ~gtb
        d_ndvi = float(np.median((idx(yh / 1e4, 3, 2) - idx(obs / 1e4, 3, 2))[bgm]))
        print(f"[홀드아웃] {dates[i0]} 외삽 RMSE: " +
              " ".join(f"{b} {r:.0f}" for b, r in zip(ORDER, r_out)) +
              f" · 배경 dNDVI(예측−관측) {d_ndvi:+.4f}", flush=True)
        out["holdout"] = {"date": dates[i0].isoformat(),
                          "rmse": {b: float(r) for b, r in zip(ORDER, r_out)},
                          "bg_dndvi": d_ndvi}
        # ★대조군 캐시 — pre=조화예측@그날 · post=**같은 날 실관측**. 화재 이전 날짜이므로
        #   정답은 "변화 0" 이다. 여기서 면적이 나오면 그건 전부 **합성 pre 가 만든 헛것**
        #   (=학습분포 밖이라는 증거)이고, 0 에 가까우면 도메인 시프트는 문제가 아니다.
        fp = os.path.join(PROJ, "nifos_cache", f"{a.site}_harmfp_q{dates[i0].isoformat().replace('-','')}")
        os.makedirs(fp, exist_ok=True)
        H2, W2 = obs.shape[0], obs.shape[1]
        np.savez_compressed(os.path.join(fp, f"scene_{int(GSD)}m.npz"),
            x=np.concatenate([np.clip(yh, 0, 12000).astype(np.int16).transpose(2, 0, 1),
                              obs.astype(np.int16).transpose(2, 0, 1)], 0))
        np.save(os.path.join(fp, f"gt_{int(GSD)}m.npy"), np.zeros((H2, W2), np.uint8))  # 미소각
        np.save(os.path.join(fp, f"ute_{int(GSD)}m.npy"), np.array(ute))
        out["fp_cache"] = fp
        print(f"[대조군] {fp} — pre=조화@{dates[i0]} vs post=실관측 {dates[i0]} (정답 0ha)", flush=True)
        json.dump(out, open(os.path.join(PROJ, "data", f"harmonic_fit_{a.site}_holdout.json"), "w"),
                  ensure_ascii=False, indent=1)
        print("기록: data/harmonic_fit_%s_holdout.json" % a.site, flush=True)
        return

    # ── 합성 pre 로 nifos_eval prep 과 동일 규격 캐시 작성 ────────────────
    qd = dt.date.fromisoformat(a.post)
    pre = predict(C, ok, qd)                               # (H,W,NB)
    # post 6밴드는 **실제 관측**을 그대로 쓴다(조건 간 고정 변수). 어느 pre 로 prep 된
    # 캐시든 post 절반은 같으므로 q{POST} 를 가진 캐시 아무거나 쓴다.
    import glob as _g
    cands = sorted(_g.glob(os.path.join(PROJ, "nifos_cache",
                   f"{a.site}_p*_q{a.post.replace('-','')}", f"scene_{int(GSD)}m.npz")))
    if a.post_cache: cands = [os.path.join(a.post_cache, f"scene_{int(GSD)}m.npz")]
    if not cands or not os.path.exists(cands[0]):
        raise SystemExit(f"post={a.post} 씬 캐시가 없다 — nifos_eval prep(SCENE_PICK=fixed) 를 먼저 돌릴 것")
    src = os.path.dirname(cands[0]); print(f"[post] 실관측 출처 {os.path.basename(src)}", flush=True)
    z = np.load(cands[0])["x"]
    post = z[NB:]                                          # 실제 post 6밴드 (조건 간 동일 고정)
    H, W = post.shape[1], post.shape[2]
    xnew = np.concatenate([pre[:H, :W].transpose(2, 0, 1), post], 0).astype(np.int16)

    _sfx = "" if (NHARM == 2 and TREND) else f"_h{NHARM}{'t' if TREND else ''}"
    dst = os.path.join(PROJ, "nifos_cache", f"{a.site}_harm{_sfx}_q{a.post.replace('-','')}")
    os.makedirs(dst, exist_ok=True)
    np.savez_compressed(os.path.join(dst, f"scene_{int(GSD)}m.npz"), x=xnew)
    np.save(os.path.join(dst, f"gt_{int(GSD)}m.npy"), gt)
    np.save(os.path.join(dst, f"ute_{int(GSD)}m.npy"), np.array(ute))
    np.save(os.path.join(dst, f"scene_dates_{int(GSD)}m.npy"), np.array([f"harmonic@{a.post}", a.post]))
    np.savez_compressed(os.path.join(dst, "harm_coef.npz"), C=C.astype(np.float32), ok=ok, n=n)

    xf = xnew.astype(np.float32) / 1e4
    ix = lambda b, i, j: (b[i] - b[j]) / (b[i] + b[j] + 1e-6)
    gtb = gt[:H, :W].astype(bool)
    bg_ndvi = float(np.median((ix(xf[:NB], 3, 2) - ix(xf[NB:], 3, 2))[~gtb]))
    bg_nbr  = float(np.median((ix(xf[:NB], 3, 5) - ix(xf[NB:], 3, 5))[~gtb]))
    out.update({"cache": dst, "bg_dndvi": bg_ndvi, "bg_dnbr": bg_nbr,
                "coef_bytes_per_px": int(NPAR * NB * 4)})
    print(f"[합성 pre] {dst}\n  배경 dNDVI {bg_ndvi:+.4f} · 배경 dNBR {bg_nbr:+.4f} "
          f"· 계수 {NPAR*NB*4}B/화소", flush=True)
    json.dump(out, open(os.path.join(PROJ, "data", f"harmonic_fit_{a.site}{_sfx}.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"기록: data/harmonic_fit_{a.site}{_sfx}.json", flush=True)


if __name__ == "__main__":
    main()
