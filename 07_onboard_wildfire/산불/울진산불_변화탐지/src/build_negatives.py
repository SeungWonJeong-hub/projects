"""음성 타일 — **불이 안 난 같은 장소**를 학습에 넣는다 (§13.4·§16.3 의 다음 실험).

왜: 학습 타일이 전부 화재 중심 크롭이라 모델은 **소각 화소가 0 인 타일을 본 적이 없다.**
그래서 "이 장면 어딘가는 탔다"를 전제하고, 화재가 없어도 장면의 21~33% 를 소각으로 찍는다
(§13.2). 게다가 학습이 봄에 몰려 있어 가을·여름 외형을 소각으로 착각한다(§13.5·§16.3).

설계 — 교란을 하나만 남긴다:
  · **같은 화재 위치**의 같은 크롭을 쓴다 → "장소"가 아니라 **"불의 유무"** 만 달라진다.
  · pre/post 를 **둘 다 화재 이전**에서 뽑는다 → 정답은 언제나 소각 0.
  · 화재마다 계절을 **봄/여름/가을로 돌려가며** 잡는다 → 봄 편중을 깬다.

가짜 화재일 = 실제 화재일 − 1년 + SHIFT(0/90/180일). 그 뒤 학습과 **같은 창 규칙**으로
pre(−1년±) / post(+15~100일) 를 뽑는다. 가장 늦은 post 도 실제 화재일보다 85일 이상 앞선다.

출력: ko_neg_<pool>/<fire>/*.npz — x 12ch, y=0, valid. 학습 타일과 같은 규격.
사용: conda run -n geo python src/build_negatives.py [N_FIRES]
      MIN_HA=50 SHARD_N=4 SHARD_I=0 … (ko_build_refl 과 같은 env)
"""
from __future__ import annotations
import os, sys, math, glob, datetime as dt
import numpy as np

os.environ.setdefault("PROJ_DATA", os.path.expanduser("~/miniforge3/envs/geo/share/proj"))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_av = sys.argv; sys.argv = [_av[0]]
import ko_build_refl as K                      # 크롭·STAC·밴드 규칙을 그대로 상속
sys.argv = _av
from osgeo import ogr, osr

SHIFTS = [0, 90, 180]                          # 봄/여름/가을로 돌린다
GAP_D = int(os.environ.get("GAP_D", "60"))     # 실제 화재일로부터 최소 이격
OUT = os.environ.get("NEG_OUT", os.path.join(HERE, "..", "ko_neg_" + os.path.basename(K.OUT)))
N_FIRES = int(sys.argv[1]) if len(sys.argv) > 1 else 245


def build_neg(name, gu, fdate, shift):
    """가짜 화재일 기준으로 pre/post 를 뽑아 **소각 0** 타일을 만든다."""
    e = gu.GetEnvelope(); cx = (e[0] + e[1]) / 2; cy = (e[2] + e[3]) / 2
    hx = max((e[1] - e[0]) / 2 + 500, K.CROP_M); hy = max((e[3] - e[2]) / 2 + 500, K.CROP_M)
    ute = [math.floor(cx - hx), math.floor(cy - hy), math.ceil(cx + hx), math.ceil(cy + hy)]
    dst = osr.SpatialReference(); dst.ImportFromEPSG(4326); dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src = osr.SpatialReference(); src.ImportFromEPSG(K.UTM); src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)
    lo1, la1, _ = ct.TransformPoint(e[0], e[2]); lo2, la2, _ = ct.TransformPoint(e[1], e[3])
    bb = [min(lo1, lo2) - 0.02, min(la1, la2) - 0.02, max(lo1, lo2) + 0.02, max(la1, la2) + 0.02]

    pseudo = fdate - dt.timedelta(days=365) + dt.timedelta(days=shift)
    if (fdate - pseudo).days < GAP_D + 100:
        return 0, "가짜 화재일이 실제 화재일에 너무 가깝다"
    pc = pseudo - dt.timedelta(days=365)
    pre = K.stac_refl(bb, K.rfc(pc - dt.timedelta(days=30), pc + dt.timedelta(days=45)), ute)
    pre_src = "prevyear"
    if pre is None:
        # ko_build_refl 이 2016~2017 화재에 쓰는 **같은 해 pre** 폴백과 같은 규칙.
        # 양성 풀도 옛 화재는 이 구조라, 음성만 다른 간격을 쓰면 "간격" 이 지름길 단서가 된다.
        pre = K.stac_refl(bb, K.rfc(pseudo - dt.timedelta(days=60), pseudo - dt.timedelta(days=8)), ute)
        pre_src = "sameyear"
    qhi = min(pseudo + dt.timedelta(days=100), fdate - dt.timedelta(days=GAP_D))
    post = K.stac_refl(bb, K.rfc(pseudo + dt.timedelta(days=15), qhi), ute)
    if pre is None or post is None:
        return 0, "장면 없음"
    H = min(pre['B02'].shape[0], post['B02'].shape[0]); W = min(pre['B02'].shape[1], post['B02'].shape[1])
    for d in (pre, post):
        for k in d: d[k] = d[k][:H, :W]
    valid = (pre['B02'] > 0) & (pre[K.NIR] > 0) & (pre['B12'] > 0) & \
            (post['B02'] > 0) & (post[K.NIR] > 0) & (post['B12'] > 0)
    valid &= ~np.isin(pre['SCL'], [0, 1, 3, 8, 9, 10, 11]) & ~np.isin(post['SCL'], [0, 1, 3, 8, 9, 10, 11])
    x = np.stack([pre[k] for k in K.ORDER] + [post[k] for k in K.ORDER], 0).astype(np.int16)
    T = K.T
    if H < T or W < T: return 0, f"크롭<{T}"
    od = os.path.join(OUT, name); os.makedirs(od, exist_ok=True); n = 0
    ys = sorted(set(min(y, H - T) for y in range(0, H - T + 1, K.STRIDE)))
    xs = sorted(set(min(x_, W - T) for x_ in range(0, W - T + 1, K.STRIDE)))
    for y0 in ys:
        for x0 in xs:
            v = valid[y0:y0 + T, x0:x0 + T]
            if v.mean() < 0.4: continue
            np.savez_compressed(os.path.join(od, f"{name}_neg{shift}_{y0}_{x0}.npz"),
                                x=x[:, y0:y0 + T, x0:x0 + T],
                                y=np.zeros((T, T), np.uint8),          # ★소각 0
                                valid=v.astype(np.uint8))
            n += 1
    if n == 0:
        try: os.rmdir(od)
        except OSError: pass
    return n, f"pseudo {pseudo} (shift {shift}d, pre={pre_src})"


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = ogr.Open(K.SHP); ly = ds.GetLayer(); cand = []
    for f in ly:
        if f.GetField("OCCU_YEAR") not in K.YEARS: continue
        g = f.GetGeometryRef()
        if not g: continue
        s = osr.SpatialReference(); s.ImportFromEPSG(5186); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        d = osr.SpatialReference(); d.ImportFromEPSG(K.UTM); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        gu = g.Clone(); gu.Transform(osr.CoordinateTransformation(s, d)); ha = gu.GetArea() / 1e4
        if not (K.MIN_HA <= ha <= K.MAX_HA): continue
        od = f.GetField("OCCU_DATE")
        try: fdate = dt.date(int(od[:4]), int(od[4:6]), int(od[6:8]))
        except Exception: continue
        cand.append((ha, fdate, gu.Clone()))
    cand.sort(key=lambda c: c[0], reverse=True)
    sel = cand[:N_FIRES][K.SHARD_I::K.SHARD_N]
    print(f"음성 타일 빌드 — 후보 {len(cand)} 중 상위 {N_FIRES}, 샤드 {K.SHARD_I}/{K.SHARD_N} = {len(sel)}건",
          flush=True)
    print(f"출력 {OUT} · 계절 shift {SHIFTS} 를 화재마다 돌려가며", flush=True)
    tot = ok = 0
    for i, (ha, fdate, gu) in enumerate(sel):
        name = f"kf_{fdate.isoformat().replace('-','')}_{int(ha)}ha"
        if glob.glob(os.path.join(OUT, name, "*.npz")):
            print(f"[{name}] skip", flush=True); ok += 1; continue
        shift = SHIFTS[(i + K.SHARD_I) % len(SHIFTS)]
        try:
            n, why = build_neg(name, gu, fdate, shift)
        except Exception as ex:
            n, why = 0, f"실패 {type(ex).__name__}: {str(ex)[:60]}"
        tot += n; ok += (n > 0)
        print(f"[{name}] 타일 {n} · {why}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  진행 {i+1}/{len(sel)} 누적 {tot}", flush=True)
    print(f"완성 화재 {ok} | 총 음성타일 {len(glob.glob(os.path.join(OUT,'*','*.npz')))}", flush=True)


if __name__ == "__main__":
    main()
