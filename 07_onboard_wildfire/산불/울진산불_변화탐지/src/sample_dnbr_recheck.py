"""판독 후 재검 — 경계 사례를 dNBR 로 일괄 재확인한다.

왜 필요한가: 고성 2019_74 는 RE CIR 에 **담황갈·담회 능선**이 넓어(낙엽활엽·초지의 계절 모습)
어두워짐이 약한 화소를 눈으로 못 가른다. 판독 중 그런 점은 `conf: 하` 로 적어 두고 여기서 일괄로
다시 본다(READING_STATE.md).

★자의 독립성 주의: dNBR 은 **S2** 로 계산한다 = 모델 입력과 같은 센서다. 그래서 이 재검은
판정을 **뒤집는 도구가 아니라** ①눈 판독이 dNBR 분포와 어긋나는 점을 **골라내는** 체이고
②`basis` 를 RE+S2 로 남겨 `sample_estimate.py` 의 'RE 단독' 민감도 분석이 그 영향을 재게 한다.

기준선은 절대 임계값이 아니라 **같은 사이트의 확신(상) 판독 분포**다 — 씬마다 계절·센서 상태가
달라 절대 dNBR 컷은 사이트 간에 옮겨 다니지 못한다(§dNBR 라벨 함정).

사용: ~/miniforge3/envs/ml/bin/python src/sample_dnbr_recheck.py [--site goseong] [--conf 하]
"""
import os, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
NIR, SWIR2 = 3, 5          # ko_build_refl.ORDER = B02,B03,B04,B8A,B11,B12
PRE, POST = 0, 6


def nbr(x, off):
    a = x[off + NIR].astype(np.float32); b = x[off + SWIR2].astype(np.float32)
    return (a - b) / np.maximum(a + b, 1e-6)


def dnbr_scene(site, cache={}):
    if site not in cache:
        z = np.load(os.path.join(PROJ, "nifos_cache", f"{site}_refoot", "scene_20m.npz"))
        x = z["x"]
        cache[site] = nbr(x, PRE) - nbr(x, POST)
    return cache[site]


def reads(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for l in fh:
            l = l.strip()
            if l:
                r = json.loads(l); out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", default=os.path.join(PROJ, "sample", "claude_read_main.jsonl"))
    ap.add_argument("--site", default="goseong")
    ap.add_argument("--conf", default="하")
    ap.add_argument("--out", default=os.path.join(PROJ, "sample", "dnbr_recheck.json"))
    a = ap.parse_args()
    S = json.load(open(os.path.join(PROJ, "sample", "sample_points.json"), encoding="utf-8"))
    by = {p["id"]: p for p in S["points"]}
    R = reads(a.reads)
    d = dnbr_scene(a.site)

    def val(i):
        p = by[i]
        return float(d[p["row"], p["col"]])

    site_ids = [i for i in R if i in by and by[i]["site"] == a.site and not by[i].get("qc")]
    ref = {}
    for kind in ("소각", "비변화", "벌채·개간"):
        v = [val(i) for i in site_ids if R[i]["read"] == kind and R[i].get("conf") == "상"]
        ref[kind] = np.array(v)
    print(f"[{a.site}] 확신(상) 판독의 dNBR 분포 — 이것이 이 씬의 자다")
    for k, v in ref.items():
        if v.size:
            q = np.percentile(v, [10, 50, 90])
            print(f"  {k:<7} n={v.size:<4} 중앙 {q[1]:+.3f}   10~90% {q[0]:+.3f}~{q[2]:+.3f}")
        else:
            print(f"  {k:<7} n=0  — 기준선 없음")

    # ★규칙은 비대칭이다. 확신(상) **소각**은 몇 점 안 되는 씬이 있어(고성 n=2) 소각 쪽 분포로는
    #   자를 못 만든다. 대신 표본이 두꺼운 **비변화를 귀무분포**로 두고 "이 화소가 안 탄 이웃들의
    #   dNBR 범위 안에 있는가"만 묻는다 — 판정을 뒤집는 게 아니라 **다시 볼 점을 고르는** 체다.
    n0 = ref["비변화"]
    if n0.size < 10:
        print("  ⚠비변화 기준선이 10점 미만이다 — 이 사이트는 재검을 돌리지 말 것."); return
    hi = float(np.percentile(n0, 95))     # 비변화라 읽었는데 이보다 위면 = 안 탄 분포 밖
    mid = float(np.percentile(n0, 90))    # 소각이라 읽었는데 이보다 아래면 = 안 탄 분포 안
    print(f"  → 귀무(비변화) 분포: p90 {mid:+.3f} · p95 {hi:+.3f}  "
          f"(소각 확신(상) 중앙 {np.median(ref['소각']):+.3f} · n={ref['소각'].size})\n")

    tgt = sorted(i for i in site_ids if R[i].get("conf") == a.conf)
    rows = []
    for i in tgt:
        v = val(i); rd = R[i]["read"]
        if rd == "비변화":
            flag, why = v > hi, f"안 탄 분포(p95 {hi:+.3f}) 밖"
        elif rd == "소각":
            flag, why = v < mid, f"안 탄 분포(p90 {mid:+.3f}) 안"
        else:
            flag, why = False, ""
        rows.append(dict(id=i, read=rd, dnbr=round(v, 4), flag=bool(flag),
                         why=why if flag else "", note=R[i].get("note", "")))
    fl = [r for r in rows if r["flag"]]
    print(f"conf={a.conf} 대상 {len(rows)}점 · 다시 볼 점 {len(fl)}점")
    for r in rows:
        print(f"  {r['id']}  판독 {r['read']:<7} dNBR {r['dnbr']:+.3f}"
              + (f"  ←다시 볼 것: {r['why']}" if r["flag"] else ""))
    json.dump(dict(site=a.site, conf=a.conf, p90=mid, p95=hi, n_null=int(n0.size),
                   med_burn=(float(np.median(ref["소각"])) if ref["소각"].size else None),
                   n_burn=int(ref["소각"].size), rows=rows),
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ {a.out}")
    print("※플래그가 붙은 점은 칩을 다시 열어 눈으로 정한다. dNBR 이 판독을 자동으로 덮어쓰지 않는다 —"
          " 그러면 표본이 S2(모델 입력)로 잰 것이 되어 모델과 순환한다.")


if __name__ == "__main__":
    main()
