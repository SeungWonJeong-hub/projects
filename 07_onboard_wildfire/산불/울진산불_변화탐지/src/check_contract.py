"""학습/서빙 계약 검사기 — 어긋나면 **에러 없이 성능만 조용히 깎이는** 종류의 버그를 잡는다.

실제로 났던 버그들이 이 검사의 근거다:
  · `acquire.WANT` 에 B8A 가 없어 `_rgbnir` 이 항상 B08 로 폴백 (2026-09-08 발견)
  · 가장자리 부분타일 제로패딩 (F1 0.828→0.847 손실)
  · `eval_boundary_tolerant.py` 의 `tau*20` 하드코딩 (10m 에서 허용오차 2배)
전부 예외를 던지지 않아 로그만 봐서는 알 수 없었다. 그래서 사람이 아니라 검사가 지켜야 한다.

계약 출처(단일 소스): `ko_build_refl.py:9` · `nifos_eval.py:21` · `웹앱/weights/wildfire_cd.json`

사용: python check_contract.py          # 위반 있으면 exit 1
      python check_contract.py --warn   # 경고만(미완성 항목은 실패로 치지 않음)
git 훅으로 걸려면: ln -s ../../Steven/산불/울진산불_변화탐지/src/check_contract.py .git/hooks/pre-commit
"""
from __future__ import annotations
import os, re, sys, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.join(HERE, "..")
WEBAPP = os.path.abspath(os.path.join(PROJ, "..", "..", "웹앱"))
SPEC = os.path.join(WEBAPP, "weights", "wildfire_cd.json")

# ── 계약표: 이 값이 진실이다. 바꾸려면 학습부터 바꾸고 여기를 고쳐라 ──────────────
CONTRACT = {"20m": dict(gsd=20, nir="B8A", tile=256),    # B8A=865nm 네이티브 20m
            "10m": dict(gsd=10, nir="B08", tile=512)}    # B08=842nm 네이티브 10m
FOV_M = 5120                                              # 지상시야 T*GSD — 해상도 A/B 공정성의 근거
NIR_ASSET = {"B8A": "nir08", "B08": "nir"}                # earth-search asset key

FAIL, WARN = [], []


def bad(msg):
    FAIL.append(msg)


def warn(msg):
    WARN.append(msg)


def _code_lines(src):
    """주석·독스트링을 지운 줄 목록 — 문서에 남은 버그 설명을 위반으로 오인하지 않기 위해."""
    src = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', "", src)
    return [ln.split("#")[0] for ln in src.split("\n")]


def read(path):
    return open(path, encoding="utf-8").read() if os.path.exists(path) else ""


def check_spec():
    """① 배포 스펙이 계약표와 같은가 ② 지상시야가 고정인가 ③ 가중치가 실제로 있는가"""
    if not os.path.exists(SPEC):
        return bad(f"스펙 없음: {SPEC}")
    doc = json.load(open(SPEC))
    variants = doc.get("variants") or {"20m": doc}
    for res, want in CONTRACT.items():
        v = variants.get(res)
        if v is None:
            warn(f"[스펙] {res} 변형이 없다 (아직 미도입이면 정상)"); continue
        for k, exp in want.items():
            got = v.get(k)
            if got is not None and str(got) != str(exp):
                bad(f"[스펙] {res}.{k} = {got} — 계약은 {exp} "
                    f"(학습이 {want['nir']}@{want['tile']}px 로 됐다면 서빙도 같아야 한다)")
        fov = float(v.get("gsd", want["gsd"])) * int(v.get("tile", want["tile"]))
        if abs(fov - FOV_M) > 1:
            bad(f"[스펙] {res} 지상시야 {fov:.0f}m ≠ {FOV_M}m — 해상도 비교가 문맥량까지 바꾼다")
        ens = v.get("ensemble")
        if ens is None:
            continue
        if not ens:
            warn(f"[스펙] {res} 앙상블 0개 — 배관만 있고 가중치 미연결(UI 는 '가중치 없음' 표시)")
            continue
        miss = [e["ckpt"] for e in ens
                if not os.path.exists(os.path.join(WEBAPP, "weights", e["ckpt"]))]
        if miss:
            bad(f"[스펙] {res} 가중치 없음 {len(miss)}개: {miss[:2]}")
        if v.get("thr") is None:
            bad(f"[스펙] {res} thr 없음 — 화재별 best-thr 로 잰 F1 은 배포 성능이 아니다")


def check_acquire():
    """④ 서빙이 각 변형에 필요한 NIR 밴드를 실제로 내려받는가 (이번 버그의 진원지)"""
    src = read(os.path.join(WEBAPP, "core", "acquire.py"))
    if not src:
        return warn("[취득] acquire.py 없음 — 웹앱 미배치 환경")
    m = re.search(r"WANT\s*=\s*\{(.*?)\}", src, re.S)
    if not m:
        return bad("[취득] acquire.WANT 를 못 찾음")
    want = m.group(1)
    for res, c in CONTRACT.items():
        band, asset = c["nir"], NIR_ASSET[c["nir"]]
        if f'"{band}"' not in want:
            bad(f"[취득] WANT 에 {band} 가 없다 — {res} 모델이 학습된 밴드를 못 받는다")
        elif f'"{asset}"' not in want:
            bad(f"[취득] WANT 의 {band} asset key 가 '{asset}' 이 아니다")


def check_no_silent_fallback():
    """⑤ 밴드가 없을 때 조용히 다른 밴드로 대체하지 않는가.
    **엔진만 보면 안 된다** — app.py 의 심각도 dNBR 이 같은 폴백을 쓰고 있던 걸
    2026-09-09 에 놓쳤다(검사 범위가 wildfire_engine.py 하나뿐이었다)."""
    for rel in (("core", "wildfire_engine.py"), ("app.py",)):
        src = read(os.path.join(WEBAPP, *rel))
        where = "/".join(rel)
        if not src:
            warn(f"[엔진] {where} 없음"); continue
        if where.endswith("wildfire_engine.py") and "BandMissing" not in src:
            bad("[엔진] BandMissing 예외가 없다 — 밴드 누락이 조용한 대체로 넘어간다")
        # spec 경로(band 인자 있음)에서 get(A, get(B)) 폴백이 살아있으면 안 된다
        for mm in re.finditer(r'\.get\(\s*"(B8A|B08)"\s*,\s*\w+\.get\(\s*"(B8A|B08)"', src):
            ln = src[:mm.start()].count("\n") + 1
            seg = src[max(0, mm.start() - 400):mm.start()]
            if "band is None" in seg:          # _nir(band=None) 레거시 관용 경로만 허용
                continue
            fn = "?"
            for fm in re.finditer(r"^def (\w+)", src[:mm.start()], re.M):
                fn = fm.group(1)
            bad(f"[엔진] {where}:{ln} ({fn}) NIR 암묵 폴백 `get(B8A, get(B08))` — "
                f"스펙이 정한 밴드를 무시한다. 10m 변형에서 모델은 B08 을 보는데 여기만 B8A 를 쓴다")
    # 지수(dNBR)도 스펙 NIR 을 써야 한다 — `bands["B08"]` 하드코딩은 폴백 정규식에 안 걸린다
    ana = "\n".join(_code_lines(read(os.path.join(WEBAPP, "core", "analyze.py"))))
    if ana:
        nb = re.search(r"def _nbr\(.*?\n(?=\ndef |\Z)", ana, re.S)
        if nb and "spec_nir" not in nb.group(0):
            bad("[엔진] analyze._nbr 이 스펙 NIR 을 안 쓴다 — 20m 모델(B8A)과 지수가 어긋난다")


def check_area_from_grid():
    """⑨ 면적이 **선택한 해상도의 실제 격자**에서 나오는가.

    소각면적은 이 제품의 산출물 자체다(피해평가 근거). 픽셀면적을 20m 로 하드코딩해 두면
    10m 을 골랐을 때 면적이 4배로 나오는데 에러는 안 난다 — 2026-09-09 에 실제로 그랬다.

    2026-09-10 등급 절대면적을 미표시로 정하면서(487c381) app.py 의 `_pxa` 가 사라졌다.
    그래서 `_pxa` 가 **있기를** 요구하지 않고 두 가지를 본다:
      ① `_pxa` 가 다시 생기면 격자에서 와야 한다(상수·비격자 금지).
      ② 웹앱 어디서든 '화소수 × 상수' 면적은 해상도가 고정된 경로라고 같은 줄이나 바로 위
         두 줄에 `고정` 으로 선언해야 한다 — 선언 없으면 위반, 있으면 매번 경고로 보인다."""
    import glob
    app = read(os.path.join(WEBAPP, "app.py"))
    if not app:
        return warn("[면적] app.py 없음")
    code = "\n".join(_code_lines(app))
    m = re.search(r"^\s*_pxa\s*=\s*(.+)$", code, re.M)
    if m:
        expr = m.group(1)
        if re.fullmatch(r"[\d.eE+-]+\s*", expr):
            bad(f"[면적] _pxa 가 상수({expr.strip()})다 — 해상도를 바꾸면 면적이 틀린다. "
                f"실제 격자(profile transform) 나 선택 gsd 에서 유도할 것")
        elif "transform" not in expr and "gsd" not in expr.lower():
            bad(f"[면적] _pxa 가 격자에서 오지 않는다: {expr.strip()}")

    const = re.compile(r"(?:\.sum\(\)\)?|count\w*)\s*\*\s*(?:0\.04|0\.0004|4e-2|4e-4|400|0\.01|100)\b"
                       r"|\b(?:0\.0004|0\.04)\s*\*")
    files = ([os.path.join(WEBAPP, "app.py")] + sorted(glob.glob(os.path.join(WEBAPP, "core", "*.py")))
             + sorted(glob.glob(os.path.join(WEBAPP, "pages", "*.py"))))
    for f in files:
        src = read(f)
        # 독스트링을 줄 수를 유지한 채 지운다 — 원본 줄(주석 포함)과 번호를 맞춰 `고정` 선언을 읽기 위해
        body = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', lambda s: "\n" * s.group(0).count("\n"), src)
        raw = src.split("\n")
        for i, ln in enumerate(body.split("\n")):
            if not const.search(ln.split("#")[0]):
                continue
            where = f"{os.path.relpath(f, WEBAPP)}:{i + 1}"
            if any("고정" in raw[j] for j in range(max(0, i - 2), i + 1)):
                warn(f"[면적] {where} 상수 픽셀면적 — 해상도 고정 경로로 선언됨. 해상도 선택이 생기면 격자에서 유도할 것")
            else:
                bad(f"[면적] {where} 화소수×상수 면적인데 해상도 고정 선언이 없다 — 10m 을 고르면 면적이 틀린다")


def check_tiling():
    """⑥ 타일 추론이 가장자리 정렬+겹침평균인가 (제로패딩이면 F1 -0.018)"""
    for f, need in [(os.path.join(WEBAPP, "core", "wildfire_engine.py"), "_tile_apply"),
                    (os.path.join(HERE, "nifos_eval.py"), "def infer")]:
        src = read(f)
        if not src:
            warn(f"[타일] {os.path.basename(f)} 없음"); continue
        if need not in src:
            bad(f"[타일] {os.path.basename(f)} 에 {need} 가 없다")
        if re.search(r"np\.zeros\(\(\s*\d+\s*,\s*T\s*,\s*T\s*\)\)", src):
            bad(f"[타일] {os.path.basename(f)} 에 제로패딩 부분타일이 남아 있다")


def check_tau_meters():
    """⑦ 경계허용 오차가 픽셀이 아니라 미터 고정인가 (10m 에서 허용오차가 2배가 된다)"""
    src = read(os.path.join(HERE, "eval_boundary_tolerant.py"))
    if not src:
        return warn("[허용오차] eval_boundary_tolerant.py 없음")
    # 주석·독스트링에 남은 "예전엔 tau*20 이었다" 설명을 위반으로 오인하면 안 된다 → 실행줄만 본다.
    for i, ln in enumerate(_code_lines(src), 1):
        if re.search(r"tau\s*\*\s*(20|10)\b", ln):
            bad(f"[허용오차] eval_boundary_tolerant.py:{i} tau*GSD 하드코딩 — tau_m/GSD 로 환산해야 한다")


def check_valid_def():
    """⑧ 서빙 valid 가 학습 정의(pre/post 반사율 + 양쪽 SCL)와 같은가.

    2026-09-09 수정 후에는 **부재 검사가 아니라 존재 검사**다 — 옛 문자열이 사라진 것만 보면
    다른 형태로 느슨해질 때(예: `valid = ba["B02"] > 0`) 못 잡는다."""
    eng = read(os.path.join(WEBAPP, "core", "wildfire_engine.py"))
    ana = read(os.path.join(WEBAPP, "core", "analyze.py"))
    if not eng or not ana:
        return
    # ★실행줄만 본다 — 독스트링에 "scl_before 도 받는다" 라고 써 두면 검사가 통과해 버린다
    # (2026-09-09 음성테스트로 실제로 걸린 오탐. tau 검사와 같은 이유다.)
    eng = "\n".join(_code_lines(eng))
    ana = "\n".join(_code_lines(ana))
    # (a) 학습의 무효 SCL 목록 — no-data(0)·결함(1) 포함
    m = re.search(r"^BAD_SCL\s*=\s*\(([^)]*)\)", eng, re.M)
    if not m:
        bad("[valid] wildfire_engine 에 BAD_SCL 이 없다 — 무효 SCL 정의는 한 곳에서만")
    else:
        got = {int(x) for x in re.findall(r"\d+", m.group(1))}
        want = {0, 1, 3, 8, 9, 10, 11}          # ko_build_refl.py:114
        if got != want:
            bad(f"[valid] BAD_SCL={sorted(got)} 가 학습(ko_build_refl.py:114)의 {sorted(want)} 와 다르다")
    # (b) valid_mask 를 **실제로 돌려** 8개 항이 각각 먹는지 본다.
    #     정적 토큰 검사는 너무 약했다 — 루프에서 pre 를 빼도 시그니처의 `bb` 때문에 통과했다
    #     (2026-09-09 음성테스트 ④). 행동 검사는 구현 형태가 바뀌어도 계약만 본다.
    if "def valid_mask(" not in eng:
        bad("[valid] wildfire_engine.valid_mask 가 없다 — 학습 valid 정의를 서빙이 갖고 있어야 한다")
    else:
        _check_valid_behavior()


    # (c) 소각 마스크·소각비율이 그 valid 를 쓰는가 (분모가 느슨하면 UI 비율이 낮게 나온다)
    bm = re.search(r"def burn_mask_model\(.*?\n(?=\n\ndef |\Z)", eng, re.S)
    if bm and "valid_mask(" not in bm.group(0):
        bad("[valid] burn_mask_model 이 valid_mask 를 안 쓴다 — burn_pct 분모가 느슨해진다")
    if re.search(r'valid\s*=\s*\(?\s*b[ab]\["B0[23]"\]\s*>\s*0\s*\)?\s*$', eng, re.M):
        bad("[valid] wildfire_engine 에 단일밴드 valid 가 남아 있다 — valid_mask 를 쓸 것")
    # (d) analyze 가 pre SCL 을 받아 실제로 쓰는가
    # 시그니처는 `):` 까지로 잘라 본다 — 기본값 안의 괄호(engine="지수(NDWI)")에 걸리면 안 된다.
    def _sig(src, name):
        m = re.search(r"def %s\(.*?\):" % name, src, re.S)
        return m.group(0) if m else ""
    if "scl_before" not in _sig(ana, "_ok"):
        bad("[valid] analyze._ok() 가 pre SCL 을 안 받는다 — 학습은 pre/post 양쪽 SCL 을 뺀다")
    ok = re.search(r"def _ok\(.*?\n(?=\ndef |\Z)", ana, re.S)
    if ok and "scl_before" not in ok.group(0).split("):", 1)[-1]:
        bad("[valid] analyze._ok() 가 scl_before 를 받기만 하고 쓰지 않는다")
    if "scl_before" not in _sig(ana, "analyze"):
        bad("[valid] analyze() 가 scl_before 를 안 받는다 — 배관이 pre SCL 을 버린다")
    if not re.search(r"scl_before\s*=\s*scl_before", ana):
        bad("[valid] analyze 가 scl_before 를 burn_mask_model 로 안 넘긴다")



def _check_valid_behavior():
    """valid_mask 가 pre/post × (B02·NIR·B12) + 양쪽 SCL 8개 항 전부에 반응하는가.
    항 하나를 무효로 만들었을 때 그 화소가 valid 에서 빠져야 한다."""
    try:
        import numpy as np
        sys.path.insert(0, WEBAPP)
        from core.wildfire_engine import valid_mask, BAD_SCL     # torch 미임포트
    except Exception as e:                                       # 웹앱이 없는 체크아웃 등
        warn(f"[valid] 행동 검사를 건너뜀({type(e).__name__}: {e})")
        return
    H = W = 4
    mk = lambda: {k: np.ones((H, W), "float32") for k in
                  ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")}
    clean = np.full((H, W), 4, "uint8")                          # 4=식생, 유효
    base = valid_mask(mk(), mk(), clean.copy(), clean.copy())
    if not base.all():
        bad("[valid] 결함 없는 씬인데 valid 가 전부 True 가 아니다 — valid_mask 가 과하게 깎는다")
    nir = {"pre": "B8A", "post": "B8A"}                          # 20m 스펙 기준(_spec_nir)
    for tag in ("pre", "post"):
        for band in ("B02", nir[tag], "B12"):
            bb, ba = mk(), mk()
            (bb if tag == "pre" else ba)[band][0, 0] = 0
            if valid_mask(bb, ba, clean.copy(), clean.copy())[0, 0]:
                bad(f"[valid] {tag}/{band}=0 인 화소가 유효로 남는다 — 학습 정의(ko_build_refl.py:113)와 다르다")
    for tag in ("pre", "post"):
        sb, sa = clean.copy(), clean.copy()
        (sb if tag == "pre" else sa)[0, 0] = BAD_SCL[-1]          # 눈/구름 등 무효 클래스
        if valid_mask(mk(), mk(), sb, sa)[0, 0]:
            bad(f"[valid] {tag} SCL 이 무효인 화소가 유효로 남는다 — 학습은 양쪽 SCL 을 뺀다")


def _reexec_with_numpy():
    """numpy 가 없는 파이썬으로 불렸으면 numpy 가 있는 쪽으로 한 번만 갈아탄다.

    훅은 맨 `python3` 로 부르는데 이 머신의 python3 엔 numpy 가 없어서 **행동 검사가 조용히
    건너뛰어졌다**(2026-09-09 발견). 검사를 지키는 자리에서 안 도는 검사는 없는 것과 같다.
    후보는 `CONTRACT_PY` 로 덮어쓸 수 있다 — runs_sh/*.sh 의 `PY=${PY:-...}` 와 같은 규약."""
    if os.environ.get("_CONTRACT_REEXEC"):
        return                                   # 이미 갈아탔다 — 무한재귀 방지
    try:
        import numpy  # noqa: F401
        return
    except ImportError:
        pass
    home = os.path.expanduser("~")
    cands = [os.environ.get("CONTRACT_PY"),
             os.path.join(home, "envs", "torch", "bin", "python"),
             os.path.join(home, "miniforge3", "envs", "geo", "bin", "python"),
             os.path.join(home, "miniforge3", "envs", "ml", "bin", "python")]
    for py in cands:
        if not py or not os.path.exists(py) or os.path.realpath(py) == os.path.realpath(sys.executable):
            continue
        if subprocess.call([py, "-c", "import numpy"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
            continue
        env = dict(os.environ, _CONTRACT_REEXEC="1")
        sys.exit(subprocess.call([py, os.path.abspath(__file__)] + sys.argv[1:], env=env))


def main():
    _reexec_with_numpy()
    warn_only = "--warn" in sys.argv
    for fn in (check_spec, check_acquire, check_no_silent_fallback,
               check_tiling, check_tau_meters, check_valid_def, check_area_from_grid):
        try:
            fn()
        except Exception as e:
            bad(f"[{fn.__name__}] 검사 자체가 실패: {e}")
    print(f"학습/서빙 계약 검사 — 위반 {len(FAIL)} · 경고 {len(WARN)}")
    for m in FAIL:
        print("  ✗", m)
    for m in WARN:
        print("  !", m)
    if not FAIL and not WARN:
        print("  ✓ 계약 일치")
    return 1 if (FAIL and not warn_only) else 0


if __name__ == "__main__":
    sys.exit(main())
