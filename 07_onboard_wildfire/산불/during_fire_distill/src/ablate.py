"""② student F1 굳히기 ablation — footprint 반경 sweep + SWIR 기여도.

두 실험을 오케스트레이션(train_student.py 를 env 바꿔 subprocess 호출):
 A) footprint 반경 sweep: teacher 마스크 dilate 반경(VIIRS_M)을 바꿔 재래스터화 →
    각 마스크로 LORO 학습 → 어느 반경이 광학 student 와 가장 잘 맞나(거친 vs 촘촘).
 B) SWIR ablation: 6밴드(RGB+NIR+SWIR) vs no-SWIR(RGB+NIR) 학습 → SWIR(B11/B12)가
    active-fire 신호를 얼마나 지는가(메모리: SWIR=char/burn 채널).

※2이벤트 스모크선 신호가 약함 — 진짜 결론은 ①(35이벤트) 취득 후/B200 재실행.
단일 인터프리터: /usr/bin/python3

env: WHICH=all|footprint|swir   EPOCHS=25   VAL_EVENT=<eid>
"""
import os, sys, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = "/usr/bin/python3"
WHICH  = os.environ.get("WHICH", "all")
EPOCHS = os.environ.get("EPOCHS", "25")
VAL_EVENT = os.environ.get("VAL_EVENT", "")
RESULT = os.path.join(ROOT, "runs", "ablation.json")


def run_train(tag, extra_env):
    """train_student 를 격리된 OUT 으로 실행 → result.json 반환."""
    out = os.path.join("runs", f"ablate_{tag}")
    env = dict(os.environ, EPOCHS=EPOCHS, OUT=out)
    if VAL_EVENT:
        env["VAL_EVENT"] = VAL_EVENT
    env.update(extra_env)
    print(f"\n=== 학습 [{tag}]  {extra_env} ===", flush=True)
    subprocess.run([PY, os.path.join(HERE, "train_student.py")], env=env,
                   cwd=ROOT, check=False)
    rp = os.path.join(ROOT, out, "result.json")
    if os.path.exists(rp):
        r = json.load(open(rp))
        return {"best_f1": r.get("best_f1"), "final": r.get("final_metrics"),
                "val_event": r.get("val_event")}
    return {"error": "result.json 없음"}


def run_rasterize(mask_name, viirs_m):
    env = dict(os.environ, MASK_NAME=mask_name, VIIRS_M=str(viirs_m), FORCE="1")
    print(f"\n--- 래스터화 반경 {viirs_m}m → {mask_name} ---", flush=True)
    subprocess.run([PY, os.path.join(HERE, "rasterize_teacher.py")], env=env,
                   cwd=ROOT, check=False)


def main():
    res = {}
    if os.path.exists(RESULT):
        try:
            res = json.load(open(RESULT))
        except Exception:
            pass

    if WHICH in ("all", "footprint"):
        res.setdefault("footprint_sweep", {})
        for v in (120, 200, 280):
            mask = f"FIRMS_MASK_r{v}.tif"
            run_rasterize(mask, v)
            res["footprint_sweep"][f"r{v}"] = run_train(
                f"fp_r{v}", {"MASK_NAME": mask, "BANDS_USE": "B02,B03,B04,B08,B11,B12"})
            json.dump(res, open(RESULT, "w"), ensure_ascii=False, indent=1)

    if WHICH in ("all", "swir"):
        res.setdefault("swir_ablation", {})
        configs = {"all6_RGBNIRSWIR": "B02,B03,B04,B08,B11,B12",
                   "noSWIR_RGBNIR":  "B02,B03,B04,B08",
                   "SWIRonly":       "B11,B12"}
        for tag, bands in configs.items():
            res["swir_ablation"][tag] = run_train(
                f"swir_{tag}", {"BANDS_USE": bands, "MASK_NAME": "FIRMS_MASK.tif"})
            json.dump(res, open(RESULT, "w"), ensure_ascii=False, indent=1)

    print("\n==== ABLATION 요약 ====", flush=True)
    for grp, d in res.items():
        print(f"[{grp}]", flush=True)
        for k, v in d.items():
            f1 = v.get("best_f1"); fm = v.get("final") or {}
            print(f"  {k:22s} best_F1={f1}  P={fm.get('precision')} R={fm.get('recall')}",
                  flush=True)
    json.dump(res, open(RESULT, "w"), ensure_ascii=False, indent=1)
    print(f"→ {RESULT}", flush=True)


if __name__ == "__main__":
    main()
