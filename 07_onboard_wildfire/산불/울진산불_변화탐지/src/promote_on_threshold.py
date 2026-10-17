"""산불 CD 승격 게이트 — 미학습 홀드아웃 F1 >= THRESHOLD 달성 시에만 발동.

발동 시:
  1) 최고 모델 체크포인트 + spec 을 산불 웹앱 weights/ 로 복사
  2) PROMOTED.json 플래그 기록 (idempotent — 더 좋은 모델일 때만 갱신)
  3) git add/commit/push (Steven 폴더, park-hyun-su/syntax_inference main)
미달이면: 현재 best 만 보고하고 아무것도 하지 않음 (git·복사 전부 skip).

Notion 은 MCP 가 대화형이라 무인 cron 에서 불가 → 이 스크립트는 PROMOTED.json 만 남기고,
Notion 반영은 Claude 세션이 그 플래그를 보고 수행한다.

사용:
  python3 promote_on_threshold.py            # 스캔→미달이면 no-op, 달성이면 승격+push
  python3 promote_on_threshold.py --no-push  # 커밋만, push 안 함
  python3 promote_on_threshold.py --threshold 0.70 --dry-run
"""
import os, sys, json, glob, shutil, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(HERE, "..")                       # 울진산불_변화탐지
RUNS = os.path.join(PROJ, "runs")
GIT_ROOT = os.path.abspath(os.path.join(PROJ, "..", "..", ".."))   # syntax_inference
WEBAPP = os.path.abspath(os.path.join(PROJ, "..", "..", "홍수", "물변화탐지_웹앱"))
WEIGHTS = os.path.join(WEBAPP, "weights")
FLAG = os.path.join(WEBAPP, "PROMOTED.json")
THRESHOLD = 0.70


def scan_best():
    """runs/*/results.json 전체를 훑어 최고 홀드아웃 F1 결과를 고른다."""
    best = None
    for rj in glob.glob(os.path.join(RUNS, "*", "results.json")):
        try:
            r = json.load(open(rj))
        except Exception:
            continue
        # 새 held-out 포맷만 인정: spec+per_season+f1+ckpt 필수.
        # (구식 in-domain results.json — 예: cd_unet 0.772 — 을 미학습 통과로 오인 방지)
        if not all(k in r for k in ("spec", "per_season", "f1", "ckpt")):
            continue
        r["f1"] = float(r["f1"])
        ckpt = os.path.join(os.path.dirname(rj), r["ckpt"])
        if not os.path.exists(ckpt):
            continue
        r["_results_json"] = rj
        r["_ckpt_path"] = ckpt
        if best is None or r["f1"] > best["f1"]:
            best = r
    return best


def run(cmd, cwd=GIT_ROOT):
    print("  $", " ".join(cmd), flush=True)
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")  # cron 무인: 자격증명 프롬프트로 멈추지 않게
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env, timeout=120)


def promote(best, do_push=True, dry=False):
    f1 = float(best["f1"])
    os.makedirs(WEIGHTS, exist_ok=True)
    dst_ckpt = os.path.join(WEIGHTS, "wildfire_cd.pt")
    spec = {k: best[k] for k in ("model", "script", "spec", "in_ch", "arch",
                                 "thr", "f1", "precision", "recall", "per_season")
            if k in best}
    if dry:
        print(f"[DRY] {best['_ckpt_path']} → {dst_ckpt}\n[DRY] spec={json.dumps(spec, ensure_ascii=False)}")
        return
    shutil.copy2(best["_ckpt_path"], dst_ckpt)
    json.dump(spec, open(os.path.join(WEIGHTS, "wildfire_cd.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump({"promoted": True, "f1": f1, "threshold": THRESHOLD,
               "model": best.get("model"), "spec": best.get("spec"),
               "per_season": best.get("per_season"),
               "source": os.path.relpath(best["_ckpt_path"], GIT_ROOT),
               "notion_synced": False},
              open(FLAG, "w"), ensure_ascii=False, indent=2)
    print(f"[승격] F1={f1:.3f} 모델 → 웹앱 weights/ 복사 + PROMOTED.json 기록", flush=True)

    # ── git commit + push (승격 아티팩트만; 웹앱 폴더 통째 add 금지 = 대용량 샘플 쓸어담기 버그 방지) ──
    for rel in [os.path.relpath(os.path.join(WEIGHTS, "wildfire_cd.pt"), GIT_ROOT),
                os.path.relpath(os.path.join(WEIGHTS, "wildfire_cd.json"), GIT_ROOT),
                os.path.relpath(FLAG, GIT_ROOT)]:
        run(["git", "add", "-f", rel])
    msg = (f"산불 CD 승격: 미학습 홀드아웃 F1 {f1:.3f} (>= {THRESHOLD}) — 웹앱 배포 모델 갱신\n\n"
           f"model={best.get('model')} spec={best.get('spec')} "
           f"P={best.get('precision')} R={best.get('recall')}\n"
           f"per_season={json.dumps(best.get('per_season'), ensure_ascii=False)}\n\n"
           "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>")
    c = run(["git", "commit", "-m", msg])
    print(c.stdout or c.stderr, flush=True)
    if "nothing to commit" in (c.stdout + c.stderr):
        print("[git] 변경 없음(이미 커밋됨)", flush=True)
    print("[git] 로컬 커밋 완료", flush=True)
    pushed = False
    if do_push:
        p = run(["git", "push", "origin", "main"])
        pushed = (p.returncode == 0)
        print(p.stdout or p.stderr, flush=True)
        print("[git] push 완료" if pushed else
              "[git] push 실패 — 로컬 커밋은 됨. Claude 세션/수동으로 push 필요(cron엔 자격증명 없음).",
              flush=True)
    # 플래그에 push 결과 반영(Claude 세션이 이걸 보고 push+Notion 마무리)
    flag = json.load(open(FLAG))
    flag["pushed"] = pushed
    json.dump(flag, open(FLAG, "w"), ensure_ascii=False, indent=2)


def main():
    global THRESHOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="임계 미달이어도 최고 모델 강제 승격")
    a = ap.parse_args()
    THRESHOLD = a.threshold

    best = scan_best()
    if best is None:
        print("[게이트] 저장된 results.json/ckpt 없음 — 학습이 먼저 체크포인트를 저장해야 함.")
        return 0
    f1 = float(best["f1"])
    print(f"[게이트] 현재 최고: {best.get('model')} F1={f1:.3f} "
          f"(P={best.get('precision')} R={best.get('recall')}) thr>={THRESHOLD}")

    if f1 < THRESHOLD and not a.force:
        print(f"[게이트] 미달 ({f1:.3f} < {THRESHOLD}) — 승격/커밋 없음. 대기.")
        return 0

    # idempotency: 이미 같거나 더 좋은 모델로 승격돼 있으면 skip
    if os.path.exists(FLAG):
        prev = json.load(open(FLAG))
        if float(prev.get("f1", -1)) >= f1:
            print(f"[게이트] 이미 F1={prev['f1']} 로 승격됨(>= 현재 {f1:.3f}) — skip.")
            return 0

    promote(best, do_push=not a.no_push, dry=a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
