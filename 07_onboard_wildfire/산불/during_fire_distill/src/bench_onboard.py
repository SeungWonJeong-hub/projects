"""온보드 추론 효율화 벤치 — detect_during(student U-Net)를 FP32/FP16/TensorRT로 대결.

컴퓨트 분담: 온보드 실시간 탐지=Jetson. 위성 전력·지연 예산이 실제 제약이므로
정밀도별 지연(ms/tile·Mpix/s)·전력(tegrastats VDD)·정확도 손실(확률 diff)을 실측.
[[gpu-detect-power-v2-2026-09-09]] 전력측정 방법론 계승, duty-cycle 게이팅 연계.

단일 인터프리터: /usr/bin/python3 (torch 2.11 cuda · tensorrt 10.3 · onnx)

산출: runs/bench/onboard_bench.json  + ONNX/엔진은 runs/bench/
env: TILE=256 ITERS=100 WARMUP=20 SCENE=<dir>(실타일 소스)
"""
import os, sys, json, time, subprocess, threading, re
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from train_student import UNet, BANDS
from detect_during import load_bands

TILE   = int(os.environ.get("TILE", 256))
ITERS  = int(os.environ.get("ITERS", 100))
WARMUP = int(os.environ.get("WARMUP", 20))
CKPT   = os.environ.get("CKPT", os.path.join(ROOT, "runs", "student", "best.pt"))
SCENE  = os.environ.get("SCENE",
    os.path.join(ROOT, "data", "s2_during", "dixie_2021_us", "S2B_10TFK_20210802_0_L2A"))
BENCH  = os.path.join(ROOT, "runs", "bench"); os.makedirs(BENCH, exist_ok=True)
DEV = "cuda"


class PowerSampler:
    """tegrastats 백그라운드 샘플링 → VDD_IN / VDD_GPU 평균(mW)."""
    def __init__(self, interval_ms=50):
        self.iv = interval_ms; self.samples = {"IN": [], "GPU": [], "CPU": []}
        self.p = None; self.stop = False; self.th = None
    def _run(self):
        self.p = subprocess.Popen(["tegrastats", "--interval", str(self.iv)],
                                  stdout=subprocess.PIPE, text=True)
        for line in self.p.stdout:
            if self.stop:
                break
            for key, pat in (("IN", r"VDD_IN (\d+)"), ("GPU", r"VDD_GPU_SOC (\d+)"),
                             ("CPU", r"VDD_CPU_CV (\d+)")):
                m = re.search(pat, line)
                if m:
                    self.samples[key].append(int(m.group(1)))
    def __enter__(self):
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start()
        time.sleep(0.3); return self
    def __exit__(self, *a):
        self.stop = True
        if self.p:
            self.p.terminate()
        time.sleep(0.2)
    def mean_w(self, key="IN"):
        s = self.samples.get(key, [])
        return round(sum(s) / len(s) / 1000.0, 2) if s else None


def get_tiles(n=16):
    x, valid, ref = load_bands(SCENE)
    C, H, W = x.shape
    tiles = []
    step = TILE
    for r0 in range(0, H - TILE, step):
        for c0 in range(0, W - TILE, step):
            t = x[:, r0:r0 + TILE, c0:c0 + TILE]
            if valid[r0:r0 + TILE, c0:c0 + TILE].mean() > 0.8:
                tiles.append(t)
            if len(tiles) >= n:
                return np.stack(tiles), ref
    return np.stack(tiles), ref


def bench_torch(model, batch, half=False):
    model = model.half() if half else model.float()
    b = batch.half() if half else batch.float()
    with torch.no_grad():
        for _ in range(WARMUP):
            model(b)
        torch.cuda.synchronize()
        with PowerSampler() as ps:
            t0 = time.perf_counter()
            for _ in range(ITERS):
                out = model(b)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            pw = ps.mean_w("IN"); pg = ps.mean_w("GPU")
    n_tiles = ITERS * b.shape[0]
    mpix = n_tiles * TILE * TILE / 1e6 / dt
    return {"ms_per_tile": round(1e3 * dt / n_tiles, 3), "mpix_per_s": round(mpix, 2),
            "power_in_w": pw, "power_gpu_w": pg,
            "mj_per_mpix": round(pw * dt / (n_tiles * TILE * TILE / 1e6) * 1000, 1) if pw else None,
            "out": (torch.sigmoid(out.float())).detach().cpu().numpy()}


def export_onnx(model, path):
    model = model.float().eval()
    dummy = torch.randn(1, len(BANDS), TILE, TILE, device=DEV)
    # dynamo=False → 레거시 TorchScript 익스포터(onnxscript 불필요)
    torch.onnx.export(model, dummy, path, input_names=["x"], output_names=["y"],
                      opset_version=17, dynamic_axes=None, dynamo=False)
    return path


def trtexec_bench(onnx_path, flag, tag):
    """trtexec 로 엔진 빌드+프로파일. flag='--fp16' 등. 지연만(정확도는 별도)."""
    eng = os.path.join(BENCH, f"student_{tag}.engine")
    cmd = ["/usr/src/tensorrt/bin/trtexec", f"--onnx={onnx_path}",
           f"--saveEngine={eng}", f"--iterations={ITERS}", "--warmUp=500",
           "--avgRuns=50", "--noDataTransfers"]
    if flag:
        cmd.append(flag)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    out = r.stdout + r.stderr
    # "Latency: ... mean = X ms" / "Throughput: X qps"
    lat = re.search(r"GPU Compute Time:.*?mean = ([\d.]+) ms", out) or \
          re.search(r"Latency:.*?mean = ([\d.]+) ms", out)
    thr = re.search(r"Throughput: ([\d.]+) qps", out)
    ms = float(lat.group(1)) if lat else None
    qps = float(thr.group(1)) if thr else None
    return {"tag": tag, "ms_per_tile": round(ms, 3) if ms else None,
            "mpix_per_s": round(qps * TILE * TILE / 1e6, 2) if qps else None,
            "engine": eng if os.path.exists(eng) else None,
            "ok": r.returncode == 0}


def main():
    model = UNet().to(DEV)
    st = torch.load(CKPT, map_location=DEV); model.load_state_dict(st["model"])
    batch, ref = get_tiles(16)
    b = torch.from_numpy(batch).to(DEV)
    print(f"device={torch.cuda.get_device_name(0)}  tile={TILE}  batch={b.shape[0]}  "
          f"iters={ITERS}  ckpt_f1={st.get('f1')}", flush=True)

    res = {}
    jpath = os.path.join(BENCH, "onboard_bench.json")
    skip_torch = os.environ.get("SKIP_TORCH", "") == "1"
    if skip_torch and os.path.exists(jpath):
        res = json.load(open(jpath))                       # 기존 torch 결과 유지, TRT만 갱신
        print("SKIP_TORCH: 기존 FP32/FP16 유지, TRT만 재실행", flush=True)
    else:
        print("== FP32 ==", flush=True)
        r32 = bench_torch(model, b, half=False); out32 = r32.pop("out")
        res["torch_fp32"] = r32; print(f"  {r32}", flush=True)

        print("== FP16 ==", flush=True)
        r16 = bench_torch(model, b, half=True); out16 = r16.pop("out")
        res["torch_fp16"] = r16; print(f"  {r16}", flush=True)

        # 정확도 손실(FP16 vs FP32 확률맵)
        d = np.abs(out32 - out16)
        res["fp16_accuracy"] = {"max_prob_diff": round(float(d.max()), 5),
                                "mean_prob_diff": round(float(d.mean()), 6),
                                "mask_disagree_frac_at_0.5":
                                round(float(((out32 > 0.5) != (out16 > 0.5)).mean()), 6)}
        print(f"  정확도 FP16: {res['fp16_accuracy']}", flush=True)

    # TensorRT (ONNX → trtexec) — 지연만
    try:
        onnx_p = export_onnx(model.to(DEV), os.path.join(BENCH, "student.onnx"))
        print("== TensorRT FP16 ==", flush=True)
        res["trt_fp16"] = trtexec_bench(onnx_p, "--fp16", "fp16")
        print(f"  {res['trt_fp16']}", flush=True)
        print("== TensorRT INT8(무보정·지연만) ==", flush=True)
        res["trt_int8"] = trtexec_bench(onnx_p, "--int8", "int8")
        print(f"  {res['trt_int8']}", flush=True)
    except Exception as e:
        res["trt_error"] = str(e)[:200]
        print(f"  TRT 실패: {e}", flush=True)

    # 요약: FP32 대비 배속
    base = res["torch_fp32"]["mpix_per_s"]
    res["speedup_vs_fp32"] = {
        k: round(res[k]["mpix_per_s"] / base, 2)
        for k in ("torch_fp16", "trt_fp16", "trt_int8")
        if res.get(k, {}).get("mpix_per_s")}
    json.dump(res, open(os.path.join(BENCH, "onboard_bench.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n요약 배속(FP32 기준): {res['speedup_vs_fp32']}\n→ {BENCH}/onboard_bench.json",
          flush=True)


if __name__ == "__main__":
    main()
