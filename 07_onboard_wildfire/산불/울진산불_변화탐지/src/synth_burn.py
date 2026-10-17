"""실측-분광 소각 합성기 (우리 sim2real).
B200 procedural 트윈이 죽은 이유=합성 소각 분광변화가 실제의 5~7배 약함(dNIR≈0).
회피책=실제 화재의 pre→post 8ch 델타를 실측 무화재 배경(ko_neg_s50)에 이식.
→ 합성 소각이 실제 세기(dNIR≈−0.10)를 그대로 가짐 + 픽셀완벽 라벨 + 무한/다양.

출력: cir_synth/synth_<i>/gt_stac.npz (x 8ch, gt, valid) — cir_load/cir_tiles 호환.
학습 전 go/no-go: 합성 소각 dNIR이 실제(−0.10)와 맞는지 출력. 안 맞으면 중단.
사용: N_SYNTH=200 CIR_RAW=0 python synth_burn.py
"""
from __future__ import annotations
import os, glob, random, numpy as np
from scipy import ndimage
import step4_rehearsal as S

OUT=os.path.join(S.PROJ,"cir_synth")
NEG_DIR=os.path.join(S.PROJ,"ko_neg_s50")
N_TILES=int(os.environ.get("N_SYNTH","200"))
TILE=256
random.seed(0); np.random.seed(0)

def collect_real():
    """실제 화재서 (a) burned pre/post 4ch 쌍 풀, (b) 소각 마스크 모양."""
    pre_pool=[]; post_pool=[]; shapes=[]
    for n in S.CIR_TRAIN:
        try: x,gt,vd=S.cir_load(n)
        except Exception as e: print(f"  skip {n}: {e}"); continue
        b=gt&vd
        if b.sum()<200: continue
        pre_pool.append(x[0:4][:,b].T.astype(np.float32))   # (Npix,4)
        post_pool.append(x[4:8][:,b].T.astype(np.float32))
        lab,ncc=ndimage.label(gt)
        for c in range(1,ncc+1):
            m=lab==c
            if m.sum()<80: continue
            ys,xs=np.where(m); crop=m[ys.min():ys.max()+1,xs.min():xs.max()+1]
            if 8<=crop.shape[0]<=TILE and 8<=crop.shape[1]<=TILE: shapes.append(crop)
    pre=np.concatenate(pre_pool); post=np.concatenate(post_pool)
    print(f"실측 소각 픽셀 풀 {len(pre)} · 마스크 모양 {len(shapes)}개 ({len(S.CIR_TRAIN)}화재)")
    return pre, post, shapes

def bg_tiles():
    """ko_neg_s50 무화재 2시기 타일 → 8ch 배경."""
    out=[]
    for f in sorted(glob.glob(os.path.join(NEG_DIR,"*","*.npz"))):
        z=np.load(f); x=np.clip(S.sel12(z["x"].astype(np.float32)/1e4),0,1)  # 8ch
        v=z["valid"].astype(bool) if "valid" in z else np.ones(x.shape[1:],bool)
        out.append((x,v))
    print(f"배경 타일 {len(out)}개 (ko_neg_s50)")
    return out

def place(bg, valid, delta_pool, shapes):
    """배경에 소각 모양 얹고 post = pre + 실측델타 이식."""
    x=bg.copy(); H,W=x.shape[1:]
    mask=np.zeros((H,W),bool)
    ncomp=random.randint(1,3)
    for _ in range(ncomp):
        crop=random.choice(shapes)
        k=random.randint(0,3); crop=np.rot90(crop,k)
        if random.random()<0.5: crop=crop[:,::-1]
        ch,cw=crop.shape
        if ch>=H or cw>=W: continue
        r0=random.randint(0,H-ch); c0=random.randint(0,W-cw)
        mask[r0:r0+ch, c0:c0+cw] |= crop
    mask &= valid
    if mask.sum()<50: return None
    idx=np.where(mask)
    npix=len(idx[0])
    sel=np.random.randint(0,len(delta_pool),npix)
    d=delta_pool[sel]                      # (npix,4) 실측 pre→post 델타
    # post = 배경 pre + 실측 소각 델타 (마스크 내부만)
    for c in range(4):
        x[4+c][idx]=np.clip(x[c][idx]+d[:,c],0,1)
    return x, mask, valid

def main():
    pre,post,shapes=collect_real()
    delta_pool=(post-pre)                   # (Npix,4) 실측 소각 분광변화
    print(f"실측 소각 델타 평균 [B02 B03 B04 NIR]: {np.round(delta_pool.mean(0),3).tolist()}  (NIR가 핵심 신호)")
    bgs=bg_tiles()
    if not bgs or len(shapes)==0: print("데이터 부족 중단"); return
    os.makedirs(OUT,exist_ok=True)
    made=0; dnir_syn=[]; dnir_bg=[]
    for i in range(N_TILES):
        bg,v=bgs[i%len(bgs)]
        r=place(bg,v,delta_pool,shapes)
        if r is None: continue
        x,mask,valid=r
        # 검증용 dNIR 채취
        b=mask; u=(~mask)&valid
        if b.sum()>10: dnir_syn.append(float((x[7][b]-x[3][b]).mean()))
        if u.sum()>10: dnir_bg.append(float((x[7][u]-x[3][u]).mean()))
        d=os.path.join(OUT,f"synth_{i:04d}"); os.makedirs(d,exist_ok=True)
        np.savez_compressed(os.path.join(d,"gt_stac.npz"),
                            x=x.astype(np.float32),gt=mask,valid=valid)
        made+=1
    print(f"\n생성 {made} 타일 → {OUT}")
    real_dnir=float(delta_pool[:,3].mean())
    print(f"=== go/no-go: 합성 소각 신호 세기 ===")
    print(f"  합성 소각 dNIR = {np.mean(dnir_syn):+.3f}  (실측 소각 {real_dnir:+.3f})")
    print(f"  합성 무화재 dNIR = {np.mean(dnir_bg):+.3f}  (실측 무화재 ~−0.002)")
    ok=abs(np.mean(dnir_syn)-real_dnir)<0.03
    print(f"  판정: {'PASS — 실측 세기 일치(≠B200 약함)' if ok else 'FAIL — 신호 약함, B200 재현'}")

if __name__=="__main__": main()
