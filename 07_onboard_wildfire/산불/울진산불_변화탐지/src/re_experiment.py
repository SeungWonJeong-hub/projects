"""적색경계(RE) 효과 실험: base(12ch) vs +RE(18ch), 동일 타일·seed·홀드아웃 짝비교.
ko_refl_re(18ch [B02,B03,B04,B05,B06,B07,B8A,B11,B12]×pre,post) 사용.
GroupNorm UNet + recall-Tversky(a0.7/b0.3) = v3 레시피. 봄 화재 F1 집중 리포트.
사용: KO_DATA_RE=../ko_refl_re python re_experiment.py
"""
import os, glob, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("KO_DATA_RE",os.path.join(HERE,"..","ko_refl_re"))
HOLD=os.path.join(HERE,"..","data","holdout_fixed.txt")
DEV="cuda" if torch.cuda.is_available() else "cpu"
# 18ch 인덱스: pre 0-8, post 9-17. RE=B05,B06,B07 → 3,4,5,12,13,14
RE_IDX=[3,4,5,12,13,14]
BASE_IDX=[i for i in range(18) if i not in RE_IDX]  # 12ch (RE 제외)

def gn(c): return nn.GroupNorm(min(8,c),c)
class Blk(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.c=nn.Sequential(nn.Conv2d(i,o,3,padding=1),gn(o),nn.GELU(),
                                              nn.Conv2d(o,o,3,padding=1),gn(o),nn.GELU())
    def forward(s,x): return s.c(x)
class UNet(nn.Module):
    def __init__(s,ic):
        super().__init__()
        s.d1=Blk(ic,32); s.d2=Blk(32,64); s.d3=Blk(64,128); s.d4=Blk(128,256)
        s.p=nn.MaxPool2d(2)
        s.u3=Blk(256+128,128); s.u2=Blk(128+64,64); s.u1=Blk(64+32,32)
        s.o=nn.Conv2d(32,1,1)
    def up(s,x): return F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)
    def forward(s,x):
        c1=s.d1(x); c2=s.d2(s.p(c1)); c3=s.d3(s.p(c2)); c4=s.d4(s.p(c3))
        x=s.u3(torch.cat([s.up(c4),c3],1)); x=s.u2(torch.cat([s.up(x),c2],1)); x=s.u1(torch.cat([s.up(x),c1],1))
        return s.o(x)

class DS(torch.utils.data.Dataset):
    def __init__(s,files,idx,aug=False): s.f=files; s.idx=idx; s.aug=aug
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        d=np.load(s.f[i]); x=d["x"].astype(np.float32)[s.idx]/10000.0; y=d["y"].astype(np.float32)
        if s.aug and random.random()<0.5: x,y=x[:,:,::-1].copy(),y[:,::-1].copy()
        if s.aug and random.random()<0.5: x,y=x[:,::-1].copy(),y[::-1].copy()
        return torch.from_numpy(x),torch.from_numpy(y)

def tversky_bce(logit,y,a=0.7,b=0.3):
    p=torch.sigmoid(logit).squeeze(1); e=1e-6
    tp=(p*y).sum((1,2)); fn=(y*(1-p)).sum((1,2)); fp=((1-y)*p).sum((1,2))
    tv=1-((tp+e)/(tp+a*fn+b*fp+e)).mean()
    return tv+F.binary_cross_entropy_with_logits(logit.squeeze(1),y)

def stats(P,Y,t):
    p=(P>=t); tp=(p&(Y>0)).sum(); fp=(p&(Y==0)).sum(); fn=((~p)&(Y>0)).sum()
    pr=tp/(tp+fp+1e-9); rc=tp/(tp+fn+1e-9); f1=2*pr*rc/(pr+rc+1e-9)
    return dict(f1=float(f1),precision=float(pr),recall=float(rc))

@torch.no_grad()
def predict(model,files,idx):
    model.eval(); P=[];Y=[]
    for f in files:
        d=np.load(f); x=torch.from_numpy(d["x"].astype(np.float32)[idx]/10000.0).unsqueeze(0).to(DEV)
        p=torch.sigmoid(model(x)).squeeze().cpu().numpy()
        P.append(p.reshape(-1)); Y.append(d["y"].astype(np.uint8).reshape(-1))
    return np.concatenate(P),np.concatenate(Y)

def season(m): return "봄" if 3<=m<=5 else "여름" if 6<=m<=8 else "가을" if 9<=m<=11 else "겨울"

def run(tag,idx):
    random.seed(42); torch.manual_seed(42); np.random.seed(42)
    hold=set(l.strip() for l in open(HOLD) if l.strip())
    fires=sorted([d for d in glob.glob(os.path.join(DATA,"*")) if glob.glob(os.path.join(d,"*.npz"))])
    te_f=[d for d in fires if os.path.basename(d) in hold]
    tr_all=[d for d in fires if os.path.basename(d) not in hold]; random.shuffle(tr_all)
    nval=max(3,len(tr_all)//6); va_f=tr_all[:nval]; trn_f=tr_all[nval:]
    tf=lambda ds:[f for d in ds for f in glob.glob(os.path.join(d,"*.npz"))]
    trn,va=tf(trn_f),tf(va_f)
    print(f"[{tag}] ic={len(idx)} train {len(trn_f)}화재/{len(trn)}타일 val {len(va)} TEST {len(te_f)}화재",flush=True)
    model=UNet(len(idx)).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    dl=torch.utils.data.DataLoader(DS(trn,idx,aug=True),batch_size=8,shuffle=True,num_workers=2)
    best=-1;bestsd=None
    Pv,Yv=None,None
    for ep in range(40):
        model.train()
        for xb,yb in dl:
            xb,yb=xb.to(DEV),yb.to(DEV); opt.zero_grad()
            loss=tversky_bce(model(xb),yb); loss.backward(); opt.step()
        Pv,Yv=predict(model,tf(va_f),idx)
        vb=max([stats(Pv,Yv,t)["f1"] for t in np.arange(0.3,0.85,0.05)])
        if vb>best: best=vb; bestsd={k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(bestsd)
    # 전체 홀드아웃 best-thr
    allP=[];allY=[];per=[]
    for d in te_f:
        pp,yy=predict(model,glob.glob(os.path.join(d,"*.npz")),idx); allP.append(pp);allY.append(yy)
        per.append((os.path.basename(d),season(int(os.path.basename(d).split("_")[1][4:6])),pp,yy))
    Pa=np.concatenate(allP);Ya=np.concatenate(allY)
    thr=max(np.arange(0.3,0.85,0.05),key=lambda t:stats(Pa,Ya,t)["f1"])
    ov=stats(Pa,Ya,thr)
    from collections import defaultdict
    bs=defaultdict(lambda:[[],[]])
    for fn,se,pp,yy in per: bs[se][0].append(pp);bs[se][1].append(yy)
    res={"tag":tag,"thr":round(float(thr),3),"overall":ov}
    for se in ["봄","여름","가을","겨울"]:
        if se in bs:
            pp=np.concatenate(bs[se][0]);yy=np.concatenate(bs[se][1]); res[se]=stats(pp,yy,thr)
    return res

if __name__=="__main__":
    import json
    R={}
    for tag,idx in [("base_12ch",BASE_IDX),("re_18ch",list(range(18)))]:
        R[tag]=run(tag,idx)
        print(f"=== {tag} thr{R[tag]['thr']} overall F1 {R[tag]['overall']['f1']:.3f} | "
              f"봄 {R[tag].get('봄',{}).get('f1',float('nan')):.3f} 여름 {R[tag].get('여름',{}).get('f1',float('nan')):.3f}",flush=True)
    os.makedirs(os.path.join(HERE,"..","runs","re_exp"),exist_ok=True)
    json.dump(R,open(os.path.join(HERE,"..","runs","re_exp","results.json"),"w"),ensure_ascii=False,indent=1)
    print("\n=== 봄 화재 F1 (핵심) ===")
    print(f"base(12ch): {R['base_12ch'].get('봄',{}).get('f1',0):.3f}  +RE(18ch): {R['re_18ch'].get('봄',{}).get('f1',0):.3f}  "
          f"Δ={R['re_18ch'].get('봄',{}).get('f1',0)-R['base_12ch'].get('봄',{}).get('f1',0):+.3f}")
