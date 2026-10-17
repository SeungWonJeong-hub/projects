"""임의 지역 실측 타일에서 v2(US 하위48 학습) 모델 vs dNBR-echo. 사용: region_eval.py <dir> [kx ky]"""
import os,glob,sys,numpy as np,torch,segmentation_models_pytorch as smp
HERE=os.path.dirname(os.path.abspath(__file__)); DEV="cuda"
D=sys.argv[1]; KX=sys.argv[2] if len(sys.argv)>2 else "x"; KY=sys.argv[3] if len(sys.argv)>3 else "y"
MODEL=os.path.join(HERE,"..","runs","mtbs_cd","mtbs_v2_unet.pt")
def met(P,Y):
    def at(t):
        pr=P>=t;tp=(pr&(Y>0)).sum();fp=(pr&(Y==0)).sum();fn=((~pr)&(Y>0)).sum()
        pe=tp/(tp+fp+1e-9);re=tp/(tp+fn+1e-9);return 2*pe*re/(pe+re+1e-9),tp/(tp+fp+fn+1e-9),pe,re
    t=max(np.linspace(0.1,0.9,33),key=lambda t:at(t)[0]);f1,i,pe,re=at(t)
    return dict(thr=round(float(t),3),f1=round(float(f1),3),iou=round(float(i),3),precision=round(float(pe),3),recall=round(float(re),3))
fs=sorted(glob.glob(os.path.join(D,"*","*.npz")) or glob.glob(os.path.join(D,"*.npz")))
X=[];Y=[]
for f in fs: d=np.load(f);X.append(d[KX].astype(np.float32));Y.append(d[KY].astype(np.uint8))
X=np.stack(X);Y=np.stack(Y)
print(f"{os.path.basename(D)} 실측: {len(fs)}타일 불탄 {Y.mean()*100:.1f}%")
print("  dNBR-echo:",met(X[:,12].reshape(-1),Y.reshape(-1)))
m=smp.Unet("resnet34",encoder_weights=None,in_channels=13,classes=1).to(DEV)
m.load_state_dict(torch.load(MODEL,map_location=DEV,weights_only=True));m.eval()
P=[]
with torch.no_grad():
    for i in range(0,len(X),8):
        import warnings;warnings.filterwarnings("ignore")
        p=torch.sigmoid(m(torch.from_numpy(X[i:i+8]).to(DEV))).float()[:,0].cpu().numpy();P.append(p)
P=np.concatenate(P);mm=met(P.reshape(-1),Y.reshape(-1))
print("  US모델→해당지역:",mm)
print(f"  ▶ 마진(모델-echo margin이 일반화 신호)")
