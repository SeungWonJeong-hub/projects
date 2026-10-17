import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
fp="/home/syntax_kmu/.fonts/NanumGothic.ttf"; fm.fontManager.addfont(fp)
plt.rcParams["font.family"]=fm.FontProperties(fname=fp).get_name(); plt.rcParams["axes.unicode_minus"]=False
OUT="out"
encs=["ResNet18","ResNet34","ResNet50"]
F1={"ResNet18":[0.669,0.656,0.646],"ResNet34":[0.604,0.594,0.633],"ResNet50":[0.659,0.624,0.564]}
SP={"ResNet18":[0.636,0.626,0.633],"ResNet34":[0.594,0.567,0.590],"ResNet50":[0.637,0.574,0.466]}
PAR={"ResNet18":14.4,"ResNet34":24.4,"ResNet50":32.5}
cols={"ResNet18":"#2a7d2a","ResNet34":"#c07000","ResNet50":"#7a4fbf"}
fig,ax=plt.subplots(1,2,figsize=(13,5.4))
for k,(title,D) in enumerate([("전체 F1 (계절 홀드아웃 8화재)",F1),("봄 cross-fire F1",SP)]):
    a=ax[k]
    for i,e in enumerate(encs):
        v=np.array(D[e]); m,s=v.mean(),v.std()
        a.bar(i,m,0.6,color=cols[e],alpha=0.85,yerr=s,capsize=6,ecolor="#333")
        a.scatter([i]*3,v,color="black",zorder=5,s=28)
        a.text(i,m+s+0.006,f"{m:.3f}\n±{s:.3f}",ha="center",fontsize=11,fontweight="bold")
    a.set_xticks(range(3)); a.set_xticklabels([f"{e}\n({PAR[e]:.0f}M)" for e in encs],fontsize=11)
    a.set_ylim(0.45,0.72); a.set_title(title,fontsize=12,fontweight="bold"); a.grid(axis="y",alpha=0.3)
    a.set_ylabel("F1 (3시드 평균±표준편차)")
fig.suptitle("인코더 A/B: U-Net 고정 · 백본만 교체 · 나머지 전부 동일 (한국 61화재, imagenet, 40ep, 3시드)\n"
             "→ ResNet18이 최고·가장 일관 · ResNet50은 과적합(시드 편차 큼) · 더 큰 백본이 이득 없음",
             fontsize=12.5,y=1.02)
plt.tight_layout(); plt.savefig(OUT+"/encoder_ablation.png",dpi=115,bbox_inches="tight")
print("saved encoder_ablation.png")
for e in encs:
    print(f"{e}: 전체F1 {np.mean(F1[e]):.3f}±{np.std(F1[e]):.3f} | 봄F1 {np.mean(SP[e]):.3f}±{np.std(SP[e]):.3f}")
