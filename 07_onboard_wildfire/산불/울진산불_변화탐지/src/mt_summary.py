"""다중시기 실험 5단계 — runs/mt_*_s*/results.json 을 모아 설정별 평균과 표준편차를 낸다. ko_mt/RESULT.md 에 기록.
사용: python3 src/mt_summary.py
"""
import os, glob, json, re, statistics as st, datetime as dt
HERE=os.path.dirname(os.path.abspath(__file__)); PROJ=os.path.join(HERE,"..")
COV=["kf_20200422_1753ha","kf_20200608_2433ha"]      # 젯슨 자료가 있는 홀드아웃 화재 (봄, 여름)
rows={}
for p in sorted(glob.glob(os.path.join(PROJ,"runs","mt_*_s*","results.json"))):
    r=json.load(open(p)); m=re.match(r"mt_(.+)_s(\d+)$",os.path.basename(os.path.dirname(p)))
    rows.setdefault(m.group(1),[]).append(r)
def ms(v): return (st.mean(v), st.pstdev(v) if len(v)>1 else 0.0)
lines=[f"# 다중시기 채널 실험 결과 ({dt.date.today().isoformat()})","",
       "설정마다 seed 1, 2, 3 의 평균 (표준편차). 모델 smp U-Net resnet34, bignore 2, 40 epoch, CPU 32 스레드. 학습 50 타일, 검증 10, 홀드아웃 8 화재 (고정).",
       "mt = 젯슨 보관 자료(RGB+NIR 20 m) 로 만든 9 채널 (화재 전 NDVI 중앙값, 90 백분위, 화재 후 중앙값, 20 백분위, 두 낙차, 장면 수 2, flag). 자료 없는 화재는 0.",""]
out=[]
for cfg in ["bitemporal","bitemporal_mt","rgbnir","rgbnir_mt"]:
    rs=rows.get(cfg,[])
    if not rs: continue
    f1=ms([r["f1"] for r in rs]); spring=ms([r["per_season"].get("봄",float("nan")) for r in rs])
    cov={k:ms([r["per_fire"][k] for r in rs]) for k in COV}
    out.append((cfg,f1,spring,cov,len(rs)))
    lines.append(f"- {cfg} (n={len(rs)}): 전체 F1 {f1[0]:.3f} ({f1[1]:.3f}), 봄 {spring[0]:.3f} ({spring[1]:.3f}), "
                 f"{COV[0]} {cov[COV[0]][0]:.3f} ({cov[COV[0]][1]:.3f}), {COV[1]} {cov[COV[1]][0]:.3f} ({cov[COV[1]][1]:.3f})")
lines.append("")
d={o[0]:o for o in out}
for base in ["bitemporal","rgbnir"]:
    if base in d and base+"_mt" in d:
        b,m=d[base],d[base+"_mt"]
        lines.append(f"- {base} 대비 mt 차이: 전체 {m[1][0]-b[1][0]:+.3f}, 봄 {m[2][0]-b[2][0]:+.3f}, {COV[0]} {m[3][COV[0]][0]-b[3][COV[0]][0]:+.3f}, {COV[1]} {m[3][COV[1]][0]-b[3][COV[1]][0]:+.3f}")
lines+=["","seed 별 전체 F1:"]
for cfg,rs in rows.items(): lines.append(f"- {cfg}: "+", ".join(f"s{r['seed']} {r['f1']:.3f}" for r in sorted(rs,key=lambda r:r['seed'])))
txt="\n".join(lines); print(txt); os.makedirs(os.path.join(PROJ,"ko_mt"),exist_ok=True)
open(os.path.join(PROJ,"ko_mt","RESULT.md"),"a",encoding="utf-8").write(txt+"\n\n")
