import base64, os
EV=os.path.expanduser("~/syntax_inference/Steven/산불/근거사진")
def b64(p):
    with open(p,"rb") as f: return "data:image/png;base64,"+base64.b64encode(f.read()).decode()
arch=b64(EV+"/아키텍처비교_UNet_vs_FPN_vs_DeepLabV3.png")
enc=b64(EV+"/인코더A_B_ResNet18_34_50.png")

CSS = """
<style>
:root{
  --ground:#eef0f3; --paper:#ffffff; --ink:#191d23; --muted:#5c6470;
  --faint:#8a919c; --line:#e0e3e9; --line2:#eceef2;
  --ember:#cf5a17; --ember-soft:#fbeee3;
  --good:#1c7a3c; --good-soft:#e8f4ec;
  --warn:#9a6400; --warn-soft:#fbf3e0;
}
*{box-sizing:border-box}
body{margin:0}
.wrap{
  font-family:'Apple SD Gothic Neo','Malgun Gothic','Noto Sans KR','Pretendard',system-ui,-apple-system,'Segoe UI',sans-serif;
  color:var(--ink); background:var(--ground);
  line-height:1.65; font-size:16px; -webkit-font-smoothing:antialiased;
}
.col{max-width:900px;margin:0 auto;padding:0 24px}
.mono{font-family:ui-monospace,'SF Mono','Cascadia Code',Menlo,monospace;font-variant-numeric:tabular-nums}
/* header */
header{background:#22272e;color:#eef0f3;padding:56px 0 44px}
header .col{padding-top:0}
.kicker{font-size:12.5px;letter-spacing:.16em;text-transform:uppercase;color:#e9a878;font-weight:600;margin:0 0 14px}
header h1{font-size:33px;line-height:1.2;margin:0 0 14px;font-weight:800;text-wrap:balance;letter-spacing:-.01em}
header p{margin:0;color:#aeb6c2;font-size:16px;max-width:62ch}
.meta{margin-top:22px;display:flex;flex-wrap:wrap;gap:10px}
.meta span{font-size:12.5px;color:#c4ccd6;background:#2e343d;border:1px solid #3a424c;border-radius:100px;padding:5px 13px}
.meta span b{color:#e9a878;font-weight:600}
/* section */
section{padding:52px 0 8px}
.eyebrow{display:flex;align-items:baseline;gap:14px;margin-bottom:8px}
.eyebrow .n{font-family:ui-monospace,monospace;font-size:13px;font-weight:700;color:var(--ember);
  border:1.5px solid var(--ember);border-radius:6px;padding:2px 9px;flex:none}
.eyebrow .q{font-size:13px;letter-spacing:.04em;color:var(--faint);text-transform:uppercase;font-weight:600}
h2{font-size:25px;font-weight:800;margin:2px 0 6px;letter-spacing:-.01em;text-wrap:balance}
.lead{font-size:17px;color:var(--muted);margin:0 0 26px;max-width:64ch}
.verdict{font-weight:700;color:var(--ink)}
.verdict.good{color:var(--good)}
/* card / figure */
.fig{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:14px;
  box-shadow:0 1px 2px rgba(20,25,35,.04),0 8px 24px rgba(20,25,35,.05);margin-bottom:14px}
.fig img{display:block;width:100%;height:auto;border-radius:6px}
.cap{font-size:13.5px;color:var(--faint);margin:12px 4px 2px;line-height:1.5}
.cap b{color:var(--muted);font-weight:600}
/* table */
.tw{overflow-x:auto;margin:22px 0;border:1px solid var(--line);border-radius:12px;background:var(--paper)}
table{border-collapse:collapse;width:100%;font-size:14.5px}
th,td{padding:11px 16px;text-align:right;border-bottom:1px solid var(--line2);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{background:#f7f8fa;font-size:12.5px;letter-spacing:.03em;color:var(--muted);font-weight:700;
  border-bottom:1.5px solid var(--line)}
tbody tr:last-child td{border-bottom:none}
td.num{font-family:ui-monospace,monospace;font-variant-numeric:tabular-nums}
tr.chosen{background:var(--good-soft)}
tr.chosen td:first-child{font-weight:700;color:var(--good);box-shadow:inset 3px 0 0 var(--good)}
.win{color:var(--good);font-weight:700}
.tag{font-size:11px;font-weight:700;padding:2px 8px;border-radius:100px;letter-spacing:.02em;vertical-align:middle}
.tag.dep{background:var(--good-soft);color:var(--good);border:1px solid #bfe0c9}
.tag.bad{background:#f4f5f7;color:var(--faint);border:1px solid var(--line)}
/* callout */
.note{border-left:4px solid var(--warn);background:var(--warn-soft);border-radius:0 10px 10px 0;
  padding:16px 20px;margin:24px 0;font-size:14.5px;line-height:1.6}
.note b{color:var(--warn)}
.note .h{font-weight:800;color:var(--warn);font-size:13px;letter-spacing:.05em;text-transform:uppercase;
  display:block;margin-bottom:6px}
/* key-values */
.kv{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line);
  border-radius:12px;overflow:hidden;margin:22px 0}
.kv div{background:var(--paper);padding:15px 18px}
.kv .k{font-size:12.5px;color:var(--faint);text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px;font-weight:600}
.kv .v{font-size:15px;color:var(--ink)}
.kv .v .mono{font-size:15px;font-weight:600}
@media(max-width:640px){.kv{grid-template-columns:1fr}header h1{font-size:26px}h2{font-size:21px}}
hr.div{border:none;border-top:1px solid var(--line);margin:40px 0 0}
footer{padding:44px 0 64px;color:var(--faint);font-size:13px}
footer .col > div{margin-bottom:6px}
footer code{font-family:ui-monospace,monospace;background:var(--paper);border:1px solid var(--line);
  border-radius:5px;padding:1px 6px;font-size:12px;color:var(--muted)}
strong.em{color:var(--ember);font-weight:700}
</style>
"""

HTML = CSS + """
<div class="wrap">
<header><div class="col">
  <p class="kicker">배포 모델 설계 근거 · KMU 산불 온보드 변화탐지</p>
  <h1>산불 소각탐지 모델은 왜 이렇게 설계됐나</h1>
  <p>세 가지 핵심 선택 — <b style="color:#e9a878">아키텍처 · 인코더 · 파인튜닝</b> — 을 미학습 화재 실측으로 검증한다. 유리한 숫자만이 아니라 결정을 뒤집을 수 있는 반례까지 함께 싣는다.</p>
  <div class="meta">
    <span>평가 <b>LORO 미학습 홀드아웃</b></span>
    <span>모델 <b>ET-02 · RGB+NIR 20m</b></span>
    <span>정답 <b>국토정보플랫폼 CIR-GT · NIFoS</b></span>
    <span>2026-09-22</span>
  </div>
</div></header>

<section><div class="col">
  <div class="eyebrow"><span class="n">01</span><span class="q">아키텍처 — 왜 U-Net</span></div>
  <h2>U-Net은 매 화재 1등은 아니지만, 무너지지 않는다</h2>
  <p class="lead">같은 데이터로 U-Net · FPN · DeepLabV3+ 를 미학습 4화재에 적용했다. 주황=모델 예측, 빨강선=정답 경계.</p>
  <div class="fig"><img src="__ARCH__" alt="세 아키텍처의 소각영역 예측 비교"/>
    <div class="cap"><b>그림 1.</b> 미학습 홀드아웃 4화재 × [촬영영상+정답 · U-Net · FPN · DeepLabV3+]. 하단 바=화재별 F1. 각 모델 best-thr.</div>
  </div>
  <div class="tw"><table>
    <thead><tr><th>화재 (미학습)</th><th>U-Net</th><th>FPN</th><th>DeepLabV3+</th><th>화재별 승자</th></tr></thead>
    <tbody>
      <tr><td>안동 ’25</td><td class="num">0.826</td><td class="num win">0.862</td><td class="num">0.805</td><td>FPN</td></tr>
      <tr><td>산청 ’25</td><td class="num win">0.811</td><td class="num">0.778</td><td class="num">0.786</td><td>U-Net</td></tr>
      <tr><td>홍성 ’23</td><td class="num">0.841</td><td class="num win">0.851</td><td class="num">0.809</td><td>FPN</td></tr>
      <tr><td>양구 ’22</td><td class="num win">0.810</td><td class="num">0.770</td><td class="num">0.756</td><td>U-Net</td></tr>
      <tr class="chosen"><td>평균</td><td class="num">0.822</td><td class="num">0.815</td><td class="num">0.789</td><td>U-Net</td></tr>
      <tr class="chosen"><td>최악화재 (하한)</td><td class="num">0.810</td><td class="num">0.770</td><td class="num">0.756</td><td>U-Net</td></tr>
      <tr><td>표준편차 (작을수록 일관)</td><td class="num win">0.013</td><td class="num">0.041</td><td class="num">0.021</td><td>U-Net</td></tr>
    </tbody>
  </table></div>
  <p>화재별로 보면 <b>FPN이 안동·홍성에서 U-Net을 이긴다.</b> 하지만 FPN은 양구에서 0.770까지 무너지는 <b>boom-bust</b>(표준편차 0.041)다. U-Net은 <span class="verdict good">평균 최고(0.822)·최악화재 최고(0.810)·가장 일관(0.013)</span> — 어떤 화재가 와도 바닥이 높다. 배포는 평균이 아니라 <strong class="em">하한</strong>이 중요하므로 U-Net을 택했다. 그림 안동 행을 보면 FPN·DeepLabV3+는 비소각지까지 주황이 범람(과예측)하는 반면 U-Net은 경계에 붙어 있다.</p>
</div></section>

<hr class="div"/>

<section><div class="col">
  <div class="eyebrow"><span class="n">02</span><span class="q">인코더 — ResNet18 / 34 / 50</span></div>
  <h2>더 큰 백본은 이득이 없었다 — 오히려 반대다</h2>
  <p class="lead">U-Net 고정, 백본만 교체, 나머지 전부 동일(한국 61화재 · ImageNet · 40에폭 · 3시드). 이 실험은 배포된 ResNet34를 <b>지지하지 않는다</b> — 정직하게 싣는다.</p>
  <div class="fig"><img src="__ENC__" alt="ResNet18/34/50 인코더 A/B 결과"/>
    <div class="cap"><b>그림 2.</b> 3시드 평균±표준편차, 점=개별 시드. 괄호=학습 파라미터 수.</div>
  </div>
  <div class="tw"><table>
    <thead><tr><th>인코더</th><th>파라미터</th><th>전체 F1</th><th>봄 cross-fire F1</th><th>일관성(σ)</th></tr></thead>
    <tbody>
      <tr class="chosen"><td>ResNet18</td><td class="num">14M</td><td class="num win">0.657</td><td class="num win">0.632</td><td class="num">±0.009</td></tr>
      <tr><td>ResNet34 <span class="tag dep">현 배포</span></td><td class="num">24M</td><td class="num">0.610</td><td class="num">0.584</td><td class="num">±0.017</td></tr>
      <tr><td>ResNet50</td><td class="num">32M</td><td class="num">0.616</td><td class="num">0.559</td><td class="num">±0.039</td></tr>
    </tbody>
  </table></div>
  <p><b>ResNet18이 최고이자 가장 싸고 가장 안정적이다.</b> ResNet50은 파라미터가 2배지만 이득이 없고 시드 편차가 커진다(0.564~0.659) — 작은 데이터(50타일)에서 <strong class="em">과적합</strong>하는 전형이다. 즉 “큰 백본이 낫다”는 통념은 이 과제에서 틀렸다.</p>
  <div class="note"><span class="h">정직한 한계</span>
    이 ablation은 smp 표준 셋업(<b>12채널/SWIR · 계절 홀드아웃 8화재</b>)이라 배포 파이프라인(8채널 RGB+NIR · CIR rehearsal)과 셋업이 달라 절대수치를 배포모델에 직접 이식할 수 없다. 하지만 <b>인코더 용량 질문은 공정하게 통제</b>됐고, 방향은 분명하다 — <b>ResNet34는 실측으로 이겨서 뽑힌 게 아니라 중간값 기본선이었고, 데이터는 오히려 ResNet18(더 가벼운 온보드)을 가리킨다.</b> 온보드 경량화 후보로 ResNet18 재검토 가치가 있다.
  </div>
</div></section>

<hr class="div"/>

<section><div class="col">
  <div class="eyebrow"><span class="n">03</span><span class="q">파인튜닝 — 왜 rehearsal</span></div>
  <h2>안 하면 봄 산불에서 붕괴한다 — 그런데 순진하게 하면 기존을 잃는다</h2>
  <p class="lead">파인튜닝은 성능 미세조정이 아니라, 미학습 봄 산불을 아예 못 잡던 모델을 살리는 유일한 레버였다.</p>
  <div class="kv">
    <div><div class="k">문제 — 봄 붕괴</div><div class="v">US/여름 모델 → 한국 직투입 <span class="mono">F1 0.000</span> (−0.609 완전붕괴). 봄 cross-fire 베이스 <span class="mono">0.097</span>.</div></div>
    <div><div class="k">원인 — phenology</div><div class="v">봄 개엽(green-up)으로 <b>dNBR 부호 반전</b> → 여름 신호로 학습한 모델이 소각을 반대로 읽음.</div></div>
  </div>
  <p>여러 파인튜닝 방식을 실제로 시도했고, 대부분 기각됐다:</p>
  <div class="tw"><table>
    <thead><tr><th>방식</th><th>봄 결과</th><th>기존(산림청) 영향</th><th>판정</th></tr></thead>
    <tbody>
      <tr><td>naive FT <span style="color:var(--faint);font-size:12px">(봄만 미세조정)</span></td><td class="num">0.805~0.840</td><td>in-domain 붕괴</td><td><span class="tag bad">파국망각 → 롤백</span></td></tr>
      <tr><td>계절매칭 사전학습 <span style="color:var(--faint);font-size:12px">(일본·미동부)</span></td><td>—</td><td>—</td><td><span class="tag bad">negative</span></td></tr>
      <tr><td>파운데이션 FT <span style="color:var(--faint);font-size:12px">(Prithvi·Presto·AlphaEarth)</span></td><td class="num">0.62~0.70</td><td>—</td><td><span class="tag bad">supervised 못 넘음</span></td></tr>
      <tr class="chosen"><td>rehearsal <span style="color:var(--faint);font-size:12px">(봄 CIR-GT + 산림청 replay 혼합)</span></td><td class="num">0.097 → 0.828</td><td>0.687 → 0.682 (유지)</td><td><span class="tag dep">채택·배포</span></td></tr>
    </tbody>
  </table></div>
  <p>핵심은 <strong class="em">파국망각 회피</strong>다. naive FT는 봄을 얻는 대신 기존 성능을 잃는다(연속학습의 고전적 함정) — 봄 단독 숫자만 보면 0.840으로 더 좋아 보이는 함정이 있으니 반드시 “기존 유지 여부”와 함께 봐야 한다. rehearsal은 옛 데이터를 일부 다시 섞어 <span class="verdict good">봄을 +0.731 올리면서 기존을 지킨다.</span> 결정 기준 자체가 코드에 박혀 있다: <i>“봄 올리되 산림청 유지, 둘 다 좋으면 배포.”</i></p>
  <div class="kv">
    <div><div class="k">망각 억제 레시피</div><div class="v"><span class="mono">LR 3e-5</span> · <span class="mono">15 epoch</span> (약하게=옛 지식 덮어쓰기 방지)</div></div>
    <div><div class="k">추가 안정화</div><div class="v">9모델 앙상블 · 오프셋 이중보정 버그 수정(raw DN) → 봄 <span class="mono">0.165 → 0.828</span></div></div>
  </div>
  <div class="note"><span class="h">정직한 한계</span>
    봄 GT(CIR-GT)는 dNBR 기반이라 <b>준순환</b> — 이 파인튜닝의 절대수치는 임계값·순위 결정에 쓰지 않고, 독립 NIFoS(안동 <b>F1 0.909</b>)로 별도 검증했다.
  </div>
</div></section>

<footer><div class="col">
  <div><strong style="color:var(--muted)">세 줄 요약</strong> — ① U-Net = 최고 봉우리가 아니라 최고 하한·일관성. ② 인코더는 크다고 낫지 않다(ResNet18 최선, 배포 ResNet34는 미검증 기본선). ③ 파인튜닝은 봄 붕괴를 살리되 파국망각을 피한 rehearsal.</div>
  <div style="margin-top:14px">근거·재현: <code>Steven/산불/근거사진/</code> (그림·스크립트·결과 원자료) · 인코더 결과 <code>runs/enc_*/results.json</code></div>
</div></footer>
</div>
""".replace("__ARCH__",arch).replace("__ENC__",enc)

out=EV+"/근거_모델설계.html"
open(out,"w").write(HTML)
print("written",out,round(len(HTML)/1e6,2),"MB")
