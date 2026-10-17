# 근거사진 작업 재개 스냅샷 (2026-09-22)

세션 끊겨도 이어가기 위한 기록. 사용자 요청 = **아키텍처 비교 + 인코더 A/B + 파인튜닝 근거**를 하나의 볼 수 있는 페이지(아티팩트)로 묶기.

## 목표 산출물
브라우저로 볼 수 있는 근거 페이지 1장에 3섹션:
1. **아키텍처 비교** (U-Net vs FPN vs DeepLabV3+) — 완성됨
2. **인코더 A/B** (ResNet18/34/50) — 학습 진행 중(아래)
3. **파인튜닝 근거** (rehearsal, 텍스트+표) — 근거 확보됨, 아래 요지

## 진행 상태 (2026-09-22 02:34 기준)

### 1) 아키텍처 비교 — 완료
- 그림: `근거사진/아키텍처비교_UNet_vs_FPN_vs_DeepLabV3.png`
- 스크립트: `근거사진/fig_arch_v2.py` (실행: `LD_LIBRARY_PATH=$HOME/wildfire-seg/libs PYTHONPATH=<uljin>/src $HOME/wildfire-seg/.venv/bin/python fig_arch_v2.py`)
- 데이터: LORO 홀드아웃 4화재(안동'25·산청'25·홍성'23·양구'22), cir_reh_ens 의 s1 3모델(unet/fpn/dlv3+), best-thr.
- **결론(정직)**: U-Net이 매 화재 1등은 아님(안동·홍성은 FPN 우세). 하지만 **평균 0.822(최고)·최악화재 0.810(최고)·표준편차 0.013(가장 일관)**. FPN=boom-bust(0.770까지 붕괴), DeepLabV3+=일관 최하. → worst-case/일관성 기준으로 U-Net 채택 근거.
- per-fire F1 (U/F/D): 안동 .826/.862/.805 · 산청 .811/.778/.786 · 홍성 .841/.851/.809 · 양구 .810/.770/.756

### 2) 인코더 A/B — 백그라운드 학습 중
- 실행 스크립트: `근거사진/enc_ablation.sh` (unet 고정, ENCODER=resnet18/34/50 × SEED 1/2/3, TRAIN=korea 61화재, INIT=imagenet, EPOCHS=40, LOSS=tversky 0.7/0.3). 평가=smp_cd 내장 holdout_fixed 8화재(계절별).
- 라이브 로그: scratchpad/enc_ablation.log (사본 = `근거사진/enc_ablation.log`, 스냅샷이라 재복사 필요)
- **영속 결과**: `<uljin>/runs/enc_{resnet18,resnet34,resnet50}_s{1,2,3}/results.json` (F1 저장). P/R 필요시 체크포인트로 재평가.
- 완료분(02:34): r18_s1 F1 0.669 · r18_s2 0.656 · r18_s3 0.646 (평균 0.657) / r34_s1 0.604
- **남은 것**: r34_s2, r34_s3, r50_s1, r50_s2, r50_s3 (예상 완료 ~03:00~03:05)
- ★주의: 작은 데이터(50타일)라 시드 편차 큼 → 3시드 평균으로만 비교.

### 3) 파인튜닝 근거 — 확보(요지)
- **왜**: 베이스는 여름/산림청 라벨 학습 → 미학습 봄 산불서 붕괴(US→한국 F1 0.000 −0.609, 봄 cross-fire baseline 0.097). 원인=개엽 phenology로 dNBR 부호반전. `GENERALIZATION.md:117`, step4_rehearsal 도크스트링.
- **왜 rehearsal**: naive FT(봄만 미세조정)=봄 0.805/0.840 얻지만 산림청 in-domain 붕괴=파국망각→롤백(배포된 적 없음). rehearsal(새 봄 CIR-GT + 산림청 replay 혼합)=봄 0.097→0.828, 산림청 0.687→0.682 유지. 도크스트링 결정기준: "봄 올리되 산림청 유지, 둘 다 좋으면 배포."
- **왜 그 세부설정**: LR 3e-5·EPOCHS 15(약하게=망각억제), 9모델 앙상블, 오프셋 이중보정 버그 수정(raw DN→봄 0.165→0.828).
- **기각 대안**: 계절매칭 사전학습(일본/미동부)=neg, 파운데이션 FT(Prithvi FT4 0.624·Presto 0.634·AlphaEarth)=supervised 못 넘음. `FINAL_RESULTS.md:59`.
- caveat: 봄 GT=CIR-GT dNBR 준순환 → 절대수치는 임계값·순위에 쓰지 말고 독립 NIFoS(안동 0.909)로 검증.

## 완료 시 남은 작업 (재개 체크리스트)
1. enc_ablation 9개 완료 확인: `ls <uljin>/runs/enc_*/results.json | wc -l` == 9, 로그 끝에 `ABLATION DONE`.
2. 인코더 F1 표/바차트 그림 생성(3시드 평균±표준편차, r18/34/50). P/R 필요시 체크포인트 재평가.
3. artifact-design 스킬 로드 → HTML 한 페이지에 3섹션(그림 base64 임베드) → Artifact 도구로 배포.
4. 파인튜닝 섹션은 위 요지 표로.

## 환경 메모
- venv: `$HOME/wildfire-seg/.venv/bin/python`, torch 실행엔 `LD_LIBRARY_PATH=$HOME/wildfire-seg/libs` 필수.
- CJK 폰트: `/home/syntax_kmu/.fonts/NanumGothic.ttf` (matplotlib fontManager.addfont).
- Jetson 7W 모드(전력상향 sudo 비번 필요=세션서 불가). 1학습 ~4.5~7분.
- uljin 프로젝트 루트: `~/syntax_inference/Steven/산불/울진산불_변화탐지`
- 배경 Bash task id: b4rhwi79m

---
## ✅ 완료 (2026-09-22 03:0x)
- 인코더 A/B 9개 완료. 결과: **ResNet18 0.657±0.009 (최고·최저비용) > ResNet50 0.616±0.039(과적합) > ResNet34 0.610±0.017**. 봄F1: 18=0.632·34=0.584·50=0.559. ★배포 ResNet34는 이 ablation이 지지 안 함(중간 기본선). caveat=12ch/holdout_fixed 셋업.
- 인코더 그림: `근거사진/인코더A_B_ResNet18_34_50.png` (스크립트 fig_encoder.py)
- 통합 페이지: `근거사진/근거_모델설계.html` (자체완결 1.93MB) → 아티팩트 배포 https://claude.ai/code/artifact/9c4e38c4-7634-4a2c-9707-b295f203b940
- 빌더: `근거사진/build_html.py`. 재배포=이 py 재실행 후 Artifact 같은 file_path.
