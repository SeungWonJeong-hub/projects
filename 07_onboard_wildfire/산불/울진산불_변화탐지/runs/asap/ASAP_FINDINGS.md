# ASAP-pdSI 한국 산불 재현·검증 (2026-09-19)

**대상 논문**: Sung, T., Yang, S., Kim, W., Kang, Y., Son, B., Lee, J., Im, J. (2026).
"Enhancing satellite-based wildfire monitoring: An advanced framework for burn severity
analysis with phenology-detrended indices." *Remote Sensing of Environment* **343**, 115488.
(UNIST 임정호팀. 계기: 한국일보 A2026090112560000598)

**pdSI 방법**: `dSI = SI_pre − SI_post` (표준), `pdSI = dSI − median_{unburned,class}(dSI)`.
주변 unburned를 화재-독립 phenology-class(pre-fire 다계절 NDVI 분위)별로 계절보정 →
소각분만 잔류. 논문은 CBI 야외플롯+MTBS(미국)로 severity만 검증, 한국 봄·이진 extent 미검증.

## 데이터
- **severity GT (NIFoS 3등급 심/중/경)**: 울진2022·안동2020·봉화·군위·밀양·양구(2022).
- **extent GT**: 위 + 산청2025(sancheong2025_GT).
- **영상**: 로컬 아카이브(~/Desktop/sentinel2_data, B02/03/04/08만 SWIR無) + STAC 다운로드
  (Earth Search v1, B04/08/11/12, **RFC3339 datetime 필수**). 울진 타일=52SEG, 안동=52SDF.

## 핵심 결과

### 1. Extent(이진 소각탐지): pdSI 게이팅 법칙 — 견고 ✅
- pdSI 이득은 **계절 불일치 크기 |median unburned dSI|의 선형함수**.
- **ΔF1 = 0.50·|mm| − 0.066**, 영교차 **임계 |mm| ≈ 0.124, 95%CI [0.104, 0.145]**, slope p=0.0.
- **6화재(경북·경남 여러 지역, 2020/2022/2025)서 임계 재현** = 일반화 확증.
- 해석: 불일치 임계 이하선 detrend가 노이즈로 해롭고, 이상서 phenology 오탐 제거로 이롭다.
  → 조작 가능한 gating 규칙(|mm|>임계일 때만 pdSI 적용).

### 2. Severity(등급): pdSI 이득 없음 — negative ❌
- 초기 2화재 NDVI: slope 0.42 R²0.53 (양성처럼 보임) → **과적합이었음**.
- 6화재 NDVI: R² 0.53→0.21, 임계 CI가 0 접촉.
- **5화재 dNBR (n=41): 완전 평평** (slope −0.07, p=0.54, R²0.03; dNDVI도 p=0.25).
- 안동 +0.045 등은 화재-특정, 일반 법칙 아님.
- **dNBR은 dNDVI보다 나은 severity 지수**(baseline ρ 0.60 vs 0.52, SWIR 값어치) — 그러나
  detrend 이득은 severity에 안 생김.

### 3. 기각된 중간 가설
- "severity↑ / extent↓ 발산" (asap_divergence.py): 통계 비유의(p=0.45/0.39), 기각.
  → 이득은 태스크가 아니라 '불일치 크기 × 기저신호'가 좌우.

## 결론
**phenology-detrend의 게이팅 이득은 detection(extent) 현상이지 severity 현상이 아니다**
— ASAP 논문이 강조한 방향(severity)과 반대. 우리 고유 기여 =
(a) extent 게이팅 임계 정량화(|mm|≈0.12, 6화재 재현),
(b) severity에는 이득 없음(한국 NIFoS 3등급 기준),
(c) ASAP를 extent로 확장 + 한국 검증 + 적용 범위 경계.
caveat: NIFoS 3등급 서수는 ASAP CBI(연속 야외측정)보다 severity 감도 낮음.

## 산출물
- 코드: `src/asap_{pdsi,mismatch,faithful,divergence,curve,curve_dl,curve_nbr}.py`,
  `src/dl_{nifos2022,severity}.py`
- 결과: `runs/asap/{mismatch_gain_curve.png, mismatch_gain_curve_7fires.png,
  dnbr_severity_curve.png, curve_points*.json, curve_fit*.json,
  {pdsi,mismatch,faithful,divergence}_result.json}`
