# 한국 산불 실측라벨 탐색 — NGII 정사영상 & 대안 라벨 조사 (2026-08-26)

## 배경
산림청 MTfire 폴리곤(현 GT)의 under-label로 미학습 홀드아웃 F1이 **0.697로 저평가**된다
(손라벨 검증: 모델 vs 손라벨 0.739 > 모델 vs 산림청 0.697 → 진짜 실력 ~0.74).
더 정확한 무료·비순환 실측 소각라벨을 찾아 이 라벨 천장을 뚫는 것이 목표.

## 1. 국토정보플랫폼(NGII) 항공정사영상
- **접근**: 승인키 불필요. 사이트 공개키로 WMTS 연도별 영상 확보.
  - 엔드포인트 `https://map.ngii.go.kr/airmapprime/map/wmts`, LAYER=`mapprime:air_{YYYY}`,
    matrixSet=`NGIS_AIR`(EPSG:5179), 연도목록 `/openapi/AirPhotoYearList.do`(2011–2025).
- **커버리지 (`ngii_coverage_check.py`)**: 51개 화재 전부 **화재 당해년(gap=0)** 고해상 영상 존재.
  2018–2019 화재=0.5m(L17), 2020–2025 화재=0.25m(L18). 연도별 실제 다른 영상(md5 검증).
- **육안검증 (`ngii_chip.py`, gold 여름/가을 4화재)**: **소각흔적 육안 식별 불가**.
  폴리곤 내부가 pre/화재년/post 전부 초록 산림. 0.5m 원본 확대도 동일.
  - 원인: ①연1회 촬영이 화재를 못 bracket ②**RGB-only** — 보이는 차이(단풍)와 소각 구분불가
    ③나디르 캐노피뷰(지표화재 은폐).
- **판정**: **RGB 정사영상은 산불 픽셀GT로 부적합**. 병목은 해상도가 아니었음
  (0.25m RGB가 S2 dNBR보다 소각정보 적음).

### ★단, 완전 종료 아님 — NIR 밴드 (미확인)
- NGII 원시영상은 **RGB + NIR 4밴드 촬영**됨 (ngii.go.kr sq=202: "종류별 원시영상(RGB, NIR) DB 구축").
- 공개 WMTS는 RGB JPEG만 서빙 → NIR 미테스트.
- **배포 정사영상 TIFF에 NIR이 포함되면** 0.25–0.5m에서 NDVI/NBR/dNBR 계산 가능 → S2처럼 소각 가시.
- **다음 작업**: 승인키/원본 TIFF 다운로드 시 `gdalinfo`로 밴드수(3 vs 4) 확인.
  4밴드 → NBR로 소각 재테스트, 3밴드 → NGII 종결.

### ★해소 (2026-08-27) — 승인 OpenAPI 인증키로 NIR 경로 전수검증 = CLOSED
사용자 OpenAPI 인증키 승인(`7DCA…B81A2`, ~/.secrets/ngii.env). map.ngii.go.kr OGC 3종 전수 실측:
- **키 유효**: `AirPhotoYearList.do` 정상(2011–2025), 커버리지체커로 실제 0.5m RGB same-year 타일 흐름 확인(울진 2019·2021 전부 O).
- **WMTS**: 지원 FORMAT = `image/jpeg`, `image/png`뿐 → **RGB**. (GetCapabilities 실측)
- **WMS**: `GetMap FORMAT=image/geotiff/tiff` → `WMS_INVALID_FORMAT` 거부, 출력 래스터는 렌더된 RGB PNG뿐. 원시밴드 불가.
- **WCS**(원시 밴드값 서빙 유일 경로): 서버는 응답(1.0/1.1.1/2.0.1)하나 **커버리지 목록 공백**(`ContentMetadata`/`Contents` empty), 레이어명 직접지정(`mapprime:air_2022` 등) DescribeCoverage 전부 **HTTP 500**. → 원시 멀티밴드 접근 0.
- **판정**: 이 인증키는 **RGB 타일/렌더 서비스 전용**. NIR/4밴드 배포경로 없음 = 문서 §1(RGB 부적합) 벽과 동일, 새 능력 없음.
- **NIR 유일 잔여경로**: 오프라인 **원본 정사영상 자료 신청**(파일 납품, apikey 아닌 별도 인적 승인). → 트랙 종료, NIFoS 문의만 잔존.

## 2. VWorld
- 위성/항공영상은 같은 NGII 원본(RGB 열화판, 연도 얕음).
- 재난방재 레이어 중 산불 관련은 `lt_c_kfdrssigugrade`(산불위험예측지도) 하나뿐 = **위험 예보**, 소각 extent 아님.
- **탈락**.

## 3. 딥리서치 — 기타 한국 산불 실측라벨 (25클레임 전수검증)
- **국내 유일 extent 후보 = NIFoS 산불피해강도등급 지도**(5m, GeoTiff/Shapefile, 심/중/경).
  BUT ①순환(NIFoS=산림청 소속기관) ②미공개(온디맨드, 다운로드 경로 없음) ③S2 순환위험(KCBI 광학 기반).
- 기타 국내 공개데이터: 산불상태별 이력·통계·위험예보 = 전부 tabular/행정동/예보 (extent 아님).
- 학계: 채&최 2024 = S2 육안 손라벨(순환)+미공개, AI-Hub 국립공원 = 산불 클래스 없음.
- 국제 소각면적: MCD64A1/VNP64A1(500m, 누락 71–72%), GABAM(30m, 누락 ~30%+광학순환) — 한국 소형 봄산불 구조적 누락.
- **판정: 은탄환 없음** (무료·비순환·extent·2022–25 커버 셋 다 만족하는 소스 0).

## 4. ★반전 — 오프라인 화재이벤트 다운로드로 GT 파이프라인 성립 (2026-08-27)
승인 OpenAPI 인증키(항공영상)로 NIR 경로는 전수검증 결과 **CLOSED**(WMTS/WMS 렌더 RGB뿐, WCS 커버리지 공백/500).
그러나 **국토정보플랫폼 화재이벤트별 자료 다운로드**가 다른 길을 열었다(2020–2026 산불 **29건** 다운로드 가능).

첫 번들 `200424_안동풍천면_산불`(2020-04-24) 실측:
- **구성**: `CompResult_SEN_MMDD_CIR_s.tif` CIR pre/post 2장(0414 화재전·0429 화재후) + 수치지도 벡터(하천 N3A_E·도로 N3A_A·농경지 N3A_D, EPSG:5179).
- **CIR = NIR 있음**: 8-bit 위색 3밴드, band1=NIR/band2=Red/band3=Green → NDVI 계산 가능. 단 **"SEN"=Sentinel-2 10m**(픽셀 10.16m), 지오레퍼런싱 없음(`_s`=미리보기), SWIR無→NBR불가·상대NDVI만.
- **★지오레퍼런싱 해결**: 세 벡터 extent 동일 = CIR 지상범위 → 벡터 extent로 CIR에 geotransform 부여. 도로/하천 오버레이 정렬 검증 통과.
- **소각 가시성 = YES**(RGB정사와 정반대): CIR pre/post 15일 브래킷이라 봄 개엽 confound 최소 → dNDVI로 소각흔 선명·자동분리. **봄 phenology 벽을 타이트 브래킷으로 우회**.
- **GT 자동생성**: dNDVI<-0.12 & NDVIpost<0.35 + 형태학정리 → 소각폴리곤 ~1,700ha(공식 1,944ha 근접). `src/ngii_cir_gt.py`.
- **모델 실측 채점**(`src/andong_model_eval.py`): STAC/AWS S2 pre2019/post2020 6밴드 → Prithvi FT4 추론 → CIR-GT와 **F1=0.480(P0.37/R0.69)**, 봄 과예측(3,231ha, best-thr 하한). 대조 **산림청 MTfire 폴리곤 vs CIR-GT = F1 0.000**(이 화재 폴리곤이 geom 48.5ha 조각, AR속성 1944 ≠ 실제) → **이 화재에선 CIR-GT가 산림청보다 정확, 모델 0.48 > 산림청 0.00**.
- **함의**: 2022–2025 GT부재 벽을 CIR-GT로 메워 **미학습 최신화재 모델 채점 최초 가능**. 개선 후보=농경지레이어 마스킹(precision↑)·여름가을 채점·손보정.
- **다음 리드 = 재난공간정보 API 키**(발급 대기): 29건은 미리보기, API는 더 많이/원본 추정. 키 오면 엔드포인트 probe로 화재수·좌표·해상도·피해벡터 유무 실측 판별.

## 종합
- RGB 정사영상·OpenAPI NIR 경로 = CLOSED. 그러나 **오프라인 CIR 번들(Sentinel 10m+벡터)로 GT 파이프라인 성립** = 새 무료 라벨 경로 확보.
- 산림청 MTfire는 화재마다 품질 편차(안동=48.5ha 조각), CIR-GT가 검증·보정 도구.
- 남은 리드: ① 재난공간정보 API(대기) ② NIFoS 직접 문의.

## 스크립트
- `src/ngii_fire_targets.py` — MTfire.shp → 화재 51개 좌표·날짜·폴리곤 목록 (GEE 불필요)
- `src/ngii_coverage_check.py` — 화재별 연도별 항공영상 커버리지 스캔 (LEVEL: L17=0.5m, L18=0.25m 2020+)
- `src/ngii_chip.py` — gold 화재 pre/화재년/post 정사영상 chip + 산림청 폴리곤 오버레이 (육안검증)
- `src/ngii_cir_gt.py` — ★화재 이벤트 번들(CIR pre/post+벡터) → 지오레퍼런싱된 소각 GT(shp/mask/qa.png)
- `src/andong_model_eval.py` — ★안동 화재 STAC S2 취득 → Prithvi FT4 추론 → CIR-GT와 F1
