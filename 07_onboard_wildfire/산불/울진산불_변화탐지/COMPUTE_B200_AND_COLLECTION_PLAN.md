# 산불 소각영역 — 컴퓨트(B200) & 나라단위 데이터셋 수집 계획

_작성 2026-09-07 · 상태: 착수 직전(수집기 미구현)_

## 0. 컴퓨트 환경 (2026-09-07 전환)
- **실행/스케일 = NVIDIA B200** (Blackwell, ~180GB HBM3e급). 대용량 배치·풀해상도·나라단위 수집을 여기서 돌린다.
- **개발/소규모 검증 = Jetson Orin** (이 머신). 코드 작성·파이프라인 검증만.
- 코드 규칙: **device-agnostic**(`cuda` 자동감지, Orin 하드코딩 금지)·**경로 상대화**·**재개가능(idempotent) 배치**.
- 산출물: `~/syntax_inference/Steven/` · `main` 브랜치 · GitHub `park-hyun-su/syntax_inference`.

## 1. 목표
한 화재만이 아니라 **한 나라의 모든 화재를 빠짐없이 카탈로그화**하여 각각 **소각영역(extent) + 면적(ha)** 을 산출하는 배치 수집기.
※ 지금 단계는 **전이학습이 아님** — 순수 데이터셋 수집/소각영역 산출.

## 2. 수집 단위 = "화재 이벤트"
레코드: `{나라, 발화일, bbox, pre 씬id, post 씬id, 소각마스크, area_ha, 라벨출처, QC플래그}`

## 3. 이벤트 발견 (어디·언제 불났나) — 2계층
| 계층 | 출처 | 역할 | 상태 |
|---|---|---|---|
| Tier-1 권위 소각면적 산출물 | US=**MTBS**, EU=**EFFIS/CEMS**, Canada=NBAC, AUS=DEA/NAFI | 이벤트 목록 + 폴리곤 라벨(GT) 동시 | MTBS·CEMS 로컬 보유(`mtbs/`,`cems/`) |
| Tier-2 활성화재 아카이브 | **FIRMS** VIIRS 375m 열점 | 산출물 없는 나라 → 시공간 DBSCAN 클러스터 → 이벤트 bbox+기간 | 한국서 완료(`firms_korea_bbox_*.geojson`), 전지구 일반화만 |

## 4. 이벤트별 영상·소각영역 산출 (기존 전처리 재사용)
1. S2 L2A STAC 검색 → 발화 전 최저구름 pre + 진화 후 post (계절 매칭)
2. 10m 격자 (RGB+NIR 네이티브, SWIR B11/B12는 20m→10m 업샘플)
3. 구름/그림자: SCL + 다시점 마스크 `mt_cloud.py`
4. 소각마스크: Tier-1=권위 폴리곤 래스터화 GT / Tier-2=dNBR-Otsu 정제 self-label (`build_dataset.refine_label`)
5. area_ha 산출

## 5. 품질·함정
- **계절/phenology**: 한국 봄 개엽기 dNBR 반전 → 나라별 화재계절 인지 창 설정. Tier-1 폴리곤은 반전 영향 없음.
- **구름·연무**(캐나다/북방): 날짜창 확대 + 다시점 컴포지팅.
- **라벨 하모나이제이션**: 나라별 정의 차이 → 픽셀단위 소각/비소각 통일 (`harmonize_labels.py`, MTBS 7생태권 방식).
- dedup·최소면적·지오정합 필터.

## 6. 자동화
`카탈로그 → 이벤트 루프 → 다운로드 → 빌드 → QC`, 재개가능·rate-limit. 나라별 **매니페스트 CSV** 산출.

## 7. 재사용 vs 신규
- **있음**: `stac.py`·`build_dataset.refine_label`·`mt_cloud.py`·MTBS·CEMS·FIRMS클러스터링(한국)·`harmonize_labels.py`·`download_scenes_10m.py`(10m 다운로드)
- **신규**: ① FIRMS 클러스터링 전지구 일반화 ② 나라단위 수집 오케스트레이터 ③ Canada/AUS 산출물 커넥터

## 8. 시작점
**US(MTBS)** — 권위 GT 로컬 보유 + 여름체제라 dNBR 강건 → "완전 일반화" 첫 나라 리스크 최저. 그다음 EU(CEMS).
