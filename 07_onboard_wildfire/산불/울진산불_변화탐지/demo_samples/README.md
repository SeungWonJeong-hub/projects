# 산불 웹앱 데모 샘플 (한국 실제 화재)

kf20190826_3047ha — 2019-08-26 발생 여름 대형화재(3047ha), 실제 소각 47%.
6밴드 GeoTIFF [B02,B03,B04,B8A,B11,B12] 반사율×1e4.

## 사용
물변화탐지_웹앱 → 🔥 산불 → GeoTIFF 업로드:
- Before: kf20190826_before.tif
- After : kf20190826_after.tif
→ smp DeepLabV3+ 모델이 소각흔 마스크 출력.
preview PNG는 눈으로 before(초록숲)/after(소각흔) 확인용.
