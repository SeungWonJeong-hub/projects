"""산불 소각면적(ha) -> 피해액(원화) 환산 모듈.

지상국(B200) 파이프라인용. device-agnostic 순수 파이썬(표준 라이브러리만).
Sentinel-2 변화탐지가 산출하는 area_ha(예: core/vectorize.py의 area_ha)를 받아
국가(공식) 기준 피해액과 민간(보험/감정평가) 기준 피해액을 함께 산정한다.

출력 규칙(사용자 요구): 국가 기준액을 주로 표시하고, 민간 기준액은 괄호로 병기.
    예) "1조 2,345억원 (민간 2,100억원)"

국가 vs 민간 개념(중요 — 2026-09-10 정정):
    - 국가(공식) = **사회적 총손실**. 임목직접피해 + 복구비 + 진화비 + 공익기능 손실.
      공익기능(4,110만원/ha)이 압도적 비중.
    - 민간(산주 사적 실손) = **산주가 실제로 잃거나 지출하는 손실**.
      = 임목 자산가치 + 재조림비 실손 + 임야 지가 하락분.
      공익기능·진화비는 산주가 회수 못 하므로 제외. 대신 기존에 빠졌던
      재조림비 전액과 지가하락을 포함(→ 임목값만일 때보다 민간액이 커짐).
    겹치는 항목: 임목(양쪽), 재조림/복구(국가=복구비, 민간=재조림 실손).
    안 겹침: 공익기능·진화비(국가만), 임야 지가하락(민간만; 토지는 국가 산불피해 대상 아님).

단가 근거·연도·확신도는 같은 폴더의 `피해액기준_조사.md` 참조.
=====================================================================================
정직성: 아래 상수 중 PUBLIC_BENEFIT_PER_HA(공익기능)·TIMBER_VOLUME_PER_HA(재적)는
        공식 발표치 기반. TIMBER_PRICE_PER_M3(임목가)·RECOVERY_FIRE_PER_HA(복구/진화)·
        PRIVATE_TIMBER_RATIO·REFORESTATION_PER_HA(재조림)·LAND_DEPRECIATION_RATE(지가)는
        [추정/미확인] 보수적 기본값이다. 특히 지가하락은 한국 실증 정량자료 미확보 →
        임야 단가 미입력 시 0 처리(과대추정 방지). TODO 참조.
=====================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# =====================================================================================
# 단가 상수 (근거는 피해액기준_조사.md)
# 갱신 시 이 블록만 수정하면 됨.
# =====================================================================================

#: 산림 공익기능 평가액 (원/ha). [원문확인 총액 259조 + 역산]
#: 국립산림과학원 2020년 기준 259조원 / 산림면적 630만ha ≈ 4,110만원/ha.
#: 경향신문/정부 2025 산불 추정에서 실제 사용된 계수.
#: 출처: korea.kr policyNewsView newsId=156559826, khan.co.kr 202504022039005
#: TODO: 차기 종합 공익기능 평가(2025 기준, 2027 발표 예정) 나오면 교체.
PUBLIC_BENEFIT_PER_HA: float = 41_100_000.0

#: 단위면적당 임목축적 (m³/ha). [원문확인]
#: 2020년 말, 제7차 국가산림자원조사 전국 평균 165.2 m³/ha.
#: 출처: 산림청 산림기본통계 (forest.go.kr NKFS_04_05_01)
#: TODO: 제8차 국가산림자원조사 갱신치로 교체.
TIMBER_VOLUME_PER_HA: float = 165.2

#: 평균 임목가(입목가) (원/m³). [미확인/추정 - 보수적 기본값]
#: 시장가역산법(원목시장가 - 벌채·반출비)상 원목가보다 낮음. 혼효림·소경목 포함 평균 가정.
#: 출처(원목가 참고): woodkorea.co.kr (표가 이미지라 원문 단가 미확정)
#: TODO: 산림청 '국산재 원목 시장가격 동향' 최신 수종별 단가로 교체.
TIMBER_PRICE_PER_M3: float = 60_000.0

#: 복구비(조림·응급) + 진화비(인력·헬기)의 면적당 근사 (원/ha). [추정]
#: 통계연보 실측 미확보. 울진 역산(9,086억/16,302ha≈5,573만/ha)에서
#: 임목(≈990만)+공익(4,110만)을 뺀 잔차(약 470만/ha)에 정합하도록 보수적으로 500만/ha 설정.
#: TODO: 산불통계연보 연도별 복구비·진화비 실수치로 분해 교체.
RECOVERY_FIRE_PER_HA: float = 5_000_000.0

#: 민간(보험/감정) 임목 평가액 / 국가 임목 직접피해 비율. [추정]
#: 민간은 시장가역산·감정으로 다소 보수적. 0.85 가정.
#: TODO: 감정평가 사례·보험 약관 단가 확보 시 교체.
PRIVATE_TIMBER_RATIO: float = 0.85

#: 재조림비 실손 (원/ha) — 산불로 죽은 임분 재조림 전액(식재+5년 사후관리). [부분확인/추정]
#: anchor: 울진·삼척 산림생태복원 325억원 / 4,789ha ≈ 679만원/ha(조림49%+자연복원51% 혼합).
#:         순수 인공조림+사후관리는 통상 ha당 500만~1,000만원 → 보수적 700만원/ha.
#: 출처: south.forest.go.kr(울진 복원), 금산군 보조조림 공고(자부담 10%)
#: TODO: 산림청 조림 표준품셈/토목단가 CSV의 수종별 ha단가로 교체.
REFORESTATION_PER_HA: float = 7_000_000.0

#: 재조림 국고보조 하 산주 자부담 비율 (국고70%+지방20%=90% 보조 → 자부담 10%). [원문확인]
#: 출처: 산림조합 iforest.nfcf.or.kr, 금산군 2026 보조조림 공고문.
#: 민간 실손을 '전액(사회적)'으로 볼지 '현금지출(자부담)'로 볼지 선택용.
REFOREST_OWNER_SHARE: float = 0.10

#: 임야(토지) 단가 (원/ha). [미상 — 기본 0]
#: 한국은 임지·임목 분리평가라 임목 소실이 토지 공시지가를 직접 낮추지 않음.
#: 대상지 공시지가를 알면 estimate_damage(land_price_per_ha=...)로 주입. 미입력 시 0 → 지가하락 0.
LAND_PRICE_PER_HA: float = 0.0

#: 산불 후 임야 지가 하락률. [미확인 — 한국 실증 정량자료 미확보, 보수적]
#: 분리평가 원칙상 하락은 있어도 작을 가능성. land_price 미입력 시 어차피 0원.
#: 출처(원칙): first-a.kr, pagarlic.com(임지·임목 분리)
#: TODO: 산불 전후 임야 실거래/공시지가 변화 실증자료 확보 시 교체.
LAND_DEPRECIATION_RATE: float = 0.05

#: 기본 피해율(burn fraction). 심각도맵이 없으면 전소(1.0) 가정.
DEFAULT_BURN_FRACTION: float = 1.0

# 참고 상수 (역산 검증용, 계산에는 미사용)
_REF_TOTAL_PUBLIC_BENEFIT_KRW = 259e12       # 259조원 (2020)
_REF_FOREST_AREA_HA = 6_300_000.0            # 산림면적 630만ha


@dataclass
class DamageResult:
    """피해액 산정 결과. 금액 단위는 모두 원(KRW)."""
    area_ha: float
    burn_fraction: float
    species: Optional[str]
    age_class: Optional[str]

    # 국가(공식) 기준 분해
    national_timber_krw: float          # 임목 직접피해
    national_public_krw: float          # 공익기능 손실
    national_recovery_fire_krw: float   # 복구비 + 진화비
    national_total_krw: float           # 국가 종합

    # 민간(산주 사적 실손) 기준 분해
    private_timber_krw: float           # 임목 자산가치
    private_reforestation_krw: float    # 재조림비 실손(전액 또는 자부담)
    private_land_loss_krw: float        # 임야 지가 하락분
    private_total_krw: float            # 민간 종합

    unit_prices: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def estimate_damage(
    area_ha: float,
    species: Optional[str] = None,
    age_class: Optional[str] = None,
    burn_fraction: float = DEFAULT_BURN_FRACTION,
    timber_volume_per_ha: Optional[float] = None,
    land_price_per_ha: Optional[float] = None,
    reforestation_basis: str = "full",
) -> DamageResult:
    """소각면적(ha)으로부터 국가(사회적 총손실)/민간(산주 사적 실손) 피해액을 산정한다.

    Args:
        area_ha: 소각(피해) 면적 [ha]. 파이프라인 area_ha.
        species: 수종 힌트. 'conifer'|'침엽수', 'broadleaf'|'활엽수', 'mixed'|'혼효림'.
                 재적/임목가 보정에 사용(현재는 재적만 보정, 미지정이면 전국 평균).
        age_class: 영급 힌트(문자열, 예 'III'). 현재 로깅용(정보 부족으로 계수 미적용).
        burn_fraction: 피해율 0~1. 심각도맵이 있으면 평균 피해율을 넣는다. 기본 1.0(전소).
        timber_volume_per_ha: 재적을 외부에서 알면 직접 지정(m³/ha). 없으면 상수/수종보정 사용.
        land_price_per_ha: 대상 임야 단가(원/ha). 주면 지가하락분을 계상, 없으면 0(과대추정 방지).
        reforestation_basis: 민간 재조림비 계상 기준.
            'full'  = 재조림 전액(사회적 실손 관점, 기본).
            'owner' = 산주 자부담분만(국고보조 90% 제외 → 실제 현금지출 관점).

    Returns:
        DamageResult. 모든 금액은 원(KRW).

    Note:
        area_ha가 '영향구역'인지 '실소각면적'인지에 따라 결과 성격이 다르다(조사.md 3절).
        영향구역이면 실제 확정면적은 통상 그보다 작다(울진 2022는 약 78%).
    """
    if area_ha < 0:
        raise ValueError("area_ha must be >= 0")
    if not (0.0 <= burn_fraction <= 1.0):
        raise ValueError("burn_fraction must be in [0, 1]")
    if reforestation_basis not in ("full", "owner"):
        raise ValueError("reforestation_basis must be 'full' or 'owner'")

    notes: list = []

    # --- 재적(m³/ha): 수종 보정 (전국 평균 165.2 대비) ---
    if timber_volume_per_ha is not None:
        vol = float(timber_volume_per_ha)
        notes.append(f"재적 외부지정 {vol} m³/ha")
    else:
        vol = TIMBER_VOLUME_PER_HA
        sp = (species or "").strip().lower()
        # 보수적 근사 계수 [추정]: 침엽수>혼효>활엽수 대략 경향. 미확인이면 1.0.
        vol_factor = {
            "conifer": 1.05, "침엽수": 1.05,
            "mixed": 1.00, "혼효림": 1.00,
            "broadleaf": 0.95, "활엽수": 0.95,
        }.get(sp, 1.0)
        if vol_factor != 1.0:
            notes.append(f"수종({species}) 재적보정 x{vol_factor} [추정]")
        vol = vol * vol_factor

    effective_ha = area_ha * burn_fraction
    if burn_fraction != 1.0:
        notes.append(f"피해율 {burn_fraction} 적용 -> 유효면적 {effective_ha:.2f} ha")

    # --- 국가(공식) 기준 ---
    national_timber = effective_ha * vol * TIMBER_PRICE_PER_M3
    national_public = effective_ha * PUBLIC_BENEFIT_PER_HA
    national_recovery_fire = effective_ha * RECOVERY_FIRE_PER_HA
    national_total = national_timber + national_public + national_recovery_fire

    # --- 민간(산주 사적 실손): 임목자산 + 재조림비 + 지가하락. 공익·진화 미계상 ---
    # (A) 임목 자산가치 (감정 보수계수)
    private_timber = national_timber * PRIVATE_TIMBER_RATIO
    # (B) 재조림비 실손: 전액(사회적) 또는 자부담(현금지출)
    reforest_share = 1.0 if reforestation_basis == "full" else REFOREST_OWNER_SHARE
    private_reforestation = effective_ha * REFORESTATION_PER_HA * reforest_share
    if reforestation_basis == "owner":
        notes.append(f"재조림비=산주 자부담분만(국고보조 {int((1-REFOREST_OWNER_SHARE)*100)}% 제외).")
    else:
        notes.append("재조림비=전액(사회적 실손). 산주 현금지출은 자부담분만은 basis='owner'.")
    # (C) 임야 지가 하락분. land_price 미입력이면 0 (과대추정 방지)
    lp = LAND_PRICE_PER_HA if land_price_per_ha is None else float(land_price_per_ha)
    # 지가하락은 소각 유무(면적 전체) 기준, 피해율에 비례시킴
    private_land_loss = effective_ha * lp * LAND_DEPRECIATION_RATE
    if lp <= 0:
        notes.append("임야 지가 미입력 → 지가하락 0원(한국은 임지·임목 분리평가, 직접하락 소액·미확인).")
    else:
        notes.append(f"지가하락 {LAND_DEPRECIATION_RATE:.0%} 가정 [미확인].")

    private_total = private_timber + private_reforestation + private_land_loss

    if age_class:
        notes.append(f"영급 힌트 '{age_class}' 수신(현재 계수 미적용 [미확인]).")

    return DamageResult(
        area_ha=area_ha,
        burn_fraction=burn_fraction,
        species=species,
        age_class=age_class,
        national_timber_krw=national_timber,
        national_public_krw=national_public,
        national_recovery_fire_krw=national_recovery_fire,
        national_total_krw=national_total,
        private_timber_krw=private_timber,
        private_reforestation_krw=private_reforestation,
        private_land_loss_krw=private_land_loss,
        private_total_krw=private_total,
        unit_prices={
            "public_benefit_per_ha": PUBLIC_BENEFIT_PER_HA,
            "timber_volume_per_ha": vol,
            "timber_price_per_m3": TIMBER_PRICE_PER_M3,
            "recovery_fire_per_ha": RECOVERY_FIRE_PER_HA,
            "private_timber_ratio": PRIVATE_TIMBER_RATIO,
            "reforestation_per_ha": REFORESTATION_PER_HA,
            "reforestation_basis": reforestation_basis,
            "land_price_per_ha": lp,
            "land_depreciation_rate": LAND_DEPRECIATION_RATE,
        },
        notes=notes,
    )


# =====================================================================================
# 한국어 금액 포맷
# =====================================================================================

def format_krw(amount: float, digits: int = 1) -> str:
    """원 금액을 조/억/만원 단위 한국어 문자열로. 예 123456789012 -> '1,234억 5,679만원'."""
    a = int(round(amount))
    if a == 0:
        return "0원"
    sign = "-" if a < 0 else ""
    a = abs(a)

    jo = a // 1_0000_0000_0000            # 조
    rem = a % 1_0000_0000_0000
    eok = rem // 1_0000_0000             # 억
    rem2 = rem % 1_0000_0000
    man = rem2 // 1_0000                 # 만
    won = rem2 % 1_0000                  # 원

    parts = []
    if jo:
        parts.append(f"{jo:,}조")
    if eok:
        parts.append(f"{eok:,}억")
    if man:
        parts.append(f"{man:,}만")
    if won and not (jo or eok):          # 소액일 때만 원단위 표기
        parts.append(f"{won:,}")
    s = " ".join(parts)
    # 마지막 토큰에 '원' 부착
    if s.endswith("조") or s.endswith("억") or s.endswith("만"):
        s += "원"
    else:
        s += "원"
    return sign + s


def format_result(result: DamageResult, style: str = "short") -> str:
    """사용자 요구 포맷: '국가기준액 (민간 민간기준액)'.

    style='short'  -> "1조 2,345억원 (민간 2,100억원)"
    style='detail' -> 내역까지 여러 줄로.
    """
    nat = format_krw(result.national_total_krw)
    priv = format_krw(result.private_total_krw)
    headline = f"{nat} (민간 {priv})"
    if style == "short":
        return headline

    lines = [
        f"소각면적: {result.area_ha:,.1f} ha (피해율 {result.burn_fraction:g})",
        f"피해액(국가 종합): {nat}",
        f"  - 임목 직접피해 : {format_krw(result.national_timber_krw)}",
        f"  - 공익기능 손실 : {format_krw(result.national_public_krw)}",
        f"  - 복구비+진화비 : {format_krw(result.national_recovery_fire_krw)}",
        f"피해액(민간 실손): {priv}   [산주 사적 손실: 공익·진화 제외, 재조림·지가 포함]",
        f"  - 임목 자산가치 : {format_krw(result.private_timber_krw)}",
        f"  - 재조림비 실손 : {format_krw(result.private_reforestation_krw)}",
        f"  - 임야 지가하락 : {format_krw(result.private_land_loss_krw)}",
        f"표시용: {headline}",
    ]
    if result.notes:
        lines.append("비고:")
        lines.extend(f"  * {n}" for n in result.notes)
    return "\n".join(lines)


# =====================================================================================
# 데모 (울진 2022 사례 등)
# =====================================================================================

if __name__ == "__main__":
    print("=" * 74)
    print("산불 피해액 환산 데모  (단가 근거: 피해액기준_조사.md)")
    print("=" * 74)

    # 사례 1: 울진·삼척 2022 (영향구역 16,302 ha) [원문확인 면적]
    print("\n[사례1] 울진·삼척 2022, 영향구역 16,302 ha")
    r1 = estimate_damage(area_ha=16_302.0)
    print(format_result(r1, style="detail"))
    # sanity-check: 정부 발표 종합 피해액 약 9,086억원 [스니펫]
    print(f"  (참고) 정부 발표 종합 피해액 약 9,086억원, ha당 역산 약 5,573만원/ha")
    print(f"  (본모듈) ha당 국가 종합 = {r1.national_total_krw / r1.area_ha:,.0f} 원/ha")
    # 민간 before/after (재설계 효과)
    before_private = r1.national_timber_krw * PRIVATE_TIMBER_RATIO   # 구: 임목값만
    print(f"  [민간 before/after] 임목값만 {format_krw(before_private)}"
          f" -> 재설계(+재조림) {format_krw(r1.private_total_krw)}")

    # 사례 2: 확정면적 근사(영향구역의 78%) 적용 시
    print("\n[사례2] 울진 확정면적 근사 = 16,302 x 0.78")
    r2 = estimate_damage(area_ha=16_302.0 * 0.78)
    print(format_result(r2, style="short"))

    # 사례 3: 2025 경북 등 48,238.61 ha
    print("\n[사례3] 2025 경북 등 48,238.61 ha")
    r3 = estimate_damage(area_ha=48_238.61)
    print(format_result(r3, style="short"))
    print(f"  (참고) 경향신문 공익가치만 추정 약 1조 9,826억원(=면적x4,110만/ha)")
    print(f"  (본모듈) 공익기능 손실분만 = {format_krw(r3.national_public_krw)}")

    # 사례 4: 소규모 + 피해율(심각도) + 수종 + 임야단가(지가하락 반영)
    print("\n[사례4] 소규모 320 ha, 침엽수, 피해율 0.6, 임야단가 3천만원/ha")
    r4 = estimate_damage(area_ha=320.0, species="침엽수", burn_fraction=0.6,
                         land_price_per_ha=30_000_000.0)
    print(format_result(r4, style="detail"))

    # 사례 5: 산주 현금지출 관점(재조림 자부담분만)
    print("\n[사례5] 320 ha, 산주 자부담 관점(basis='owner')")
    r5 = estimate_damage(area_ha=320.0, reforestation_basis="owner")
    print(format_result(r5, style="short"))
