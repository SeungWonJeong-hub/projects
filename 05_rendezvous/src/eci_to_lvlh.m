function [rho_lvlh, vrel_lvlh, C_LVLH_to_ECI] = eci_to_lvlh(r_t, v_t, r_c, v_c)
% eci_to_lvlh
% Target/Chaser의 ECI 절대 상태에서 사용자 정의 LVLH 상대상태를 계산한다.
%
% 입력:
% r_t, v_t : Target ECI 위치/속도 [km], [km/s]
% r_c, v_c : Chaser ECI 위치/속도 [km], [km/s]
%
% 출력:
% rho_lvlh     : LVLH 상대위치 [km]
% vrel_lvlh    : LVLH 상대속도 [km/s]
% C_LVLH_to_ECI: LVLH -> ECI 회전행렬

%% =========================
% 1. LVLH 회전행렬 생성
%% =========================

C_LVLH_to_ECI = make_lvlh_frame(r_t, v_t);
C_ECI_to_LVLH = C_LVLH_to_ECI';

%% =========================
% 2. ECI 상대위치/상대속도 계산
%% =========================

rho_eci  = r_c - r_t;
vrel_eci = v_c - v_t;

%% =========================
% 3. LVLH 회전 각속도 계산
%% =========================

% LVLH 프레임의 ECI 기준 각속도. 2체에서는 궤도면이 고정이라 h/r^2 (법선 방향)이 전부다.
% 섭동이 있으면 궤도면 자체가 세차해 반경 방향 성분이 추가로 생기지만(h_dot = r x a_pert),
% 그 항은 여기서 쓰지 않는다. CW STM이 2체 회전좌표계 모델이라 계획 단계와 일관되게 맞추기 위함이다.
% 한계: J2 조건에서 아래 vrel_lvlh의 법선(y) 성분이 약 3.5% (전체 1.1%) 과소/과대 평가된다.
%       위치(rho_lvlh)와 궤적/ΔV에는 영향이 없다.
h = cross(r_t, v_t);
omega_eci = h / norm(r_t)^2;

%% =========================
% 4. 위치와 속도 변환
%% =========================

rho_lvlh = C_ECI_to_LVLH * rho_eci;

% 회전좌표계 상대속도:
% (d rho / dt)_LVLH = C_ECI_to_LVLH * [vrel_eci - omega x rho_eci]
vrel_lvlh = C_ECI_to_LVLH * (vrel_eci - cross(omega_eci, rho_eci));

end
