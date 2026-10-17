function [dv1_lvlh, dv2_lvlh] = cw_stm_delta_v(rho0, vrel0_current, n, tf)
% cw_stm_delta_v
% 1차 출발 델타V와 2차 도착 델타V를 모두 계산하여 반환한다.
%
% 입력:
% rho0          : 기동 시작 시점의 상대 위치 [km]
% vrel0_current : 기동 시작 시점의 현재 상대 속도 [km/s]
% n             : 궤도 평균 각속도 [rad/s]
% tf            : 비행 소요 시간 (TOF) [s]
%
% 출력:
% dv1_lvlh : 1차 출발 델타V 벡터 (3x1) [km/s]
% dv2_lvlh : 2차 도착 델타V 벡터 (3x1) [km/s]

%% =========================
% 1. 기본 삼각함수 계산
%% =========================
nt = n * tf;
c = cos(nt);
s = sin(nt);

%% =========================
% 2. 사용자 축 기준 CW 위치 STM 구성 (X:전진, Y:수직, Z:반경)
%% =========================
Phi_rr = [ ...
    1,       0,       6*(s - nt);
    0,       c,       0;
    0,       0,       4 - 3*c ];
    
Phi_rv = [ ...
    (4*s - 3*nt)/n,      0,       -2*(1-c)/n;
    0,                   s/n,      0;
    2*(1-c)/n,           0,        s/n ];
    
%% =========================
% 3. 1차 기동 (출발 Delta-V) 계산
%% =========================
% 목표 지점(target_offset을 뺀 상대위치가 0이 되는 곳)으로 가기 위해 요구되는 초기 상대속도
vrel0_required = -Phi_rv \ (Phi_rr * rho0);
dv1_lvlh = vrel0_required - vrel0_current;

%% =========================
% 4. 2차 기동 (도착 Delta-V) 예측 계산
%% =========================
Phi_vr = [ ...
    0,       0,       6*n*(c - 1);
    0,      -n*s,     0;
    0,       0,       3*n*s ];
    
% ✨수정됨: Phi_vv의 (1,3)과 (3,1) 요소 부호 교정
Phi_vv = [ ...
    4*c - 3,             0,      -2*s; 
    0,                   c,       0;
    2*s,                 0,       c ]; 
    
% 도착 순간 체이서가 가지게 될 예상 상대 속도
vrel_target_expected = Phi_vr * rho0 + Phi_vv * vrel0_required;

% 도착했을 때 타겟과의 상대 속도를 0으로 만들기 위한 제동 델타V
dv2_lvlh = -vrel_target_expected;

end