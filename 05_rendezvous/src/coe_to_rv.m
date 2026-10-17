function [r_eci, v_eci] = coe_to_rv(a, e, i, RAAN, argp, nu, mu)
% coe_to_rv
% 궤도 6요소를 ECI 위치/속도로 변환한다.
%
% 입력:
% a    : 장반경 [km]
% e    : 이심률 [-]
% i    : 경사각 [rad]
% RAAN : 승교점 적경 [rad]
% argp : 근지점 인수 [rad]
% nu   : 진근점이각 [rad]
% mu   : 중력상수 [km^3/s^2]
%
% 출력:
% r_eci : ECI 위치 [km]
% v_eci : ECI 속도 [km/s]

%% =========================
% 1. 반통경 p 계산
%% =========================

p = a * (1 - e^2);

%% =========================
% 2. PQW 좌표계 위치/속도 계산
%% =========================

r_pqw = (p / (1 + e*cos(nu))) * [cos(nu); sin(nu); 0];
v_pqw = sqrt(mu/p) * [-sin(nu); e + cos(nu); 0];

%% =========================
% 3. PQW -> ECI 회전행렬 계산
%% =========================

R3_RAAN = [ cos(RAAN), -sin(RAAN), 0;
            sin(RAAN),  cos(RAAN), 0;
            0,          0,         1 ];

R1_i = [ 1, 0,       0;
         0, cos(i), -sin(i);
         0, sin(i),  cos(i) ];

R3_argp = [ cos(argp), -sin(argp), 0;
            sin(argp),  cos(argp), 0;
            0,          0,         1 ];

Q_pqw_to_eci = R3_RAAN * R1_i * R3_argp;

%% =========================
% 4. ECI 위치/속도 도출
%% =========================

r_eci = Q_pqw_to_eci * r_pqw;
v_eci = Q_pqw_to_eci * v_pqw;

end
