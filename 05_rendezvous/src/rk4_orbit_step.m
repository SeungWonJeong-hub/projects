function [r_next, v_next] = rk4_orbit_step(r, v, dt, mu, a_thrust, pert)
% rk4_orbit_step
% 현재 ECI 위치/속도를 dt초 뒤로 RK4 방식으로 전파한다.
% a_thrust가 주어지면 해당 가속도를 추가하여 적분하고, 없으면 자유 낙하(관성 비행)를 수행한다.
% pert: 섭동 모델 'none'(기본, 2체), 'J2', 'J2~J5' (gravity_accel 참고)

%% =========================
% 0. 외부 추력(가속도) 설정
%% =========================
% 입력 인자가 4개 이하일 경우(가속도 입력이 없을 경우) 추력을 0으로 설정
if nargin < 5 || isempty(a_thrust)
    a_thrust = [0; 0; 0];
end
if nargin < 6
    pert = 'none';
end

% a_thrust가 행벡터/열벡터 혼용될 수 있으므로 열벡터로 통일
a_thrust = reshape(a_thrust, 3, 1);
r = reshape(r, 3, 1);
v = reshape(v, 3, 1);

%% =========================
% 1. k1 계산
%% =========================
k1_r = v;
k1_v = gravity_accel(r, mu, pert) + a_thrust;

%% =========================
% 2. k2 계산
%% =========================
k2_r = v + 0.5*dt*k1_v;  % 수정됨: k1_r -> k1_v
k2_v = gravity_accel(r + 0.5*dt*k1_r, mu, pert) + a_thrust;

%% =========================
% 3. k3 계산
%% =========================
k3_r = v + 0.5*dt*k2_v;  % 수정됨: k2_r -> k2_v
k3_v = gravity_accel(r + 0.5*dt*k2_r, mu, pert) + a_thrust;

%% =========================
% 4. k4 계산
%% =========================
k4_r = v + dt*k3_v;      % 수정됨: k3_r -> k3_v
k4_v = gravity_accel(r + dt*k3_r, mu, pert) + a_thrust;

%% =========================
% 5. 가중평균으로 상태 업데이트
%% =========================
r_next = r + dt/6*(k1_r + 2*k2_r + 2*k3_r + k4_r);
v_next = v + dt/6*(k1_v + 2*k2_v + 2*k3_v + k4_v);

end