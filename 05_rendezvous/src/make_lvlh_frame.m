function C_LVLH_to_ECI = make_lvlh_frame(r_t, v_t)
% make_lvlh_frame
% 타겟 위성의 ECI 위치와 속도로 사용자 정의 LVLH 좌표계를 만든다.
%
% 사용자 정의 LVLH 축:
% x축 = 위성 진행방향
% y축 = 궤도평면 수직방향
% z축 = 지구 반대방향
%
% 출력:
% C_LVLH_to_ECI : LVLH 성분 벡터를 ECI 성분 벡터로 변환하는 행렬

%% =========================
% 1. z축: radial outward
%% =========================

e_z = r_t / norm(r_t);

%% =========================
% 2. y축: orbit normal
%% =========================

h = cross(r_t, v_t);
e_y = h / norm(h);

%% =========================
% 3. x축: along-track
%% =========================

% 오른손 좌표계를 만족하도록 x = y cross z 로 정의한다.
e_x = cross(e_y, e_z);
e_x = e_x / norm(e_x);

%% =========================
% 4. 회전행렬 구성
%% =========================

C_LVLH_to_ECI = [e_x, e_y, e_z];

end
