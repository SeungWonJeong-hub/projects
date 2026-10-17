function a = gravity_accel(r, mu, pert)
% gravity_accel
% ECI 위치 r에서의 중력가속도를 계산한다.
%
% 입력:
% r    : 위치 벡터 [km]
% mu   : 지구 중력상수 [km^3/s^2]
% pert : 섭동 모델 'none'(2체, 기본), 'J2', 'J2~J5'(J2,J3,J4,J5)
%
% 출력:
% a  : 중력가속도 [km/s^2]
%
% 대역(zonal) 조화항 Jn의 퍼텐셜과 가속도 (u = z/r, Pn = 르장드르 다항식):
%   U_n = -mu*Jn*Re^n / r^(n+1) * Pn(u)
%   a_n = mu*Jn*Re^n / r^(n+2) * [ ((n+1)*Pn(u) + u*Pn'(u)) * r_hat - Pn'(u) * z_hat ]
% 자전축(ECI z축) 대칭이라 지구 자전과 무관하게 ECI에서 바로 계산할 수 있다.

r = reshape(r, 3, 1);
rn = norm(r);
a = -mu * r / rn^3;

if nargin < 3 || strcmp(pert, 'none')
    return;
end
switch pert
    case 'J2',    terms = 2;
    case 'J2~J5', terms = 2:5;
    otherwise,    error('gravity_accel:pert', 'Unknown perturbation model: %s', pert);
end
nmax = max(terms);

% EGM2008 비정규화 대역 계수 J2~J5, WGS-84 적도 반경
J  = [0, 1.08262668e-3, -2.53265649e-6, -1.61962159e-6, -2.27296083e-7];
Re = 6378.137;

u = r(3) / rn;
rhat = r / rn;
zhat = [0; 0; 1];

% 르장드르 다항식 P0~P5와 도함수 (점화식)
P  = zeros(1, nmax + 1);  dP = zeros(1, nmax + 1);   % 인덱스 k -> 차수 k-1
P(1) = 1;  P(2) = u;
dP(1) = 0; dP(2) = 1;
for n = 2:nmax
    P(n+1)  = ((2*n - 1) * u * P(n) - (n - 1) * P(n-1)) / n;
    dP(n+1) = dP(n-1) + (2*n - 1) * P(n);
end

for n = terms
    k = mu * J(n) * Re^n / rn^(n+2);
    a = a + k * (((n + 1) * P(n+1) + u * dP(n+1)) * rhat - dP(n+1) * zhat);
end

end
