function [best_idx,sel_rho,sel_rc,sel_vc,min_err,DV1,DV2,info] = ...
         evaluate_maneuver_module(rt0,vt0,rc0,vc0,t1,tof,tf,a_max,dt,mu, ...
                                  method,sel_idx,pert,mode,t_mcc)
% evaluate_maneuver_module
%   mode='finite'  : 유한 연소 9개 시나리오 비교 (N회 기동 지원)
%   mode='impulse' : 순수 임펄스 기동 (투입 → 중간 보정 0~n회 → 도착 제동)
%                    t_mcc = 중간 기동 시각[s] 벡터, 비우면 기동 2회
% 계획(plan)은 두 모드 모두 2체. 실제 전파에만 pert 적용.

%% 0. 기본값 및 공용 입력 검증
if nargin<12, sel_idx=[]; end
if nargin<13 || isempty(pert),  pert='none';  end
if nargin<14 || isempty(mode),  mode='finite'; end
if nargin<15, t_mcc = t1 + 0.5*tof; end   % Fehse Fig.6.4 (p.182)의 중간 보정 예시를 참고한 기본값

validateattributes(t1,{'numeric'},{'scalar','real','finite','nonnegative'});
for value=[tof,tf,dt,mu]
    validateattributes(value,{'numeric'},{'scalar','real','finite','positive'});
end
method = validatestring(method,{'CW','LAM'});
mode   = validatestring(mode,{'finite','impulse'});
for state={rt0,vt0,rc0,vc0}
    validateattributes(state{1},{'numeric'},{'real','finite','numel',3});
end
if ~ismember(pert,{'none','J2','J2~J5'})
    error('Maneuver:Pert','pert must be none, J2, or J2~J5.');
end
rt0=rt0(:); vt0=vt0(:); rc0=rc0(:); vc0=vc0(:);
if norm(rt0)==0 || norm(rc0)==0 || norm(cross(rt0,vt0))==0
    error('Maneuver:State','Nonzero radius and target angular momentum required.');
end
ecc = cross(vt0,cross(rt0,vt0))/mu - rt0/norm(rt0);
if norm(ecc)>1e-8
    error('Maneuver:CircularTarget','CW comparison requires a circular target orbit.');
end
t2 = t1 + tof;
if tf < t2
    error('Maneuver:Schedule','종료 시간(%.6g s)이 2차 기동 시각(%.6g s)보다 빠릅니다.',tf,t2);
end
offset = [1;0;0];

%% 1. 모드 분기
switch mode
    case 'finite'
        [best_idx,sel_rho,sel_rc,sel_vc,min_err,DV1,DV2,info] = ...
            run_finite(rt0,vt0,rc0,vc0,t1,tof,tf,a_max,dt,mu,method,sel_idx,pert,offset,t_mcc);
    case 'impulse'
        [best_idx,sel_rho,sel_rc,sel_vc,min_err,DV1,DV2,info] = ...
            run_impulse(rt0,vt0,rc0,vc0,t1,tof,tf,dt,mu,method,pert,t_mcc,offset);
end
info.mode = mode;
end


%% ======================= 유한 연소 (N회 기동) =======================
function [best_idx,sel_rho,sel_rc,sel_vc,min_err,DV1,DV2,info] = ...
         run_finite(rt0,vt0,rc0,vc0,t1,tof,tf,a_max,dt,mu,method,sel_idx,pert,offset,t_mcc)

validateattributes(a_max,{'numeric'},{'scalar','real','finite','positive'});
if ~isempty(sel_idx)
    validateattributes(sel_idx,{'numeric'},{'scalar','integer','>=',1,'<=',9});
end
t2 = t1 + tof;
t_mcc = sort(t_mcc(:)).';
if any(t_mcc<=t1 | t_mcc>=t1+0.95*tof)
    error('Maneuver:MCC','중간 기동 시각은 %g < t < %g s 여야 합니다.', t1, t1+0.95*tof);
end
if numel(unique(t_mcc)) < numel(t_mcc)
    error('Maneuver:MCC','중간 기동 시각이 중복됩니다.');
end

t = unique([0:dt:tf, tf, t1, t_mcc, t2]);
N = numel(t);
t_burn = [t1, t_mcc, t2];
num_burns = numel(t_burn);
i_burn = arrayfun(@(x) find(t==x,1), t_burn);

[rt,vt] = propagate(t,rt0,vt0,mu,pert);

% 1. N회 임펄스 기준 궤적(ri, vi) 계산 
rc_imp = zeros(3,N); vc_imp = rc_imp; rc_imp(:,1)=rc0; vc_imp(:,1)=vc0;
dv = zeros(3, num_burns);
goal_r = []; goal_v = []; transfer_angle = NaN;

for k = 1:N
    if k == i_burn(1)
        rem_t = t2 - t(k);
        [d,~,goal_r,goal_v,transfer_angle] = plan(rt(:,k),vt(:,k),rc_imp(:,k),vc_imp(:,k),rem_t,dt,mu,method,offset);
        vc_imp(:,k) = vc_imp(:,k) + d;
        dv(:,1) = d;
    elseif any(k == i_burn(2:end-1))
        idx_b = find(i_burn == k);
        rem_t = t2 - t(k);
        [d,~,goal_r,goal_v] = plan(rt(:,k),vt(:,k),rc_imp(:,k),vc_imp(:,k),rem_t,dt,mu,method,offset);
        vc_imp(:,k) = vc_imp(:,k) + d;
        dv(:,idx_b) = d;
    elseif k == i_burn(end)
        C2 = make_lvlh_frame(rt(:,k),vt(:,k));
        om = cross(rt(:,k),vt(:,k))/norm(rt(:,k))^2;
        d  = (vt(:,k) + cross(om,C2*offset)) - vc_imp(:,k);
        vc_imp(:,k) = vc_imp(:,k) + d;
        dv(:,end) = d;
    end
    if k < N
        [rc_imp(:,k+1),vc_imp(:,k+1)] = rk4_orbit_step(rc_imp(:,k),vc_imp(:,k),t(k+1)-t(k),mu,[],pert);
    end
end
ri = rc_imp; vi = vc_imp;

% 2. 연소 시간 및 9개 시나리오 시간 창 계산 (마지막 기동만 '후자' 적용)
dur = vecnorm(dv) / a_max;
if t1 < dur(1)
    error('Maneuver:Schedule','1차 연소 시간 %.3f s가 점화 시각 %.6g s보다 깁니다.',dur(1),t1);
end
if tf < t2 + dur(end)
    error('Maneuver:Schedule','종료 시간이 마지막 연소 종료(최대 %.6g s)보다 빠릅니다.',t2+dur(end));
end
for j = 1:num_burns-1
    if t_burn(j) + dur(j) > t_burn(j+1) - dur(j+1)
        error('Maneuver:Schedule','연소 시간이 너무 길어 기동간 간격이 부족합니다. 최대 가속도를 높이세요.');
    end
end

shifts = @(d) [-d, 0; -d/2, d/2; 0, d];
names = {'Pre','Centered','Post'};
windows = zeros(9, num_burns, 2);
labels = cell(9,1);
for a=1:3
    for b=1:3
        s = (a-1)*3 + b;
        labels{s} = [names{a},' / ',names{b}];
        for j = 1:num_burns
            if j < num_burns % 초기 및 중간 기동은 '전자(a)' 시나리오 적용
                w = shifts(dur(j));
                windows(s, j, :) = t_burn(j) + w(a, :);
            else             % 마지막 기동은 '후자(b)' 시나리오 적용
                w = shifts(dur(j));
                windows(s, j, :) = t_burn(j) + w(b, :);
            end
        end
    end
end

% 3. 연소 경계를 모두 포함한 격자로 재전파 및 이상적인 실시간 기준 궤도 생성
t = unique([t, reshape(windows,1,[])]);
N = numel(t);
i_burn = arrayfun(@(x) find(t==x,1), t_burn);

[rt,vt] = propagate(t,rt0,vt0,mu,pert);

rc_imp = zeros(3,N); vc_imp = rc_imp; rc_imp(:,1)=rc0; vc_imp(:,1)=vc0;
for k = 1:N
    if any(k == i_burn)
        idx_b = find(i_burn == k);
        vc_imp(:,k) = vc_imp(:,k) + dv(:,idx_b);
    end
    if k < N
        [rc_imp(:,k+1),vc_imp(:,k+1)] = rk4_orbit_step(rc_imp(:,k),vc_imp(:,k),t(k+1)-t(k),mu,[],pert);
    end
end
ri = rc_imp; vi = vc_imp;

r_ideal = zeros(3, N, num_burns);
v_ideal = zeros(3, N, num_burns);
for j = 1:num_burns
    idx = i_burn(j);
    r_ideal(:, idx, j) = ri(:, idx);
    v_ideal(:, idx, j) = vi(:, idx); % 이미 해당 기동의 DV가 반영된 이상적 상태
    for k = idx-1:-1:1
        [r_ideal(:,k,j), v_ideal(:,k,j)] = rk4_orbit_step(r_ideal(:,k+1,j), v_ideal(:,k+1,j), t(k)-t(k+1), mu, [], pert);
    end
    for k = idx:N-1
        [r_ideal(:,k+1,j), v_ideal(:,k+1,j)] = rk4_orbit_step(r_ideal(:,k,j), v_ideal(:,k,j), t(k+1)-t(k), mu, [], pert);
    end
end

% 4. 유한 연소 N회 시뮬레이션 및 실시간 동적 Cut-off 적용
errors = zeros(9,1); goal_errors = zeros(9,1);
durations = zeros(9, num_burns); all_trajs = cell(9,1);
actual_windows = windows;
u_dv = zeros(3, num_burns);
for j = 1:num_burns, u_dv(:,j) = unit(dv(:,j)); end

for s = 1:9
    rc = zeros(3,N); vc = rc; rho = rc; vrel = rc;
    rc(:,1)=rc0; vc(:,1)=vc0;
    
    b_active = false(num_burns, 1);
    b_done   = false(num_burns, 1);

    for k = 1:N
        [rho(:,k),vrel(:,k)] = eci_to_lvlh(rt(:,k),vt(:,k),rc(:,k),vc(:,k));
        if k == N, break; end
        h = t(k+1)-t(k); acc = zeros(3,1);

        for j = 1:num_burns
            if t(k) >= windows(s,j,1) && ~b_done(j) && ~b_active(j)
                b_active(j) = true;
            end
        end

        for j = 1:num_burns
            if b_active(j)
                dp = dot(v_ideal(:,k,j) - vc(:,k), u_dv(:,j));
                if dp <= 0
                    b_active(j) = false; b_done(j) = true;
                    actual_windows(s,j,2) = t(k);
                elseif dp <= a_max * h
                    frac = dp / a_max;
                    acc = a_max * u_dv(:,j) * (frac / h);
                    durations(s,j) = durations(s,j) + frac;
                    b_active(j) = false; b_done(j) = true;
                    actual_windows(s,j,2) = t(k) + frac;
                else
                    acc = a_max * u_dv(:,j);
                    durations(s,j) = durations(s,j) + h;
                end
            end
        end
        
        [rc(:,k+1),vc(:,k+1)] = rk4_orbit_step(rc(:,k),vc(:,k),h,mu,acc,pert);
    end
    errors(s)      = norm(rc(:,end)-ri(:,end));
    goal_errors(s) = norm(rho(:,end)-offset);
    all_trajs{s}   = struct('rho',rho,'rc',rc,'vc',vc,'vrel',vrel);
end

[min_err,best_idx] = min(errors);
[min_goal_err,min_goal_idx] = min(goal_errors);
if isempty(sel_idx), sel_idx = best_idx; end
sel = all_trajs{sel_idx};
sel_rho = sel.rho; sel_rc = sel.rc; sel_vc = sel.vc;

% Flatten actual_windows for UI representation
sel_win_flat = reshape(actual_windows(sel_idx, :, :), num_burns, 2);
sel_win_flat = reshape(sel_win_flat', 1, []); 

info = struct('t',t,'vrel',sel.vrel,'impulse_rc',ri,'impulse_vc',vi,...
    'target_r',rt,'target_v',vt,'goal_r',goal_r,'goal_v',goal_v,'offset',offset,...
    'windows',actual_windows,'labels',{labels},'errors',errors,'goal_errors',goal_errors,...
    'burn_durations',durations,'dur',dur,'dv1',dv(:,1),'dv2',dv(:,end),...
    'best_idx',best_idx,'best_window',reshape(reshape(actual_windows(best_idx,:,:),num_burns,2)',1,[]),'best_name',labels{best_idx},...
    'min_goal_idx',min_goal_idx,'min_goal_err',min_goal_err,...
    'sel_idx',sel_idx,'sel_window',sel_win_flat,'sel_name',labels{sel_idx},...
    'transfer_angle',transfer_angle,'pert',pert,'idx_burn',i_burn, ...
    'n_burns', num_burns, 't_mcc', t_mcc);

fprintf('\n[%s, pert=%s] a_max=%.6g km/s^2, 유한연소 %d회 기동, 전이각 %.2f deg\n', ...
        method,pert,a_max,num_burns,transfer_angle);
fprintf('  %-2s %-20s %16s %16s\n','#','scenario','유한연소 오차 [km]','목표점 오차 [km]');
tags = {' (유한연소오차 최소)',' (목표점오차 최소)',' (선택)'};
for s=1:9
    mark = [tags{[s==best_idx, s==min_goal_idx, s==sel_idx]}];
    fprintf('  %d  %-20s %16.6f %16.6f%s\n',s,labels{s},errors(s),goal_errors(s),mark);
end

DV1 = dv(:,1);
DV2 = dv(:,end);
end


%% ==================== 임펄스 기동 (중간 = 경로 보정) ====================
function [best_idx,sel_rho,sel_rc,sel_vc,min_err,DV1,DV2,info] = ...
         run_impulse(rt0,vt0,rc0,vc0,t1,tof,tf,dt,mu,method,pert,t_mcc,offset)

t2 = t1 + tof;
t_mcc = sort(t_mcc(:)).';
% 1차 기동 이후, 도착 전 잔여 전이각이 남아 있는 구간에서만 보정이 성립한다.
if any(t_mcc<=t1 | t_mcc>=t1+0.95*tof)
    error('Maneuver:MCC','중간 기동 시각은 %g < t < %g s 여야 합니다.', t1, t1+0.95*tof);
end
% 같은 시각은 아래 unique 격자에서 하나로 합쳐져 임펄스 수가 줄어든다.
if numel(unique(t_mcc)) < numel(t_mcc)
    error('Maneuver:MCC','중간 기동 시각이 중복됩니다.');
end

t = unique([0:dt:tf, tf, t1, t_mcc, t2]);
N = numel(t);
i1 = find(t==t1,1); i2 = find(t==t2,1);
if isempty(t_mcc), imcc = [];
else,              imcc = arrayfun(@(x) find(t==x,1), t_mcc);
end

[rt,vt] = propagate(t,rt0,vt0,mu,pert);

rc = zeros(3,N); vc = rc; rc(:,1)=rc0; vc(:,1)=vc0;
dv = zeros(3,0); dv_label = {}; dv_time = [];
transfer_angle = NaN; goal_r = []; goal_v = [];
for k = 1:N
    if k == i1
        [d,~,goal_r,goal_v,transfer_angle] = ...
            plan(rt(:,k),vt(:,k),rc(:,k),vc(:,k),tof,dt,mu,method,offset);
        vc(:,k) = vc(:,k) + d;
        dv(:,end+1)=d; dv_label{end+1}='투입'; dv_time(end+1)=t(k);
    elseif any(k == imcc)
        rem_t = t2 - t(k);
        [d,~,goal_r,goal_v] = ...
            plan(rt(:,k),vt(:,k),rc(:,k),vc(:,k),rem_t,dt,mu,method,offset);
        vc(:,k) = vc(:,k) + d;
        dv(:,end+1)=d;
        dv_label{end+1}=sprintf('중간기동%d@%gs',numel(dv_label),t(k));
        dv_time(end+1)=t(k);
    elseif k == i2
        C2 = make_lvlh_frame(rt(:,k),vt(:,k));
        om = cross(rt(:,k),vt(:,k))/norm(rt(:,k))^2;
        d  = (vt(:,k) + cross(om,C2*offset)) - vc(:,k);
        vc(:,k) = vc(:,k) + d;
        dv(:,end+1)=d; dv_label{end+1}='도착'; dv_time(end+1)=t(k);
    end
    if k < N
        [rc(:,k+1),vc(:,k+1)] = rk4_orbit_step(rc(:,k),vc(:,k),t(k+1)-t(k),mu,[],pert);
    end
end

rho = zeros(3,N); vrel = rho;
for k = 1:N
    [rho(:,k),vrel(:,k)] = eci_to_lvlh(rt(:,k),vt(:,k),rc(:,k),vc(:,k));
end

dv_mag  = vecnorm(dv);
min_err = norm(rho(:,i2) - offset);        % 도착 시각(t2) 기준 오차
goal_err_tf = norm(rho(:,end) - offset);   % 시뮬 종료(tf) 기준. 화면 라벨과 같은 기준.
best_idx = 1;
sel_rho = rho; sel_rc = rc; sel_vc = vc;
DV1 = dv(:,1); DV2 = dv(:,end);
if numel(dv_mag)>2, dv_mcc = sum(dv_mag(2:end-1)); else, dv_mcc = 0; end

info = struct('t',t,'vrel',vrel,'target_r',rt,'target_v',vt, ...
    'goal_r',goal_r,'goal_v',goal_v,'offset',offset, ...
    'transfer_angle',transfer_angle,'pert',pert, ...
    'dv',dv,'dv_label',{dv_label},'dv_time',dv_time,'dv_mag',dv_mag, ...
    'dv_total',sum(dv_mag),'dv_mcc',dv_mcc,'n_impulse',numel(dv_mag), ...
    't_mcc',t_mcc,'idx_burn',[i1 imcc i2], ...
    'goal_err',min_err,'goal_err_tf',goal_err_tf);

fprintf('\n[%s, pert=%s] 임펄스 %d회, 전이각 %.2f deg\n', ...
        method,pert,info.n_impulse,transfer_angle);
for j = 1:numel(dv_mag)
    fprintf('  %-12s t=%8.1f s   |dV| = %10.4f mm/s\n', ...
            dv_label{j}, dv_time(j), dv_mag(j)*1e6);
end
fprintf('  총 dV = %.6f km/s,   목표점 오차: 도착(t2) %.6f km / 종료(tf) %.6f km\n', ...
        info.dv_total, min_err, goal_err_tf);
end


%% ========================= 공용 =========================
function [DV1,DV2,goal_r,goal_v,transfer_angle] = ...
         plan(rt,vt,rc,vc,tof,dt,mu,method,offset)
tg = unique([0:dt:tof, tof]);
[rp,vp] = propagate(tg,rt,vt,mu,'none');
rt2 = rp(:,end); vt2 = vp(:,end);
C2 = make_lvlh_frame(rt2,vt2);
goal_r = rt2 + C2*offset;
omega  = cross(rt2,vt2)/norm(rt2)^2;
goal_v = vt2 + cross(omega,C2*offset);

h1 = cross(rc,vc);
transfer_angle = atan2d(dot(cross(rc,goal_r),h1/norm(h1)),dot(rc,goal_r));
transfer_angle = mod(transfer_angle,360);

if strcmp(method,'CW')
    [rho,vrel,C1] = eci_to_lvlh(rt,vt,rc,vc);
    n = sqrt(mu/norm(rt)^3);
    nt = n*tof; c = cos(nt); sn = sin(nt);
    rv = [(4*sn-3*nt)/n,0,-2*(1-c)/n; 0,sn/n,0; 2*(1-c)/n,0,sn/n];
    if rcond(rv)<1e-10
        error('Maneuver:SingularCW','CW transfer time produces a singular targeting matrix.');
    end
    [d1,d2] = cw_stm_delta_v(rho-offset,vrel,n,tof);
    DV1 = C1*d1; DV2 = C2*d2;
else
    tm = 1;
    if transfer_angle>180, tm = -1; end
    [DV1,DV2,alarm] = LambertSolver(rc,vc,goal_r,goal_v,tof,mu,tm);
    if alarm~=0
        error('Maneuver:Lambert','Lambert solver failed (code %d, angle %.2f deg).',alarm,transfer_angle);
    end
    DV1 = DV1(:); DV2 = DV2(:);
end
end

function [r,v] = propagate(t,r0,v0,mu,pert)
r = zeros(3,numel(t)); v = r; r(:,1)=r0; v(:,1)=v0;
for k = 1:numel(t)-1
    [r(:,k+1),v(:,k+1)] = rk4_orbit_step(r(:,k),v(:,k),t(k+1)-t(k),mu,[],pert);
end
end

function u = unit(x)
if norm(x)>0, u = x/norm(x); else, u = zeros(3,1); end
end