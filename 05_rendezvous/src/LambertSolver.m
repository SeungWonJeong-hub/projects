function [DELTA_V1, DELTA_V2, ALARM] = LambertSolver(R_CHASER_T1, V_CHASER_T1, R_TARGET_T2, V_TARGET_T2, DEL_T_TRAN, EARTH_MU, TM)
% LambertSolver: 출발 및 도착 델타 V를 모두 계산하는 랑데부 솔버
% [입력 파라미터]
% R_CHASER_T1 : 기동 시작 시점(t1)의 추적 위성 현재 위치 (3x1) [km]
% V_CHASER_T1 : 기동 시작 시점(t1)의 추적 위성 현재 속도 (3x1) [km/s]
% R_TARGET_T2 : 타겟 도달 시점(t2)의 타겟 위성 예상 위치 (3x1) [km]
% V_TARGET_T2 : 타겟 도달 시점(t2)의 타겟 위성 예상 속도 (3x1) [km/s]
% DEL_T_TRAN  : 비행 소요 시간 (TOF) [초]
% EARTH_MU    : 중심 천체의 중력 상수 [km^3/s^2]
% TM          : 비행 방향 (1: Short-way / -1: Long-way)
%
% [출력 파라미터]
% DELTA_V1    : 출발 시 엔진을 켜야 할 속도 증분 벡터 (3x1) [km/s]
% DELTA_V2    : 도착 시 타겟 궤도에 안착하기 위한 속도 증분 벡터 (3x1) [km/s]
% ALARM       : 상태 코드 (0: 정상, 1: 시간오류, 2: 각도오류, 5: 수렴실패)

    % 알고리즘 제어 상수 설정
    EPS_TRANSFER = 1e-6; 
    DU = 1e-5;           
    NMAX = 50;           
    EPS_U = 1e-6;        

    % a. 알람 플래그 및 출력 변수 초기화
    ALARM = 0;
    DELTA_V1 = zeros(3,1);
    DELTA_V2 = zeros(3,1);

    % b. 초기 벡터 및 최종 벡터의 크기 계산
    R0_MAG = norm(R_CHASER_T1);
    R1_MAG = norm(R_TARGET_T2);

    % c. 전이 삼각형의 반둘레 (정규화 상수) 계산
    R_PARABOLA = (R0_MAG + R1_MAG + norm(R_TARGET_T2 - R_CHASER_T1)) / 2;

    % d. 포물선 속도 (정규화 인자) 계산
    V_PARABOLA = sqrt(2 * EARTH_MU / R_PARABOLA);

    % e. Z 파라미터 계산 및 각도 검사
    Z = R0_MAG * R1_MAG - dot(R_CHASER_T1, R_TARGET_T2);
    if Z <= EPS_TRANSFER * R0_MAG * R1_MAG
        ALARM = 2; 
        return;
    end

    % 궤도 평면의 법선 벡터 (UN) 계산
    UN = cross(R_CHASER_T1, R_TARGET_T2);
    UN_MAG = norm(UN);
    if UN_MAG < 1e-8
        ALARM = 2; 
        return;
    end
    UN = TM * (UN / UN_MAG);

    % f. 쎄타/2의 코탄젠트 계산 
    Z = dot(cross(R_CHASER_T1, R_TARGET_T2), UN) / Z;

    % g. 파라미터 VH 계산
    VH = sqrt(R0_MAG * R1_MAG / (1 + Z^2));

    % h. 반복 계산 카운터 초기화
    N = 0;

    % i. 독립 변수 U의 상한 및 하한 설정
    U_MAX = 1 - DU / 2;
    U_MIN = -1.0;

    % j. 상수 파라미터 LAMBDA 계산
    LAMBDA = (VH / R_PARABOLA) * Z;

    % k. 독립 변수 U의 첫 번째 추정값
    U = LAMBDA / sqrt(1 + LAMBDA^2);

    % l. 정규화된 비행 소요 시간 계산
    T_TILDA_DESIRED = (V_PARABOLA / R_PARABOLA) * DEL_T_TRAN;

    % m. 최소 비행 시간 검사
    T_MIN = (2/3) * (1 - LAMBDA^3) + (0.4) * (1 - LAMBDA^5) * DU;
    if T_TILDA_DESIRED <= T_MIN
        ALARM = 1; 
        return;
    end

    % n. 뉴턴-랩슨 반복 계산 루프 시작
    while true
        N = N + 1;
        if N == NMAX
            ALARM = 5; 
            return;
        end

        W = sqrt(1 - U^2);
        X = LAMBDA * W;
        Y = sqrt(abs(1 - X^2)); 
        F = W * Y - U * X;
        G = U * Y + W * X;

        theta_angle = atan2(F, G);
        if theta_angle < 0
            theta_angle = theta_angle + 2 * pi; 
        end
        T_TILDA = (theta_angle - (U * W - X * Y)) / W^3;

        S_TILDA = (3 * U * T_TILDA - 2 * (1 - (U / Y) * LAMBDA^3)) / W^2;

        T_TILDA_ERROR = T_TILDA_DESIRED - T_TILDA;

        if abs(T_TILDA_ERROR) < (T_TILDA_DESIRED * EPS_U)
            break; 
        end

        U_STEP = T_TILDA_ERROR / S_TILDA;
        if U_STEP > 0
            U_MIN = U;
            U_NEW = U + U_STEP;
            if U_NEW > U_MAX
                U = (U_MIN + U_MAX) / 2; 
            else
                U = U_NEW;
            end
        else
            U_MAX = U;
            U_NEW = U + U_STEP;
            if U_NEW < U_MIN
                U = (U_MIN + U_MAX) / 2; 
            else
                U = U_NEW;
            end
        end
    end % while 루프 종료

    % o. 요구되는 전이 궤도 초기 속도 벡터 (VS_REQUIRED_T1) 산출
    VH_FINAL = VH / R0_MAG;
    VR_FINAL = (R_PARABOLA / R0_MAG) * LAMBDA - G;
    COEF = V_PARABOLA / (Y - LAMBDA * U);

    VS_REQUIRED_T1 = (COEF / R0_MAG) * (VR_FINAL * R_CHASER_T1 + VH_FINAL * cross(UN, R_CHASER_T1));

    % ==========================================
    % 최종 델타 V 계산 로직 (출발 & 도착)
    % ==========================================
    
    % 1. 출발 델타 V (Delta V1)
    DELTA_V1 = VS_REQUIRED_T1 - V_CHASER_T1;

    % 2. 도착 시점의 전이 궤도 속도 벡터 (VS_REQUIRED_T2) 산출
    % 기존 해석적 방법 대신 RK4 수치적분을 통해 비행 소요 시간(DEL_T_TRAN) 동안 전파합니다.
    dt_step = 1; % 1초 단위로 적분
    time_elapsed = 0;
    
    r_trans = R_CHASER_T1;
    v_trans = VS_REQUIRED_T1;
    
    while time_elapsed < DEL_T_TRAN
        step = min(dt_step, DEL_T_TRAN - time_elapsed);
        [r_trans, v_trans] = rk4_orbit_step(r_trans, v_trans, step, EARTH_MU);
        time_elapsed = time_elapsed + step;
    end
    
    VS_REQUIRED_T2 = v_trans;

    % 3. 도착 델타 V (Delta V2)
    % 타겟의 속도에 내 위성의 속도를 맞춰 궤도 안착
    DELTA_V2 = V_TARGET_T2 - VS_REQUIRED_T2;
end