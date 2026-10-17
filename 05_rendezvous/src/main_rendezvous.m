function main_rendezvous
    % 1. UI 및 레이아웃 설정
    fig = uifigure('Name', '우주 랑데부: CW vs Lambert 듀얼 시뮬레이션', 'Position', [20 75 1500 750]);
    if isprop(fig, 'Theme'), fig.Theme = 'light'; end   % MATLAB 다크 모드여도 흰 화면으로 표시 (R2025a+)

    % 좌측 패널 (Mission Control Panel)
    pnlCtrl = uipanel(fig, 'Title', '임무 제어판', 'Position', [20 20 310 710]);

    % 1. 기동 모델 (라디오)
    bgMode = uibuttongroup(pnlCtrl, 'Title', '1. 기동 모델', 'Position', [8 610 294 74], ...
        'SelectionChangedFcn', @onModeChange);
    uiradiobutton(bgMode, 'Text', '유한연소 기동 (9 시나리오)', 'Position', [10 29 270 22]);
    rbImpulse = uiradiobutton(bgMode, 'Text', '임펄스 기동', 'Position', [10 6 270 22]);

    uilabel(pnlCtrl, 'Text', '2. 타겟 [a(km), e, i, Ω, ω, ν(°)]:', 'Position', [10 580 280 20], 'FontWeight', 'bold');
    efTargetCoe = uieditfield(pnlCtrl, 'text', 'Value', '[6878, 0, 45, 0, 0, 0]', 'Position', [10 555 280 25]);

    uilabel(pnlCtrl, 'Text', '3. 체이서 [a(km), e, i, Ω, ω, ν(°)]:', 'Position', [10 520 280 20], 'FontWeight', 'bold');
    efChaserCoe = uieditfield(pnlCtrl, 'text', 'Value', '[6858, 0, 45, 0, 0, -0.1]', 'Position', [10 495 280 25]);

    uilabel(pnlCtrl, 'Text', '4. 1차 임펄스 기준 시각 [s]:', 'Position', [10 460 280 20], 'FontWeight', 'bold');
    efBurn1 = uieditfield(pnlCtrl, 'numeric', 'Value', 500, 'Position', [10 435 280 25]);

    uilabel(pnlCtrl, 'Text', '5. 목표 도달 소요 시간 (TOF) [s]:', 'Position', [10 400 280 20], 'FontWeight', 'bold');
    efTOF = uieditfield(pnlCtrl, 'numeric', 'Value', 2700, 'Position', [10 375 280 25]);

    uilabel(pnlCtrl, 'Text', '6. 전체 시뮬레이션 종료 시간 [s]:', 'Position', [10 340 280 20], 'FontWeight', 'bold');
    efTotalTime = uieditfield(pnlCtrl, 'numeric', 'Value', 6000, 'Position', [10 315 280 25]);

    uilabel(pnlCtrl, 'Text', '7. 섭동 모델:', 'Position', [10 280 280 20], 'FontWeight', 'bold');
    ddPert = uidropdown(pnlCtrl, 'Items', {'없음 (2체)', 'J2', 'J2~J5'}, 'ItemsData', {'none', 'J2', 'J2~J5'}, 'Value', 'none', 'Position', [10 255 280 25]);

    % 공용 패널 1: 기동 횟수 및 중간 기동 (두 모드 모두 사용)
    pnlImp = uipanel(pnlCtrl, 'Title', '기동 횟수 및 중간 기동 시각', 'Position', [8 165 294 85]);
    uilabel(pnlImp, 'Text', '기동 횟수:', 'Position', [10 38 68 20], 'FontWeight', 'bold');
    efImpCount = uieditfield(pnlImp, 'numeric', 'Value', 3, 'Limits', [2 Inf], ...
                             'Position', [78 36 60 25], 'ValueChangedFcn', @onImpOpt);
    uilabel(pnlImp, 'Text', 'MCC [s]:', 'Position', [10 12 68 20], 'FontWeight', 'bold');
    efMCC = uieditfield(pnlImp, 'text', 'Value', '1850', 'Position', [78 10 200 25]);

    % 공용 패널 2: 유한연소 전용 설정 (임펄스일 때 숨김 처리됨)
    scenario_items = {'[1] Pre-burn / Pre-burn', '[2] Pre-burn / Centered-burn', '[3] Pre-burn / Post-burn', ...
                      '[4] Centered-burn / Pre-burn', '[5] Centered-burn / Centered-burn (기본)', '[6] Centered-burn / Post-burn', ...
                      '[7] Post-burn / Pre-burn', '[8] Post-burn / Centered-burn', '[9] Post-burn / Post-burn'};
    pnlFin = uipanel(pnlCtrl, 'Title', '유한연소 설정', 'Position', [8 75 294 85]);
    uilabel(pnlFin, 'Text', '최대 가속도 [km/s²]:', 'Position', [10 38 150 20], 'FontWeight', 'bold');
    efAccel = uieditfield(pnlFin, 'numeric', 'Value', 0.001, 'Position', [160 36 120 25]);
    ddScenario = uidropdown(pnlFin, 'Items', scenario_items, 'Value', scenario_items{5}, 'Position', [10 8 270 25]);

    btnRun = uibutton(pnlCtrl, 'push', 'Text', '▶ 유한연소 3회 기동 실행', ...
        'Position', [10 10 280 60], 'BackgroundColor', [0.1 0.4 0.8], 'FontColor', 'w', ...
        'FontSize', 15, 'FontWeight', 'bold', 'ButtonPushedFcn', @runSim);

    % 상단 텔레메트리 패널 
    pnlInfo = uipanel(fig, 'Title', '듀얼 임무 텔레메트리', 'Position', [340 480 1140 250]);
    cB = [0 0.45 0.9]; cR = [0.9 0.25 0.25]; cP = [0.7 0.3 0.8];
    lblTime    = uilabel(pnlInfo, 'Text', '비행 시간: 0 / - 초', 'Position', [10 195 300 22], 'FontSize', 12, 'FontWeight', 'bold');
    lblStatus  = uilabel(pnlInfo, 'Text', '상태: 대기 중...', 'Position', [10 175 310 22], 'FontSize', 12);
    lblPeriod  = uilabel(pnlInfo, 'Text', '궤도주기: -', 'Position', [10 155 310 20], 'FontSize', 11, 'FontColor', cP);
    lblInitial = uilabel(pnlInfo, 'Text', '초기 상대위치: -', 'Position', [10 135 310 20], 'FontSize', 11, 'FontColor', cP);
    
    % 좌측 하단 유동 텍스트들
    lblDur_CW  = uilabel(pnlInfo, 'Text', 'CW 연소 시간: -', 'Position', [10 115 310 20], 'FontSize', 11, 'FontColor', cB);
    lblDur_LAM = uilabel(pnlInfo, 'Text', 'LAM 연소 시간: -', 'Position', [10 95 310 20], 'FontSize', 11, 'FontColor', cR);
    lblPosStart_CW  = uilabel(pnlInfo, 'Text', '▶ CW 점화 시작: -', 'Position', [10 75 310 20], 'FontSize', 11, 'FontColor', cB);
    lblPosEnd_CW    = uilabel(pnlInfo, 'Text', '▶ CW 연소 종료: -', 'Position', [10 55 310 20], 'FontSize', 11, 'FontColor', cB);
    lblPosStart_LAM = uilabel(pnlInfo, 'Text', '▶ LAM 점화 시작: -', 'Position', [10 35 310 20], 'FontSize', 11, 'FontColor', cR);
    lblPosEnd_LAM   = uilabel(pnlInfo, 'Text', '▶ LAM 연소 종료: -', 'Position', [10 15 310 20], 'FontSize', 11, 'FontColor', cR);
    lblSummary      = uilabel(pnlInfo, 'Text', '', 'Position', [350 25 700 20], 'FontSize', 11, 'FontColor', cP, 'FontWeight', 'bold');

    % 중앙 열 (CW) / 우측 열 (LAM) 배치
    colCW  = makeColumn(pnlInfo, 350, 'CW', cB);
    colLAM = makeColumn(pnlInfo, 750, 'LAM', cR);

    % 하단 그래프 영역
    ax3D = uiaxes(fig, 'Position', [340 20 450 450]);
    axLVLH = uiaxes(fig, 'Position', [830 20 450 450]);

    mu = 398600.4418;
    efTargetCoe.ValueChangedFcn = @updatePeriod;
    efChaserCoe.ValueChangedFcn = @updatePeriod;
    efTOF.ValueChangedFcn       = @onTimeChange;
    efBurn1.ValueChangedFcn     = @onTimeChange;
    lastAutoMCC = '';   
    updatePeriod();
    onImpOpt();

    function n = impCount()
        n = round(efImpCount.Value);
        efImpCount.Value = n;
    end

    function onImpOpt(~, ~)
        n = impCount();
        efMCC.Enable = onoff(n > 2);
        switch n
            case 2
                efMCC.Value = '';                 
            case 3
                if ~staleMCC(1), efMCC.Value = defaultMCC(0.5); end
            case 4
                if ~staleMCC(2), efMCC.Value = defaultMCC([0.5 0.75]); end
            otherwise
                nMCC = n - 2;
                if ~staleMCC(nMCC)
                    fr = (1:nMCC)/(nMCC+1);
                    if fr(end) >= 0.95, fr = linspace(1/(nMCC+1), 18/19, nMCC); end
                    efMCC.Value = defaultMCC(fr);
                end
        end
        isImp = rbImpulse.Value;
        if isImp
            btnRun.Text = sprintf('▶ 임펄스 %d회 기동 실행', n);
        else
            btnRun.Text = sprintf('▶ 유한연소 %d회 기동 실행', n);
        end
    end

    function onTimeChange(~, ~)
        updatePeriod();
        if strcmp(strtrim(efMCC.Value), strtrim(lastAutoMCC))
            efMCC.Value = '';
            onImpOpt();
        end
    end

    function ok = staleMCC(nWant)
        f  = safeMCC();
        t1 = efBurn1.Value;
        ok = numel(f) == nWant && all(f > t1 & f < t1 + 0.95*efTOF.Value);
    end

    function str = defaultMCC(fracs)
        str = strjoin(compose('%g', efBurn1.Value + efTOF.Value*fracs), ' ');
        lastAutoMCC = str;
    end

    function f = safeMCC()
        try, f = parseMCC(efMCC.Value); catch, f = []; end
    end

    function updatePeriod(~, ~)
        try
            ct = parseCoe(efTargetCoe.Value); cc = parseCoe(efChaserCoe.Value);
            Tt = 2*pi*sqrt(ct(1)^3/mu); Tc = 2*pi*sqrt(cc(1)^3/mu);
            lblPeriod.Text = sprintf('주기: 타겟 %.1f / 체이서 %.1f s  (TOF %.1f%%)', Tt, Tc, 100*efTOF.Value/Tt);
        catch
            lblPeriod.Text = '궤도주기: - (궤도요소 확인)';
        end
    end

    function resetCanvas()
        cla(ax3D); cla(axLVLH); hold(ax3D, 'on'); hold(axLVLH, 'on');
        grid(ax3D, 'on'); grid(axLVLH, 'on');
        
        % Data Aspect Ratio 강제 적용 (물리적 1:1:1 스케일 보장)
        daspect(ax3D, [1 1 1]);
        daspect(axLVLH, [1 1 1]);

        xlabel(ax3D, '전진 (X) [km]'); ylabel(ax3D, '수직 (Y) [km]'); zlabel(ax3D, '반경 (Z) [km]');
        title(ax3D, '3D 상대궤적 (LVLH)'); view(ax3D, 3);
        xlabel(axLVLH, '전진 (X) [km]'); ylabel(axLVLH, '반경 (Z) [km]');
        title(axLVLH, '2D 궤도면 내 상대궤적 (LVLH)');
    end

    function setScale(s3, s2)
        ax3D.XLim = [-s3, s3]; ax3D.YLim = [-s3, s3]; ax3D.ZLim = [-s3, s3];
        axLVLH.XLim = [-s2, s2]; axLVLH.YLim = [-s2, s2];
    end

    function onModeChange(~, ~)
        isImp = rbImpulse.Value;
        pnlFin.Visible = onoff(~isImp);
        onImpOpt(); 
    end

    function runSim(~, ~)
        btnRun.Enable = 'off';
        restoreButton = onCleanup(@() enableButton(btnRun));
        try
            ct = parseCoe(efTargetCoe.Value); cc = parseCoe(efChaserCoe.Value);
            t_burn1 = efBurn1.Value; transfer_time = efTOF.Value;
            t_burn2 = t_burn1 + transfer_time; t_final = efTotalTime.Value;
            pert = ddPert.Value;

            mode = 'finite';
            if rbImpulse.Value
                mode = 'impulse';
            end
            
            nImp = impCount();
            if nImp == 2
                mcc = [];
            else
                mcc = parseMCC(efMCC.Value);
                if numel(mcc) ~= nImp - 2
                    error('Maneuver:MCC', '기동 %d회에는 중간 기동 시각 %d개가 필요합니다 (현재 %d개).', nImp, nImp-2, numel(mcc));
                end
                if numel(unique(mcc)) < numel(mcc)
                    error('Maneuver:MCC', '중간 기동 시각이 중복됩니다 (현재 %s). 서로 다른 시각을 넣으세요.', strjoin(compose('%g', mcc), ' '));
                end
                if any(mcc <= t_burn1 | mcc >= t_burn1 + 0.95*transfer_time)
                    error('Maneuver:MCC', '중간 기동 시각은 %g < t < %g s 여야 합니다 (현재 %s).', t_burn1, t_burn1 + 0.95*transfer_time, strjoin(compose('%g', mcc), ' '));
                end
            end

            if strcmp(mode, 'impulse')
                a_max = Inf;  sel_idx = 1;          
            else
                a_max = efAccel.Value;
                sel_idx = find(strcmp(ddScenario.Items, ddScenario.Value));
            end

            if t_final < t_burn2
                uialert(fig, '전체 시뮬레이션 종료 시간이 2차 도착 시간보다 짧습니다!', '입력 오류');
                return;
            end

            T_orbit = 2*pi*sqrt(ct(1)^3/mu);
            if transfer_time >= T_orbit
                uialert(fig, sprintf(['목표 도달 소요 시간(TOF) %.6g s가 타겟 궤도주기 %.6g s 이상입니다.\n' ...
                    'CW/Lambert 단일 리볼루션 계획은 TOF < 1주기에서만 유효합니다.'], transfer_time, T_orbit), '입력 오류');
                return;
            end

            resetCanvas(); deg = pi/180;
            [r_t, v_t] = coe_to_rv(ct(1), ct(2), ct(3)*deg, ct(4)*deg, ct(5)*deg, ct(6)*deg, mu);
            [r_c_0, v_c_0] = coe_to_rv(cc(1), cc(2), cc(3)*deg, cc(4)*deg, cc(5)*deg, cc(6)*deg, mu);
            dt = 1;

            [rho_start, ~, ~] = eci_to_lvlh(r_t, v_t, r_c_0, v_c_0);
            lblInitial.Text = sprintf('초기 상대위치: [%.3f, %.3f, %.3f] km', rho_start);
            p3D_Init = plot3(ax3D, rho_start(1), rho_start(2), rho_start(3), 'o', 'MarkerEdgeColor', [0.75 0 0.75], 'MarkerFaceColor', 'none', 'LineWidth', 1.8, 'MarkerSize', 9);
            plot(axLVLH, rho_start(1), rho_start(3), 'o', 'MarkerEdgeColor', [0.75 0 0.75], 'MarkerFaceColor', 'none', 'LineWidth', 1.8, 'MarkerSize', 9);
            setScale(max([abs(rho_start); 10]) * 1.5, max([abs(rho_start([1 3])); 10]) * 1.5);

            if strcmp(mode, 'impulse')
                lblStatus.Text = '상태: 백그라운드 연산 진행 중... (임펄스 기동)';
            else
                lblStatus.Text = sprintf('상태: 백그라운드 연산 진행 중... (표출 시나리오: %d번)', sel_idx);
            end
            drawnow;

            [~, cw_rho, cw_rc, cw_vc, ~, cw_DV1, cw_DV2, cw_info] = ...
                evaluate_maneuver_module(r_t, v_t, r_c_0, v_c_0, t_burn1, transfer_time, t_final, a_max, dt, mu, 'CW', sel_idx, pert, mode, mcc);
            [~, lam_rho, lam_rc, lam_vc, ~, lam_DV1, lam_DV2, lam_info] = ...
                evaluate_maneuver_module(r_t, v_t, r_c_0, v_c_0, t_burn1, transfer_time, t_final, a_max, dt, mu, 'LAM', sel_idx, pert, mode, mcc);

            t_span = union(cw_info.t, lam_info.t);
            [cw_rho, cw_rc, cw_vc] = resampleTo(cw_info.t, t_span, cw_rho, cw_rc, cw_vc);
            [lam_rho, lam_rc, lam_vc] = resampleTo(lam_info.t, t_span, lam_rho, lam_rc, lam_vc);

            span = abs([cw_rho, lam_rho, rho_start]);
            setScale(max(max(span, [], 'all', 'omitnan'), 10) * 1.15, max(max(span([1 3],:), [], 'all', 'omitnan'), 10) * 1.15);

            if strcmp(mode, 'impulse')
                cw_se_times  = cw_info.dv_time;
                lam_se_times = lam_info.dv_time;
                lblDur_CW.Text  = sprintf('CW 중간 기동 ΔV 합: %.6f km/s', cw_info.dv_mcc);
                lblDur_LAM.Text = sprintf('LAM 중간 기동 ΔV 합: %.6f km/s', lam_info.dv_mcc);
            else
                cw_se_times  = cw_info.sel_window;
                lam_se_times = lam_info.sel_window;
                lblDur_CW.Text  = sprintf('CW 총 연소시간: %.3f s (기동 %d회)', sum(cw_info.dur), cw_info.n_burns);
                lblDur_LAM.Text = sprintf('LAM 총 연소시간: %.3f s (기동 %d회)', sum(lam_info.dur), lam_info.n_burns);
            end
            
            cw_se = interp1(t_span, cw_rho', cw_se_times)';
            lam_se = interp1(t_span, lam_rho', lam_se_times)';
            leftRows = [lblDur_CW, lblDur_LAM, lblPosStart_CW, lblPosEnd_CW, lblPosStart_LAM, lblPosEnd_LAM];
            
            % 동적 텍스트 (연소/점화 정보) Y위치 재할당
            if strcmp(mode, 'impulse')
                lblPosStart_CW.Text = sprintf('▶ 기동 시각: %s s', join(string(round(cw_se_times)),' / '));
                lblPosStart_CW.FontColor = cP;
                nUsed = 3;
                if isempty(cw_info.t_mcc)
                    leftRows = [lblPosStart_CW, lblDur_CW, lblDur_LAM, lblPosEnd_CW, lblPosStart_LAM, lblPosEnd_LAM];
                    nUsed = 1;
                end
            else
                lblPosStart_CW.FontColor = cB;
                lblPosStart_CW.Text  = sprintf('▶ CW 점화 시작: 1차 [%.2f, %.2f, %.2f] km', cw_se(:,1));
                lblPosEnd_CW.Text    = sprintf('▶ CW 연소 종료: 도착 [%.2f, %.2f, %.2f] km', cw_se(:,end));
                lblPosStart_LAM.Text = sprintf('▶ LAM 점화 시작: 1차 [%.2f, %.2f, %.2f] km', lam_se(:,1));
                lblPosEnd_LAM.Text   = sprintf('▶ LAM 연소 종료: 도착 [%.2f, %.2f, %.2f] km', lam_se(:,end));
                nUsed = 6;
            end
            
            for k = 1:numel(leftRows)
                leftRows(k).Visible = onoff(k <= nUsed);
                leftRows(k).Position(2) = 115 - 20*(k-1); 
            end

            showDV(colCW, cw_DV1, cw_DV2, mode);
            showDV(colLAM, lam_DV1, lam_DV2, mode);
            showSummary(colCW, cw_info);
            showSummary(colLAM, lam_info);
            lblSummary.Text = '';

            %% 2단계: 선택 시나리오 애니메이션 재생 준비
            if strcmp(mode, 'impulse')
                lblStatus.Text = sprintf('상태: 임펄스 %d회 재생 시작', cw_info.n_impulse);
            else
                lblStatus.Text = sprintf('상태: 선택된 %d번 시나리오 재생 시작 (유한연소 %d회)', sel_idx, cw_info.n_burns);
            end

            line3D_CW = animatedline(ax3D, 'Color', 'b', 'LineWidth', 1.5);
            line3D_LAM = animatedline(ax3D, 'Color', 'r', 'LineWidth', 1.5, 'LineStyle', '--');
            line2D_CW = animatedline(axLVLH, 'Color', 'b', 'LineWidth', 1.5);
            line2D_LAM = animatedline(axLVLH, 'Color', 'r', 'LineWidth', 1.5, 'LineStyle', '--');

            p3D_T = plot3(ax3D, 0, 0, 0, 'kp', 'MarkerFaceColor', 'y', 'MarkerSize', 12);
            plot(axLVLH, 0, 0, 'kp', 'MarkerFaceColor', 'y', 'MarkerSize', 15);

            off = cw_info.offset;
            p3D_Goal = plot3(ax3D, off(1), off(2), off(3), 'kd', 'MarkerFaceColor', [0 0.8 0.8], 'MarkerSize', 9);
            plot(axLVLH, off(1), off(3), 'kd', 'MarkerFaceColor', [0 0.8 0.8], 'MarkerSize', 10);

            p3D_C_CW = plot3(ax3D, 0, 0, 0, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 6);
            p3D_C_LAM = plot3(ax3D, 0, 0, 0, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
            p2D_C_CW = plot(axLVLH, 0, 0, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 8);
            p2D_C_LAM = plot(axLVLH, 0, 0, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8);

            % 기동 지점 마커들: 1차(녹색), 중간(청록/마젠타), 도착(파랑/빨강)
            p3D_Burn1 = plot3(ax3D, NaN, NaN, NaN, 'kp', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
            p2D_Burn1 = plot(axLVLH, NaN, NaN, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
            
            p3D_MCC_CW  = plot3(ax3D, NaN, NaN, NaN, 'kp', 'MarkerSize', 10, 'MarkerFaceColor', 'c');
            p2D_MCC_CW  = plot(axLVLH, NaN, NaN, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'c');
            p3D_MCC_LAM = plot3(ax3D, NaN, NaN, NaN, 'kp', 'MarkerSize', 10, 'MarkerFaceColor', 'm');
            p2D_MCC_LAM = plot(axLVLH, NaN, NaN, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'm');

            p3D_Burn2_CW = plot3(ax3D, NaN, NaN, NaN, 'kp', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
            p2D_Burn2_CW = plot(axLVLH, NaN, NaN, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'b');
            p3D_Burn2_LAM = plot3(ax3D, NaN, NaN, NaN, 'kp', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
            p2D_Burn2_LAM = plot(axLVLH, NaN, NaN, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');

            % 연소 S/E 'x' 마커 (유한연소 전용)
            p3D_CW_SE = plot3(ax3D, NaN, NaN, NaN, 'bx', 'MarkerSize', 10, 'LineWidth', 2);
            p2D_CW_SE = plot(axLVLH, NaN, NaN, 'bx', 'MarkerSize', 10, 'LineWidth', 2);
            p3D_LAM_SE = plot3(ax3D, NaN, NaN, NaN, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
            p2D_LAM_SE = plot(axLVLH, NaN, NaN, 'rx', 'MarkerSize', 10, 'LineWidth', 2);

            % 범례(Legend)
            lgd = legend(ax3D, [p3D_T, p3D_Goal, p3D_Init, line3D_CW, line3D_LAM, p3D_Burn1, p3D_MCC_CW, p3D_Burn2_CW, p3D_CW_SE, p3D_LAM_SE], ...
                   {'타겟', '목표점', '초기(0s)', 'CW 궤적', 'LAM 궤적', '1차 투입점', 'MCC 기준점', t2Label(mode,'CW'), seLabel(mode,'CW'), seLabel(mode,'LAM')});
            lgd.Orientation = 'vertical';
            lgd.NumColumns = 1;
            lgd.FontSize = 10;
            lgd.Units = 'pixels';
            lgd.Position = [1320, 150, 160, 240]; % 화면 우측 패널 옆 여백 공간에 절대 배치

            max_frames = 600;
            stride = max(1, ceil(numel(t_span)/max_frames));
            key_t = unique([t_burn1, t_burn2, cw_se_times, lam_se_times]);
            show = ismember(t_span, key_t) | mod(1:numel(t_span), stride) == 0;
            show(1) = true; show(end) = true;

            play_seconds = 25;
            frame_dt = play_seconds / max(1, nnz(show));
            shown = 0;
            play_clock = tic;

            %% 3단계: 애니메이션 Loop
            for i = 1:length(t_span)
                if ~isvalid(fig), return; end
                t = t_span(i);

                rho_cw_curr = cw_rho(:, i); rho_lam_curr = lam_rho(:, i);
                addpoints(line3D_CW, rho_cw_curr(1), rho_cw_curr(2), rho_cw_curr(3));
                addpoints(line3D_LAM, rho_lam_curr(1), rho_lam_curr(2), rho_lam_curr(3));
                addpoints(line2D_CW, rho_cw_curr(1), rho_cw_curr(3));
                addpoints(line2D_LAM, rho_lam_curr(1), rho_lam_curr(3));

                if ~show(i), continue; end   

                if t == t_burn1
                    lblStatus.Text = '상태: 1차 임펄스 기준 시각';
                    p3D_Burn1.XData = rho_cw_curr(1); p3D_Burn1.YData = rho_cw_curr(2); p3D_Burn1.ZData = rho_cw_curr(3);
                    p2D_Burn1.XData = rho_cw_curr(1); p2D_Burn1.YData = rho_cw_curr(3);
                elseif t == t_burn2
                    if strcmp(mode, 'impulse'), lblStatus.Text = '상태: 도착 임펄스 시각';
                    else, lblStatus.Text = '상태: 2차(도착) 임펄스 기준 시각'; end
                    p3D_Burn2_CW.XData = rho_cw_curr(1); p3D_Burn2_CW.YData = rho_cw_curr(2); p3D_Burn2_CW.ZData = rho_cw_curr(3);
                    p2D_Burn2_CW.XData = rho_cw_curr(1); p2D_Burn2_CW.YData = rho_cw_curr(3);
                    p3D_Burn2_LAM.XData = rho_lam_curr(1); p3D_Burn2_LAM.YData = rho_lam_curr(2); p3D_Burn2_LAM.ZData = rho_lam_curr(3);
                    p2D_Burn2_LAM.XData = rho_lam_curr(1); p2D_Burn2_LAM.YData = rho_lam_curr(3);
                end
                
                if ~isempty(cw_info.t_mcc)
                    idx_mcc_cw = cw_info.t_mcc <= t;
                    if any(idx_mcc_cw)
                        mcc_t = cw_info.t_mcc(idx_mcc_cw);
                        mcc_p_cw = interp1(t_span, cw_rho', mcc_t)';
                        p3D_MCC_CW.XData = mcc_p_cw(1,:); p3D_MCC_CW.YData = mcc_p_cw(2,:); p3D_MCC_CW.ZData = mcc_p_cw(3,:);
                        p2D_MCC_CW.XData = mcc_p_cw(1,:); p2D_MCC_CW.YData = mcc_p_cw(3,:);
                        
                        mcc_p_lam = interp1(t_span, lam_rho', mcc_t)';
                        p3D_MCC_LAM.XData = mcc_p_lam(1,:); p3D_MCC_LAM.YData = mcc_p_lam(2,:); p3D_MCC_LAM.ZData = mcc_p_lam(3,:);
                        p2D_MCC_LAM.XData = mcc_p_lam(1,:); p2D_MCC_LAM.YData = mcc_p_lam(3,:);
                    end
                end
                
                if strcmp(mode, 'impulse') && any(t == cw_info.t_mcc)
                    k = find(t == cw_info.t_mcc, 1);
                    lblStatus.Text = sprintf('상태: 중간 기동 %d/%d (t=%g s)', k, numel(cw_info.t_mcc), t);
                end

                idx_cw = cw_se_times <= t;
                if any(idx_cw)
                    p3D_CW_SE.XData = cw_se(1,idx_cw); p3D_CW_SE.YData = cw_se(2,idx_cw); p3D_CW_SE.ZData = cw_se(3,idx_cw);
                    p2D_CW_SE.XData = cw_se(1,idx_cw); p2D_CW_SE.YData = cw_se(3,idx_cw);
                end
                idx_lam = lam_se_times <= t;
                if any(idx_lam)
                    p3D_LAM_SE.XData = lam_se(1,idx_lam); p3D_LAM_SE.YData = lam_se(2,idx_lam); p3D_LAM_SE.ZData = lam_se(3,idx_lam);
                    p2D_LAM_SE.XData = lam_se(1,idx_lam); p2D_LAM_SE.YData = lam_se(3,idx_lam);
                end

                p3D_C_CW.XData = rho_cw_curr(1); p3D_C_CW.YData = rho_cw_curr(2); p3D_C_CW.ZData = rho_cw_curr(3);
                p3D_C_LAM.XData = rho_lam_curr(1); p3D_C_LAM.YData = rho_lam_curr(2); p3D_C_LAM.ZData = rho_lam_curr(3);
                p2D_C_CW.XData = rho_cw_curr(1); p2D_C_CW.YData = rho_cw_curr(3);
                p2D_C_LAM.XData = rho_lam_curr(1); p2D_C_LAM.YData = rho_lam_curr(3);

                lblTime.Text = sprintf('비행 시간: %.6g / %.6g 초', t, t_final);
                showState(colCW, 'CW', rho_cw_curr, cw_rc(:,i), cw_vc(:,i), cw_info.offset);
                showState(colLAM, 'LAM', rho_lam_curr, lam_rc(:,i), lam_vc(:,i), lam_info.offset);

                drawnow;
                shown = shown + 1;
                lag = shown * frame_dt - toc(play_clock);
                if lag > 0, pause(lag); end
            end

            if strcmp(mode, 'impulse')
                lblStatus.Text  = sprintf('상태: 임펄스 %d회 재생 완료', cw_info.n_impulse);
                if isempty(cw_info.t_mcc)
                    lblSummary.Text = '중간 기동 없음 (1차 투입 + 도착 제동)';
                else
                    lblSummary.Text = sprintf('중간 기동 시각: %s s', join(string(cw_info.t_mcc),' / '));
                end
                cw_goal  = cw_info.goal_err_tf;   
                lam_goal = lam_info.goal_err_tf;
                tag = sprintf('임펄스 %d회', cw_info.n_impulse);
            else
                lblStatus.Text  = sprintf('상태: %d번 재생 완료', sel_idx);
                lblSummary.Text = sprintf('자연표류 시작: CW %.6g s / LAM %.6g s', cw_se_times(end), lam_se_times(end));
                cw_goal  = cw_info.goal_errors(sel_idx);
                lam_goal = lam_info.goal_errors(sel_idx);
                tag = sprintf('%d번(유한연소 %d회)', sel_idx, cw_info.n_burns);
            end

            fprintf('\n============================================================\n');
            fprintf(' [최종 절대 상태 리포트 - 비행 종료(%.6g초) 시점]\n', t_final);
            fprintf('============================================================\n');
            fprintf(' [CW 선택 체이서 (%s)]\n', tag);
            fprintf('  - 절대 위치 [km]   : %13.4f, %13.4f, %13.4f\n', cw_rc(:,end));
            fprintf('  - 절대 속도 [km/s] : %13.6f, %13.6f, %13.6f\n', cw_vc(:,end));
            fprintf('  - 목표점 오차 [km]   : %13.6f\n\n', cw_goal);
            fprintf(' [Lambert 선택 체이서 (%s)]\n', tag);
            fprintf('  - 절대 위치 [km]   : %13.4f, %13.4f, %13.4f\n', lam_rc(:,end));
            fprintf('  - 절대 속도 [km/s] : %13.6f, %13.6f, %13.6f\n', lam_vc(:,end));
            fprintf('  - 목표점 오차 [km]   : %13.6f\n', lam_goal);
            fprintf('============================================================\n\n');

        catch ME
            if isvalid(fig)
                lblStatus.Text = '상태: 오류 - 입력값을 확인하세요';
                uialert(fig, ME.message, 'Simulation Error');
            end
        end
    end
end

function col = makeColumn(parent, x, name, color)
    mk = @(txt, y, w) uilabel(parent, 'Text', txt, 'Position', [x y 285 20], 'FontColor', color, 'FontSize', 11, 'FontWeight', w);
    col.dist   = mk(sprintf('[%s] 목표점 오차: - km', name), 195, 'bold');
    col.dvTot  = mk('총 ΔV: - km/s', 175, 'bold');
    col.posRel = mk('상대위치: [-, -, -] km', 155, 'normal');
    col.posAbs = mk('절대위치: [-, -, -] km', 135, 'normal');
    col.velAbs = mk('절대속도: [-, -, -] km/s', 115, 'normal');
    col.dvMag  = mk('ΔV 크기: 1차(-), 도착(-) km/s', 95, 'bold');
    col.dv1    = mk('ΔV1: [-, -, -] km/s', 75, 'normal');
    col.dv2    = mk('ΔV2: [-, -, -] km/s', 55, 'normal');
end

function showSummary(col, info)
    if strcmp(info.mode, 'impulse')
        col.dvTot.Text = sprintf('총 ΔV: %.6f km/s  (기동 %d회)', info.dv_total, info.n_impulse);
    else
        col.dvTot.Text = sprintf('총 ΔV: %.6f km/s  (기동 %d회)', norm(info.dv1) + norm(info.dv2), info.n_burns);
    end
end

function showState(col, name, rho, rc, vc, offset)
    col.dist.Text   = sprintf('[%s] 목표점 오차: %.4f km', name, norm(rho - offset));
    col.posRel.Text = sprintf('상대위치: [%.3f, %.3f, %.3f] km', rho);
    col.posAbs.Text = sprintf('절대위치: [%.1f, %.1f, %.1f] km', rc);
    col.velAbs.Text = sprintf('절대속도: [%.4f, %.4f, %.4f] km/s', vc);
end

function showDV(col, DV1, DV2, mode)
    n1 = '1차'; n2 = '도착'; 
    col.dvMag.Text = sprintf('ΔV 크기: %s(%.4f), %s(%.4f) km/s', n1, norm(DV1), n2, norm(DV2));
    col.dv1.Text   = sprintf('%s ΔV(ECI): [%.4f, %.4f, %.4f] km/s', n1, DV1);
    col.dv2.Text   = sprintf('%s ΔV(ECI): [%.4f, %.4f, %.4f] km/s', n2, DV2);
end

function varargout = resampleTo(t_src, t_dst, varargin)
    varargout = cell(1, numel(varargin));
    for n = 1:numel(varargin)
        varargout{n} = interp1(t_src, varargin{n}', t_dst)';
    end
end

function enableButton(btn), if isvalid(btn), btn.Enable = 'on'; end, end
function s = seLabel(mode, name), if strcmp(mode,'impulse'), s = [name ' 기동점']; else, s = [name ' S/E 점화']; end, end
function s = t2Label(mode, name), if strcmp(mode,'impulse'), s = [name ' 도착']; else, s = [name ' 도착 기준']; end, end
function s = onoff(b), if b, s = 'on'; else, s = 'off'; end, end

function f = parseMCC(value)
    cleaned = regexprep(value, '[\[\],;]', ' ');
    if isempty(strtrim(cleaned)), f = []; return; end
    if ~isempty(regexp(cleaned, '[^0-9eE+\-.\s]', 'once'))
        error('Maneuver:MCC','중간 기동 시각에는 숫자만 넣으세요. 예) 3160 또는 [1850 3160]');
    end
    f = sscanf(cleaned, '%f').';
    if any(~isfinite(f)), error('Maneuver:MCC','중간 기동 시각을 숫자로 읽을 수 없습니다.'); end
end

function coe = parseCoe(value)
    cleaned = regexprep(value, '[\[\],;]', ' ');
    if ~isempty(regexp(cleaned, '[^0-9eE+\-.\s]', 'once'))
        error('FiniteBurn:COE','Enter six numeric orbital elements.');
    end
    coe = sscanf(cleaned, '%f');
    if numel(coe)~=6 || any(~isfinite(coe)) || coe(1)<=0 || coe(2)<0 || coe(2)>=1
        error('FiniteBurn:COE','Require six finite elements, a > 0, and 0 <= e < 1.');
    end
end