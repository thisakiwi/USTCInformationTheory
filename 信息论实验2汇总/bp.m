main_bp_polar_simulation();
function [u_hat, num_iter] = BP_decoder_with_iter_count(L_ch, frozen_bits, max_iter)
    % L_ch: 信道LLR输入
    % frozen_bits: 1表示冻结位，0表示信息位
    % max_iter: 最大迭代次数
    % u_hat: 译码输出（硬判决）
    % num_iter: 实际迭代次数
    
    N = length(L_ch);
    n = log2(N);
    
    % 初始化L和R矩阵
    L = zeros(N, n+1);
    R = zeros(N, n+1);
    
    % 冻结位处理：R(冻结位,1)=inf，R(信息位,1)=0
    frozen_idx = find(frozen_bits == 1);
    info_idx = find(frozen_bits == 0);
    
    R(info_idx, 1) = 0;
    R(frozen_idx, 1) = inf;
    
    % 信道LLR输入到L矩阵最后一列
    L(:, n+1) = L_ch(:);
    
    % BP迭代译码
    for iter = 1:max_iter
        % 左向传播 (L更新)
        for i = n:-1:1
            for j = 1:N/2
                L(j, i) = fun_g1(L(2*j-1, i+1), L(2*j, i+1) + R(j+N/2, i));
                L(j+N/2, i) = fun_g1(R(j, i), L(2*j-1, i+1)) + L(2*j, i+1);
            end
        end
        
        % 右向传播 (R更新)
        for i = 1:n
            for j = 1:N/2
                R(2*j-1, i+1) = fun_g1(R(j, i), R(j+N/2, i) + L(2*j, i+1));
                R(2*j, i+1) = fun_g1(R(j, i), L(2*j-1, i+1)) + R(j+N/2, i);
            end
        end
        
        % 计算当前估计的LLR
        current_llr = L(:, 1);
        temp_decision = zeros(N, 1);
        temp_decision(current_llr >= 0) = 0;
        temp_decision(current_llr < 0) = 1;
        
        % 检查冻结位是否满足
        if all(temp_decision(frozen_idx) == 0)
            % 冻结位正确，可提前终止
            num_iter = iter;
            break;
        end
    end
    
    if iter == max_iter
        num_iter = max_iter;
    end
    
    % 输出LLR并硬判决
    llr_out = L(:, 1);
    u_hat = zeros(N, 1);
    u_hat(llr_out < 0) = 1;
    u_hat(llr_out >= 0) = 0;
end

function L_R = fun_g1(x1, x2)
    % BP译码中的g函数（最小和近似）
    L_R = sign(x1) * sign(x2) * min(abs(x1), abs(x2));
end

function main_bp_polar_simulation()
    clear; clc; close all;

    %% 1. 仿真参数
    N = 256;                     % 码长
    K = N/2;                    % 信息位
    R = K/N;                    % 码率
    n = log2(N);                % 层数
    EbN0_des = 2.5;             % 构造信噪比/5.5
    EbN0 = 5:1:14;            % 仿真信噪比
    max_frame = 100000;
    min_error = 100;
    
    % BP参数
    max_iter = 50;              % BP最大迭代次数

    %% 2. 极化码构造（GA）
    fprintf('正在构造极化码...\n');
    [info_idx, frozen_idx] = polar_construct_GA(N, K, EbN0_des);
    frozen_bits = ones(N, 1);
    frozen_bits(info_idx) = 0;
    fprintf('信息位范围：%d-%d\n', min(info_idx), max(info_idx));
    disp(info_idx);
    fprintf('冻结位范围：%d-%d\n', min(frozen_idx), max(frozen_idx));
    disp(frozen_idx);
    fprintf('N=%d, K=%d\n', N, K);

    %% 3. 测试无噪声BP译码
    fprintf('\n===== 测试无噪声BP译码 =====\n');
    
    % 测试数据
    test_info = randi([0,1], K, 1);
    u_test = zeros(N, 1);
    u_test(info_idx) = test_info;
    x_test = my_polar_encode(u_test);
    L_ch_test = 20 * (1 - 2*x_test);  % 高置信度LLR
    
    % BP译码测试
    fprintf('测试BP译码...\n');
    [u_hat_bp, num_iter_used] = BP_decoder_with_iter_count(L_ch_test, frozen_bits, max_iter);
    info_hat_bp = u_hat_bp(info_idx);
    
    if all(info_hat_bp == test_info)
        fprintf('BP译码正确（无噪声）\n');
    else
        fprintf('BP译码错误！错误比特数: %d\n', sum(info_hat_bp ~= test_info));
        fprintf('  BP译码输出: '); fprintf('%d ', info_hat_bp(1:min(20,end))'); fprintf('\n');
        fprintf('  期望输出:   '); fprintf('%d ', test_info(1:min(20,end))'); fprintf('\n');
    end

    %% 4. 蒙特卡洛仿真
    fprintf('\n===== 开始蒙特卡洛仿真 =====\n');

    % 打开文件用于保存结果
    fid = fopen('BP_simulation_results.txt', 'w');
    fprintf(fid, 'BP译码仿真结果\n');
    fprintf(fid, '参数: N=%d, K=%d, 最大迭代=%d\n', N, K, max_iter);
    fprintf(fid, '仿真时间: %s\n', datestr(now));
    fprintf(fid, '最大帧数: %d, 最小错误帧: %d\n\n', max_frame, min_error);

    BLER = zeros(length(EbN0), 1);
    BER = zeros(length(EbN0), 1);
    frame_errors = zeros(length(EbN0), 1);
    bit_errors = zeros(length(EbN0), 1);
    num_runs = zeros(length(EbN0), 1);
    total_iterations = zeros(length(EbN0), 1);  % 统计总迭代次数
    
    tic
    for i_run = 1:max_frame
        % 检查是否所有EbN0点都已完成
        all_done = true;
        for i_ebno = 1:length(EbN0)
            if ~(frame_errors(i_ebno) >= min_error && num_runs(i_ebno) > 200)
                all_done = false;
                break;
            end
        end
        if all_done
            fprintf('  所有EbN0点已达到停止条件\n');
            break;
        end
        
        info = randi([0, 1], K, 1);
        u = zeros(N, 1);
        u(info_idx) = info;
        x = my_polar_encode(u);
        bpsk = 1 - 2 * x;
        
        for i_ebno = 1:length(EbN0)
            if frame_errors(i_ebno) >= min_error && num_runs(i_ebno) > 200
                continue;
            end
            
            num_runs(i_ebno) = num_runs(i_ebno) + 1;
            
            EbN0_lin = 10^(EbN0(i_ebno) / 10);
            sigma = sqrt(1 / (2 * R * EbN0_lin));
            y = bpsk + sigma * randn(N, 1);
            L_ch = (2 / sigma^2) * y;
            
            % BP译码
            [u_hat, num_iter] = BP_decoder_with_iter_count(L_ch, frozen_bits, max_iter);
            total_iterations(i_ebno) = total_iterations(i_ebno) + num_iter;
            
            info_hat = u_hat(info_idx);
            bit_err = sum(info_hat ~= info);
            bit_errors(i_ebno) = bit_errors(i_ebno) + bit_err;
            if bit_err > 0
                frame_errors(i_ebno) = frame_errors(i_ebno) + 1;
            end
        end
        
        if mod(i_run, 100) == 0
            fprintf('  进度: %d/%d 帧\n', i_run, max_frame);
            for i_ebno = 1:length(EbN0)
                if num_runs(i_ebno) > 0
                    current_bler = frame_errors(i_ebno)/num_runs(i_ebno);
                    fprintf('    EbN0=%.1fdB: 帧数=%d, 错误=%d, BLER=%.4f\n', ...
                        EbN0(i_ebno), num_runs(i_ebno), frame_errors(i_ebno), current_bler);
                end
            end
        end
    end
    elapsed_time = toc;
    
    % 计算统计结果
    bit_errors_count = bit_errors ./ num_runs;
    ber_curve = bit_errors ./ max(1, num_runs * K);
    bler_curve = frame_errors ./ max(1, num_runs);
    avg_iterations = total_iterations ./ max(1, num_runs);
    
    % 打印结果
    fprintf('\nBP译码结果:\n');
    fprintf('运行时间: %.1f 秒\n', elapsed_time);
    fprintf('Eb/N0 (dB)\terros\tBLER\t\tBER\t\t平均迭代次数\t仿真帧数\n');
    for i = 1:length(EbN0)
        fprintf('%.1f\t%d\t%.4e\t%.4e\t%.2f\t\t%d\n', ...
            EbN0(i), bit_errors_count(i), bler_curve(i), ber_curve(i), avg_iterations(i), num_runs(i));
    end
    
    % 保存结果到文件
    fprintf(fid, '\nBP译码结果:\n');
    fprintf(fid, '运行时间: %.1f 秒\n', elapsed_time);
    fprintf(fid, 'Eb/N0 (dB)\tBLER\t\tBER\t\t平均迭代次数\n');
    for i = 1:length(EbN0)
        fprintf(fid, '%.1f\t%d\t%.4e\t%.4e\t%.2f\n', ...
            EbN0(i), bit_errors_count(i), bler_curve(i), ber_curve(i), avg_iterations(i));
    end
    
    fclose(fid);
    fprintf('\n结果已保存到 BP_simulation_results.txt\n');

    %% 5. 绘图
    figure('Position', [100, 100, 800, 600]);
    semilogy(EbN0, bler_curve, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    grid on;
    xlabel('E_b/N_0 (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('BLER', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('BP译码性能 N=%d, K=%d, 最大迭代=%d', N, K, max_iter), ...
        'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'FontSize', 10);
    xlim([min(EbN0)-0.2, max(EbN0)+0.2]);
    ylim([1e-4, 1]);
end

function [info_idx, frozen_idx] = polar_construct_GA(N, K, EbN0)
    n = log2(N);
    R = K / N;
    sigma = 1 / sqrt(2 * R) * 10^(-EbN0 / 20);
    m = zeros(1, N);
    m(1) = 2 / sigma^2;

    for layer = 1:n
        j = 2^(layer - 1);
        m_new = zeros(1, 2 * j);
        for i = 1:j
            tmp = m(i);
            m_new(2 * i - 1) = phi_inv(1 - (1 - phi(tmp))^2);
            m_new(2 * i) = 2 * tmp;
        end
        m = m_new;
    end

    m = bitrevorder(m);
    [~, idx] = sort(m, 'descend');
    info_idx = sort(idx(1:K));
    frozen_idx = sort(idx(K + 1:end));
end

function y = phi(x)
    if x < 10
        y = exp(-0.4527 * x^0.86) + 0.0218;
    else
        y = sqrt(pi ./ (x + 1e-4)) .* (1 - 1.4286 ./ (x + 1e-4)) .* exp(-x / 4);
    end
    y = min(y, 1);
end

function x = phi_inv(y)
    y = min(max(y, 1e-10), 1 - 1e-10);
    y_cut = exp(-0.4527 * 10^0.86) + 0.0218;
    if y >= y_cut
        x = ((-log(y - 0.0218)) / 0.4527)^(1/0.86);
    else
        x = -4 * log(y);
    end
    x = min(max(x, 0), 50);
end

function x = my_polar_encode(u)
    u = u(:);
    N = length(u);
    if N == 1
        x = u;
    else
        u1 = u(1:N/2);
        u2 = u(N/2 + 1:end);
        x1 = mod(u1 + u2, 2);
        x = [my_polar_encode(x1); my_polar_encode(u2)];
    end
end   