clear; clc; close all;
% 若选择SCL无CRC，需要注释代码417-435行
%% 1. 仿真参数
N       = 64;                % 码长
K       = N/2;                % 信息位（含CRC）
R       = K/N;                % 码率
n       = log2(N);            % 层数
EbN0_des= 2.5;                % 构造信噪比
EbN0    = 5:0.5:8;           % 仿真信噪比
max_frame = 100000;
min_error = 100;

% SCL参数
L_list = [1,2,4];        % 列表大小 1, 2, 4
CRC_len = 8;                  % CRC长度 8

%% 2. 极化码构造（GA）
fprintf('正在构造极化码...\n');
[info_idx, frozen_idx] = polar_construct_GA(N, K, EbN0_des);
frozen_bits = ones(N, 1);
frozen_bits(info_idx) = 0;
fprintf('信息位范围：%d-%d\n', min(info_idx), max(info_idx));
disp(info_idx);
fprintf('冻结位范围：%d-%d\n', min(frozen_idx), max(frozen_idx));
disp(frozen_idx);
fprintf('N=%d, K=%d, CRC=%d\n', N, K, CRC_len);

%% 3. 测试无噪声译码
fprintf('\n===== 测试无噪声译码 =====\n');
g_crc = generate_crc_polynomial(CRC_len);
K_info = K - CRC_len;

% 测试数据
test_info = randi([0,1], K_info, 1);
test_with_crc = crc_encode(test_info, g_crc);
u_test = zeros(N, 1);
u_test(info_idx) = test_with_crc;
x_test = my_polar_encode(u_test);
L_ch_test = 20 * (1 - 2*x_test);  % 高置信度LLR

% 先测试标准SC译码
fprintf('测试标准SC译码...\n');
u_hat_sc = SC_decoder(L_ch_test, frozen_bits);
info_hat_sc = u_hat_sc(info_idx);
info_hat_no_crc_sc = info_hat_sc(1:K_info);
if all(info_hat_no_crc_sc == test_info)
    fprintf('SC译码正确（无噪声）\n');
else
    fprintf('SC译码错误！错误比特数: %d\n', sum(info_hat_no_crc_sc ~= test_info));
    fprintf('  SC译码输出: '); fprintf('%d ', info_hat_no_crc_sc(1:min(20,end))'); fprintf('\n');
    fprintf('  期望输出:   '); fprintf('%d ', test_info(1:min(20,end))'); fprintf('\n');
    return;
end

% 测试SCL译码
fprintf('测试SCL译码...\n');
for L_test = L_list
    [u_hat_scl, ~] = SCL_decoder_based_on_SC(L_ch_test, frozen_bits, info_idx, L_test, g_crc);
    info_hat_scl = u_hat_scl(info_idx);
    info_hat_no_crc_scl = info_hat_scl(1:K_info);
    
    if all(info_hat_no_crc_scl == test_info)
        fprintf('SCL L=%d 译码正确\n', L_test);
    else
        fprintf('SCL L=%d 译码错误！错误比特数: %d\n', L_test, sum(info_hat_no_crc_scl ~= test_info));
    end
end

%% 4. 蒙特卡洛仿真
fprintf('\n===== 开始蒙特卡洛仿真 =====\n');

% 打开文件用于保存结果
fid = fopen('SCL_simulation_results.txt', 'w');
fprintf(fid, 'SCL译码仿真结果\n');
fprintf(fid, '参数: N=%d, K=%d, CRC=%d\n', N, K, CRC_len);
fprintf(fid, '仿真时间: %s\n', datestr(now));
fprintf(fid, '最大帧数: %d, 最小错误帧: %d\n\n', max_frame, min_error);

results = struct();

for l_idx = 1:length(L_list)
    L = L_list(l_idx);
    fprintf('\n--- 仿真 L=%d ---\n', L);
    fprintf(fid, '\n--- 仿真 L=%d ---\n', L);
    
    BLER = zeros(length(EbN0), 1);
    BER = zeros(length(EbN0), 1);
    frame_errors = zeros(length(EbN0), 1);
    bit_errors = zeros(length(EbN0), 1);
    num_runs = zeros(length(EbN0), 1);
    total_active_paths = zeros(length(EbN0), 1);  % 统计总活跃路径数
    
    tic
    for i_run = 1:max_frame
        % 检查是否所有EbN0点都已完成
        all_done = true;
        for i_ebno = 1:length(EbN0)
            if ~(frame_errors(i_ebno) >= min_error && num_runs(i_ebno) > 200) %此处调整min_frame
                all_done = false;
                break;
            end
        end
        if all_done
            fprintf('  所有EbN0点已达到停止条件\n');
            break;
        end
        
        info = randi([0, 1], K_info, 1);
        info_with_crc = crc_encode(info, g_crc);
        u = zeros(N, 1);
        u(info_idx) = info_with_crc;
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
            
            [u_hat, num_active] = SCL_decoder_based_on_SC(L_ch, frozen_bits, info_idx, L, g_crc);
            total_active_paths(i_ebno) = total_active_paths(i_ebno) + num_active;
            
            info_hat = u_hat(info_idx);
            info_hat_no_crc = info_hat(1:K_info);
            bit_err = sum(info_hat_no_crc ~= info);
            bit_errors(i_ebno) = bit_errors(i_ebno) + bit_err;
            if bit_err > 0
                frame_errors(i_ebno) = frame_errors(i_ebno) + 1;
            end
        end
        
        if mod(i_run, 100) == 0
            fprintf('  进度: %d/%d 帧, L=%d\n', i_run, max_frame, L);
            % 显示当前所有EbN0点的BLER
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
    ber_curve = bit_errors ./ max(1, num_runs * K_info);
    bler_curve = frame_errors ./ max(1, num_runs);
    avg_paths = total_active_paths ./ max(1, num_runs);  % 平均活跃路径数
    
    % 保存结果到结构体
    results(l_idx).L = L;
    results(l_idx).BLER = bler_curve;
    results(l_idx).BER = ber_curve;
    results(l_idx).avg_paths = avg_paths;
    results(l_idx).EbN0 = EbN0;
    results(l_idx).num_runs = num_runs;
    results(l_idx).frame_errors = frame_errors;
    results(l_idx).elapsed_time = elapsed_time;
    
    % 打印结果到屏幕
    fprintf('\n结果 (L=%d, CRC=%d):\n', L, CRC_len);
    fprintf('运行时间: %.1f 秒\n', elapsed_time);
    fprintf('Eb/N0 (dB)\terrors\tBLER\t\tBER\t\t平均活跃路径数\t仿真帧数\n');
    for i = 1:length(EbN0)
        fprintf('%.1f\t\t%d\t%.4e\t%.4e\t%.2f\t\t%d\n', ...
            EbN0(i), bit_errors_count(i), bler_curve(i), ber_curve(i), avg_paths(i), num_runs(i));
    end
    
    % 打印结果到文件
    fprintf(fid, '\n结果 (L=%d, CRC=%d):\n', L, CRC_len);
    fprintf(fid, '运行时间: %.1f 秒\n', elapsed_time);
    fprintf(fid, 'Eb/N0 (dB)\terrors\tBLER\t\tBER\t\t平均活跃路径数\n');
    for i = 1:length(EbN0)
        fprintf(fid, '%.1f\t\t%d\t%.4e\t%.4e\t%.2f\n', ...
            EbN0(i), bit_errors_count(i), bler_curve(i), ber_curve(i), avg_paths(i));
    end
end

% 关闭文件
fclose(fid);
fprintf('\n结果已保存到 SCL_simulation_results.txt\n');

%% 5. 绘图 - BLER曲线
figure('Position', [100, 100, 800, 600]);
hold on;
colors = {'b', 'r', 'g', 'm', 'k'};
markers = {'-o', '-s', '-^', '-d', '-v'};
h_legend = [];  % 存储图例句柄
legend_str = {};  % 存储图例文本

for l_idx = 1:length(L_list)
    if l_idx <= length(results) && ~isempty(results(l_idx).BLER)
        h = semilogy(results(l_idx).EbN0, results(l_idx).BLER, ...
            [colors{l_idx} markers{l_idx}], 'LineWidth', 2, ...
            'MarkerSize', 8, 'MarkerFaceColor', colors{l_idx}, ...
            'DisplayName', sprintf('L=%d', L_list(l_idx)));
        h_legend = [h_legend, h];
        legend_str{end+1} = sprintf('L=%d', L_list(l_idx));
    end
end

grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('BLER', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('SCL译码性能 N=%d, K=%d, CRC=%d', N, K, CRC_len), ...
    'FontSize', 14, 'FontWeight', 'bold');
legend(h_legend, legend_str, 'Location', 'southwest', 'FontSize', 10);
set(gca, 'FontSize', 10);
xlim([min(EbN0)-0.2, max(EbN0)+0.2]);
ylim([1e-4, 1]);
saveas(gcf, 'SCL_BLER.png');
fprintf('BLER曲线已保存到 SCL_BLER.png\n');

%% ============================================================
% SC译码器
% ============================================================
function u_hat = SC_decoder(initial_llr, frozen_bits)
    code_length = length(initial_llr);
    num_stages = log2(code_length);
    info_positions = find(frozen_bits == 0);
    
    llr_matrix = initialize_llr_matrix(initial_llr, num_stages, info_positions);
    
    current_stage = 1;
    current_node_group = 1;
    
    while true
        if current_stage <= num_stages
            [group_indices, stage_nodes] = get_polar_groups(code_length, num_stages, ...
                                                           current_stage, current_node_group);
            
            if is_uncomputed(llr_matrix(stage_nodes(1), current_stage + 1))
                for group = 1:size(group_indices, 1)
                    left_input = llr_matrix(group_indices(group, 1), current_stage);
                    right_input = llr_matrix(group_indices(group, 2), current_stage);
                    llr_matrix(group_indices(group, 1), current_stage + 1) = ...
                        min_sum_f(left_input, right_input);
                end
                current_node_group = current_node_group * 2 - 1;
                current_stage = current_stage + 1;
                
            elseif is_uncomputed(llr_matrix(stage_nodes(end), current_stage + 1))
                for group = 1:size(group_indices, 1)
                    left_input = llr_matrix(group_indices(group, 1), current_stage);
                    right_input = llr_matrix(group_indices(group, 2), current_stage);
                    left_child_llr = llr_matrix(group_indices(group, 1), current_stage + 1);
                    left_u = hard_decision(left_child_llr);
                    right_llr = (1 - 2*left_u) .* left_input + right_input;
                    llr_matrix(group_indices(group, 2), current_stage + 1) = right_llr;
                end
                current_node_group = current_node_group * 2;
                current_stage = current_stage + 1;
                
            else
                for group = 1:size(group_indices, 1)
                    left_child = llr_matrix(group_indices(group, 1), current_stage + 1);
                    right_child = llr_matrix(group_indices(group, 2), current_stage + 1);
                    llr_matrix(group_indices(group, 1), current_stage) = ...
                        min_sum_f(left_child, right_child);
                    llr_matrix(group_indices(group, 2), current_stage) = right_child;
                end
                current_node_group = ceil(current_node_group / 2);
                current_stage = current_stage - 1;
            end
            
        elseif current_node_group == code_length
            break;
        else
            current_node_group = ceil(current_node_group / 2);
            current_stage = current_stage - 1;
        end
    end
    
    u_hat = hard_decision(llr_matrix(:, end));
    u_hat = u_hat(:);
end

%% ============================================================
% 基于SC译码器的SCL译码器
% ============================================================
function [u_hat, best_path] = SCL_decoder_based_on_SC(initial_llr, frozen_bits, info_idx, L, g_crc)
    N = length(initial_llr);
    n = log2(N);
    info_positions = find(frozen_bits == 0);
    
    % ===== Lazy Copy 数据结构 =====
    % 使用单一的LLR矩阵存储，避免复制
    % 存储所有路径的中间LLR值
    % llr_store: 每列对应一个比特位置的LLR计算
    % 使用三维数组存储：第3维是路径
    llr_store = NaN(N, n+1, L);  % N个节点 x (n+1)层 x L条路径
    
    % 初始化所有路径的第一层（信道LLR）
    for l = 1:L
        llr_store(:, 1, l) = initial_llr;
        llr_store(info_positions, n+1, l) = NaN;
        llr_store(setdiff(1:N, info_positions), n+1, l) = Inf;
    end
    
    % Lazy copy 索引：lazy_copy(l) 表示路径l的数据来源
    lazy_copy = 1:L;  % 初始时每条路径指向自己
    next_free = L + 1;  % 下一个空闲位置
    
    % 路径的比特估计和度量
    u_paths = zeros(L, N);
    PM = zeros(1, L);
    PM(2:end) = Inf;
    active = L;  % 使用固定数量路径，非活跃的PM设为Inf
    
    % 逐比特SCL译码
    for bit_idx = 1:N
        new_lazy = zeros(1, 2*L);
        new_u = zeros(2*L, N);
        new_PM = inf(1, 2*L);
        new_llr_store = NaN(N, n+1, 2*L);
        
        cand_count = 0;
        
        for l = 1:active
            if isinf(PM(l))
                continue;
            end
            
            % 使用lazy copy获取当前路径的LLR矩阵
            src = lazy_copy(l);
            llr_mat = llr_store(:, :, src);
            
            % 计算当前比特的LLR
            llr_val = compute_bit_llr_using_SC(llr_mat, frozen_bits, bit_idx, N, n);
            
            if frozen_bits(bit_idx) == 1
                % 冻结位：只有0
                u_est = 0;
                new_pm = PM(l);
                if llr_val < 0
                    new_pm = new_pm + abs(llr_val);
                end
                
                cand_count = cand_count + 1;
                idx = cand_count;
                new_lazy(idx) = src;  % 直接指向源数据
                new_PM(idx) = new_pm;
                new_u(idx, :) = u_paths(l, :);
                new_u(idx, bit_idx) = u_est;
                new_llr_store(:, :, idx) = llr_mat;
                new_llr_store(bit_idx, n+1, idx) = Inf;
                
            else
                % 信息位：分裂为0和1
                for bit_val = [0, 1]
                    new_pm = PM(l);
                    hard = double(llr_val < 0);
                    if bit_val ~= hard
                        new_pm = new_pm + abs(llr_val);
                    end
                    
                    cand_count = cand_count + 1;
                    idx = cand_count;
                    new_lazy(idx) = src;  % 指向源数据
                    new_PM(idx) = new_pm;
                    new_u(idx, :) = u_paths(l, :);
                    new_u(idx, bit_idx) = bit_val;
                    
                    % Lazy copy: 只复制引用
                    new_llr_store(:, :, idx) = llr_mat;
                    if bit_val == 0
                        new_llr_store(bit_idx, n+1, idx) = Inf;
                    else
                        new_llr_store(bit_idx, n+1, idx) = -Inf;
                    end
                end
            end
        end
        
        % 裁剪：保留PM最小的L条路径
        num_cand = cand_count;
        if num_cand > L
            [~, sorted] = sort(new_PM(1:num_cand));
            keep = sorted(1:L);
        else
            keep = 1:num_cand;
        end
        
        % 更新活跃路径
        active = length(keep);
        new_active = active;
        
        % 重新组织数据
        temp_lazy = lazy_copy;
        temp_llr = llr_store;
        temp_u = u_paths;
        temp_PM = PM;
        
        for l = 1:active
            idx = keep(l);
            lazy_copy(l) = l;  % 新路径指向自己
            PM(l) = new_PM(idx);
            u_paths(l, :) = new_u(idx, :);
            llr_store(:, :, l) = new_llr_store(:, :, idx);
        end
        
        % 未使用的路径设为非活跃
        for l = active+1:L
            PM(l) = Inf;
        end
    end
    
    % CRC校验选择最佳路径
    crc_ok = false(1, active);
    for l = 1:active
        if ~isinf(PM(l))
            candidate = u_paths(l, info_idx);
            crc_ok(l) = crc_check(candidate, g_crc);
        end
    end
    
    if any(crc_ok)
        [~, best] = min(PM(crc_ok));
        best_path = find(crc_ok, best);
        best_path = best_path(end);
    else
        [~, best_path] = min(PM(1:active));
    end

%     % 无CRC
%     [~, best_path] = min(PM(1:active));

    u_hat = u_paths(best_path, :)';
end

%% ============================================================
% 使用SC译码器的逻辑计算单个比特的LLR
% ============================================================
function llr_val = compute_bit_llr_using_SC(llr_matrix, frozen_bits, target_bit, N, n)
    % 计算第target_bit位的LLR
    % 注意：不修改输入的llr_matrix
    
    % 避免修改原始数据
    llr_mat = llr_matrix; 
    
    current_stage = 1;
    current_node_group = 1;
    max_iter = 10000;  % 防止无限循环
    iter = 0;
    
    while iter < max_iter
        iter = iter + 1;
        
        if current_stage <= n
            [group_indices, stage_nodes] = get_polar_groups(N, n, current_stage, current_node_group);
            
            % 检查是否到达目标比特所在节点
            if current_stage == n + 1
                llr_val = llr_mat(target_bit, end);
                return;
            end
            
            % 如果在最底层并且是目标比特
            if current_stage == n && current_node_group == target_bit
                if ~isnan(llr_mat(target_bit, end))
                    llr_val = llr_mat(target_bit, end);
                    return;
                end
            end
            
            if is_uncomputed(llr_mat(stage_nodes(1), current_stage + 1))
                for group = 1:size(group_indices, 1)
                    left_input = llr_mat(group_indices(group, 1), current_stage);
                    right_input = llr_mat(group_indices(group, 2), current_stage);
                    llr_mat(group_indices(group, 1), current_stage + 1) = ...
                        min_sum_f(left_input, right_input);
                end
                current_node_group = current_node_group * 2 - 1;
                current_stage = current_stage + 1;
                
            elseif is_uncomputed(llr_mat(stage_nodes(end), current_stage + 1))
                for group = 1:size(group_indices, 1)
                    left_input = llr_mat(group_indices(group, 1), current_stage);
                    right_input = llr_mat(group_indices(group, 2), current_stage);
                    left_child_llr = llr_mat(group_indices(group, 1), current_stage + 1);
                    left_u = hard_decision(left_child_llr);
                    right_llr = (1 - 2*left_u) * left_input + right_input;
                    llr_mat(group_indices(group, 2), current_stage + 1) = right_llr;
                end
                current_node_group = current_node_group * 2;
                current_stage = current_stage + 1;
                
            else
                for group = 1:size(group_indices, 1)
                    left_child = llr_mat(group_indices(group, 1), current_stage + 1);
                    right_child = llr_mat(group_indices(group, 2), current_stage + 1);
                    llr_mat(group_indices(group, 1), current_stage) = ...
                        min_sum_f(left_child, right_child);
                    llr_mat(group_indices(group, 2), current_stage) = right_child;
                end
                
                % 检查目标比特
                visited_nodes = unique(stage_nodes);
                if any(visited_nodes == target_bit)
                    if ~isnan(llr_mat(target_bit, end))
                        llr_val = llr_mat(target_bit, end);
                        return;
                    end
                end
                
                current_node_group = ceil(current_node_group / 2);
                current_stage = current_stage - 1;
            end
            
        elseif current_node_group == N
            llr_val = llr_mat(target_bit, end);
            return;
        else
            current_node_group = ceil(current_node_group / 2);
            current_stage = current_stage - 1;
        end
    end
    
    % 如果循环结束还没找到，返回当前值
    llr_val = llr_mat(target_bit, end);
    if isnan(llr_val)
        llr_val = 0;  % 默认值
    end
end

%% ============================================================
% SC译码器的辅助函数
% ============================================================
function llr_matrix = initialize_llr_matrix(channel_llr, num_stages, info_positions)
    code_length = length(channel_llr);
    llr_matrix = [channel_llr(:), NaN(code_length, num_stages - 1), Inf(code_length, 1)];
    llr_matrix(info_positions, end) = NaN;
end

function [group_indices, first_group_nodes] = get_polar_groups(code_length, num_stages, ...
                                                                current_stage, current_node_group)
    stage_size = 2^(num_stages - current_stage);
    num_groups = 2^(current_stage - 1);
    all_groups = reshape(1:code_length, stage_size, 2, num_groups);
    group_indices = all_groups(:, :, current_node_group);
    first_group_nodes = group_indices(1, :);
end

function is_nan = is_uncomputed(value)
    is_nan = isnan(value);
end

function bits = hard_decision(llr_values)
    bits = (1 - sign(llr_values)) / 2;
    bits(isnan(bits)) = 0;
end

function result = min_sum_f(a, b)
    sign_a = sign(a);
    sign_b = sign(b);
    sign_a(sign_a == 0) = 1;
    sign_b(sign_b == 0) = 1;
    result = sign_a .* sign_b .* min(abs(a), abs(b));
end

%% ============================================================
% 其他辅助函数
% ============================================================
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

function g = generate_crc_polynomial(crc_len)
    switch crc_len
        case 8
            g = [1 0 0 0 0 0 1 1 1];
        case 16
            g = [1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1];
        otherwise
            error('不支持的CRC长度');
    end
end

function coded_bits = crc_encode(info_bits, g)
    K = length(info_bits);
    r = length(g) - 1;
    padded = [info_bits; zeros(r, 1)];
    for i = 1:K
        if padded(i) == 1
            for j = 1:r+1
                padded(i+j-1) = xor(padded(i+j-1), g(j));
            end
        end
    end
    coded_bits = [info_bits; padded(K+1:end)];
end

function passed = crc_check(received_bits, g)
    K = length(received_bits) - (length(g) - 1);
    if K <= 0
        passed = false;
        return;
    end
    padded = received_bits;
    for i = 1:K
        if padded(i) == 1
            for j = 1:length(g)
                padded(i+j-1) = xor(padded(i+j-1), g(j));
            end
        end
    end
    passed = all(padded(K+1:end) == 0);
end