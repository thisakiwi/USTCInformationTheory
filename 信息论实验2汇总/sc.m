clear; clc; close all;

%% 1. 仿真参数
N       = 64;                % 码长 /512
K       = N/2;                % 信息位
R       = K/N;                % 码率
n       = log2(N);            % 层数
EbN0_des= 2.5;                % 构造信噪比
EbN0    = 5:0.5:11;            % 仿真信噪比
max_frame = 10000;
min_error = 100;

%% 2. 极化码构造（GA）
fprintf('正在构造极化码...\n');
[info_idx, frozen_idx] = polar_construct_GA(N, K, EbN0_des);
frozen_bits = ones(N, 1);
frozen_bits(info_idx) = 0;

fprintf('信息位范围：%d-%d\n', min(info_idx), max(info_idx));
disp(info_idx);
fprintf('冻结位范围：%d-%d\n', min(frozen_idx), max(frozen_idx));
disp(frozen_idx);

%% 3. 蒙特卡洛仿真
BLER = zeros(length(EbN0), 1);
BER = zeros(length(EbN0), 1);
error_count = zeros(length(EbN0), 1);
num_runs = zeros(length(EbN0), 1);
decode_time_total = zeros(length(EbN0), 1);

tic
for i_run = 1 : max_frame
    info = randi([0, 1], K, 1);
    u = zeros(N, 1);
    u(info_idx) = info;

    x = my_polar_encode(u);
    bpsk = 1 - 2 * x;

    for i_ebno = 1 : length(EbN0)
        if BLER(i_ebno) >= min_error
            continue;
        end

        num_runs(i_ebno) = num_runs(i_ebno) + 1;
        noise = randn(N, 1);

        EbN0_lin = 10^(EbN0(i_ebno) / 10);
        sigma = sqrt(1 / (2 * R * EbN0_lin));
        y = bpsk + sigma * noise;
        L_ch = (2 / sigma^2) * y;

        t_start = tic;
        u_hat = SC_decoder(L_ch, frozen_bits);
        t_decode = toc(t_start);

        decode_time_total(i_ebno) = decode_time_total(i_ebno) + t_decode;

        info_hat = u_hat(info_idx);
        current_err = sum(info_hat ~= info);
        error_count(i_ebno) = error_count(i_ebno) + current_err;
        BER(i_ebno) = BER(i_ebno) + current_err;

        if current_err > 0
            BLER(i_ebno) = BLER(i_ebno) + 1;
        end
    end
end
toc

error_curve = error_count ./ num_runs;
for i = 1:length(EbN0)
    fprintf('Eb/N0 = %.1f | error = %.6e\n', EbN0(i), error_curve(i));
end

bler_curve = BLER ./ num_runs;
for i = 1:length(EbN0)
    fprintf('Eb/N0 = %.1f | BLER = %.4f\n', EbN0(i), bler_curve(i));
end

ber_curve = BER ./ (num_runs * K);
for i = 1:length(EbN0)
    fprintf('Eb/N0 = %.1f dB | BER = %.4f\n', EbN0(i), ber_curve(i));
end

avg_decode_time = decode_time_total ./ num_runs;
for i = 1:length(EbN0)
    fprintf('Eb/N0 = %.1f dB | 平均译码时间 = %.4f s\n', EbN0(i), avg_decode_time(i));
end

%% 绘图
figure;
semilogy(EbN0, bler_curve, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'SC译码');
grid on;
xlabel('E_b / N_0 (dB)');
ylabel('BLER');
title(sprintf('极化码 SC译码 N=%d, K=%d', N, K));
legend;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GA 构造
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
%     plot(m, 'LineWidth',1.5);
%     xlabel('idx');
%     title('极化现象');
%     grid on;
    m = bitrevorder(m);
    [~, idx] = sort(m, 'descend');
    info_idx = sort(idx(1:K));
    frozen_idx = sort(idx(K + 1:end));
end

function y = phi(x)
    if x < 10
%         y = exp(-0.4527 * x^0.86 + 0.0218);
        y = exp(-0.4527 * x^0.86) + 0.0218;
    else
        y = sqrt(pi ./ (x + 1e-4)) .* (1 - 1.4286 ./ (x + 1e-4)) .* exp(-x / 4);
    end
    y = min(y, 1);
end

function x = phi_inv(y)
    y = min(max(y, 1e-10), 1 - 1e-10);
%     y_cut = exp(-0.4527 * 10^0.86 + 0.0218);   % φ(10)
    y_cut = exp(-0.4527 * 10^0.86)+0.0218;
    if y >= y_cut
%         x = ((-log(y) + 0.0218) / 0.4527)^(1 / 0.86);
        x = ((-log(y - 0.0218)) / 0.4527)^(1/0.86);

    else
        x = -4 * log(y);                        % 大 x 近似支路
    end
    x = min(max(x, 0), 50);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 递归极化编码：x = mod(u' * G_N, 2)'
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SC 译码
function u_hat = SC_decoder(initial_llr, frozen_bits)
    % 输入:
    %   initial_llr - 信道输出的对数似然比
    %   frozen_bits - 冻结位指示 (1=冻结, 0=信息位)
    % 输出:
    %   u_hat - 译码后的比特序列
    
    code_length = length(initial_llr);
    num_stages = log2(code_length);
    info_positions = find(frozen_bits == 0);  % 信息位位置
    
    % 初始化LLR矩阵
    llr_matrix = initialize_llr_matrix(initial_llr, num_stages, info_positions);
    
    current_stage = 1;
    current_node_group = 1;
    
    while true
        if current_stage <= num_stages
            [group_indices, stage_nodes] = get_polar_groups(code_length, num_stages, ...
                                                           current_stage, current_node_group);
            
            % 检查左子节点的LLR是否未计算
            if is_uncomputed(llr_matrix(stage_nodes(1), current_stage + 1))
                % 从左向右传播，计算左子节点LLR
                for group = 1:size(group_indices, 1)
                    left_input = llr_matrix(group_indices(group, 1), current_stage);
                    right_input = llr_matrix(group_indices(group, 2), current_stage);
                    llr_matrix(group_indices(group, 1), current_stage + 1) = ...
                        min_sum_f(left_input, right_input);
                end
                % 移动到左子节点
                current_node_group = current_node_group * 2 - 1;
                current_stage = current_stage + 1;
                
            % 检查右子节点的LLR是否未计算
            elseif is_uncomputed(llr_matrix(stage_nodes(end), current_stage + 1))
                % 从左向右传播，计算右子节点LLR
                for group = 1:size(group_indices, 1)
                    left_input = llr_matrix(group_indices(group, 1), current_stage);
                    right_input = llr_matrix(group_indices(group, 2), current_stage);
                    left_child_llr = llr_matrix(group_indices(group, 1), current_stage + 1);
%                     llr_matrix(group_indices(group, 2), current_stage + 1) = ...
%                         min_sum_f(left_child_llr, left_input) + right_input;
                    left_u = hard_decision(left_child_llr);  % 对左子节点LLR做硬判决
                    right_llr = (1 - 2*left_u) .* left_input + right_input;
                    llr_matrix(group_indices(group, 2), current_stage + 1) = right_llr;
                end
                % 移动到右子节点
                current_node_group = current_node_group * 2;
                current_stage = current_stage + 1;
                
            else
                % 左右子节点都已计算，从右向左传播
                for group = 1:size(group_indices, 1)
                    left_child = llr_matrix(group_indices(group, 1), current_stage + 1);
                    right_child = llr_matrix(group_indices(group, 2), current_stage + 1);
                    llr_matrix(group_indices(group, 1), current_stage) = ...
                        min_sum_f(left_child, right_child);
                    llr_matrix(group_indices(group, 2), current_stage) = right_child;
                end
                % 返回到父节点
                current_node_group = ceil(current_node_group / 2);
                current_stage = current_stage - 1;
            end
            
        elseif current_node_group == code_length
            % 完成所有译码
            break;
        else
            % 在根层级继续返回
            current_node_group = ceil(current_node_group / 2);
            current_stage = current_stage - 1;
        end
    end
    
    % 硬判决得到最终比特
    u_hat = hard_decision(llr_matrix(:, end));
    u_hat = u_hat(:);
end

function llr_matrix = initialize_llr_matrix(channel_llr, num_stages, info_positions)
    % 初始化LLR矩阵
    % 第1列: 信道LLR输入
    % 中间列: 待计算的中间节点 (初始化为NaN)
    % 最后一列: 译码输出 (信息位初始化为NaN，冻结位初始化为Inf表示已知为0)
    
    code_length = length(channel_llr);
    
    % 创建矩阵并填充初始值
    llr_matrix = [channel_llr(:), NaN(code_length, num_stages - 1), Inf(code_length, 1)];
    
    % 信息位需要被计算，所以设为NaN
    llr_matrix(info_positions, end) = NaN;
end

function [group_indices, first_group_nodes] = get_polar_groups(code_length, num_stages, ...
                                                                current_stage, current_node_group)
    % 获取当前阶段当前节点组的所有极化单元索引
    stage_size = 2^(num_stages - current_stage);
    num_groups = 2^(current_stage - 1);
    
    all_groups = reshape(1:code_length, stage_size, 2, num_groups);
    group_indices = all_groups(:, :, current_node_group);
    
    % 获取第一个组的前两个节点用于状态检查
    first_group_nodes = group_indices(1, :);
end

function is_nan = is_uncomputed(value)
    % 检查LLR值是否未计算（NaN）
    is_nan = isnan(value);
end

function bits = hard_decision(llr_values)
    % LLR到比特的硬判决
    % LLR < 0 => bit = 1, LLR >= 0 => bit = 0
    bits = (1 - sign(llr_values)) / 2;
    bits(isnan(bits)) = 0;  % 处理可能的NaN
end

function result = min_sum_f(a, b)
    % f(a,b) = sign(a)·sign(b)·min(|a|,|b|)
    sign_a = sign(a);
    sign_b = sign(b);
    sign_a(sign_a == 0) = 1;  % 处理零值
    sign_b(sign_b == 0) = 1;
    
    result = sign_a .* sign_b .* min(abs(a), abs(b));
end