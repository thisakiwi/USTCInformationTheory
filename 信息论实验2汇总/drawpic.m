% 清空环境
clear; close all; clc;

%% BP译码数据 (从 BP_64_results.txt)
EbN0_BP = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
BLER_BP = [2.521739e-01, 1.412371e-01, 7.878788e-02, 5.528846e-02, ...
           3.093003e-02, 8.829589e-03, 2.340000e-03, 6.400000e-04, ...
           6.000000e-05, 0, 0];

%% SC译码数据 (从 SC64.txt)
EbN0_SC = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0];
BLER_SC = [0.2141, 0.1427, 0.0878, 0.0649, 0.0295, 0.0073, ...
           0.0036, 0.0007, 0.0003, 0, 0.0001, 0, 0];

%% CA-SCL译码数据 (从 CA-SCL_results.txt)
% L=1
EbN0_SCL1 = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0];
BLER_SCL1 = [1.8083e-01, 1.3021e-01, 1.0787e-01, 5.3163e-02, ...
             2.6925e-02, 1.1235e-02, 4.7000e-03];

% L=2
EbN0_SCL2 = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0];
BLER_SCL2 = [1.1947e-01, 7.2464e-02, 3.3772e-02, 2.0786e-02, ...
             8.1000e-03, 3.7000e-03, 1.0000e-03];

% L=4
EbN0_SCL4 = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0];
BLER_SCL4 = [4.7642e-02, 2.4950e-02, 1.1787e-02, 5.9000e-03, ...
             1.7000e-03, 8.0000e-04, 2.0000e-04];

%% 绘图
figure('Position', [100, 100, 800, 600]);
semilogy(EbN0_BP, BLER_BP, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
hold on;
semilogy(EbN0_SC, BLER_SC, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
semilogy(EbN0_SCL1, BLER_SCL1, 'g-^', 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'g');
semilogy(EbN0_SCL2, BLER_SCL2, 'm-d', 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'm');
semilogy(EbN0_SCL4, BLER_SCL4, 'k-v', 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'k');

% 图形设置
grid on;
xlabel('Eb/N0 (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('BLER', 'FontSize', 12, 'FontWeight', 'bold');
title('极化码译码性能对比 (N=64, K=32)', 'FontSize', 14, 'FontWeight', 'bold');
legend('BP译码', 'SC译码', 'CA-SCL (L=1)', 'CA-SCL (L=2)', 'CA-SCL (L=4)', ...
       'Location', 'northeast', 'FontSize', 10);
axis([4.5 15.5 1e-6 1]);
set(gca, 'FontSize', 11);

% 保存图片
saveas(gcf, 'BLER_comparison_N64.png');