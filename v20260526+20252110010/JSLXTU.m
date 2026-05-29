% 创建长比例的画布
figure('Position', [100, 100, 650, 600], 'Color', 'w');  % 画布高度减小
axes('Position', [0.13, 0.15, 0.8, 0.8]);               % 坐标区占满画布

% ================= 1. 数据精确构造 =================
K = 9;
n_vals = 2.^(0:K-1); 
h_vals = 1 ./ n_vals;

% 蓝线数据
pi_n = zeros(1, K);
pi_n(1) = 0;
pi_n(2:end) = n_vals(2:end) .* sin(pi ./ n_vals(2:end));
e_n = abs(pi - pi_n);

% 橙线数据 (4个点，从n=8开始)
h_extrap = h_vals(4:7);          
e_extrap = zeros(1, 4);
e_extrap(1) = e_n(4);            % 交点 (n=8)
e_extrap(2) = e_extrap(1) / (2^5.31);
e_extrap(3) = e_extrap(2) / (2^7.50);
e_extrap(4) = e_extrap(3) / (2^9.76);

% ================= 2. 绘图 =================
loglog(h_vals, e_n, '-v', 'Color', '#0072BD', 'MarkerFaceColor', 'none', ...
       'LineWidth', 1.2, 'MarkerSize', 6);
hold on;
loglog(h_extrap, e_extrap, '-^', 'Color', '#D95319', 'MarkerFaceColor', 'none', ...
       'LineWidth', 1.2, 'MarkerSize', 6);
loglog(h_vals(4), e_n(4), 'p', 'MarkerFaceColor', 'r', ...
       'MarkerEdgeColor', 'r', 'MarkerSize', 14);

% ================= 3. 线段上方标注 + 实心箭头（加长版） =================
% --- 蓝线标注 ---
x_mid_blue = 10^((log10(h_vals(5)) + log10(h_vals(6))) / 2);
y_mid_blue = 10^((log10(e_n(5))   + log10(e_n(6)))   / 2);
y_text_blue = 10^(log10(y_mid_blue) + 2.0);
x_text_blue = x_mid_blue;
plot([x_text_blue, x_mid_blue], [y_text_blue, y_mid_blue], 'r-', 'LineWidth', 1.5);
plot(x_mid_blue, y_mid_blue, 'rv', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', 'MarkerSize', 8);
text(x_text_blue, y_text_blue, 'slope: 2.00', 'Color', 'r', 'FontSize', 11, ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

% --- 橙线标注 ---
x_mid_orange = 10^((log10(h_extrap(3)) + log10(h_extrap(4))) / 2);
y_mid_orange = 10^((log10(e_extrap(3)) + log10(e_extrap(4))) / 2);
y_text_orange = 10^(log10(y_mid_orange) + 2.0);
x_text_orange = x_mid_orange;
plot([x_text_orange, x_mid_orange], [y_text_orange, y_mid_orange], 'r-', 'LineWidth', 1.5);
plot(x_mid_orange, y_mid_orange, 'rv', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', 'MarkerSize', 8);
text(x_text_orange, y_text_orange, 'slope: 9.76', 'Color', 'r', 'FontSize', 11, ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

% ================= 4. 样式与防遮挡排版 =================
xlim([1e-3, 1.5]);
ylim([1e-16, 1e6]);
xlabel('$h = 1/n$', 'FontSize', 13, 'Interpreter', 'latex');
ylabel('$e_n$', 'FontSize', 13, 'Interpreter', 'latex');
grid on;
set(gca, 'TickDir', 'in', 'FontSize', 11, 'XMinorGrid','on','YMinorGrid','on');
text(0.015, 5e4, '1.46, 1.87, 1.97, 1.99, 2.00, 2.00, 2.00, 2.00', 'FontSize', 11);
text(0.15, 1e-12, ' 5.31, 7.50, 9.76', 'FontSize', 11);
text(0.1, 1e-6, '$e_n = |\pi - \pi_n|$', 'Interpreter', 'latex', 'FontSize', 14);
hold off;