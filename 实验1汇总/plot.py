import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# 硬编码所有实测数据（1KB、10KB、100KB、1MB）
# --------------------------
# 文本长度（字节）
text_lengths = [1024, 10240, 102400, 1048576]

# BPB（压缩效率，越低越好）
gzip_bpb   = [2.8203, 2.9922, 2.8997, 2.8312]
xz_bpb     = [3.2188, 2.9281, 2.6372, 2.3099]
zstd_bpb   = [2.7578, 2.9805, 2.8029, 2.5491]
bzip2_bpb  = [3.1328, 2.9875, 2.5008, 2.2441]
gpt2_bpb       = [1.4688, 1.8195, 1.6636, 1.7246]
gpt22 = [1.875,2.0187,1.6145,1.6007]
tiny =[0.9375,1.2641,1.2897,1.2707]

# 压缩吞吐量 MB/s
gzip_tp    = [0.11, 0.98, 4.14, 4.78]
xz_tp      = [0.04, 0.34, 1.58, 2.12]
zstd_tp    = [0.13, 0.56, 4.29, 15.33]
bzip2_tp   = [0.13, 0.81, 4.17, 5.59]

# 样式：4种算法 + 4种长度点形状
colors     = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # gzip, xz, zstd, bzip2
markers    = ['o', 's', '^', 'D']                          # 1KB,10KB,100KB,1MB
labels     = ['1KB', '10KB', '100KB', '1MB']

# ==========================
# 图 1：文本长度 — 压缩效率（所有长度）
# ==========================
plt.figure(figsize=(10, 6))

plt.plot(text_lengths, gzip_bpb,  color=colors[0], linewidth=2.5, marker='o', markersize=7, label='gzip')
plt.plot(text_lengths, xz_bpb,    color=colors[1], linewidth=2.5, marker='s', markersize=7, label='xz')
plt.plot(text_lengths, zstd_bpb,  color=colors[2], linewidth=2.5, marker='^', markersize=7, label='zstd')
plt.plot(text_lengths, bzip2_bpb, color=colors[3], linewidth=2.5, marker='D', markersize=7, label='bzip2')
plt.plot(text_lengths, gpt2_bpb,  color='purple', linestyle='--', linewidth=2.5, marker='*', markersize=8, label='gpt2-gptzip')
plt.plot(text_lengths, gpt22,  color='black', linestyle='--', linewidth=2.5, marker='*', markersize=8, label='gpt2-finezip')
plt.plot(text_lengths, tiny,  color='yellow', linestyle='--', linewidth=2.5, marker='*', markersize=8, label='tinyllama-gptzip')



plt.xlabel('Text Length (Bytes)', fontsize=12)
plt.ylabel('BPB (Compression Efficiency)', fontsize=12)
plt.title('Text Length — Compression Efficiency', fontsize=14, fontweight='bold')
plt.legend(fontsize=9, loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('length_vs_efficiency.png', dpi=300)
plt.show()

# # ==========================
# # 图 2：压缩效率—吞吐量（4个长度：不同形状+颜色）
# # ==========================
# plt.figure(figsize=(10, 6))

# # 绘制 4个长度点：颜色区分算法，形状区分长度
# data_sets = [
#     (gzip_bpb, gzip_tp, colors[0], 'gzip'),
#     (xz_bpb, xz_tp, colors[1], 'xz'),
#     (zstd_bpb, zstd_tp, colors[2], 'zstd'),
#     (bzip2_bpb, bzip2_tp, colors[3], 'bzip2')
# ]

# for i, (bpb_list, tp_list, color, name) in enumerate(data_sets):
#     for idx in range(4):
#         plt.scatter(bpb_list[idx], tp_list[idx], 
#                     s=160, color=color, marker=markers[idx], 
#                     edgecolors='black', linewidth=1)

# # ====================== ✅ 关键：手动创建清晰图例（方法+长度）
# from matplotlib.lines import Line2D
# legend_elements = []

# # 1. 先加 方法（颜色）
# legend_elements.append(Line2D([0], [0], marker='o', color=colors[0], linestyle='', markersize=10, label='gzip'))
# legend_elements.append(Line2D([0], [0], marker='o', color=colors[1], linestyle='', markersize=10, label='xz'))
# legend_elements.append(Line2D([0], [0], marker='o', color=colors[2], linestyle='', markersize=10, label='zstd'))
# legend_elements.append(Line2D([0], [0], marker='o', color=colors[3], linestyle='', markersize=10, label='bzip2'))

# # 2. 再加 长度（形状）
# legend_elements.append(Line2D([0], [0], marker=markers[0], color='gray', linestyle='', markersize=10, label='1KB'))
# legend_elements.append(Line2D([0], [0], marker=markers[1], color='gray', linestyle='', markersize=10, label='10KB'))
# legend_elements.append(Line2D([0], [0], marker=markers[2], color='gray', linestyle='', markersize=10, label='100KB'))
# legend_elements.append(Line2D([0], [0], marker=markers[3], color='gray', linestyle='', markersize=10, label='1MB'))

# # 显示图例
# plt.legend(handles=legend_elements, loc='best', fontsize=11)

# plt.xlabel('BPB (Compression Efficiency)', fontsize=12)
# plt.ylabel('Compression Throughput (MB/s)', fontsize=12)
# plt.title('Compression Efficiency — Throughput', fontsize=14, fontweight='bold')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('efficiency_vs_throughput.png', dpi=300)
# plt.show()
