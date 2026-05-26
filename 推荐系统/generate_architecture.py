"""
生成推荐系统架构图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('TensorFlow 商品推荐系统架构', fontsize=18, fontweight='bold', pad=20)

# 颜色定义
colors = {
    'data': '#E3F2FD',
    'model': '#E8F5E9',
    'output': '#FFF3E0',
    'arrow': '#455A64'
}

# ===== 用户行为数据 =====
user_data = FancyBboxPatch((0.5, 6.5), 2.5, 2, boxstyle="round,pad=0.1",
                            facecolor=colors['data'], edgecolor='#1976D2', linewidth=2)
ax.add_patch(user_data)
ax.text(1.75, 7.8, '用户行为数据', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(1.75, 7.3, '(user_id,', ha='center', va='center', fontsize=9)
ax.text(1.75, 7.0, ' product_id,', ha='center', va='center', fontsize=9)
ax.text(1.75, 6.7, ' rating)', ha='center', va='center', fontsize=9)

# ===== 商品图像数据 =====
product_data = FancyBboxPatch((0.5, 1), 2.5, 2, boxstyle="round,pad=0.1",
                               facecolor=colors['data'], edgecolor='#1976D2', linewidth=2)
ax.add_patch(product_data)
ax.text(1.75, 2.5, '商品图像数据', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(1.75, 2.0, '(商品图片)', ha='center', va='center', fontsize=9)

# ===== ProductRecommender =====
recommender = FancyBboxPatch((4.5, 5), 3.5, 4, boxstyle="round,pad=0.1",
                              facecolor=colors['model'], edgecolor='#388E3C', linewidth=2)
ax.add_patch(recommender)
ax.text(6.25, 8.3, 'ProductRecommender', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(6.25, 7.8, '(协同过滤模型)', ha='center', va='center', fontsize=9)

# GMF 模块
gmf_box = FancyBboxPatch((4.8, 6.5), 2.9, 1, boxstyle="round,pad=0.05",
                          facecolor='white', edgecolor='#4CAF50', linewidth=1)
ax.add_patch(gmf_box)
ax.text(6.25, 7.0, 'GMF 模块', ha='center', va='center', fontsize=9)

# MLP 模块
mlp_box = FancyBboxPatch((4.8, 5.3), 2.9, 1, boxstyle="round,pad=0.05",
                          facecolor='white', edgecolor='#4CAF50', linewidth=1)
ax.add_patch(mlp_box)
ax.text(6.25, 5.8, 'MLP 模块', ha='center', va='center', fontsize=9)

# 融合层
fusion_box = FancyBboxPatch((5.5, 4.5), 1.5, 0.6, boxstyle="round,pad=0.05",
                             facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=1)
ax.add_patch(fusion_box)
ax.text(6.25, 4.8, '融合层', ha='center', va='center', fontsize=8)

# ===== ProductFeatureExtractor =====
feature_ext = FancyBboxPatch((4.5, 0.5), 3.5, 2.5, boxstyle="round,pad=0.1",
                               facecolor=colors['model'], edgecolor='#388E3C', linewidth=2)
ax.add_patch(feature_ext)
ax.text(6.25, 2.5, 'ProductFeatureExtractor', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(6.25, 2.0, '(图像特征提取)', ha='center', va='center', fontsize=9)

# MobileNetV2
mobilenet_box = FancyBboxPatch((4.8, 0.8), 2.9, 0.8, boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor='#4CAF50', linewidth=1)
ax.add_patch(mobilenet_box)
ax.text(6.25, 1.2, 'MobileNetV2', ha='center', va='center', fontsize=9)

# 相似度计算
sim_box = FancyBboxPatch((5.5, 0.8), 1.5, 0.6, boxstyle="round,pad=0.05",
                          facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=1)
ax.add_patch(sim_box)
ax.text(6.25, 1.1, '相似度计算', ha='center', va='center', fontsize=8)

# ===== 推荐结果输出 =====
output1 = FancyBboxPatch((9.5, 5.5), 2.5, 2, boxstyle="round,pad=0.1",
                          facecolor=colors['output'], edgecolor='#F57C00', linewidth=2)
ax.add_patch(output1)
ax.text(10.75, 6.8, '推荐结果', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(10.75, 6.3, '(top-k 商品)', ha='center', va='center', fontsize=9)

# ===== 相似商品推荐 =====
output2 = FancyBboxPatch((9.5, 1), 2.5, 2, boxstyle="round,pad=0.1",
                          facecolor=colors['output'], edgecolor='#F57C00', linewidth=2)
ax.add_patch(output2)
ax.text(10.75, 2.3, '相似商品', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(10.75, 1.8, '(余弦相似度)', ha='center', va='center', fontsize=9)

# ===== 箭头 =====
# 用户数据 -> Recommender
ax.annotate('', xy=(4.5, 7), xytext=(3, 7),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
# 商品数据 -> FeatureExtractor
ax.annotate('', xy=(4.5, 1.5), xytext=(3, 1.5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
# Recommender -> 输出1
ax.annotate('', xy=(9.5, 6.5), xytext=(8, 6.5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
# FeatureExtractor -> 输出2
ax.annotate('', xy=(9.5, 2), xytext=(8, 2),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

# ===== 图例 =====
legend_items = [
    (colors['data'], '#1976D2', '数据输入'),
    (colors['model'], '#388E3C', '核心模型'),
    (colors['output'], '#F57C00', '输出结果'),
]
for i, (fc, ec, label) in enumerate(legend_items):
    legend_box = FancyBboxPatch((11, 8.5 - i * 0.5), 0.3, 0.3, boxstyle="round,pad=0.02",
                                 facecolor=fc, edgecolor=ec, linewidth=1)
    ax.add_patch(legend_box)
    ax.text(11.5, 8.65 - i * 0.5, label, ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('d:/tensorflow_machinelearning_app/推荐系统/architecture.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('架构图已保存至: d:/tensorflow_machinelearning_app/推荐系统/architecture.png')
