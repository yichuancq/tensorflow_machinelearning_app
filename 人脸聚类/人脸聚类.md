# TensorFlow 机器学习应用

### TensorFlow 电商商品自动打标 + 以图搜图

---

## 代码逻辑说明

### 1. 核心架构
本项目使用 **MobileNetV2** 预训练模型实现两个功能：
- **特征提取模型** (`base_model`)：去除了分类层，输出 1280 维特征向量，用于以图搜图
- **分类模型** (`classify_model`)：完整模型，用于 ImageNet 1000 类分类

### 2. 主要流程

```
商品图片 → 图片预处理(224x224) → MobileNetV2特征提取 → 商品库索引
                                                    ↓
查询图片 → 特征提取 → 余弦相似度计算 → 返回Top-N相似商品
```

### 3. 关键函数

| 函数 | 功能 |
|------|------|
| `load_and_preprocess()` | 加载图片并缩放到 224x224，执行 MobileNetV2 预处理 |
| `extract_feature()` | 提取图片的 1280 维特征向量 |
| `auto_tag_product()` | 获取 ImageNet Top-5 预测结果，匹配商品类别标签 |
| `build_product_database()` | 遍历商品文件夹，构建商品索引库 (JSON) |
| `search_similar_product()` | 计算余弦相似度，返回最相似商品 |

### 4. 相似度计算
使用 **余弦相似度**：`sim = dot(q_feat, feat) / (||q_feat|| * ||feat||)`

---

## TensorFlow 开发注意事项

### 1. Windows GPU 支持
```
⚠️ 警告：TensorFlow >= 2.11 在原生 Windows 上不支持 GPU 加速
即使安装了 CUDA/cuDNN，GPU 也不会被使用。
解决方案：
- 推荐：使用 WSL2 (Windows Subsystem for Linux)
- 替代方案：安装 TensorFlow-DirectML 插件
```

### 2. 模型加载
- `weights="imagenet"`：首次运行会自动下载预训练权重
- `include_top=False`：不包含原始分类层，自定义 pooling
- `pooling="avg"`：输出全局平均池化后的特征向量

### 3. 图片预处理
必须使用对应的 `preprocess_input()`，MobileNetV2 使用的是 `(-1, 1)` 范围的标准化。

### 4. 预测解码
```python
decode_predictions(preds, top=5)  # 返回 [(class_id, label, score), ...]
```

### 5. 商品标签匹配逻辑
- 将 ImageNet 标签（如 `fur_coat`）与关键词列表进行子串匹配
- 匹配到则添加对应商品类别，最多返回 3 个标签

---

## 运行示例

```text
       TensorFlow 电商商品自动打标 + 以图搜图
==================================================
正在构建商品索引...
处理：coat.jpg
【模型识别】: ['fur_coat', 'wool', 'cardigan', 'velvet', 'sweatshirt']
处理：dress.jpg
【模型识别】: ['hoopskirt', 'Windsor_tie', 'overskirt', 'apron', 'gown']

✅ 商品库构建完成！共 2 个商品

===== 以图搜图测试 =====

🔍 搜索结果（Top 5）：
1. coat.jpg | 标签：['衣服'] | 相似度：1.0
2. dress.jpg | 标签：['衣服'] | 相似度：0.36
```