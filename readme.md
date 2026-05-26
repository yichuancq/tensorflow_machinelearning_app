# TensorFlow 机器学习应用项目

基于 TensorFlow 的机器学习应用集合，包含商品推荐系统、目标检测、人脸识别、智能相册等功能。

## 项目结构

```
tensorflow_machinelearning_app/
├── my_photos/                  # 测试图片目录
│   ├── coat.jpg
│   └── dress.jpg
├── shop_products/              # 商品图片目录
│   ├── coat.jpg
│   └── dress.jpg
├── 人脸聚类/                    # 人脸聚类模块
│   ├── face.py
│   ├── face2.py
│   └── 人脸聚类.md
├── 推荐系统/                    # 商品推荐系统
│   ├── recommender.py
│   ├── recommendation_service.py
│   └── 推荐系统.md
├── 智能相册/                    # 智能相册（以图搜图）
│   └── product_search2.py
├── 目标检测/                    # 目标检测模块
│   ├── models/
│   │   └── yolo/
│   │       └── yolov3.cfg
│   ├── __init__.py
│   ├── object_detection.py
│   ├── readme.md
│   ├── test_detection.py
│   └── tflite_detector.py
├── album_index.json            # 相册索引
├── product_index.json          # 商品索引
└── readme.md                   # 项目总览
```

## 功能模块

### 1. 商品推荐系统

**位置**: `推荐系统/`

**功能描述**:
- 基于神经协同过滤（Neural Collaborative Filtering）的商品推荐模型
- 结合 GMF（广义矩阵分解）和 MLP（多层感知器）
- 支持用户行为数据训练和基于图像特征的相似商品推荐

**核心组件**:
- `ProductRecommender`: 协同过滤推荐模型
- `ProductFeatureExtractor`: 基于 MobileNetV2 的图像特征提取器
- `RecommendationService`: 推荐服务封装，提供 REST API

**使用方法**:
```python
from recommendation_service import RecommendationService

service = RecommendationService()
service.load_products()
service.init_recommender(num_users=100, num_products=50)

# 训练模型
service.train_recommender(user_ids, product_ids, ratings, epochs=10)

# 获取推荐
recommendations = service.get_recommendations(user_id=0, top_k=5)
```

**详细文档**: [推荐系统.md](推荐系统/推荐系统.md)

---

### 2. 目标检测

**位置**: `目标检测/`

**功能描述**:
- 基于 TensorFlow 和 OpenCV DNN 的目标检测模块
- 支持多种预训练模型格式（YOLO、TensorFlow Lite、Caffe、ONNX）
- COCO 80类目标检测，支持批量处理和结果可视化

**核心特性**:
- 多框架支持：OpenCV DNN、TensorFlow Lite
- 多模型格式：YOLOv3、EfficientDet-Lite0
- 自动绘制检测框和标签
- 支持批量检测整个目录

**使用方法**:
```python
from 目标检测 import object_detection

# 单张图片检测
detections = object_detection.detect_objects("image.jpg", threshold=0.5)

# 批量检测
results = object_detection.detect_batch("input_dir/", output_dir="output/")
```

**详细文档**: [目标检测/readme.md](目标检测/readme.md)

---

### 3. 人脸聚类

**位置**: `人脸聚类/`

**功能描述**:
- 基于深度学习的人脸聚类功能
- 支持人脸检测、特征提取和聚类分析
- 可用于相册人脸分组、人脸搜索等场景

**核心文件**:
- `face.py`: 人脸检测和特征提取
- `face2.py`: 人脸聚类算法实现

**详细文档**: [人脸聚类/人脸聚类.md](人脸聚类/人脸聚类.md)

---

### 4. 智能相册（以图搜图）

**位置**: `智能相册/`

**功能描述**:
- 基于 MobileNetV2 预训练模型实现商品自动打标 + 以图搜图
- 1280维特征向量提取，余弦相似度匹配
- 自动识别商品类别标签

**核心功能**:
- `extract_feature()`: 提取图片特征向量
- `auto_tag_product()`: 自动生成商品标签
- `search_similar_product()`: 以图搜图，返回相似商品
- `build_product_database()`: 构建商品索引库

**使用方法**:
```python
from 智能相册.product_search2 import build_product_database, search_similar_product

# 构建商品索引
build_product_database()

# 搜索相似商品
results = search_similar_product("query.jpg", top_n=5)
```

---

## 环境要求

### 基础依赖
```bash
pip install tensorflow opencv-python pillow numpy flask
```

### 版本要求
- Python >= 3.8
- TensorFlow >= 2.0

### 可选依赖
```bash
# Flask（用于推荐系统API）
pip install flask
```

## 快速开始

### 1. 推荐系统
```bash
cd 推荐系统
python recommendation_service.py
```

### 2. 目标检测
```bash
cd 目标检测
python test_detection.py
```

### 3. 智能相册
```bash
cd 智能相册
python product_search2.py
```

## 模型下载说明

### YOLOv3 模型（目标检测）
1. 下载权重: https://pjreddie.com/media/files/yolov3.weights
2. 保存到: `目标检测/models/yolo/yolov3.weights`

### EfficientDet-Lite0（目标检测）
1. 下载: https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/int8/2
2. 保存到: `目标检测/models/efficientdet_lite0.tflite`

### MobileNetV2（推荐系统、智能相册）
- 首次运行自动下载预训练权重
- weights="imagenet"

## 技术栈

| 模块 | 框架/模型 | 用途 |
|------|-----------|------|
| 推荐系统 | TensorFlow/Keras, MobileNetV2 | 协同过滤、图像特征提取 |
| 目标检测 | TensorFlow Lite, OpenCV DNN, YOLOv3 | 多框架目标检测 |
| 人脸聚类 | TensorFlow | 人脸检测和聚类 |
| 智能相册 | MobileNetV2 | 特征提取、以图搜图 |

## 注意事项

1. **Windows GPU 支持**: TensorFlow >= 2.11 在原生 Windows 上不支持 GPU 加速，建议使用 WSL2
2. **模型文件**: 部分模型文件较大，首次使用需要下载
3. **图片格式**: 支持 JPG、PNG 格式，建议尺寸 224x224
4. **数据准备**: 运行前请确保图片目录存在并包含测试图片

## 许可证

本项目仅供学习研究使用。