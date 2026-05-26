# 目标检测模块 (Object Detection)

基于 TensorFlow 和 OpenCV DNN 的目标检测模块，支持多种预训练模型格式。

## 项目结构

```
目标检测/
├── __init__.py              # 模块初始化文件
├── object_detection.py      # 主检测模块 (OpenCV DNN + TensorFlow TFLite)
├── tflite_detector.py       # TFLite专用检测器
├── test_detection.py        # 测试脚本
├── readme.md                # 本文档
└── models/                  # 模型文件目录
    ├── yolo/                # YOLO模型
    │   ├── yolov3.cfg       # YOLOv3配置文件
    │   └── yolov3.weights   # YOLOv3权重文件 (需自行下载)
    └── efficientdet_lite0.tflite  # EfficientDet-Lite0 TFLite模型
```

## 功能特性

- **多框架支持**: OpenCV DNN、TensorFlow Lite
- **多模型格式**: TensorFlow (.pb, .tflite)、Caffe (.caffemodel)、Darknet/YOLO (.weights, .cfg)、ONNX (.onnx)
- **COCO 80类检测**: person、bicycle、car、dog、cat 等常见目标
- **批量处理**: 支持目录批量检测
- **结果可视化**: 自动在图片上绘制检测框和标签

## 安装依赖

```bash
pip install tensorflow opencv-python pillow numpy
```

## 快速开始

### 1. 下载预训练模型

#### 方案一: YOLOv3 (推荐)

1. 下载权重文件: https://pjreddie.com/media/files/yolov3.weights
2. 下载配置文件: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
3. 保存到: `目标检测/models/yolo/`

#### 方案二: EfficientDet-Lite0 (TFLite)

1. 下载模型: https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/int8/2
2. 保存到: `目标检测/models/`

### 2. 基本使用

```python
from 目标检测 import object_detection

# 单张图片检测
detections = object_detection.detect_objects("path/to/image.jpg", threshold=0.5)

# 打印检测结果
for det in detections['detections']:
    print(f"{det['class']}: {det['confidence']:.2%}")

# 保存带检测框的图片
object_detection.draw_detection_results("input.jpg", detections, "output.jpg")
```

### 3. 批量检测

```python
# 批量检测目录中的所有图片
results = object_detection.detect_batch(
    "image_dir/",                    # 输入目录
    output_dir="output/",             # 输出目录
    threshold=0.5                     # 置信度阈值
)

# 保存结果到JSON
object_detection.save_results_to_json(results, "results.json")
```

### 4. 使用不同的模型

```python
# 使用YOLO模型
detector = object_detection.ObjectDetector(
    model_path="models/yolo/yolov3.weights",
    config_path="models/yolo/yolov3.cfg",
    framework='yolo'
)

# 使用TFLite模型
detector = object_detection.ObjectDetector(
    model_path="models/efficientdet_lite0.tflite",
    framework='tflite'
)
```

## API 参考

### detect_objects(image_path, model_path=None, threshold=0.5, framework='auto')

单张图片目标检测

**参数:**
- `image_path`: 图片路径
- `model_path`: 模型文件路径 (可选)
- `threshold`: 置信度阈值，默认0.5
- `framework`: 检测框架 ('auto', 'opencv', 'tensorflow', 'tflite')

**返回:**
```python
{
    'image_path': 'path/to/image.jpg',
    'total_objects': 3,
    'detections': [
        {
            'class': 'person',
            'class_id': 1,
            'confidence': 0.95,
            'bbox': {'xmin': 100, 'ymin': 50, 'xmax': 200, 'ymax': 300}
        },
        ...
    ]
}
```

### draw_detection_results(image_path, detections, output_path=None)

在图片上绘制检测结果

**参数:**
- `image_path`: 原始图片路径
- `detections`: 检测结果字典
- `output_path`: 输出图片路径 (可选，默认原文件名+_detected)

### detect_batch(image_dir, output_dir=None, model_path=None, threshold=0.5, framework='auto')

批量检测目录中的图片

**参数:**
- `image_dir`: 图片目录路径
- `output_dir`: 输出目录 (可选)
- `model_path`: 模型路径 (可选)
- `threshold`: 置信度阈值
- `framework`: 检测框架

**返回:** 检测结果列表

### save_results_to_json(results, output_path)

保存检测结果到JSON文件

**参数:**
- `results`: 检测结果列表
- `output_path`: 输出JSON文件路径

## 支持的检测类别 (COCO 80类)

| 类别 | 说明 |
|------|------|
| person | 人 |
| bicycle, car, motorcycle, airplane, bus, train, truck, boat | 交通工具 |
| dog, cat, bird, horse, sheep, cow, elephant, bear, zebra, giraffe | 动物 |
| traffic light, stop sign, parking meter, bench | 街道设施 |
| bottle, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake | 食物/餐具 |
| chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase | 家居/物品 |

## 运行测试

```bash
python 目标检测/test_detection.py
```

## 注意事项

1. 首次使用需要下载预训练模型文件
2. YOLOv3模型文件较大 (~240MB)，请确保网络连接稳定
3. 配置文件 `yolov3.cfg` 已包含在 `models/yolo/` 目录中
4. 建议使用Python 3.8或更高版本

## 技术说明

- **OpenCV DNN**: 使用OpenCV的Deep Neural Networks模块加载各种模型格式
- **TensorFlow Lite**: 使用TensorFlow Lite Interpreter进行轻量级推理
- **非极大值抑制(NMS)**: 用于去除重复检测框
- **COCO数据集**: 80类常见目标的标准数据集

## 许可证

本项目仅供学习研究使用。
