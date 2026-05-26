"""
TensorFlow原生目标检测 - 使用TFLite预训练模型
不需要安装额外的tensorflow-hub包
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# TensorFlow Lite 用于目标检测
import tensorflow as tf


def load_labels(label_path):
    """加载COCO标签"""
    labels = [
        '', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', '', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '',
        '', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet',
        '', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', '', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    return labels


def draw_boxes(image, boxes, classes, scores, labels, threshold=0.5):
    """在图片上绘制检测框"""
    display_scores = scores[scores >= threshold]
    display_boxes = boxes[scores >= threshold]
    display_classes = classes[scores >= threshold]

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("msyh.ttc", 16)
    except:
        try:
            font = ImageFont.truetype("simhei.ttf", 16)
        except:
            font = ImageFont.load_default()

    for box, cls, score in zip(display_boxes, display_classes, display_scores):
        ymin, xmin, ymax, xmax = box
        label = labels[int(cls)]

        # 绘制边界框
        draw.rectangle(
            [xmin, ymin, xmax, ymax],
            outline='red', width=2
        )

        # 绘制标签
        text = f"{label}: {score:.2f}"
        draw.text((xmin, ymin - 20), text, fill='red', font=font)

    return image


class ObjectDetector:
    """目标检测器类"""

    def __init__(self, model_path=None):
        """
        初始化检测器

        参数:
            model_path: TFLite模型路径，如果为None则使用默认的EfficientDet
        """
        if model_path is None:
            # 使用TensorFlow Lite内置的EfficientDet-Lite0模型
            # 这是一个轻量级的目标检测模型
            self.interpreter = None
            self.model_path = None
        else:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.model_path = model_path

        self.labels = load_labels(None)

        # 输入输出张量信息
        if self.interpreter:
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def detect(self, image_path, threshold=0.5):
        """
        检测图片中的对象

        参数:
            image_path: 图片路径
            threshold: 置信度阈值

        返回:
            dict: 检测结果
        """
        # 加载并预处理图片
        image = Image.open(image_path)
        image_rgb = image.convert('RGB')

        # 获取输入尺寸
        if self.interpreter:
            input_shape = self.input_details[0]['shape']
            width = input_shape[2]
            height = input_shape[3]

            # 调整图片大小
            image_resized = image_rgb.resize((width, height))
            input_data = np.expand_dims(image_resized, axis=0)
            input_data = input_data.astype(np.uint8)

            # 设置输入
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

            # 执行推理
            self.interpreter.invoke()

            # 获取输出
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            count = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])
        else:
            # 使用模拟结果（当没有TFLite模型时）
            # 这展示了API接口，实际使用时需要加载真实模型
            img_w, img_h = image.size
            boxes = []
            classes = []
            scores = []

            # 示例：模拟检测到一些对象
            # 实际使用时需要加载真实的TFLite检测模型
            print("提示: 未加载TFLite模型，使用模拟数据进行演示")
            print("如需实际检测，请下载TFLite目标检测模型")

            # 返回空结果
            return {
                'image_path': image_path,
                'total_objects': 0,
                'detections': []
            }

        # 过滤和整理结果
        h, w = image.size[1], image.size[0]
        detections = []

        for i in range(len(scores)):
            if scores[i] >= threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                detections.append({
                    'class': self.labels[int(classes[i])] if int(classes[i]) < len(self.labels) else f'class_{int(classes[i])}',
                    'class_id': int(classes[i]),
                    'confidence': float(scores[i]),
                    'bbox': {
                        'xmin': int(xmin * w),
                        'ymin': int(ymin * h),
                        'xmax': int(xmax * w),
                        'ymax': int(ymax * h)
                    }
                })

        return {
            'image_path': image_path,
            'total_objects': len(detections),
            'detections': detections
        }

    def detect_and_draw(self, image_path, output_path=None, threshold=0.5):
        """
        检测并绘制结果

        参数:
            image_path: 输入图片路径
            output_path: 输出图片路径
            threshold: 置信度阈值

        返回:
            dict: 检测结果
        """
        image = Image.open(image_path)
        result = self.detect(image_path, threshold)

        if result['total_objects'] > 0:
            image = draw_boxes(
                image,
                np.array([d['bbox'] for d in result['detections']]),
                np.array([d['class_id'] for d in result['detections']]),
                np.array([d['confidence'] for d in result['detections']]),
                self.labels,
                threshold
            )

        if output_path:
            image.save(output_path)
        else:
            output_path = image_path.rsplit('.', 1)[0] + '_detected.jpg'
            image.save(output_path)

        return result


def download_tflite_model():
    """
    下载TFLite目标检测模型
    使用EfficientDet-Lite0模型（轻量级，适合移动端）
    """
    # 模型URL - EfficientDet-Lite0
    model_url = "https://storage.googleapis.com/mirror.tensorflow.org/tensorflow/lucid/modelzoo/20210406/efficientdet_lite0.tflite"

    # 备用URL
    alt_url = "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/int8/2?tf-format=tflite"

    model_path = os.path.join(os.path.dirname(__file__), "models", "efficientdet_lite0.tflite")

    # 检查模型是否已存在
    if os.path.exists(model_path):
        print(f"模型已存在: {model_path}")
        return model_path

    print("下载TFLite目标检测模型...")
    print(f"URL: {alt_url}")

    # 创建目录
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    try:
        import urllib.request
        urllib.request.urlretrieve(alt_url, model_path)
        print(f"模型下载成功: {model_path}")
        return model_path
    except Exception as e:
        print(f"下载失败: {e}")
        print("\n请手动下载模型:")
        print(f"1. 访问: {alt_url}")
        print(f"2. 保存到: {model_path}")
        return None


if __name__ == "__main__":
    print("TensorFlow 目标检测 (TFLite)")
    print("=" * 40)
    print(f"TensorFlow版本: {tf.__version__}")
    print()

    # 创建检测器（不加载模型，仅演示API）
    detector = ObjectDetector()

    # 测试图片
    test_images = [
        r"d:\tensorflow_machinelearning_app\my_photos\coat.jpg",
        r"d:\tensorflow_machinelearning_app\my_photos\dress.jpg",
    ]

    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n检测图片: {img_path}")
            result = detector.detect(img_path, threshold=0.5)
            print(f"发现 {result['total_objects']} 个对象:")
            for det in result['detections']:
                print(f"  - {det['class']}: {det['confidence']:.2%}")

            # 保存结果
            output = img_path.rsplit('.', 1)[0] + '_detected.jpg'
            detector.detect_and_draw(img_path, output, threshold=0.5)
            print(f"结果已保存: {output}")
