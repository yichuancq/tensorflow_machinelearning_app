"""
TensorFlow/OpenCV 目标检测模块
使用OpenCV DNN模块加载预训练模型进行目标检测

支持多种模型格式:
- TensorFlow (.pb, .tflite)
- Caffe (.caffemodel, .prototxt)
- Darknet/YOLO (.weights, .cfg)
- ONNX (.onnx)
"""

import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# COCO类别名称映射 (80类)
COCO_CLASSES = {
    0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign',
    14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog',
    19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella',
    31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis',
    36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 45: 'wine glass', 46: 'cup',
    47: 'fork', 48: 'knife', 49: 'spoon', 50: 'bowl', 51: 'banana',
    52: 'apple', 53: 'sandwich', 54: 'orange', 55: 'broccoli',
    56: 'carrot', 57: 'hot dog', 58: 'pizza', 59: 'donut', 60: 'cake',
    61: 'chair', 62: 'couch', 63: 'potted plant', 64: 'bed',
    65: 'dining table', 66: 'toilet', 67: 'tv', 68: 'laptop', 69: 'mouse',
    70: 'remote', 71: 'keyboard', 72: 'cell phone', 73: 'microwave',
    74: 'oven', 75: 'toaster', 76: 'sink', 77: 'refrigerator', 78: 'book',
    79: 'clock', 80: 'vase', 81: 'scissors', 82: 'teddy bear',
    83: 'hair drier', 84: 'toothbrush'
}

# YOLO类别 (COCO 80类 - 无背景类)
YOLO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


class OpenCVObjectDetector:
    """使用OpenCV DNN的目标检测器"""

    def __init__(self, model_path=None, config_path=None, framework='tensorflow'):
        """
        初始化OpenCV DNN检测器

        参数:
            model_path: 模型文件路径 (.pb, .caffemodel, .weights, .onnx, .tflite)
            config_path: 配置文件路径 (.prototxt, .cfg)
            framework: 框架类型 ('tensorflow', 'caffe', 'darknet', 'onnx', 'yolo')
        """
        self.net = None
        self.model_path = model_path
        self.config_path = config_path
        self.framework = framework
        self.classes = YOLO_CLASSES if framework in ['darknet', 'yolo'] else COCO_CLASSES

        if model_path and os.path.exists(model_path):
            self._load_model()

    def _load_model(self):
        """加载OpenCV DNN模型"""
        try:
            if self.framework == 'tensorflow':
                self.net = cv2.dnn.readNetFromTensorflow(self.model_path, self.config_path)
            elif self.framework == 'caffe':
                self.net = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)
            elif self.framework == 'darknet':
                self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.model_path)
            elif self.framework == 'onnx':
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
            elif self.framework == 'yolo':
                # YOLO需要配置文件
                self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.model_path)
            else:
                print(f"不支持的框架: {self.framework}")
                return

            print(f"{self.framework} 模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.net = None

    def detect(self, image_path, threshold=0.5):
        """
        检测图片中的对象

        参数:
            image_path: 图片路径
            threshold: 置信度阈值

        返回:
            dict: 检测结果
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            return {
                'image_path': image_path,
                'total_objects': 0,
                'detections': [],
                'error': '无法读取图片'
            }

        h, w = image.shape[:2]

        if self.net is None:
            return {
                'image_path': image_path,
                'total_objects': 0,
                'detections': [],
                'message': '未加载模型'
            }

        try:
            if self.framework in ['darknet', 'yolo']:
                return self._detect_yolo(image, h, w, threshold)
            else:
                return self._detect_general(image, h, w, threshold)
        except Exception as e:
            print(f"检测出错: {e}")
            return {
                'image_path': image_path,
                'total_objects': 0,
                'detections': [],
                'error': str(e)
            }

    def _detect_general(self, image, h, w, threshold):
        """通用DNN检测"""
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > threshold:
                class_id = int(detections[0, 0, i, 1])
                bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = bbox.astype(int)

                results.append({
                    'class': self.classes.get(class_id, f'class_{class_id}'),
                    'class_id': class_id,
                    'confidence': float(confidence),
                    'bbox': {
                        'xmin': int(x1),
                        'ymin': int(y1),
                        'xmax': int(x2),
                        'ymax': int(y2)
                    }
                })

        return {
            'image_path': image_path,
            'total_objects': len(results),
            'detections': results
        }

    def _detect_yolo(self, image, h, w, threshold):
        """YOLO格式检测"""
        # 创建输入blob
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        # 获取输出层名称
        layer_names = self.net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[0]]

        # 前向传播
        outputs = self.net.forward(output_layers)

        # 解析输出
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > threshold:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    bw = int(detection[2] * w)
                    bh = int(detection[3] * h)

                    x = int(center_x - bw / 2)
                    y = int(center_y - bh / 2)

                    boxes.append([x, y, bw, bh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 非极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.4)

        results = []
        if len(indices) > 0:
            for i in indices.flatten() if isinstance(indices, np.ndarray) else indices:
                x, y, bw, bh = boxes[i]
                results.append({
                    'class': self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else f'class_{class_ids[i]}',
                    'class_id': int(class_ids[i]),
                    'confidence': confidences[i],
                    'bbox': {
                        'xmin': int(x),
                        'ymin': int(y),
                        'xmax': int(x + bw),
                        'ymax': int(y + bh)
                    }
                })

        return {
            'image_path': image_path,
            'total_objects': len(results),
            'detections': results
        }


def download_yolo_model():
    """
    尝试下载YOLO模型 (COCO数据集80类)

    返回:
        tuple: (weights_path, config_path) 或 (None, None)
    """
    model_dir = os.path.join(os.path.dirname(__file__), "models", "yolo")
    os.makedirs(model_dir, exist_ok=True)

    weights_path = os.path.join(model_dir, "yolov3.weights")
    config_path = os.path.join(model_dir, "yolov3.cfg")

    # 检查是否已存在
    if os.path.exists(weights_path) and os.path.getsize(weights_path) > 20000000:
        print(f"YOLO模型已存在: {weights_path}")
        return weights_path, config_path

    # YOLOv3 COCO权重下载链接
    weights_url = "https://pjreddie.com/media/files/yolov3.weights"
    config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"

    # 尝试下载权重文件
    try:
        print("尝试下载YOLOv3模型权重...")
        import urllib.request
        urllib.request.urlretrieve(weights_url, weights_path)
        print(f"权重文件下载成功: {weights_path} ({os.path.getsize(weights_path)} bytes)")
    except Exception as e:
        print(f"权重下载失败: {e}")
        if os.path.exists(weights_path):
            os.remove(weights_path)
        weights_path = None

    # 下载配置文件
    try:
        print("下载配置文件...")
        import urllib.request
        urllib.request.urlretrieve(config_url, config_path)
        print(f"配置文件下载成功: {config_path}")
    except Exception as e:
        print(f"配置下载失败: {e}")
        config_path = None

    if weights_path and config_path:
        return weights_path, config_path

    return None, None


class ObjectDetector:
    """目标检测器主类 - 自动选择最佳可用方法"""

    def __init__(self, model_path=None, config_path=None, framework='auto'):
        """
        初始化检测器

        参数:
            model_path: 模型路径
            config_path: 配置文件路径
            framework: 检测框架 ('auto', 'opencv', 'tensorflow', 'tflite')
        """
        self.detector = None
        self.framework = framework

        if framework == 'opencv' or framework == 'auto':
            self._init_opencv_detector(model_path, config_path)
        elif framework == 'tensorflow':
            self._init_tensorflow_detector(model_path)
        elif framework == 'tflite':
            self._init_tflite_detector(model_path)

    def _init_opencv_detector(self, model_path, config_path):
        """初始化OpenCV检测器"""
        # 尝试查找本地模型
        if model_path is None:
            model_path = self._find_local_model()

        if model_path and os.path.exists(model_path):
            # 根据文件扩展名确定框架
            ext = os.path.splitext(model_path)[1].lower()
            if ext == '.pb':
                self.detector = OpenCVObjectDetector(model_path, config_path, 'tensorflow')
            elif ext == '.caffemodel':
                self.detector = OpenCVObjectDetector(model_path, config_path, 'caffe')
            elif ext == '.onnx':
                self.detector = OpenCVObjectDetector(model_path, config_path, 'onnx')
            elif ext in ['.weights', '.tflite']:
                self.detector = OpenCVObjectDetector(model_path, config_path, 'yolo' if ext == '.weights' else 'tensorflow')
        else:
            # 尝试下载YOLO模型
            print("未找到本地模型，尝试下载YOLOv3...")
            weights_path, config_path = download_yolo_model()
            if weights_path and config_path:
                self.detector = OpenCVObjectDetector(weights_path, config_path, 'yolo')
            else:
                print("警告: 无法加载任何检测模型")

    def _init_tensorflow_detector(self, model_path):
        """初始化TensorFlow检测器"""
        try:
            import tensorflow as tf
            if model_path and os.path.exists(model_path):
                self.detector = TFLiteDetector(model_path)
            else:
                print("TensorFlow检测器需要模型文件")
        except ImportError:
            print("TensorFlow不可用")

    def _init_tflite_detector(self, model_path):
        """初始化TFLite检测器"""
        self.detector = TFLiteDetector(model_path)

    def _find_local_model(self):
        """查找本地模型"""
        search_dirs = [
            os.path.join(os.path.dirname(__file__), "models"),
            os.path.join(os.path.dirname(__file__), "models", "yolo"),
            r"d:\tensorflow_machinelearning_app\目标检测\models",
        ]

        extensions = ['.pb', '.caffemodel', '.onnx', '.weights', '.tflite']

        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for f in os.listdir(search_dir):
                    if os.path.splitext(f)[1].lower() in extensions:
                        return os.path.join(search_dir, f)

        return None

    def detect(self, image_path, threshold=0.5):
        """检测图片中的对象"""
        if self.detector:
            return self.detector.detect(image_path, threshold)
        return {
            'image_path': image_path,
            'total_objects': 0,
            'detections': [],
            'message': '未初始化检测器'
        }


class TFLiteDetector:
    """TensorFlow Lite目标检测器"""

    def __init__(self, model_path=None):
        import tensorflow as tf

        self.interpreter = None
        if model_path and os.path.exists(model_path):
            try:
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"TFLite模型加载成功: {model_path}")
            except Exception as e:
                print(f"TFLite模型加载失败: {e}")

    def detect(self, image_path, threshold=0.5):
        """使用TFLite模型检测"""
        import tensorflow as tf
        import numpy as np
        from PIL import Image

        image = Image.open(image_path)
        image_rgb = image.convert('RGB')
        image_resized = image_rgb.resize((300, 300))

        input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)

        if self.interpreter:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            count = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])

            h, w = image.size[1], image.size[0]
            detections = []

            for i in range(min(count, len(scores))):
                if scores[i] >= threshold:
                    ymin, xmin, ymax, xmax = boxes[i]
                    detections.append({
                        'class': COCO_CLASSES.get(int(classes[i]), f'class_{int(classes[i])}'),
                        'class_id': int(classes[i]),
                        'confidence': float(scores[i]),
                        'bbox': {
                            'ymin': int(ymin * h),
                            'xmin': int(xmin * w),
                            'ymax': int(ymax * h),
                            'xmax': int(xmax * w)
                        }
                    })

            return {
                'image_path': image_path,
                'total_objects': len(detections),
                'detections': detections
            }

        return {
            'image_path': image_path,
            'total_objects': 0,
            'detections': [],
            'message': '模型未加载'
        }


def detect_objects(image_path, model_path=None, threshold=0.5, framework='auto'):
    """便捷检测函数"""
    detector = ObjectDetector(model_path=model_path, framework=framework)
    return detector.detect(image_path, threshold)


def draw_detection_results(image_path, detections, output_path=None):
    """在图片上绘制检测结果"""
    image = cv2.imread(image_path)
    if image is None:
        image = Image.open(image_path)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    try:
        font = ImageFont.truetype("msyh.ttc", 20)
    except:
        try:
            font = ImageFont.truetype("simhei.ttf", 20)
        except:
            font = ImageFont.load_default()

    for det in detections.get('detections', []):
        bbox = det['bbox']
        label = f"{det['class']} {det['confidence']:.2f}"

        # 绘制边界框
        draw.rectangle(
            [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']],
            outline='red', width=3
        )

        # 绘制标签背景
        text_bbox = draw.textbbox((bbox['xmin'], bbox['ymin']), label, font=font)
        draw.rectangle(text_bbox, fill='red')
        draw.text((bbox['xmin'], bbox['ymin']), label, fill='white', font=font)

    # 转换回OpenCV格式
    result_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    if output_path:
        cv2.imwrite(output_path, result_image)
    else:
        output_path = image_path.rsplit('.', 1)[0] + '_detected.jpg'
        cv2.imwrite(output_path, result_image)

    return output_path


def detect_batch(image_dir, output_dir=None, model_path=None, threshold=0.5, framework='auto'):
    """批量检测"""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    results = []

    detector = ObjectDetector(model_path=model_path, framework=framework)

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(image_dir, filename)
            detections = detector.detect(image_path, threshold)

            if output_dir:
                output_path = os.path.join(output_dir, f'detected_{filename}')
            else:
                output_path = None

            draw_detection_results(image_path, detections, output_path)
            results.append(detections)
            print(f"检测完成: {filename} - 发现 {detections['total_objects']} 个对象")

    return results


def save_results_to_json(results, output_path):
    """保存结果到JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print("TensorFlow/OpenCV 目标检测模块")
    print("=" * 50)
    print(f"OpenCV版本: {cv2.__version__}")
    print(f"TensorFlow版本: {__import__('tensorflow').__version__}")
    print()

    # 测试图片
    sample_images = [
        r"d:\tensorflow_machinelearning_app\my_photos\coat.jpg",
        r"d:\tensorflow_machinelearning_app\my_photos\dress.jpg",
    ]

    # 创建检测器（自动选择可用方法）
    detector = ObjectDetector()

    for img_path in sample_images:
        if os.path.exists(img_path):
            print(f"\n检测图片: {img_path}")
            detections = detector.detect(img_path, threshold=0.5)

            if detections['total_objects'] > 0:
                print(f"发现 {detections['total_objects']} 个对象:")
                for det in detections['detections']:
                    print(f"  - {det['class']}: {det['confidence']:.2%}")
            else:
                print(f"未发现对象")

            output = img_path.rsplit('.', 1)[0] + '_detected.jpg'
            draw_detection_results(img_path, detections, output)
            print(f"结果图片已保存: {output}")
