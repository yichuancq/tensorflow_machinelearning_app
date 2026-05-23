import os
import cv2
import numpy as np
import json
from mtcnn import MTCNN
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import load_model
import tensorflow as tf

# 禁用TensorFlow无用日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ==============================================
# 配置参数
# ==============================================
IMAGE_FOLDER = "./my_photos"  # 你的相册文件夹
FACE_SAVE_PATH = "./face_clusters"  # 人脸聚类结果保存目录
FACENET_MODEL_PATH = "facenet_keras.h5"  # FaceNet模型路径（自动下载）
CLUSTER_EPS = 0.55  # 聚类相似度阈值（越小越严格）
MIN_SAMPLES = 1  # 最少人脸数

# ==============================================
# 1. 加载 FaceNet 模型（用于提取人脸特征）
# ==============================================
def load_facenet_model():
    """加载预训练FaceNet模型，自动下载"""
    if not os.path.exists(FACENET_MODEL_PATH):
        print("正在下载 FaceNet 模型...")
        import urllib.request
        url = "https://raw.githubusercontent.com/nyoki-mtl/keras-facenet/master/model/facenet_keras.h5"
        urllib.request.urlretrieve(url, FACENET_MODEL_PATH)
    return load_model(FACENET_MODEL_PATH, compile=False)

# ==============================================
# 2. 人脸检测 + 人脸对齐
# ==============================================
def detect_face(img_path, detector, face_size=(160, 160)):
    """检测单张图片中的所有人脸，返回裁剪+标准化后的人脸"""
    try:
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)
        faces = []
        boxes = []

        for res in results:
            x1, y1, w, h = res['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + w, y1 + h
            face = img_rgb[y1:y2, x1:x2]
            face = cv2.resize(face, face_size)
            faces.append(face)
            boxes.append((x1, y1, x2, y2))

        return faces, boxes
    except:
        return [], []

# ==============================================
# 3. 人脸特征提取
# ==============================================
def get_face_embedding(model, face_pixels):
    """将人脸转为128维特征向量"""
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples, verbose=0)
    return yhat[0]

# ==============================================
# 4. 批量提取所有人脸特征
# ==============================================
def extract_all_faces_features(folder, detector, facenet_model):
    """遍历文件夹，提取所有人脸及其特征"""
    all_face_features = []
    face_info = []

    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            faces, boxes = detect_face(path, detector)

            for i, (face, box) in enumerate(zip(faces, boxes)):
                feat = get_face_embedding(facenet_model, face)
                all_face_features.append(feat)
                face_info.append({
                    "image_path": path,
                    "filename": filename,
                    "face_index": i,
                    "box": box
                })

    return np.array(all_face_features), face_info

# ==============================================
# 5. 人脸聚类（核心）
# ==============================================
def cluster_faces(features, face_info):
    """DBSCAN聚类：自动把同一个人的脸分为一组"""
    if len(features) == 0:
        return {}, []

    clustering = DBSCAN(eps=CLUSTER_EPS, min_samples=MIN_SAMPLES, metric="euclidean")
    labels = clustering.fit_predict(features)

    # 分组
    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue  # 过滤噪声
        key = f"person_{label}"
        if key not in clusters:
            clusters[key] = []
        clusters[key].append(face_info[idx])

    return clusters, labels

# ==============================================
# 6. 保存聚类结果
# ==============================================
def save_cluster_result(clusters):
    """保存分组结果到JSON"""
    if not os.path.exists(FACE_SAVE_PATH):
        os.makedirs(FACE_SAVE_PATH)

    with open("./face_clusters.json", "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 人脸聚类完成，结果已保存到 face_clusters.json")
    print(f"📁 共识别出 {len(clusters)} 个人")

# ==============================================
# 主程序
# ==============================================
if __name__ == "__main__":
    print("🔍 启动人脸聚类...")

    # 1. 加载模型
    detector = MTCNN()
    facenet = load_facenet_model()

    # 2. 提取人脸
    feats, infos = extract_all_faces_features(IMAGE_FOLDER, detector, facenet)
    print(f"✅ 共检测到 {len(infos)} 张人脸")

    # 3. 聚类
    clusters, labels = cluster_faces(feats, infos)

    # 4. 保存结果
    save_cluster_result(clusters)

    # 5. 打印分组
    print("\n===== 人脸聚类结果 =====")
    for person, faces in clusters.items():
        print(f"\n{person}：{len(faces)} 张脸")
        for f in faces[:3]:
            print(f"  - {f['filename']}")