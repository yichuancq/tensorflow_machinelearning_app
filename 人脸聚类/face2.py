import os
import cv2
import numpy as np
import json
from retinaface import RetinaFace
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import load_model
import tensorflow as tf

# 关闭 TF 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ==============================================
# 配置
# ==============================================
IMAGE_FOLDER = "./my_photos"
FACE_SAVE_PATH = "./face_clusters"
FACENET_PATH = "facenet_keras.h5"
CLUSTER_EPS = 0.55
MIN_SAMPLES = 1

# ==============================================
# 加载 FaceNet 模型
# ==============================================
def load_facenet():
    if not os.path.exists(FACENET_PATH):
        print("正在下载 FaceNet 模型...")
        import urllib.request
        url = "https://raw.githubusercontent.com/nyoki-mtl/keras-facenet/master/model/facenet_keras.h5"
        urllib.request.urlretrieve(url, FACENET_PATH)
    return load_model(FACENET_PATH, compile=False)

# ==============================================
# RetinaFace 人脸检测（替换 MTCNN）
# ==============================================
def detect_faces_retinaface(img_path, face_size=(160, 160)):
    img = cv2.imread(img_path)
    if img is None:
        return [], []

    # RGB 格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = RetinaFace.detect_faces(img_rgb)

    cropped_faces = []
    boxes = []

    if isinstance(faces, dict):
        for key in faces:
            face = faces[key]
            x1, y1, x2, y2 = face["facial_area"]
            face_crop = img_rgb[y1:y2, x1:x2]
            face_crop = cv2.resize(face_crop, face_size)
            cropped_faces.append(face_crop)
            boxes.append((x1, y1, x2, y2))

    return cropped_faces, boxes

# ==============================================
# 人脸特征提取
# ==============================================
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype("float32")
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    return model.predict(samples, verbose=0)[0]

# ==============================================
# 批量提取所有人脸
# ==============================================
def extract_all_features(folder, facenet):
    features = []
    infos = []

    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, fname)
            faces, boxes = detect_faces_retinaface(path)

            for i, (face, box) in enumerate(zip(faces, boxes)):
                emb = get_embedding(facenet, face)
                features.append(emb)
                infos.append({
                    "image_path": path,
                    "filename": fname,
                    "face_index": i,
                    "box": box
                })

    return np.array(features), infos

# ==============================================
# 人脸聚类
# ==============================================
def cluster(features, infos):
    if len(features) == 0:
        return {}, []

    cluster = DBSCAN(eps=CLUSTER_EPS, min_samples=MIN_SAMPLES, metric="euclidean")
    labels = cluster.fit_predict(features)

    groups = {}
    for idx, lab in enumerate(labels):
        if lab == -1:
            continue
        key = f"person_{lab}"
        if key not in groups:
            groups[key] = []
        groups[key].append(infos[idx])

    return groups, labels

# ==============================================
# 保存结果
# ==============================================
def save_result(groups):
    with open("face_clusters.json", "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！共识别出 {len(groups)} 个人")
    print("📄 结果已保存到 face_clusters.json")

# ==============================================
# 主程序
# ==============================================
if __name__ == "__main__":
    print("🚀 RetinaFace 人脸聚类 启动...")

    facenet = load_facenet()
    feats, infos = extract_all_features(IMAGE_FOLDER, facenet)
    print(f"✅ 检测到 {len(infos)} 张人脸")

    groups, _ = cluster(feats, infos)
    save_result(groups)

    print("\n===== 分组结果 =====")
    for pid, faces in groups.items():
        print(f"\n{pid}：{len(faces)} 张")
        for f in faces[:3]:
            print(f"  - {f['filename']}")