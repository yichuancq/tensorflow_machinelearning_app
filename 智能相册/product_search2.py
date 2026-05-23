import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# ====================== 配置 ======================
PRODUCT_FOLDER = "./shop_products"
INDEX_FILE = "product_index.json"
IMG_SIZE = (224, 224)

# 修复：扩大服装关键词，确保 coat、dress 能识别
PRODUCT_LABELS = {
    "衣服": ["shirt", "t-shirt", "sweater", "jacket", "coat", "jean",
            "dress", "skirt", "pants", "shorts", "suit", "trench",
            "frock", "gown", "overskirt", "hoopskirt", "fur"],

    "裤子": ["pant", "jean", "trouser", "short", "legging"],
    "鞋子": ["shoe", "boot", "sneaker", "sandal"],
    "包包": ["bag", "handbag", "backpack", "purse"],
    "电子产品": ["phone", "laptop", "headphone", "camera", "watch"],
    "食品": ["food", "pizza", "cake", "fruit", "drink", "snack"],
    "家居": ["chair", "table", "sofa", "bed", "pillow"],
    "美妆": ["lipstick", "perfume", "cosmetic", "makeup"]
}

# ====================== 加载模型 ======================
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(*IMG_SIZE, 3))
classify_model = MobileNetV2(weights="imagenet")

# ====================== 图片预处理 ======================
def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = preprocess_input(x)
    return x

# ====================== 提取特征 ======================
def extract_feature(img_path):
    x = load_and_preprocess(img_path)
    x = np.expand_dims(x, axis=0)
    feat = base_model.predict(x, verbose=0)
    return feat[0].tolist()

# ====================== 自动打标签 ======================
def auto_tag_product(img_path):
    x = load_and_preprocess(img_path)
    x = np.expand_dims(x, axis=0)
    preds = classify_model.predict(x, verbose=0)
    
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
    top5 = decode_predictions(preds, top=5)[0]
    
    tags = []
    # 打印模型真实识别结果
    print(f"【模型识别】: {[l for _, l, _ in top5]}")

    for _, label, score in top5:
        label = label.lower()
        
        for cate, keywords in PRODUCT_LABELS.items():
            for kw in keywords:
                if kw in label and cate not in tags:
                    tags.append(cate)
                    
    return tags[:3], [(l, round(float(s), 2)) for _, l, s in top5]

# ====================== 构建商品库 ======================
def build_product_database():
    if not os.path.exists(PRODUCT_FOLDER):
        os.makedirs(PRODUCT_FOLDER)
        print(f"请将商品图片放入 {PRODUCT_FOLDER}")
        return

    index = []
    print("正在构建商品索引...")

    for filename in os.listdir(PRODUCT_FOLDER):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(PRODUCT_FOLDER, filename)
            print(f"处理：{filename}")

            tags, top_pred = auto_tag_product(path)
            feature = extract_feature(path)

            index.append({
                "filename": filename,
                "path": path,
                "tags": tags,
                "feature": feature,
                "top_pred": top_pred
            })

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 商品库构建完成！共 {len(index)} 个商品")

# ====================== 以图搜图 ======================
def search_similar_product(query_img_path, top_n=5):
    if not os.path.exists(INDEX_FILE):
        print("❌ 请先构建商品库！")
        return

    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        index = json.load(f)

    q_feat = extract_feature(query_img_path)
    q_feat = np.array(q_feat)

    results = []
    for item in index:
        feat = np.array(item["feature"])
        sim = np.dot(q_feat, feat) / (np.linalg.norm(q_feat) * np.linalg.norm(feat))
        results.append((item["path"], item["tags"], float(sim)))

    results = sorted(results, key=lambda x: x[2], reverse=True)[:top_n]

    print(f"\n🔍 搜索结果（Top {top_n}）：")
    for i, (path, tags, score) in enumerate(results):
        print(f"{i+1}. {os.path.basename(path)} | 标签：{tags} | 相似度：{round(score, 2)}")

    return results

# ====================== 主程序 ======================
if __name__ == "__main__":
    print("="*50)
    print("       TensorFlow 电商商品自动打标 + 以图搜图")
    print("="*50)

    build_product_database()

    print("\n===== 商品自动打标演示 =====")
    for filename in os.listdir(PRODUCT_FOLDER)[:5]:
        path = os.path.join(PRODUCT_FOLDER, filename)
        tags, _ = auto_tag_product(path)
        print(f"📸 {filename} → 标签：{tags}")

    print("\n===== 以图搜图测试 =====")
    test_img = "my_photos\coat.jpg"
    if os.path.exists(test_img):
        search_similar_product(test_img)
    else:
        print(f"请放入测试图片：{test_img}，可体验以图搜图")