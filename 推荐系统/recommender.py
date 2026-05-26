"""
TensorFlow 商品推荐系统
基于神经协同过滤 (Neural Collaborative Filtering) 的推荐模型
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import os


class ProductRecommender:
    """商品推荐器"""

    def __init__(self, num_users=1000, num_products=500, embedding_dim=64):
        self.num_users = num_users
        self.num_products = num_products
        self.embedding_dim = embedding_dim
        self.model = None
        self.user_embeddings = None
        self.product_embeddings = None

    def build_model(self):
        """构建神经协同过滤模型"""
        # 用户输入
        user_input = keras.Input(shape=(1,), name='user_id')
        product_input = keras.Input(shape=(1,), name='product_id')

        # Embedding 层
        user_embedding = layers.Embedding(
            self.num_users,
            self.embedding_dim,
            name='user_embedding'
        )(user_input)
        product_embedding = layers.Embedding(
            self.num_products,
            self.embedding_dim,
            name='product_embedding'
        )(product_input)

        # 展平
        user_vec = layers.Flatten()(user_embedding)
        product_vec = layers.Flatten()(product_embedding)

        # GMF (Generalized Matrix Factorization)
        gmf = layers.Multiply()([user_vec, product_vec])

        # MLP
        mlp = layers.Concatenate()([user_vec, product_vec])
        mlp = layers.Dense(128, activation='relu')(mlp)
        mlp = layers.Dropout(0.2)(mlp)
        mlp = layers.Dense(64, activation='relu')(mlp)
        mlp = layers.Dropout(0.2)(mlp)
        mlp = layers.Dense(32, activation='relu')(mlp)

        # 合并 GMF 和 MLP
        combined = layers.Concatenate()([gmf, mlp])
        output = layers.Dense(1, activation='sigmoid', name='prediction')(combined)

        self.model = keras.Model(
            inputs=[user_input, product_input],
            outputs=output
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return self.model

    def train(self, user_ids, product_ids, ratings, epochs=10, batch_size=256, validation_split=0.1):
        """训练模型"""
        if self.model is None:
            self.build_model()

        self.model.fit(
            [user_ids, product_ids],
            ratings,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        return self.model

    def recommend(self, user_id, product_ids, top_k=10):
        """为用户推荐商品"""
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")

        user_id = np.array([[user_id]] * len(product_ids))
        product_ids = np.array([[p] for p in product_ids])

        predictions = self.model.predict([user_id, product_ids], verbose=0)
        predictions = predictions.flatten()

        # 获取 top_k 推荐的索引
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        recommended_products = [product_ids[i][0] for i in top_indices]
        scores = [predictions[i] for i in top_indices]

        return list(zip(recommended_products, scores))

    def save_model(self, path):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型不存在")
        self.model.save(path)
        print(f"模型已保存至: {path}")

    def load_model(self, path):
        """加载模型"""
        self.model = keras.models.load_model(path)
        print(f"模型已加载自: {path}")


class ProductFeatureExtractor:
    """商品特征提取器（基于图像）"""

    def __init__(self, model_path=None):
        self.feature_model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def build_feature_model(self):
        """构建特征提取模型 (使用 MobileNetV2)"""
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        self.feature_model = keras.Model(
            inputs=base_model.input,
            outputs=base_model.output
        )
        return self.feature_model

    def extract_features(self, image_paths):
        """提取商品图像特征"""
        if self.feature_model is None:
            self.build_feature_model()

        features = []
        for path in image_paths:
            try:
                img = keras.preprocessing.image.load_img(path, target_size=(224, 224))
                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)
                feature = self.feature_model.predict(img_array, verbose=0)
                features.append(feature.flatten())
            except Exception as e:
                print(f"处理图像失败 {path}: {e}")
                features.append(np.zeros(1280))  # MobileNetV2 输出维度

        return np.array(features)

    def compute_similarity(self, features1, features2):
        """计算特征相似度"""
        # 余弦相似度
        dot_product = np.dot(features1, features2.T)
        norm1 = np.linalg.norm(features1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(features2, axis=1, keepdims=True)
        similarity = dot_product / (norm1 * norm2.T + 1e-8)
        return similarity

    def find_similar_products(self, product_features, target_idx, top_k=5):
        """查找相似商品"""
        target_feature = product_features[target_idx:target_idx+1]
        similarities = self.compute_similarity(target_feature, product_features)
        similarities = similarities.flatten()

        # 排除自身
        similarities[target_idx] = -1
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return list(zip(top_indices, similarities[top_indices]))

    def save_model(self, path):
        if self.feature_model:
            self.feature_model.save(path)

    def load_model(self, path):
        self.feature_model = keras.models.load_model(path)


def create_sample_data(num_users=100, num_products=50, num_interactions=1000):
    """创建示例训练数据"""
    user_ids = np.random.randint(0, num_users, num_interactions)
    product_ids = np.random.randint(0, num_products, num_interactions)
    ratings = np.random.randint(1, 6, num_interactions) / 5.0  # 归一化到 [0, 1]
    return user_ids, product_ids, ratings


if __name__ == '__main__':
    print("=" * 50)
    print("TensorFlow 商品推荐系统")
    print("=" * 50)

    # 创建推荐模型
    recommender = ProductRecommender(num_users=100, num_products=50)
    recommender.build_model()
    recommender.model.summary()

    # 生成示例数据并训练
    print("\n生成示例训练数据...")
    user_ids, product_ids, ratings = create_sample_data()
    print(f"训练数据: {len(user_ids)} 条交互记录")

    print("\n开始训练模型...")
    recommender.train(user_ids, product_ids, ratings, epochs=5)

    # 示例推荐
    print("\n为用户 0 推荐商品:")
    recommendations = recommender.recommend(0, list(range(50)), top_k=5)
    for product_id, score in recommendations:
        print(f"  商品 {product_id}: 得分 {score:.4f}")

    print("\n训练完成！")
