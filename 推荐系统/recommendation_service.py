"""
商品推荐服务
提供商品推荐、相似商品查找等功能
"""

import os
import json
import numpy as np
from recommender import ProductRecommender, ProductFeatureExtractor


class RecommendationService:
    """推荐服务"""

    def __init__(self, model_dir=None):
        self.model_dir = model_dir or './models'
        self.recommender = None
        self.feature_extractor = None
        self.products = []
        self.product_features = None
        os.makedirs(self.model_dir, exist_ok=True)

    def load_products(self, product_file='product_index.json'):
        """加载商品数据"""
        if os.path.exists(product_file):
            with open(product_file, 'r', encoding='utf-8') as f:
                self.products = json.load(f)
            print(f"已加载 {len(self.products)} 个商品")
        else:
            # 创建示例商品数据
            self.products = [
                {"id": i, "name": f"商品 {i}", "category": f"类别 {i % 5}"}
                for i in range(50)
            ]
            print(f"已创建 {len(self.products)} 个示例商品")

    def init_recommender(self, num_users=100, num_products=50):
        """初始化推荐模型"""
        self.recommender = ProductRecommender(num_users=num_users, num_products=num_products)
        self.recommender.build_model()
        print("推荐模型已初始化")

    def init_feature_extractor(self):
        """初始化特征提取器"""
        self.feature_extractor = ProductFeatureExtractor()
        self.feature_extractor.build_feature_model()
        print("特征提取器已初始化")

    def train_recommender(self, user_ids, product_ids, ratings, epochs=10):
        """训练推荐模型"""
        if self.recommender is None:
            self.init_recommender()
        self.recommender.train(user_ids, product_ids, ratings, epochs=epochs)
        print("推荐模型训练完成")

    def recommend_for_user(self, user_id, product_ids=None, top_k=10):
        """为用户推荐商品"""
        if self.recommender is None:
            raise ValueError("推荐模型未初始化")

        if product_ids is None:
            product_ids = [p['id'] for p in self.products]

        recommendations = self.recommender.recommend(user_id, product_ids, top_k)
        results = []
        for product_id, score in recommendations:
            product_info = next((p for p in self.products if p['id'] == product_id), None)
            if product_info:
                results.append({
                    "product": product_info,
                    "score": float(score)
                })
        return results

    def find_similar_products(self, product_id, top_k=5):
        """查找相似商品"""
        if self.product_features is None:
            raise ValueError("商品特征未提取")

        product_idx = next((i for i, p in enumerate(self.products) if p['id'] == product_id), None)
        if product_idx is None:
            return []

        similar = self.feature_extractor.find_similar_products(
            self.product_features, product_idx, top_k
        )
        results = []
        for idx, score in similar:
            if idx < len(self.products):
                results.append({
                    "product": self.products[idx],
                    "similarity": float(score)
                })
        return results

    def extract_product_features(self, image_paths):
        """提取商品图像特征"""
        if self.feature_extractor is None:
            self.init_feature_extractor()
        self.product_features = self.feature_extractor.extract_features(image_paths)
        print(f"已提取 {len(self.product_features)} 个商品特征")
        return self.product_features

    def save_models(self):
        """保存模型"""
        if self.recommender:
            self.recommender.save_model(os.path.join(self.model_dir, 'recommender'))
        if self.feature_extractor:
            self.feature_extractor.save_model(os.path.join(self.model_dir, 'feature_extractor'))
        print("模型已保存")

    def load_models(self):
        """加载模型"""
        recommender_path = os.path.join(self.model_dir, 'recommender')
        feature_path = os.path.join(self.model_dir, 'feature_extractor')

        if os.path.exists(recommender_path):
            self.recommender = ProductRecommender()
            self.recommender.load_model(recommender_path)

        if os.path.exists(feature_path):
            self.feature_extractor = ProductFeatureExtractor()
            self.feature_extractor.load_model(feature_path)

        print("模型已加载")

    def get_recommendations(self, user_id, method='collaborative', top_k=10):
        """
        获取推荐结果

        Args:
            user_id: 用户ID
            method: 推荐方法 ('collaborative' 或 'content')
            top_k: 返回数量

        Returns:
            推荐商品列表
        """
        if method == 'collaborative':
            return self.recommend_for_user(user_id, top_k=top_k)
        elif method == 'content':
            # 基于内容的推荐（需要先提取特征）
            if self.product_features is None:
                return []
            # 返回高相似度商品
            return self.find_similar_products(0, top_k)
        else:
            raise ValueError(f"未知推荐方法: {method}")


# API 服务 (Flask)
def create_api_service(recommendation_service):
    """创建 Flask API 服务"""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("请安装 Flask: pip install flask")
        return None

    app = Flask(__name__)

    @app.route('/recommend', methods=['POST'])
    def recommend():
        data = request.json
        user_id = data.get('user_id')
        top_k = data.get('top_k', 10)
        method = data.get('method', 'collaborative')

        try:
            results = recommendation_service.get_recommendations(user_id, method, top_k)
            return jsonify({"success": True, "recommendations": results})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

    @app.route('/similar', methods=['POST'])
    def similar():
        data = request.json
        product_id = data.get('product_id')
        top_k = data.get('top_k', 5)

        try:
            results = recommendation_service.find_similar_products(product_id, top_k)
            return jsonify({"success": True, "similar_products": results})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok"})

    return app


if __name__ == '__main__':
    # 测试推荐服务
    print("=" * 50)
    print("商品推荐服务")
    print("=" * 50)

    service = RecommendationService()

    # 加载商品
    service.load_products()

    # 初始化并训练模型
    service.init_recommender(num_users=100, num_products=len(service.products))

    # 生成训练数据
    num_interactions = 500
    user_ids = np.random.randint(0, 100, num_interactions)
    product_ids = np.random.randint(0, len(service.products), num_interactions)
    ratings = np.random.randint(1, 6, num_interactions) / 5.0

    service.train_recommender(user_ids, product_ids, ratings, epochs=5)

    # 测试推荐
    print("\n" + "=" * 50)
    print("测试推荐")
    print("=" * 50)
    recommendations = service.get_recommendations(user_id=0, method='collaborative', top_k=5)
    for rec in recommendations:
        print(f"商品: {rec['product']['name']}, 得分: {rec['score']:.4f}")

    print("\n服务创建成功！")
