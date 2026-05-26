"""
目标检测模块测试脚本
"""
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from 目标检测 import object_detection


def test_detection():
    """测试目标检测功能"""
    print("=" * 50)
    print("TensorFlow 目标检测测试")
    print("=" * 50)

    # 测试图片路径
    test_images = [
        r"d:\tensorflow_machinelearning_app\my_photos\coat.jpg",
        r"d:\tensorflow_machinelearning_app\my_photos\dress.jpg",
    ]

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"\n图片不存在: {img_path}")
            continue

        print(f"\n检测图片: {img_path}")

        # 执行检测
        detections = object_detection.detect_objects(img_path, threshold=0.5)

        print(f"发现 {detections['total_objects']} 个对象:")
        for det in detections['detections']:
            print(f"  - {det['class']}: {det['confidence']:.2%}")

        # 保存带检测框的图片
        output_path = img_path.rsplit('.', 1)[0] + '_detected.jpg'
        object_detection.draw_detection_results(img_path, detections, output_path)
        print(f"检测结果图片已保存: {output_path}")


def test_batch_detection():
    """测试批量检测"""
    print("\n" + "=" * 50)
    print("批量检测测试")
    print("=" * 50)

    # 批量检测my_photos目录
    photo_dir = r"d:\tensorflow_machinelearning_app\my_photos"
    output_dir = os.path.join(os.path.dirname(photo_dir), "目标检测_output")

    if os.path.exists(photo_dir):
        results = object_detection.detect_batch(photo_dir, output_dir, threshold=0.5)

        # 保存结果到JSON
        json_path = os.path.join(output_dir, "detection_results.json")
        object_detection.save_results_to_json(results, json_path)
        print(f"\n检测结果已保存: {json_path}")


if __name__ == "__main__":
    test_detection()
    test_batch_detection()
    print("\n测试完成!")
