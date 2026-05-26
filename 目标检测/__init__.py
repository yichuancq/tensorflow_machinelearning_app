"""
目标检测模块
使用TensorFlow内置方法进行目标检测

主要功能:
- 使用TensorFlow Hub预训练模型进行目标检测
- 支持批量处理图片
- 支持80类COCO目标检测
"""

from . import object_detection

__all__ = ['object_detection']
