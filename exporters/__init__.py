"""
Annotation format exporters for frame_matcher.

Supports multiple export formats for object detection annotations:
- YOLO: Text files with normalized coordinates
- COCO: JSON format with absolute coordinates
"""

from .base import BaseExporter
from .yolo import YOLOExporter
from .coco import COCOExporter

__all__ = ['BaseExporter', 'YOLOExporter', 'COCOExporter']
__version__ = '1.0.0'
