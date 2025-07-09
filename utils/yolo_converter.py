"""
YOLO format converter.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import yaml


class YOLOConverter:
    def __init__(self, class_mappings: Dict[str, int]):
        """
        Initialize YOLO converter.
        
        Args:
            class_mappings: Dictionary mapping class names to their integer encodings
        """
        self.class_mappings = class_mappings
    
    def _convert_bbox_to_yolo(self, x: float, y: float, width: float, height: float, 
                             img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert bounding box from percentage coordinates to YOLO format.
        
        Args:
            x, y, width, height: Bounding box in percentage (0-100)
            img_width, img_height: Image dimensions in pixels
            
        Returns:
            Tuple of (center_x, center_y, width, height) in normalized coordinates (0-1)
        """
        # Convert percentages to normalized coordinates (0-1)
        x_norm = x / 100.0
        y_norm = y / 100.0
        width_norm = width / 100.0
        height_norm = height / 100.0
        
        # Convert to YOLO format (center coordinates)
        center_x = x_norm + (width_norm / 2.0)
        center_y = y_norm + (height_norm / 2.0)
        
        # Ensure coordinates are within bounds
        center_x = max(0.0, min(1.0, center_x))
        center_y = max(0.0, min(1.0, center_y))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))
        
        return center_x, center_y, width_norm, height_norm
    
    def create_yolo_annotation(self, annotations: List[Dict], image_shape: Tuple[int, int, int], 
                              output_path: Path):
        """
        Create a YOLO annotation file for a single image.
        
        Args:
            annotations: List of annotation dictionaries for the image
            image_shape: Shape of the image (height, width, channels)
            output_path: Path where to save the annotation file
        """
        img_height, img_width = image_shape[:2]
        
        yolo_lines = []
        
        for annotation in annotations:
            class_id = annotation['class_id']
            x = annotation['x']
            y = annotation['y']
            width = annotation['width']
            height = annotation['height']
            
            # Convert to YOLO format
            center_x, center_y, norm_width, norm_height = self._convert_bbox_to_yolo(
                x, y, width, height, img_width, img_height
            )
            
            # Create YOLO format line: class_id center_x center_y width height
            yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
            yolo_lines.append(yolo_line)
        
        # Write annotation file
        with open(output_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
    
    def create_classes_file(self, output_path: Path):
        """
        Create classes.txt file for YOLO format.
        
        Args:
            output_path: Path where to save the classes file
        """
        # Sort classes by their ID
        sorted_classes = sorted(self.class_mappings.items(), key=lambda x: x[1])
        
        class_lines = [class_name for class_name, _ in sorted_classes]
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(class_lines))
    
    def create_yaml_file(self, output_path: Path, dataset_name: str = "dataset"):
        """
        Create YAML configuration file for YOLO training.
        
        Args:
            output_path: Path where to save the YAML file
            dataset_name: Name of the dataset
        """
        # Sort classes by their ID to get the correct order
        sorted_classes = sorted(self.class_mappings.items(), key=lambda x: x[1])
        class_names = [class_name for class_name, _ in sorted_classes]
        
        # Create YAML content with relative paths
        yaml_content = {
            'path': '.',  # Relative to the YAML file location
            'train': 'images',  # Since we're not splitting, put everything in train
            'val': 'images',    # Use same directory for validation
            'test': 'images',   # Use same directory for test
            
            'nc': len(self.class_mappings),  # Number of classes
            'names': class_names  # Class names in order
        }
        
        # Write YAML file
        with open(output_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"📄 YAML config saved: {output_path}")
        print(f"   - Classes: {len(class_names)}")
        print(f"   - Names: {', '.join(class_names)}")