"""
COCO format converter.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class COCOConverter:
    def __init__(self, class_mappings: Dict[str, int]):
        """
        Initialize COCO converter.
        
        Args:
            class_mappings: Dictionary mapping class names to their integer encodings
        """
        self.class_mappings = class_mappings
        self.coco_data = {
            "info": {
                "description": "Video annotation dataset converted from LabelStudio",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Video Annotation Converter",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "categories": [],
            "images": [],
            "annotations": []
        }
        
        self.image_id = 1
        self.annotation_id = 1
        
        # Initialize categories
        self._create_categories()
    
    def _create_categories(self):
        """Create COCO categories from class mappings."""
        for class_name, class_id in self.class_mappings.items():
            category = {
                "id": class_id,
                "name": class_name,
                "supercategory": "object"
            }
            self.coco_data["categories"].append(category)
    
    def _convert_bbox_to_coco(self, x: float, y: float, width: float, height: float, 
                             img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert bounding box from percentage coordinates to COCO format.
        
        Args:
            x, y, width, height: Bounding box in percentage (0-100)
            img_width, img_height: Image dimensions in pixels
            
        Returns:
            Tuple of (x, y, width, height) in pixel coordinates (COCO format: top-left + width/height)
        """
        # Convert percentages to pixel coordinates
        x_pixels = (x / 100.0) * img_width
        y_pixels = (y / 100.0) * img_height
        width_pixels = (width / 100.0) * img_width
        height_pixels = (height / 100.0) * img_height
        
        # Ensure coordinates are within image bounds
        x_pixels = max(0, min(img_width - 1, x_pixels))
        y_pixels = max(0, min(img_height - 1, y_pixels))
        width_pixels = max(1, min(img_width - x_pixels, width_pixels))
        height_pixels = max(1, min(img_height - y_pixels, height_pixels))
        
        return x_pixels, y_pixels, width_pixels, height_pixels
    
    def add_image_with_annotations(self, image_filename: str, image_shape: Tuple[int, int, int], 
                                  annotations: List[Dict]):
        """
        Add an image and its annotations to the COCO dataset.
        
        Args:
            image_filename: Name of the image file
            image_shape: Shape of the image (height, width, channels)
            annotations: List of annotation dictionaries for the image
        """
        img_height, img_width = image_shape[:2]
        
        # Add image entry
        image_entry = {
            "id": self.image_id,
            "width": img_width,
            "height": img_height,
            "file_name": image_filename,
            "license": 1,
            "date_captured": datetime.now().isoformat()
        }
        self.coco_data["images"].append(image_entry)
        
        # Add annotation entries
        for annotation in annotations:
            class_id = annotation['class_id']
            x = annotation['x']
            y = annotation['y']
            width = annotation['width']
            height = annotation['height']
            
            # Convert to COCO format
            x_coco, y_coco, width_coco, height_coco = self._convert_bbox_to_coco(
                x, y, width, height, img_width, img_height
            )
            
            # Calculate area
            area = width_coco * height_coco
            
            # Create annotation entry
            annotation_entry = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": class_id,
                "bbox": [x_coco, y_coco, width_coco, height_coco],
                "area": area,
                "segmentation": [],  # Empty for bounding box only
                "iscrowd": 0
            }
            self.coco_data["annotations"].append(annotation_entry)
            self.annotation_id += 1
        
        self.image_id += 1
    
    def save_coco_file(self, output_path: Path):
        """
        Save the COCO dataset to a JSON file.
        
        Args:
            output_path: Path where to save the COCO JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(self.coco_data, f, indent=2)
        
        print(f"📄 COCO file saved with {len(self.coco_data['images'])} images and {len(self.coco_data['annotations'])} annotations")