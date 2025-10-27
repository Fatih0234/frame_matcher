import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from .base import BaseExporter


class COCOExporter(BaseExporter):
    """
    Exports annotations in COCO JSON format.

    COCO format specifications:
    - Single JSON file (annotations.json)
    - Images, annotations, and categories arrays
    - Bounding boxes in [x_min, y_min, width, height] absolute coordinates
    - See: https://cocodataset.org/#format-data
    """

    def __init__(self, output_dir: Path, class_mapping: Dict[str, int]):
        self.images: List[Dict] = []
        self.annotations: List[Dict] = []
        self.annotation_id = 1
        self.image_id = 1
        self.image_id_map: Dict[str, int] = {}
        super().__init__(output_dir, class_mapping)

    def setup_directories(self) -> None:
        """COCO uses a single JSON file, no subdirectories needed."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ COCO output directory: {self.output_dir}")

    def export_frame(
        self,
        frame_data: Dict[str, Any],
        image_filename: str,
        frame_width: int,
        frame_height: int
    ) -> None:
        """
        Export single frame in COCO format.

        Accumulates image and annotation data in memory.
        Actual file writing happens in finalize().

        Args:
            frame_data: Dictionary containing frame annotations with 'bboxes' key
            image_filename: Name of the image file
            frame_width: Width of the frame in pixels
            frame_height: Height of the frame in pixels
        """
        # Add image info (only once per unique image)
        if image_filename not in self.image_id_map:
            self.image_id_map[image_filename] = self.image_id

            self.images.append({
                "id": self.image_id,
                "file_name": image_filename,
                "width": frame_width,
                "height": frame_height,
            })

            current_image_id = self.image_id
            self.image_id += 1
        else:
            current_image_id = self.image_id_map[image_filename]

        # Add annotations for this frame
        for bbox in frame_data.get('bboxes', []):
            class_name = bbox['label']
            class_id = self.class_mapping[class_name]

            # Convert from normalized to absolute coordinates
            # YOLO: center_x, center_y, width, height (normalized)
            # COCO: x_min, y_min, width, height (absolute pixels)
            cx_norm, cy_norm = bbox['cx'], bbox['cy']
            w_norm, h_norm = bbox['w'], bbox['h']

            # Calculate absolute coordinates
            box_width = w_norm * frame_width
            box_height = h_norm * frame_height
            x_min = (cx_norm - w_norm / 2) * frame_width
            y_min = (cy_norm - h_norm / 2) * frame_height

            # Ensure coordinates are within image bounds
            x_min = max(0, min(x_min, frame_width))
            y_min = max(0, min(y_min, frame_height))
            box_width = min(box_width, frame_width - x_min)
            box_height = min(box_height, frame_height - y_min)

            self.annotations.append({
                "id": self.annotation_id,
                "image_id": current_image_id,
                "category_id": class_id,
                "bbox": [
                    round(x_min, 2),
                    round(y_min, 2),
                    round(box_width, 2),
                    round(box_height, 2)
                ],
                "area": round(box_width * box_height, 2),
                "iscrowd": 0
            })
            self.annotation_id += 1

    def finalize(self) -> None:
        """Write complete COCO JSON file."""
        # Create categories list
        categories = [
            {
                "id": class_id,
                "name": class_name,
                "supercategory": "object"
            }
            for class_name, class_id in sorted(
                self.class_mapping.items(),
                key=lambda x: x[1]
            )
        ]

        # Assemble final COCO structure
        coco_data = {
            "info": {
                "description": "Label Studio Video Annotations - COCO Format",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "frame_matcher",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": self.images,
            "annotations": self.annotations,
            "categories": categories
        }

        # Write to file with pretty formatting
        output_file = self.output_dir / "annotations.json"
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✓ COCO annotations saved to: {output_file}")
        print(f"  Total images: {len(self.images)}")
        print(f"  Total annotations: {len(self.annotations)}")
        print(f"  Categories: {len(categories)}")
        print(f"{'='*60}\n")
