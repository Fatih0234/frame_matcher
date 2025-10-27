from pathlib import Path
from typing import Dict, Any
import yaml
from .base import BaseExporter


class YOLOExporter(BaseExporter):
    """
    Exports annotations in YOLO format.

    YOLO format specifications:
    - One .txt file per image in labels/ directory
    - Format: class_id center_x center_y width height
    - All coordinates normalized to [0, 1]
    - Generates classes.txt and data.yaml config files
    """

    def setup_directories(self) -> None:
        """Create labels directory for YOLO format."""
        self.labels_dir = self.output_dir / "labels"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created YOLO labels directory: {self.labels_dir}")

    def export_frame(
        self,
        frame_data: Dict[str, Any],
        image_filename: str,
        frame_width: int,
        frame_height: int
    ) -> None:
        """
        Export single frame in YOLO format.

        Creates a .txt file with one line per bounding box:
        class_id cx cy w h (all normalized)

        Args:
            frame_data: Dictionary containing frame annotations with 'bboxes' key
            image_filename: Name of the image file
            frame_width: Width of the frame in pixels (unused in YOLO, kept for API compatibility)
            frame_height: Height of the frame in pixels (unused in YOLO, kept for API compatibility)
        """
        label_filename = Path(image_filename).stem + ".txt"
        label_path = self.labels_dir / label_filename

        with open(label_path, 'w') as f:
            for bbox in frame_data.get('bboxes', []):
                class_name = bbox['label']
                class_id = self.class_mapping[class_name]

                # YOLO format: normalized coordinates
                cx = bbox['cx']  # center x (normalized 0-1)
                cy = bbox['cy']  # center y (normalized 0-1)
                w = bbox['w']    # width (normalized 0-1)
                h = bbox['h']    # height (normalized 0-1)

                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def finalize(self) -> None:
        """Generate classes.txt and data.yaml files."""
        # Create classes.txt
        classes_file = self.output_dir / "classes.txt"
        sorted_classes = sorted(self.class_mapping.items(), key=lambda x: x[1])

        with open(classes_file, 'w') as f:
            for class_name, _ in sorted_classes:
                f.write(f"{class_name}\n")

        print(f"✓ Created classes.txt with {len(sorted_classes)} classes")

        # Create data.yaml
        class_names = [name for name, _ in sorted_classes]
        yaml_content = {
            'path': '.',
            'train': 'images',
            'nc': len(self.class_mapping),
            'names': class_names
        }

        yaml_file = self.output_dir / "data.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Created data.yaml configuration file")
