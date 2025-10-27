"""
Test suite specifically for COCO format export functionality.
Tests COCO exporter, format validation, and coordinate conversion.

Run with: pytest test_coco_format.py -v
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from exporters.coco import COCOExporter


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_class_mapping():
    """Sample class mapping for testing."""
    return {
        "cyclist": 0,
        "pedestrian": 1,
        "scooter-roller": 2
    }


@pytest.fixture
def sample_frame_data():
    """Sample frame annotation data in YOLO normalized format."""
    return {
        "bboxes": [
            {
                "label": "cyclist",
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.2,
                "h": 0.3
            },
            {
                "label": "pedestrian",
                "cx": 0.8,
                "cy": 0.3,
                "w": 0.15,
                "h": 0.25
            }
        ]
    }


# ============================================================================
# TEST COCO EXPORTER INITIALIZATION
# ============================================================================

class TestCOCOExporterInit:
    """Test COCOExporter initialization."""

    def test_initialization(self, temp_dir, sample_class_mapping):
        """Test that COCOExporter initializes correctly."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        assert exporter.output_dir == temp_dir
        assert exporter.class_mapping == sample_class_mapping
        assert exporter.images == []
        assert exporter.annotations == []
        assert exporter.annotation_id == 1
        assert exporter.image_id == 1
        assert exporter.image_id_map == {}

    def test_setup_directories(self, temp_dir, sample_class_mapping):
        """Test that output directory is created."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        assert temp_dir.exists()
        assert temp_dir.is_dir()


# ============================================================================
# TEST COORDINATE CONVERSION
# ============================================================================

class TestCOCOCoordinateConversion:
    """Test conversion from YOLO to COCO coordinate format."""

    def test_coordinate_conversion_centered_box(self, temp_dir, sample_class_mapping):
        """Test conversion of a centered bounding box."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        # YOLO format: center at (0.5, 0.5), size (0.2, 0.3) on 1000x1000 image
        frame_data = {
            "bboxes": [
                {
                    "label": "cyclist",
                    "cx": 0.5,
                    "cy": 0.5,
                    "w": 0.2,
                    "h": 0.3
                }
            ]
        }

        exporter.export_frame(frame_data, "test.jpg", 1000, 1000)

        # COCO format should be: [x_min, y_min, width, height] in pixels
        # x_min = (0.5 - 0.2/2) * 1000 = 400
        # y_min = (0.5 - 0.3/2) * 1000 = 350
        # width = 0.2 * 1000 = 200
        # height = 0.3 * 1000 = 300

        annotation = exporter.annotations[0]
        bbox = annotation['bbox']

        assert bbox[0] == 400.0  # x_min
        assert bbox[1] == 350.0  # y_min
        assert bbox[2] == 200.0  # width
        assert bbox[3] == 300.0  # height

    def test_coordinate_conversion_corner_box(self, temp_dir, sample_class_mapping):
        """Test conversion of a box in the top-left corner."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        # Box starting at top-left corner
        frame_data = {
            "bboxes": [
                {
                    "label": "pedestrian",
                    "cx": 0.1,  # Center at 10% from left
                    "cy": 0.1,  # Center at 10% from top
                    "w": 0.2,   # Width 20%
                    "h": 0.2    # Height 20%
                }
            ]
        }

        exporter.export_frame(frame_data, "test.jpg", 1920, 1080)

        # Expected COCO coordinates:
        # x_min = (0.1 - 0.2/2) * 1920 = 0
        # y_min = (0.1 - 0.2/2) * 1080 = 0
        # width = 0.2 * 1920 = 384
        # height = 0.2 * 1080 = 216

        annotation = exporter.annotations[0]
        bbox = annotation['bbox']

        assert bbox[0] == 0.0     # x_min clamped to 0
        assert bbox[1] == 0.0     # y_min clamped to 0
        assert bbox[2] == 384.0   # width
        assert bbox[3] == 216.0   # height


# ============================================================================
# TEST FRAME EXPORT
# ============================================================================

class TestCOCOFrameExport:
    """Test frame export functionality."""

    def test_export_single_frame(self, temp_dir, sample_class_mapping, sample_frame_data):
        """Test exporting a single frame with annotations."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        exporter.export_frame(
            sample_frame_data,
            "frame_001.jpg",
            1920,
            1080
        )

        # Check image was added
        assert len(exporter.images) == 1
        assert exporter.images[0]['file_name'] == "frame_001.jpg"
        assert exporter.images[0]['width'] == 1920
        assert exporter.images[0]['height'] == 1080

        # Check annotations were added
        assert len(exporter.annotations) == 2
        assert exporter.annotations[0]['category_id'] == 0  # cyclist
        assert exporter.annotations[1]['category_id'] == 1  # pedestrian

    def test_export_multiple_frames(self, temp_dir, sample_class_mapping):
        """Test exporting multiple frames."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        for i in range(3):
            frame_data = {
                "bboxes": [
                    {
                        "label": "cyclist",
                        "cx": 0.5,
                        "cy": 0.5,
                        "w": 0.2,
                        "h": 0.3
                    }
                ]
            }
            exporter.export_frame(frame_data, f"frame_{i:03d}.jpg", 1920, 1080)

        assert len(exporter.images) == 3
        assert len(exporter.annotations) == 3
        assert exporter.image_id == 4  # Should increment to 4

    def test_duplicate_image_handling(self, temp_dir, sample_class_mapping):
        """Test that duplicate image filenames are handled correctly."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        frame_data = {
            "bboxes": [
                {"label": "cyclist", "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.3}
            ]
        }

        # Export same filename twice
        exporter.export_frame(frame_data, "frame_001.jpg", 1920, 1080)
        exporter.export_frame(frame_data, "frame_001.jpg", 1920, 1080)

        # Should only have one image entry
        assert len(exporter.images) == 1
        # But two annotation entries
        assert len(exporter.annotations) == 2
        # Both annotations should reference the same image_id
        assert exporter.annotations[0]['image_id'] == exporter.annotations[1]['image_id']


# ============================================================================
# TEST AREA CALCULATION
# ============================================================================

class TestCOCOAreaCalculation:
    """Test bounding box area calculation."""

    def test_area_calculation(self, temp_dir, sample_class_mapping):
        """Test that bbox area is calculated correctly."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        frame_data = {
            "bboxes": [
                {
                    "label": "cyclist",
                    "cx": 0.5,
                    "cy": 0.5,
                    "w": 0.2,  # 200 pixels
                    "h": 0.3   # 300 pixels
                }
            ]
        }

        exporter.export_frame(frame_data, "test.jpg", 1000, 1000)

        annotation = exporter.annotations[0]
        # Area should be width * height = 200 * 300 = 60000
        assert annotation['area'] == 60000.0


# ============================================================================
# TEST FINALIZE AND JSON OUTPUT
# ============================================================================

class TestCOCOFinalize:
    """Test finalize method and JSON output."""

    def test_finalize_creates_json(self, temp_dir, sample_class_mapping, sample_frame_data):
        """Test that finalize creates a valid COCO JSON file."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        exporter.export_frame(sample_frame_data, "frame_001.jpg", 1920, 1080)
        exporter.finalize()

        json_file = temp_dir / "annotations.json"
        assert json_file.exists()

    def test_coco_json_structure(self, temp_dir, sample_class_mapping, sample_frame_data):
        """Test that generated COCO JSON has correct structure."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        exporter.export_frame(sample_frame_data, "frame_001.jpg", 1920, 1080)
        exporter.finalize()

        json_file = temp_dir / "annotations.json"
        with open(json_file, 'r') as f:
            coco_data = json.load(f)

        # Check required top-level keys
        assert 'info' in coco_data
        assert 'images' in coco_data
        assert 'annotations' in coco_data
        assert 'categories' in coco_data
        assert 'licenses' in coco_data

        # Check info structure
        assert 'description' in coco_data['info']
        assert 'version' in coco_data['info']
        assert 'year' in coco_data['info']

    def test_categories_structure(self, temp_dir, sample_class_mapping):
        """Test that categories are correctly formatted."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        frame_data = {"bboxes": []}
        exporter.export_frame(frame_data, "test.jpg", 1920, 1080)
        exporter.finalize()

        json_file = temp_dir / "annotations.json"
        with open(json_file, 'r') as f:
            coco_data = json.load(f)

        categories = coco_data['categories']

        assert len(categories) == 3
        assert categories[0]['id'] == 0
        assert categories[0]['name'] == 'cyclist'
        assert categories[0]['supercategory'] == 'object'

    def test_annotation_structure(self, temp_dir, sample_class_mapping, sample_frame_data):
        """Test that annotations have all required COCO fields."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        exporter.export_frame(sample_frame_data, "frame_001.jpg", 1920, 1080)
        exporter.finalize()

        json_file = temp_dir / "annotations.json"
        with open(json_file, 'r') as f:
            coco_data = json.load(f)

        annotation = coco_data['annotations'][0]

        # Check required fields
        assert 'id' in annotation
        assert 'image_id' in annotation
        assert 'category_id' in annotation
        assert 'bbox' in annotation
        assert 'area' in annotation
        assert 'iscrowd' in annotation

        # Check field types
        assert isinstance(annotation['id'], int)
        assert isinstance(annotation['image_id'], int)
        assert isinstance(annotation['category_id'], int)
        assert isinstance(annotation['bbox'], list)
        assert len(annotation['bbox']) == 4
        assert annotation['iscrowd'] == 0


# ============================================================================
# TEST EDGE CASES
# ============================================================================

class TestCOCOEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_frame(self, temp_dir, sample_class_mapping):
        """Test exporting a frame with no bounding boxes."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        frame_data = {"bboxes": []}
        exporter.export_frame(frame_data, "empty_frame.jpg", 1920, 1080)

        assert len(exporter.images) == 1
        assert len(exporter.annotations) == 0

    def test_box_at_image_boundary(self, temp_dir, sample_class_mapping):
        """Test box that extends beyond image boundaries is clamped."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        # Box centered at (0.95, 0.95) with size (0.2, 0.2)
        # Should extend beyond image boundary and be clamped
        frame_data = {
            "bboxes": [
                {
                    "label": "cyclist",
                    "cx": 0.95,
                    "cy": 0.95,
                    "w": 0.2,
                    "h": 0.2
                }
            ]
        }

        exporter.export_frame(frame_data, "test.jpg", 1000, 1000)

        annotation = exporter.annotations[0]
        bbox = annotation['bbox']

        # Check that bbox is clamped to image boundaries
        assert bbox[0] >= 0
        assert bbox[1] >= 0
        assert bbox[0] + bbox[2] <= 1000
        assert bbox[1] + bbox[3] <= 1000

    def test_very_small_box(self, temp_dir, sample_class_mapping):
        """Test handling of very small bounding boxes."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        frame_data = {
            "bboxes": [
                {
                    "label": "cyclist",
                    "cx": 0.5,
                    "cy": 0.5,
                    "w": 0.01,  # 1% of image width
                    "h": 0.01   # 1% of image height
                }
            ]
        }

        exporter.export_frame(frame_data, "test.jpg", 1920, 1080)

        assert len(exporter.annotations) == 1
        annotation = exporter.annotations[0]
        # Should have non-zero area
        assert annotation['area'] > 0


# ============================================================================
# TEST COMPARISON WITH YOLO FORMAT
# ============================================================================

class TestCOCOvsYOLO:
    """Test that COCO format produces equivalent results to YOLO."""

    def test_same_data_different_format(self, temp_dir, sample_class_mapping):
        """Verify COCO and YOLO handle the same data correctly (different coordinate systems)."""
        exporter = COCOExporter(temp_dir, sample_class_mapping)

        # Use a simple centered box
        frame_data = {
            "bboxes": [
                {
                    "label": "cyclist",
                    "cx": 0.5,
                    "cy": 0.5,
                    "w": 0.4,
                    "h": 0.6
                }
            ]
        }

        exporter.export_frame(frame_data, "test.jpg", 1000, 1000)

        # YOLO format (normalized):
        # cx=0.5, cy=0.5, w=0.4, h=0.6

        # COCO format (absolute pixels):
        # x_min = (0.5 - 0.4/2) * 1000 = 300
        # y_min = (0.5 - 0.6/2) * 1000 = 200
        # width = 0.4 * 1000 = 400
        # height = 0.6 * 1000 = 600

        annotation = exporter.annotations[0]
        bbox = annotation['bbox']

        assert bbox == [300.0, 200.0, 400.0, 600.0]


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
