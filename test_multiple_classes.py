"""
Test script to verify that our code correctly handles multiple classes per frame
"""

from utils.yolo_converter import YOLOConverter
from pathlib import Path
import tempfile

def test_multiple_classes_per_frame():
    """Test if multiple classes get saved to the same annotation file"""
    
    # Create class mappings
    class_mappings = {"cyclist": 0, "person": 1, "scooter-roller": 2}
    
    # Create YOLO converter
    yolo_converter = YOLOConverter(class_mappings)
    
    # Create test annotations for a single frame with multiple classes
    test_annotations = [
        {
            'class_id': 0,
            'class_name': 'cyclist',
            'x': 10.0,
            'y': 20.0,
            'width': 30.0,
            'height': 40.0,
            'frame': 100,
            'time': 4.0
        },
        {
            'class_id': 1,
            'class_name': 'person',
            'x': 50.0,
            'y': 60.0,
            'width': 25.0,
            'height': 35.0,
            'frame': 100,
            'time': 4.0
        },
        {
            'class_id': 2,
            'class_name': 'scooter-roller',
            'x': 70.0,
            'y': 80.0,
            'width': 20.0,
            'height': 30.0,
            'frame': 100,
            'time': 4.0
        }
    ]
    
    # Mock image shape (height, width, channels)
    image_shape = (480, 640, 3)
    
    # Create temporary file for output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        # Test the function
        yolo_converter.create_yolo_annotation(test_annotations, image_shape, temp_path)
        
        # Read the result
        with open(temp_path, 'r') as f:
            content = f.read()
        
        print("Generated YOLO annotation file content:")
        print("=" * 50)
        print(content)
        print("=" * 50)
        
        lines = content.strip().split('\n')
        print(f"\nNumber of lines in annotation file: {len(lines)}")
        print(f"Expected: 3 lines (one for each class)")
        
        for i, line in enumerate(lines):
            parts = line.split()
            class_id = parts[0]
            class_name = [name for name, id in class_mappings.items() if str(id) == class_id][0]
            print(f"Line {i+1}: Class {class_id} ({class_name}) - {' '.join(parts[1:])}")
            
        return len(lines) == 3
        
    finally:
        # Clean up
        temp_path.unlink()

if __name__ == "__main__":
    success = test_multiple_classes_per_frame()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
