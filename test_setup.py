#!/usr/bin/env python3
"""
Test script to validate the video annotation converter installation and setup.
"""

import sys
from pathlib import Path
import json

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import typer
        print("✅ typer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import typer: {e}")
        return False
    
    try:
        import cv2
        print("✅ opencv-python imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import opencv-python: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import numpy: {e}")
        return False
    
    try:
        import yaml
        print("✅ PyYAML imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PyYAML: {e}")
        return False
    
    try:
        from utils import AnnotationProcessor
        print("✅ utils package imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import utils package: {e}")
        print("Make sure you're running this from the project root directory")
        return False
    
    return True

def test_project_structure():
    """Test if the required project structure exists."""
    print("\n🏗️  Testing project structure...")
    
    required_paths = [
        "utils/",
        "utils/__init__.py",
        "utils/annotation_processor.py",
        "utils/video_matcher.py", 
        "utils/frame_extractor.py",
        "utils/yolo_converter.py",
        "utils/coco_converter.py"
    ]
    
    all_good = True
    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            print(f"✅ {path_str} exists")
        else:
            print(f"❌ {path_str} missing")
            all_good = False
    
    return all_good

def test_json_structure():
    """Test if example JSON structure can be parsed."""
    print("\n📄 Testing JSON parsing...")
    
    # Test with the provided example
    sample_json = '''[
        {
            "video": "/data/upload/4/3b780495-20250514_ride_bike_in_circles_60sec.mp4",
            "id": 19,
            "box": [
                {
                    "framesCount": 1500,
                    "duration": 60,
                    "sequence": [
                        {
                            "frame": 346,
                            "enabled": true,
                            "rotation": 0,
                            "x": 68.49907460136674,
                            "y": 41.93242217160213,
                            "width": 1.334709567198172,
                            "height": 5.979498861047836,
                            "time": 13.84
                        }
                    ],
                    "labels": ["person"]
                }
            ]
        }
    ]'''
    
    try:
        data = json.loads(sample_json)
        print("✅ JSON parsing successful")
        
        # Test class validation logic
        class_mappings = {"person": 1, "cyclist": 0}
        annotation_classes = set()
        
        for annotation in data:
            for box in annotation.get('box', []):
                labels = box.get('labels', [])
                annotation_classes.update(labels)
        
        missing_classes = annotation_classes - set(class_mappings.keys())
        if not missing_classes:
            print("✅ Class mapping validation successful")
        else:
            print(f"⚠️  Missing classes in mapping: {missing_classes}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        return False

def test_cli_help():
    """Test if the CLI help command works."""
    print("\n🖥️  Testing CLI interface...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "main.py", "--help"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "Convert video annotations" in result.stdout:
            print("✅ CLI help command successful")
            return True
        else:
            print(f"❌ CLI help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running Video Annotation Converter Tests\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Project Structure Test", test_project_structure), 
        ("JSON Parsing Test", test_json_structure),
        ("CLI Interface Test", test_cli_help)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())