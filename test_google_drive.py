#!/usr/bin/env python3
"""
Test script for Google Drive integration.
Run this to verify your Google Drive setup is working.
"""

import sys
from pathlib import Path

def test_google_drive_import():
    """Test if Google Drive modules can be imported."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("utils.google_drive_uploader")
        if spec is not None:
            print("✅ Google Drive uploader module imported successfully")
            return True
        else:
            print("❌ Google Drive uploader module not found")
            return False
    except ImportError as e:
        print(f"❌ Failed to import Google Drive uploader: {e}")
        return False

def test_credentials_file():
    """Test if credentials file exists."""
    credentials_file = Path("credentials.json")
    if credentials_file.exists():
        print("✅ credentials.json file found")
        return True
    else:
        print("⚠️  credentials.json file not found")
        print("   Please follow the setup instructions in GOOGLE_DRIVE_SETUP.md")
        return False

def test_google_drive_connection():
    """Test connection to Google Drive."""
    try:
        from utils.google_drive_uploader import GoogleDriveUploader
        
        print("🔄 Testing Google Drive connection...")
        _ = GoogleDriveUploader()  # Test authentication
        print("✅ Successfully connected to Google Drive!")
        return True
        
    except FileNotFoundError:
        print("❌ credentials.json not found")
        print("   Please follow the setup instructions in GOOGLE_DRIVE_SETUP.md")
        return False
    except Exception as e:
        print(f"❌ Failed to connect to Google Drive: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Google Drive Integration")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_google_drive_import),
        ("Credentials Test", test_credentials_file),
        ("Connection Test", test_google_drive_connection),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 40)
    print("📊 Test Summary:")
    
    all_passed = all(results)
    if all_passed:
        print("🎉 All tests passed! Google Drive integration is ready.")
        print("\nYou can now use the --upload-to-drive option:")
        print("python main.py --format yolo --classes '{...}' --output ./dataset --upload-to-drive")
    else:
        print("⚠️  Some tests failed. Please check the setup instructions.")
        print("📖 See GOOGLE_DRIVE_SETUP.md for detailed setup instructions.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
