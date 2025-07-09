"""
Frame extraction utility using OpenCV.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


class FrameExtractor:
    def __init__(self):
        """Initialize frame extractor."""
        pass
    
    def extract_frame(self, video_path: Path, frame_number: int) -> Optional[np.ndarray]:
        """
        Extract a specific frame from a video file.
        
        Args:
            video_path: Path to the video file
            frame_number: Frame number to extract (1-indexed as per annotation)
            
        Returns:
            Frame as numpy array (BGR format) or None if extraction failed
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                print(f"❌ Error: Cannot open video file {video_path}")
                return None
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Validate frame number
            if frame_number < 1 or frame_number > total_frames:
                print(f"⚠️  Warning: Frame {frame_number} out of range for video {video_path.name} (total frames: {total_frames})")
                cap.release()
                return None
            
            # Set frame position (convert to 0-indexed)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            
            # Read the frame
            ret, frame = cap.read()
            
            cap.release()
            
            if not ret:
                print(f"❌ Error: Cannot read frame {frame_number} from {video_path.name}")
                return None
            
            return frame
            
        except Exception as e:
            print(f"❌ Error extracting frame {frame_number} from {video_path}: {e}")
            return None
    
    def get_video_info(self, video_path: Path) -> dict:
        """
        Get basic information about a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video information
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                return {}
            
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            }
            
            cap.release()
            return info
            
        except Exception as e:
            print(f"❌ Error getting video info for {video_path}: {e}")
            return {}