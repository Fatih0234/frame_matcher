"""
Frame extraction utility using OpenCV.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List


class FrameExtractor:
    def __init__(self):
        """Initialize frame extractor."""
        self._video_cache = {}  # Cache for opened video captures
    
    def extract_frames_batch(self, video_path: Path, frame_numbers: List[int]) -> Dict[int, Optional[np.ndarray]]:
        """
        Extract multiple frames from a video file efficiently.
        
        Args:
            video_path: Path to the video file
            frame_numbers: List of frame numbers to extract (1-indexed)
            
        Returns:
            Dictionary mapping frame numbers to extracted frames
        """
        results = {}
        
        if not frame_numbers:
            return results
        
        try:
            # Open video file once
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                print(f"❌ Error: Cannot open video file {video_path}")
                return {fn: None for fn in frame_numbers}
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sort frame numbers for efficient sequential access
            sorted_frames = sorted(frame_numbers)
            
            # Extract frames sequentially
            for frame_number in sorted_frames:
                # Validate frame number
                if frame_number < 1 or frame_number > total_frames:
                    print(f"⚠️  Frame {frame_number} out of range for {video_path.name} (total: {total_frames})")
                    results[frame_number] = None
                    continue
                
                # Set frame position (convert to 0-indexed)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                
                # Read the frame
                ret, frame = cap.read()
                
                if not ret:
                    print(f"❌ Error: Cannot read frame {frame_number} from {video_path.name}")
                    results[frame_number] = None
                else:
                    results[frame_number] = frame
            
            cap.release()
            return results
            
        except Exception as e:
            print(f"❌ Error extracting frames from {video_path}: {e}")
            return {fn: None for fn in frame_numbers}
    
    def extract_frame(self, video_path: Path, frame_number: int) -> Optional[np.ndarray]:
        """
        Extract a specific frame from a video file.
        Fallback method for single frame extraction.
        
        Args:
            video_path: Path to the video file
            frame_number: Frame number to extract (1-indexed as per annotation)
            
        Returns:
            Frame as numpy array (BGR format) or None if extraction failed
        """
        # Use batch extraction for single frame (more efficient)
        results = self.extract_frames_batch(video_path, [frame_number])
        return results.get(frame_number)
    
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