"""
Video file matcher utility.
"""

from pathlib import Path
from typing import Optional, List
import re


class VideoMatcher:
    def __init__(self, video_files_dir: Path):
        """
        Initialize video matcher.
        
        Args:
            video_files_dir: Directory containing video files
        """
        self.video_files_dir = video_files_dir
        self.video_files = self._get_video_files()
    
    def _get_video_files(self) -> List[Path]:
        """Get all video files from the directory."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        video_files = []
        
        for file_path in self.video_files_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                video_files.append(file_path)
        
        return video_files
    
    def find_matching_video(self, json_video_path: str) -> Optional[Path]:
        """
        Find the local video file that matches the JSON video path.
        
        Args:
            json_video_path: Video path from JSON (e.g., "/data/upload/4/3b780495-20250514_ride_bike_in_circles_60sec.mp4")
            
        Returns:
            Path to matching local video file, or None if not found
        """
        # Extract filename from JSON path
        json_filename = Path(json_video_path).name
        
        # Strategy 1: Direct filename match
        for video_file in self.video_files:
            if video_file.name == json_filename:
                return video_file
        
        # Strategy 2: Extract meaningful part after the dash
        # Look for pattern like "3b780495-20250514_ride_bike_in_circles_60sec.mp4"
        # and extract "20250514_ride_bike_in_circles_60sec.mp4"
        match = re.search(r'-(.+)$', json_filename)
        if match:
            meaningful_part = match.group(1)
            for video_file in self.video_files:
                if video_file.name == meaningful_part:
                    return video_file
        
        # Strategy 3: Fuzzy matching - check if meaningful part is contained in filename
        if match:
            meaningful_part = match.group(1)
            # Remove extension for comparison
            meaningful_stem = Path(meaningful_part).stem
            for video_file in self.video_files:
                if meaningful_stem in video_file.stem:
                    return video_file
        
        # Strategy 4: Check if any part of the JSON filename matches
        json_stem = Path(json_filename).stem
        for video_file in self.video_files:
            # Check if the video filename contains any significant part of the JSON filename
            if len(json_stem) > 10:  # Only for meaningful length names
                # Remove common prefixes like hash parts
                clean_json_stem = re.sub(r'^[a-f0-9]+-', '', json_stem)
                if clean_json_stem in video_file.stem or video_file.stem in clean_json_stem:
                    return video_file
        
        # If no match found, print available files for debugging
        print(f"❌ No match found for: {json_filename}")
        print(f"Available video files: {[f.name for f in self.video_files]}")
        
        return None