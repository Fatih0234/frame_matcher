"""
Interactive video selector for Label Studio projects.
"""

import os
from typing import List, Optional
from label_studio_sdk import Client
import logging

logger = logging.getLogger(__name__)


class VideoInfo:
    """Data class to hold video information."""
    
    def __init__(self, task_id: int, filename: str, video_url: str, duration: Optional[float] = None):
        self.task_id = task_id
        self.filename = filename
        self.video_url = video_url
        self.duration = duration
    
    def __str__(self):
        duration_str = f" ({self.duration:.1f}s)" if self.duration else ""
        return f"[ID: {self.task_id}] {self.filename}{duration_str}"


class InteractiveVideoSelector:
    """Interactive selector for choosing videos from Label Studio."""
    
    def __init__(self, url: str, api_key: str, project_id: int):
        self.client = Client(url=url, api_key=api_key)
        self.project_id = project_id
        self.base_url = url
    
    def get_available_videos(self) -> List[VideoInfo]:
        """Fetch all available videos from the Label Studio project."""
        try:
            logger.info("Fetching available videos from Label Studio...")
            project = self.client.get_project(self.project_id)
            tasks = project.get_tasks()
            
            videos = []
            for task in tasks:
                if 'video' in task['data']:
                    video_url = task['data']['video']
                    
                    # Extract filename from the path
                    filename = os.path.basename(video_url)
                    if not filename.endswith('.mp4'):
                        filename = f"task_{task['id']}.mp4"
                    
                    # Try to get duration if available
                    duration = None
                    if 'annotations' in task and task['annotations']:
                        # Look for duration in annotation results
                        for annotation in task['annotations']:
                            for result in annotation.get('result', []):
                                if result.get('type') == 'videorectangle':
                                    duration = result.get('value', {}).get('duration')
                                    break
                            if duration:
                                break
                    
                    video_info = VideoInfo(
                        task_id=task['id'],
                        filename=filename,
                        video_url=video_url,
                        duration=duration
                    )
                    videos.append(video_info)
            
            logger.info(f"Found {len(videos)} videos in the project")
            return videos
            
        except Exception as e:
            logger.error(f"Failed to fetch videos: {e}")
            return []
    
    def display_video_menu(self, videos: List[VideoInfo]) -> None:
        """Display the video selection menu."""
        print("\n" + "="*80)
        print("AVAILABLE VIDEOS IN LABEL STUDIO PROJECT")
        print("="*80)
        
        if not videos:
            print("No videos found in the project.")
            return
        
        for i, video in enumerate(videos, 1):
            print(f"{i:2d}. {video}")
        
        print("\n" + "-"*80)
        print("SELECTION OPTIONS:")
        print("   • Enter numbers separated by spaces (e.g., '1 3 5')")
        print("   • Enter ranges with dash (e.g., '1-3 7-9')")
        print("   • Enter 'all' to select all videos")
        print("   • Enter 'quit' or 'q' to exit")
        print("-"*80)
    
    def parse_selection(self, selection_input: str, max_index: int) -> List[int]:
        """Parse user selection input and return list of selected indices."""
        selection_input = selection_input.strip().lower()
        
        if selection_input in ['quit', 'q']:
            return []
        
        if selection_input == 'all':
            return list(range(1, max_index + 1))
        
        selected_indices = set()
        
        try:
            # Split by spaces and process each part
            parts = selection_input.split()
            
            for part in parts:
                if '-' in part:
                    # Handle ranges like "1-3"
                    start, end = map(int, part.split('-'))
                    if start > end:
                        start, end = end, start  # Swap if reversed
                    
                    for i in range(start, end + 1):
                        if 1 <= i <= max_index:
                            selected_indices.add(i)
                else:
                    # Handle single numbers
                    num = int(part)
                    if 1 <= num <= max_index:
                        selected_indices.add(num)
            
            return sorted(list(selected_indices))
            
        except ValueError:
            print("Invalid input format. Please use numbers, ranges, or 'all'.")
            return None
    
    def select_videos_interactive(self) -> List[VideoInfo]:
        """Main interactive video selection interface."""
        videos = self.get_available_videos()
        
        if not videos:
            print("No videos available for selection.")
            return []
        
        while True:
            self.display_video_menu(videos)
            
            try:
                user_input = input("\nSelect videos to process: ").strip()
                
                if user_input.lower() in ['quit', 'q']:
                    print("Exiting video selection.")
                    return []
                
                selected_indices = self.parse_selection(user_input, len(videos))
                
                if selected_indices is None:
                    continue  # Invalid input, try again
                
                if not selected_indices:
                    if user_input.lower() not in ['quit', 'q']:
                        print("No valid videos selected. Please try again.")
                        continue
                    else:
                        return []
                
                # Get selected videos
                selected_videos = [videos[i-1] for i in selected_indices]
                
                # Confirm selection
                print(f"\nSelected {len(selected_videos)} videos:")
                for video in selected_videos:
                    print(f"   • {video}")
                
                confirm = input(f"\n🤔 Process these {len(selected_videos)} videos? (y/N): ").strip().lower()
                
                if confirm in ['y', 'yes']:
                    return selected_videos
                else:
                    print("Let's try again...\n")
                    continue
                    
            except KeyboardInterrupt:
                print("\n\nSelection cancelled by user.")
                return []
            except Exception as e:
                print(f"Error during selection: {e}")
                continue
    
    def get_selected_video_tasks(self, selected_videos: List[VideoInfo]) -> List[int]:
        """Extract task IDs from selected videos."""
        return [video.task_id for video in selected_videos]


def create_interactive_selector(url: str, api_key: str, project_id: int) -> InteractiveVideoSelector:
    """Factory function to create an interactive video selector."""
    return InteractiveVideoSelector(url, api_key, project_id)
