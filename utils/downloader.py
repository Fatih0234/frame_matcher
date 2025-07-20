import os
import requests
from typing import Tuple, List, Optional
from label_studio_sdk import Client
import logging

logger = logging.getLogger(__name__)

class LabelStudioDownloader:
    """Handles downloading videos and annotations from Label Studio"""
    
    def __init__(self, url: str, api_key: str, project_id: int):
        self.client = Client(url=url, api_key=api_key)
        self.project_id = project_id
        self.base_url = url
        
    def download_annotations(self, output_dir: str = "exported_json_annotation") -> Optional[str]:
        """Download annotations from Label Studio and save as annotations.json"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if annotations.json already exists
            output_file = os.path.join(output_dir, "annotations.json")
            if os.path.exists(output_file):
                logger.info(f"Annotations file already exists: {output_file}")
                return output_file
            
            logger.info("Downloading annotations from Label Studio...")
            project = self.client.get_project(self.project_id)
            
            # Create export snapshot
            export_result = project.export_snapshot_create(
                title='Export with Interpolated Keyframes',
                interpolate_key_frames=True
            )
            
            export_id = export_result['id']
            
            # Download the export
            status, filename = project.export_snapshot_download(
                export_id, export_type='JSON_MIN', path=output_dir
            )
            
            # Rename to standard name
            downloaded_file = os.path.join(output_dir, filename)
            if downloaded_file != output_file:
                os.rename(downloaded_file, output_file)
                logger.info(f"Renamed {filename} to annotations.json")
            
            logger.info(f"Successfully downloaded annotations to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to download annotations: {e}")
            return None
    
    def download_videos(self, output_dir: str = "exported_videos") -> Tuple[bool, List[str]]:
        """Download all videos from Label Studio project"""
        try:
            # Check if output directory exists
            if os.path.exists(output_dir):
                logger.info(f"Video directory already exists: {output_dir}. Skipping video download.")
                # Return existing video files
                existing_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
                return True, existing_files
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created video directory: {output_dir}")
            
            logger.info("Downloading videos from Label Studio...")
            project = self.client.get_project(self.project_id)
            tasks = project.get_tasks()
            
            downloaded_files = []
            failed_downloads = []
            
            for task in tasks:
                try:
                    # Get the media file URL from task data
                    if 'video' in task['data']:
                        video_url = task['data']['video']
                        
                        # If it's a relative URL, make it absolute
                        if video_url.startswith('/'):
                            video_url = f"{self.base_url}{video_url}"
                        
                        # Extract filename from the path (last component)
                        # e.g., "/data/upload/5/46763684-20250514_ride_bike_in_circles_part1.mp4" -> "46763684-20250514_ride_bike_in_circles_part1.mp4"
                        filename = os.path.basename(video_url)
                        if not filename.endswith('.mp4'):
                            filename = f"task_{task['id']}.mp4"
                        
                        filepath = os.path.join(output_dir, filename)
                        
                        # Skip if file already exists
                        if os.path.exists(filepath):
                            logger.info(f"Video already exists, skipping: {filename}")
                            downloaded_files.append(filename)
                            continue
                        
                        # Download the video file
                        headers = {'Authorization': f'Token {self.client.api_key}'}
                        response = requests.get(video_url, headers=headers, stream=True)
                        
                        if response.status_code == 200:
                            with open(filepath, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            logger.info(f"Downloaded: {filename}")
                            downloaded_files.append(filename)
                        else:
                            error_msg = f"Failed to download {video_url}: HTTP {response.status_code}"
                            logger.error(error_msg)
                            failed_downloads.append(filename)
                            
                except Exception as e:
                    error_msg = f"Error downloading video for task {task.get('id', 'unknown')}: {e}"
                    logger.error(error_msg)
                    failed_downloads.append(f"task_{task.get('id', 'unknown')}.mp4")
            
            if failed_downloads:
                logger.warning(f"Failed to download {len(failed_downloads)} videos: {failed_downloads}")
            
            logger.info(f"Successfully downloaded {len(downloaded_files)} videos")
            return len(downloaded_files) > 0, downloaded_files
            
        except Exception as e:
            logger.error(f"Failed to download videos: {e}")
            return False, []
    
    def download_all(self, video_dir: str = "exported_videos", json_dir: str = "exported_json_annotation") -> Tuple[bool, Optional[str], List[str]]:
        """Download both annotations and videos"""
        logger.info("Starting Label Studio data download...")
        
        # Download annotations first
        annotations_file = self.download_annotations(json_dir)
        if not annotations_file:
            logger.error("Failed to download annotations. Aborting.")
            return False, None, []
        
        # Download videos
        videos_success, video_files = self.download_videos(video_dir)
        if not videos_success:
            logger.warning("Video download failed, but continuing with annotations only.")
        
        return True, annotations_file, video_files
