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
        
    def download_annotations(self, output_dir: str = "exported_json_annotation", selected_task_ids: List[int] = None) -> Optional[str]:
        """Download annotations from Label Studio and save as annotations.json.
        
        Args:
            output_dir: Directory to save annotations
            selected_task_ids: List of task IDs to include. If None, includes all tasks.
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Always refresh annotations - delete existing file if present
            output_file = os.path.join(output_dir, "annotations.json")
            if os.path.exists(output_file):
                logger.info(f"Deleting existing annotations file: {output_file}")
                os.remove(output_file)
            
            logger.info("Downloading fresh annotations from Label Studio...")
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
            
            # If specific task IDs were requested, filter the annotations
            if selected_task_ids:
                logger.info(f"Filtering annotations to {len(selected_task_ids)} selected tasks...")
                self._filter_annotations_file(downloaded_file, selected_task_ids)
            
            if downloaded_file != output_file:
                os.rename(downloaded_file, output_file)
                logger.info(f"Renamed {filename} to annotations.json")
            
            logger.info(f"Successfully downloaded annotations to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to download annotations: {e}")
            return None
    
    def download_videos(self, output_dir: str = "exported_videos", selected_task_ids: List[int] = None) -> Tuple[bool, List[str]]:
        """Download videos from Label Studio project.
        
        Args:
            output_dir: Directory to save videos
            selected_task_ids: List of task IDs to download. If None, downloads all videos.
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info("Checking for existing videos and downloading missing ones...")
            project = self.client.get_project(self.project_id)
            tasks = project.get_tasks()
            
            # Filter tasks if specific task IDs were provided
            if selected_task_ids:
                tasks = [task for task in tasks if task['id'] in selected_task_ids]
                logger.info(f"Processing {len(tasks)} selected videos")
            else:
                logger.info(f"Processing all {len(tasks)} videos")
            
            downloaded_files = []
            skipped_files = []
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
                        
                        # Check if file already exists
                        if os.path.exists(filepath):
                            logger.info(f"Video already exists, skipping: {filename}")
                            skipped_files.append(filename)
                            continue
                        
                        # Download the video file
                        logger.info(f"Downloading: {filename}")
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
            
            # Summary
            all_files = downloaded_files + skipped_files
            logger.info("Video download summary:")
            logger.info(f"   - Downloaded: {len(downloaded_files)} videos")
            logger.info(f"   - Skipped (existing): {len(skipped_files)} videos") 
            logger.info(f"   - Failed: {len(failed_downloads)} videos")
            
            if failed_downloads:
                logger.warning(f"Failed to download: {failed_downloads}")
            
            return len(all_files) > 0, all_files
            
        except Exception as e:
            logger.error(f"Failed to download videos: {e}")
            return False, []
    
    def download_all(self, video_dir: str = "exported_videos", json_dir: str = "exported_json_annotation", selected_task_ids: List[int] = None) -> Tuple[bool, Optional[str], List[str]]:
        """Download both annotations and videos.
        
        Args:
            video_dir: Directory to save videos
            json_dir: Directory to save annotations
            selected_task_ids: List of task IDs to download. If None, downloads all videos.
        """
        logger.info("Starting Label Studio data download...")
        
        # Download annotations first
        annotations_file = self.download_annotations(json_dir, selected_task_ids)
        if not annotations_file:
            logger.error("Failed to download annotations. Aborting.")
            return False, None, []
        
        # Download videos
        videos_success, video_files = self.download_videos(video_dir, selected_task_ids)
        if not videos_success:
            logger.warning("Video download failed, but continuing with annotations only.")
        
        return True, annotations_file, video_files

    def _filter_annotations_file(self, annotations_file: str, selected_task_ids: List[int]):
        """Filter annotations file to only include selected task IDs."""
        try:
            import json
            
            # Read the annotations file
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            
            # Filter annotations to only include selected task IDs
            if isinstance(annotations, list):
                # Check if annotations have task IDs
                filtered_annotations = []
                for annotation in annotations:
                    # Try to find task ID in various possible locations
                    task_id = None
                    if 'task' in annotation:
                        task_id = annotation['task']
                    elif 'id' in annotation:
                        task_id = annotation['id']
                    elif 'task_id' in annotation:
                        task_id = annotation['task_id']
                    
                    if task_id in selected_task_ids:
                        filtered_annotations.append(annotation)
                
                # Write filtered annotations back
                with open(annotations_file, 'w') as f:
                    json.dump(filtered_annotations, f, indent=2)
                
                logger.info(f"Filtered annotations: {len(filtered_annotations)}/{len(annotations)} tasks included")
            else:
                logger.warning("Unexpected annotations format - skipping filtering")
                
        except Exception as e:
            logger.error(f"Failed to filter annotations: {e}")
            # Continue without filtering if there's an error
