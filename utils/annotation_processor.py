"""
Unified annotation processor that combines optimized processing with interactive selection.
This is the main processor that replaces both annotation_processor.py and optimized_annotation_processor.py
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
import logging

from .video_matcher import VideoMatcher
from .yolo_converter import YOLOConverter

logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Analyzes dataset statistics and generates reports."""
    
    def __init__(self, class_mappings: Dict[str, int]):
        self.class_mappings = class_mappings
        self.video_stats = {}  # video_name -> {frames, objects_per_class}
        
    def add_video_stats(self, video_name: str, frames_extracted: int, objects_per_class: Dict[str, int]):
        """Add statistics for a processed video."""
        self.video_stats[video_name] = {
            'frames': frames_extracted,
            'objects_per_class': objects_per_class.copy(),
            'total_objects': sum(objects_per_class.values())
        }
        
    def generate_report(self):
        """Generate and display comprehensive dataset analysis report."""
        if not self.video_stats:
            logger.info("No videos processed - skipping analysis report")
            return
            
        logger.info("DATASET ANALYSIS REPORT")
        logger.info("=" * 80)
        
        # Prepare table data
        class_names = sorted(self.class_mappings.keys())
        
        # Table header
        header = f"{'Video Name':<40} | {'Frames':<6} | {'Objects':<7} | {'Avg/Frame':<9}"
        for class_name in class_names:
            header += f" | {class_name:<10}"
        logger.info(header)
        logger.info("-" * len(header))
        
        # Video rows
        total_frames = 0
        total_objects = 0
        total_objects_per_class = {class_name: 0 for class_name in class_names}
        
        for video_name, stats in self.video_stats.items():
            frames = stats['frames']
            total_obj = stats['total_objects']
            avg_per_frame = total_obj / frames if frames > 0 else 0.0
            
            # Truncate video name if too long
            display_name = video_name[:37] + "..." if len(video_name) > 40 else video_name
            
            row = f"{display_name:<40} | {frames:<6} | {total_obj:<7} | {avg_per_frame:<9.2f}"
            
            for class_name in class_names:
                class_count = stats['objects_per_class'].get(class_name, 0)
                row += f" | {class_count:<10}"
                total_objects_per_class[class_name] += class_count
                
            logger.info(row)
            
            total_frames += frames
            total_objects += total_obj
            
        # Grand totals
        logger.info("=" * len(header))
        overall_avg = total_objects / total_frames if total_frames > 0 else 0.0
        
        total_row = f"{'TOTAL':<40} | {total_frames:<6} | {total_objects:<7} | {overall_avg:<9.2f}"
        for class_name in class_names:
            total_row += f" | {total_objects_per_class[class_name]:<10}"
        logger.info(total_row)
        
        logger.info("=" * 80)
        logger.info("📈 SUMMARY:")
        logger.info(f"   • Videos Processed: {len(self.video_stats)}")
        logger.info(f"   • Total Frames: {total_frames:,}")
        logger.info(f"   • Total Objects: {total_objects:,}")
        logger.info(f"   • Average Objects per Frame: {overall_avg:.2f}")
        
        # Class distribution
        logger.info("   • Class Distribution:")
        for class_name in class_names:
            count = total_objects_per_class[class_name]
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            logger.info(f"     - {class_name}: {count:,} objects ({percentage:.1f}%)")

logger = logging.getLogger(__name__)


class OptimizedFrameExtractor:
    """High-performance frame extractor with caching and optimized extraction."""
    
    def __init__(self, max_cache_size: int = 10):
        self.max_cache_size = max_cache_size
        self._video_metadata_cache = {}
    
    @lru_cache(maxsize=128)
    def get_video_info_cached(self, video_path_str: str) -> dict:
        """Cache video metadata to avoid repeated file access."""
        video_path = Path(video_path_str)
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
            logger.error(f"Error getting video info for {video_path}: {e}")
            return {}
    
    def calculate_optimal_batch_size(self, frame_width: int, frame_height: int, 
                                   available_memory_mb: int = 2048) -> int:
        """Calculate optimal batch size based on frame dimensions and available memory."""
        if frame_width <= 0 or frame_height <= 0:
            return 100  # Default fallback
        
        # Estimate frame size in MB (3 channels, 8 bits per channel)
        frame_size_mb = (frame_width * frame_height * 3) / (1024 * 1024)
        
        # Use 80% of available memory for safety
        usable_memory = available_memory_mb * 0.8
        
        # Calculate batch size
        batch_size = max(1, int(usable_memory / frame_size_mb))
        
        # Cap at reasonable limits
        return min(max(batch_size, 10), 1000)
    
    def extract_frames_sequential_optimized(self, video_path: Path, 
                                          frame_numbers: List[int]) -> Dict[int, Optional[np.ndarray]]:
        """
        Optimized sequential frame extraction without seeking.
        Much faster than seek-based extraction for dense frame sets.
        """
        if not frame_numbers:
            return {}
        
        results = {}
        sorted_frames = sorted(set(frame_numbers))
        frame_set = set(sorted_frames)
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video file {video_path}")
                return {fn: None for fn in frame_numbers}
            
            current_frame = 0
            frames_found = 0
            target_frames = len(sorted_frames)
            
            logger.debug(f"Sequential extraction: seeking {target_frames} frames from {video_path.name}")
            
            while cap.isOpened() and frames_found < target_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_frame += 1
                
                if current_frame in frame_set:
                    results[current_frame] = frame.copy()
                    frames_found += 1
                    
                    # Early exit if we've found all frames and they're at the beginning
                    if current_frame >= sorted_frames[-1]:
                        break
            
            cap.release()
            
            # Fill missing frames with None
            for frame_num in frame_numbers:
                if frame_num not in results:
                    results[frame_num] = None
            
            logger.debug(f"Sequential extraction complete: {frames_found}/{target_frames} frames extracted")
            return results
            
        except Exception as e:
            logger.error(f"Error in sequential frame extraction from {video_path}: {e}")
            return {fn: None for fn in frame_numbers}
    
    def extract_frames_batch_optimized(self, video_path: Path, 
                                     frame_numbers: List[int]) -> Dict[int, Optional[np.ndarray]]:
        """
        Optimized batch frame extraction with adaptive strategy.
        Chooses between sequential and seek-based extraction based on frame density.
        """
        if not frame_numbers:
            return {}
        
        # Get video info
        video_info = self.get_video_info_cached(str(video_path))
        if not video_info:
            return {fn: None for fn in frame_numbers}
        
        total_frames = video_info['total_frames']
        sorted_frames = sorted(set(frame_numbers))
        
        # Calculate frame density
        if len(sorted_frames) > 1:
            frame_span = sorted_frames[-1] - sorted_frames[0] + 1
            density = len(sorted_frames) / frame_span
        else:
            density = 1.0
        
        # Use sequential extraction for dense frame sets (> 30% density)
        # or when extracting many frames
        if density > 0.3 or len(sorted_frames) > 100:
            logger.debug(f"Using sequential extraction (density: {density:.2f})")
            return self.extract_frames_sequential_optimized(video_path, frame_numbers)
        else:
            logger.debug(f"Using seek-based extraction (density: {density:.2f})")
            return self._extract_frames_seek_based(video_path, frame_numbers, total_frames)
    
    def _extract_frames_seek_based(self, video_path: Path, frame_numbers: List[int], 
                                 total_frames: int) -> Dict[int, Optional[np.ndarray]]:
        """Traditional seek-based extraction for sparse frame sets."""
        results = {}
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {fn: None for fn in frame_numbers}
            
            for frame_number in sorted(set(frame_numbers)):
                if frame_number < 1 or frame_number > total_frames:
                    results[frame_number] = None
                    continue
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                ret, frame = cap.read()
                
                if ret:
                    results[frame_number] = frame
                else:
                    results[frame_number] = None
            
            cap.release()
            return results
            
        except Exception as e:
            logger.error(f"Error in seek-based extraction from {video_path}: {e}")
            return {fn: None for fn in frame_numbers}


class OptimizedCoordinateConverter:
    """Vectorized coordinate conversion for better performance."""
    
    @staticmethod
    def convert_bbox_batch_yolo(bboxes: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
        """
        Vectorized bbox conversion to YOLO format.
        
        Args:
            bboxes: Nx4 array of [x, y, width, height] in percentages (0-100)
            img_width, img_height: Image dimensions
            
        Returns:
            Nx4 array of [center_x, center_y, width, height] in normalized coordinates (0-1)
        """
        if len(bboxes) == 0:
            return np.array([])
        
        # Convert to numpy array if not already
        bboxes = np.asarray(bboxes, dtype=np.float32)
        
        # Convert percentages to normalized coordinates (0-1)
        normalized = bboxes / 100.0
        
        # Extract coordinates
        x, y, width, height = normalized[:, 0], normalized[:, 1], normalized[:, 2], normalized[:, 3]
        
        # Convert to YOLO format (center coordinates)
        center_x = x + (width / 2.0)
        center_y = y + (height / 2.0)
        
        # Ensure coordinates are within bounds
        center_x = np.clip(center_x, 0.0, 1.0)
        center_y = np.clip(center_y, 0.0, 1.0)
        width = np.clip(width, 0.0, 1.0)
        height = np.clip(height, 0.0, 1.0)
        
        # Stack results
        result = np.column_stack([center_x, center_y, width, height])
        return result


class AnnotationProcessor:
    """
    Unified annotation processor with optimized processing and interactive selection support.
    This replaces both the original and optimized annotation processors.
    """
    
    def __init__(self, annotations_file: Path, video_files_dir: Path, 
                 class_mappings: Dict[str, int], use_exact_matching: bool = False,
                 max_workers: int = 4, memory_limit_mb: int = 2048, fps_limit: Optional[float] = None):
        """
        Initialize the annotation processor.
        
        Args:
            annotations_file: Path to the JSON annotations file
            video_files_dir: Directory containing video files
            class_mappings: Dictionary mapping class names to their integer encodings
            use_exact_matching: If True, prefer exact filename matching for videos
            max_workers: Maximum number of parallel workers for optimization
            memory_limit_mb: Memory limit for batch processing
            fps_limit: Target frames per second for sampling (None = extract all frames)
        """
        self.annotations_file = annotations_file
        self.video_files_dir = video_files_dir
        self.class_mappings = class_mappings
        self.use_exact_matching = use_exact_matching
        self.max_workers = max_workers
        self.memory_limit_mb = memory_limit_mb
        self.fps_limit = fps_limit
        
        # Performance tracking (initialize before loading annotations)
        self.performance_stats = {
            'videos_processed': 0,
            'frames_extracted': 0,
            'total_processing_time': 0,
            'extraction_time': 0,
            'conversion_time': 0,
            'io_time': 0
        }
        
        # Load and validate annotations
        self.annotations = self._load_annotations()
        self._validate_class_mappings()
        
        # Initialize components
        self.video_matcher = VideoMatcher(video_files_dir)
        self.frame_extractor = OptimizedFrameExtractor()
        self.coord_converter = OptimizedCoordinateConverter()
        
        # Initialize dataset analyzer for statistics tracking
        self.dataset_analyzer = DatasetAnalyzer(class_mappings)
    
    def _sample_consecutive_frames(self, frame_numbers: List[int], fps_limit: float, video_fps: int = 24) -> List[int]:
        """
        Sample frames from consecutive sequences based on fps_limit.
        
        Args:
            frame_numbers: List of consecutive frame numbers
            fps_limit: Target frames per second
            video_fps: Original video frames per second (default: 24)
            
        Returns:
            List of sampled frame numbers
        """
        if fps_limit is None:
            return frame_numbers
        
        # Calculate sampling interval
        sampling_interval = int(video_fps / fps_limit)
        if sampling_interval <= 1:
            return frame_numbers  # No sampling needed if interval is 1 or less
        
        # Group consecutive sequences
        if not frame_numbers:
            return []
        
        frame_numbers = sorted(frame_numbers)
        sequences = []
        current_sequence = [frame_numbers[0]]
        
        # Detect consecutive sequences
        for i in range(1, len(frame_numbers)):
            if frame_numbers[i] == frame_numbers[i-1] + 1:
                current_sequence.append(frame_numbers[i])
            else:
                sequences.append(current_sequence)
                current_sequence = [frame_numbers[i]]
        sequences.append(current_sequence)
        
        # Sample from each consecutive sequence
        sampled_frames = []
        for sequence in sequences:
            if len(sequence) == 1:
                # Single frame - always include
                sampled_frames.extend(sequence)
            else:
                # Sample from sequence: start + every sampling_interval + end
                sequence_samples = [sequence[0]]  # Always include first frame
                
                # Sample intermediate frames
                for i in range(sampling_interval, len(sequence), sampling_interval):
                    sequence_samples.append(sequence[i])
                
                # Always include last frame if it's not already included
                if len(sequence) > 1 and sequence[-1] not in sequence_samples:
                    sequence_samples.append(sequence[-1])
                
                sampled_frames.extend(sequence_samples)
        
        return sorted(sampled_frames)
    
    def _load_annotations(self) -> List[Dict]:
        """Load and process annotations from JSON file with multi-annotator merging."""
        try:
            start_time = time.time()
            with open(self.annotations_file, 'r') as f:
                raw_annotations = json.load(f)
            
            logger.info(f"Loaded {len(raw_annotations)} raw annotation entries")
            
            # Group and merge annotations by video
            merged_annotations = self._group_and_merge_annotations(raw_annotations)
            logger.info(f"Merged into {len(merged_annotations)} unique video annotations")
            
            self.performance_stats['io_time'] = self.performance_stats.get('io_time', 0) + (time.time() - start_time)
            return merged_annotations
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Error loading annotations: {e}")
    
    def _group_and_merge_annotations(self, raw_annotations: List[Dict]) -> List[Dict]:
        """Group annotations by video file and merge multiple annotators."""
        # Group by video filename
        video_groups = defaultdict(list)
        
        for annotation in raw_annotations:
            video_path = annotation.get('video', '')
            # Extract filename from path
            video_filename = Path(video_path).name
            video_groups[video_filename].append(annotation)
        
        # Merge annotations for each video
        merged_annotations = []
        for video_filename, video_annotations in video_groups.items():
            if len(video_annotations) == 1:
                merged_annotations.append(video_annotations[0])
            else:
                merged_annotation = self._merge_annotator_data(video_annotations, video_filename)
                merged_annotations.append(merged_annotation)
        
        return merged_annotations
    
    def _merge_annotator_data(self, video_annotations: List[Dict], video_filename: str) -> Dict:
        """Merge annotations from multiple annotators for the same video."""
        logger.info(f"Merging annotations from {len(video_annotations)} annotators for {video_filename}")
        
        # Sort by updated_at to prioritize latest annotations in conflicts
        sorted_annotations = sorted(video_annotations, 
                                   key=lambda x: x.get('updated_at', ''))
        
        # Use the latest annotation as base
        merged_annotation = sorted_annotations[-1].copy()
        
        # Collect all unique frames with their annotations, resolving conflicts
        frame_annotations = {}  # frame_num -> {annotator_info, annotation_data}
        
        for annotation in sorted_annotations:
            annotator_id = annotation.get('annotator', 'unknown')
            updated_at = annotation.get('updated_at', '')
            
            for box in annotation.get('box', []):
                for sequence_item in box.get('sequence', []):
                    frame_num = sequence_item.get('frame')
                    if frame_num is not None:
                        # If frame already exists, check which annotation is newer
                        if frame_num in frame_annotations:
                            existing_updated_at = frame_annotations[frame_num]['updated_at']
                            if updated_at > existing_updated_at:
                                logger.debug(f"Frame {frame_num}: overriding annotator "
                                           f"{frame_annotations[frame_num]['annotator']} with {annotator_id} "
                                           f"(newer: {updated_at})")
                                frame_annotations[frame_num] = {
                                    'annotator': annotator_id,
                                    'updated_at': updated_at,
                                    'box': box,
                                    'sequence_item': sequence_item
                                }
                        else:
                            frame_annotations[frame_num] = {
                                'annotator': annotator_id,
                                'updated_at': updated_at,
                                'box': box,
                                'sequence_item': sequence_item
                            }
        
        # Reconstruct boxes from merged frame annotations
        merged_boxes = []
        frames_by_box = defaultdict(list)  # Group frames back by their original box structure
        
        for frame_num, frame_data in frame_annotations.items():
            box_key = id(frame_data['box'])  # Use object id as key
            frames_by_box[box_key].append({
                'frame': frame_num,
                'sequence_item': frame_data['sequence_item']
            })
        
        # Reconstruct box structure
        for box_key, frames in frames_by_box.items():
            # Get the original box structure from one of the frames
            original_box = next(f for f in frame_annotations.values() if id(f['box']) == box_key)['box']
            
            # Create new box with merged sequence
            new_box = original_box.copy()
            new_sequence = []
            
            for frame_info in sorted(frames, key=lambda x: x['frame']):
                new_sequence.append(frame_info['sequence_item'])
            
            new_box['sequence'] = new_sequence
            merged_boxes.append(new_box)
        
        # Update merged annotation with resolved boxes
        merged_annotation['box'] = merged_boxes
        
        # Add metadata about the merge
        all_annotators = list(set(ann.get('annotator') for ann in video_annotations))
        merged_annotation['merged_from_annotators'] = all_annotators
        merged_annotation['total_annotators'] = len(all_annotators)
        
        total_frames = len(frame_annotations)
        logger.info(f"Successfully merged {total_frames} frames from annotators: {all_annotators}")
        
        return merged_annotation
    
    def _validate_class_mappings(self):
        """Validate that all classes in annotations exist in class mappings."""
        annotation_classes = set()
        
        for annotation in self.annotations:
            for box in annotation.get('box', []):
                labels = box.get('labels', [])
                annotation_classes.update(labels)
        
        missing_classes = annotation_classes - set(self.class_mappings.keys())
        if missing_classes:
            raise ValueError(f"Missing class mappings for: {missing_classes}")
        
        logger.info(f"Validated class mappings for classes: {sorted(annotation_classes)}")
    
    def _process_single_video_optimized(self, annotation: Dict) -> Optional[Dict[str, Any]]:
        """Process a single video with optimized extraction."""
        video_path_str = annotation['video']
        video_file = self.video_matcher.find_matching_video(
            video_path_str, prefer_exact_match=self.use_exact_matching
        )
        
        if not video_file:
            logger.warning(f"No matching video found for {video_path_str}")
            return None
        
        logger.info(f"Processing video: {video_file.name}")
        
        # Collect all frame data across all boxes/classes
        frame_annotations = defaultdict(list)
        
        # Process all boxes for this video
        for box in annotation.get('box', []):
            class_name = box['labels'][0]  # Assuming single label per box
            class_id = self.class_mappings[class_name]
            
            for sequence_item in box.get('sequence', []):
                # Skip items without frame number or with enabled=False
                if 'frame' not in sequence_item:
                    continue
                if sequence_item.get('enabled', True) is False:
                    continue
                    
                frame_num = sequence_item['frame']
                
                bbox_data = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'x': sequence_item['x'],
                    'y': sequence_item['y'],
                    'width': sequence_item['width'],
                    'height': sequence_item['height'],
                    'frame': frame_num,
                    'time': sequence_item.get('time', 0)
                }
                
                frame_annotations[frame_num].append(bbox_data)
        
        # Apply frame sampling if fps_limit is specified
        if self.fps_limit is not None:
            original_frame_count = len(frame_annotations)
            
            # Get all frame numbers and apply sampling
            all_frame_numbers = list(frame_annotations.keys())
            sampled_frame_numbers = self._sample_consecutive_frames(all_frame_numbers, self.fps_limit)
            
            # Filter frame_annotations to keep only sampled frames
            sampled_frame_annotations = {
                frame_num: frame_annotations[frame_num] 
                for frame_num in sampled_frame_numbers
            }
            
            logger.info(f"Frame sampling applied: {original_frame_count} → {len(sampled_frame_annotations)} frames "
                       f"(target: {self.fps_limit} FPS)")
            
            frame_annotations = sampled_frame_annotations
        
        # Store processed data for this video
        return {
            'video_file': video_file,
            'frame_annotations': dict(frame_annotations),
            'frames_count': annotation.get('box', [{}])[0].get('framesCount', 0),
            'duration': annotation.get('box', [{}])[0].get('duration', 0)
        }
    
    def _process_annotations_parallel(self) -> Dict[str, Any]:
        """Process all annotations in parallel."""
        processed_data = {}
        
        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all video processing tasks
            future_to_annotation = {
                executor.submit(self._process_single_video_optimized, annotation): annotation
                for annotation in self.annotations
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_annotation):
                try:
                    result = future.result()
                    if result:
                        video_file = result['video_file']
                        processed_data[str(video_file)] = result
                        logger.info(f"Processed {len(result['frame_annotations'])} frames from {video_file.name}")
                except Exception as e:
                    annotation = future_to_annotation[future]
                    logger.error(f"Error processing video {annotation.get('video', 'unknown')}: {e}")
        
        return processed_data
    
    def _process_video_frames_batch(self, video_data: Dict[str, Any], 
                                  output_images_dir: Path, output_labels_dir: Path,
                                  converter) -> Tuple[int, int]:
        """Process all frames for a single video in optimized batches."""
        video_file = video_data['video_file']
        frame_annotations = video_data['frame_annotations']
        
        if not frame_annotations:
            return 0, 0
        
        # Initialize statistics tracking for this video
        video_objects_per_class = {class_name: 0 for class_name in self.class_mappings.keys()}
        
        # Get video info for optimal batching
        video_info = self.frame_extractor.get_video_info_cached(str(video_file))
        if not video_info:
            logger.error(f"Could not get video info for {video_file}")
            return 0, 0
        
        # Calculate optimal batch size
        optimal_batch_size = self.frame_extractor.calculate_optimal_batch_size(
            video_info['width'], video_info['height'], self.memory_limit_mb
        )
        
        frame_numbers = list(frame_annotations.keys())
        total_frames = len(frame_numbers)
        successful_extractions = 0
        
        logger.info(f"Processing {total_frames} frames from {video_file.name} "
                   f"(batch size: {optimal_batch_size})")
        
        # Process frames in optimized batches
        for i in range(0, len(frame_numbers), optimal_batch_size):
            batch_frames = frame_numbers[i:i + optimal_batch_size]
            batch_end = min(i + optimal_batch_size, len(frame_numbers))
            
            logger.debug(f"Processing batch {i//optimal_batch_size + 1}: "
                        f"frames {i+1}-{batch_end} of {total_frames}")
            
            # Extract frames using optimized method
            start_time = time.time()
            extracted_frames = self.frame_extractor.extract_frames_batch_optimized(
                video_file, batch_frames
            )
            self.performance_stats['extraction_time'] = self.performance_stats.get('extraction_time', 0) + (time.time() - start_time)
            
            # Process each frame in the batch
            start_time = time.time()
            batch_object_counts = self._process_frame_batch(
                batch_frames, extracted_frames, frame_annotations,
                output_images_dir, output_labels_dir, video_file, converter
            )
            successful_extractions += batch_object_counts['successful_frames']
            
            # Accumulate object counts for this video
            for class_name, count in batch_object_counts['objects_per_class'].items():
                video_objects_per_class[class_name] += count
                
            self.performance_stats['conversion_time'] = self.performance_stats.get('conversion_time', 0) + (time.time() - start_time)
        
        # Record statistics for this video
        self.dataset_analyzer.add_video_stats(
            video_file.name, successful_extractions, video_objects_per_class
        )
        
        return successful_extractions, total_frames
    
    def _process_frame_batch(self, batch_frames: List[int], extracted_frames: Dict[int, np.ndarray],
                           frame_annotations: Dict[int, List[Dict]], output_images_dir: Path,
                           output_labels_dir: Path, video_file: Path, converter) -> Dict[str, Any]:
        """Process a batch of extracted frames."""
        successful_count = 0
        batch_objects_per_class = {class_name: 0 for class_name in self.class_mappings.keys()}
        
        for frame_num in batch_frames:
            frame_image = extracted_frames.get(frame_num)
            if frame_image is None:
                continue
            
            annotations = frame_annotations[frame_num]
            
            # Count objects by class in this frame
            for ann in annotations:
                class_name = ann.get('class_name', '')
                if class_name in batch_objects_per_class:
                    batch_objects_per_class[class_name] += 1
            
            # Save frame image
            start_time = time.time()
            image_filename = f"frame_{video_file.stem}_{frame_num:06d}.jpg"
            image_path = output_images_dir / image_filename
            
            success = cv2.imwrite(str(image_path), frame_image)
            if not success:
                logger.warning(f"Failed to write image {image_path}")
                continue
            
            self.performance_stats['io_time'] = self.performance_stats.get('io_time', 0) + (time.time() - start_time)
            
            # Create optimized annotations (YOLO only)
            self._create_optimized_yolo_annotation(
                annotations, frame_image.shape, output_labels_dir, 
                f"frame_{video_file.stem}_{frame_num:06d}.txt", converter
            )
            
            successful_count += 1
            self.performance_stats['frames_extracted'] = self.performance_stats.get('frames_extracted', 0) + 1
        
        return {
            'successful_frames': successful_count,
            'objects_per_class': batch_objects_per_class
        }
    
    def _create_optimized_yolo_annotation(self, annotations: List[Dict], image_shape: Tuple[int, int, int],
                                        output_labels_dir: Path, label_filename: str, converter):
        """Create YOLO annotation using vectorized coordinate conversion."""
        if not annotations:
            return
        
        img_height, img_width = image_shape[:2]
        
        # Prepare batch data for vectorized conversion
        bboxes = np.array([[ann['x'], ann['y'], ann['width'], ann['height']] for ann in annotations])
        class_ids = [ann['class_id'] for ann in annotations]
        
        # Vectorized coordinate conversion
        start_time = time.time()
        yolo_coords = self.coord_converter.convert_bbox_batch_yolo(bboxes, img_width, img_height)
        self.performance_stats['conversion_time'] = self.performance_stats.get('conversion_time', 0) + (time.time() - start_time)
        
        # Create YOLO format lines
        yolo_lines = []
        for i, class_id in enumerate(class_ids):
            center_x, center_y, width, height = yolo_coords[i]
            yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
        
        # Write annotation file
        start_time = time.time()
        label_path = output_labels_dir / label_filename
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        self.performance_stats['io_time'] = self.performance_stats.get('io_time', 0) + (time.time() - start_time)
    
    def convert_to_yolo(self, output_path: Path):
        """Convert annotations to YOLO format with optimizations."""
        start_time = time.time()
        logger.info("Converting to YOLO format...")
        
        # Process annotations in parallel
        processed_data = self._process_annotations_parallel()
        yolo_converter = YOLOConverter(self.class_mappings)
        
        # Create YOLO directory structure
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        total_successful = 0
        total_frames = 0
        
        # Process each video
        for video_data in processed_data.values():
            successful, frames = self._process_video_frames_batch(
                video_data, images_dir, labels_dir, yolo_converter
            )
            total_successful += successful
            total_frames += frames
            self.performance_stats['videos_processed'] = self.performance_stats.get('videos_processed', 0) + 1
        
        # Create classes.txt and YAML config
        start_time_io = time.time()
        yolo_converter.create_classes_file(output_path / "classes.txt")
        yolo_converter.create_yaml_file(output_path / "data.yaml")
        self.performance_stats['io_time'] = self.performance_stats.get('io_time', 0) + (time.time() - start_time_io)
        
        self.performance_stats['total_processing_time'] = time.time() - start_time
        
        logger.info(f"🎉 YOLO conversion complete! "
                   f"Processed {total_successful}/{total_frames} frames from "
                   f"{self.performance_stats.get('videos_processed', 0)} videos")
        
        self._log_performance_stats()
        
        # Generate dataset analysis report
        self.dataset_analyzer.generate_report()
    
    def _log_performance_stats(self):
        """Log detailed performance statistics."""
        stats = self.performance_stats
        logger.info("PERFORMANCE STATISTICS:")
        logger.info(f"   Total Processing Time: {stats.get('total_processing_time', 0):.2f}s")
        logger.info(f"   Frame Extraction Time: {stats.get('extraction_time', 0):.2f}s")
        logger.info(f"   Coordinate Conversion Time: {stats.get('conversion_time', 0):.2f}s")
        logger.info(f"   File I/O Time: {stats.get('io_time', 0):.2f}s")
        logger.info(f"   Videos Processed: {stats.get('videos_processed', 0)}")
        logger.info(f"   Frames Extracted: {stats.get('frames_extracted', 0)}")
        
        if stats.get('frames_extracted', 0) > 0:
            fps = stats.get('frames_extracted', 0) / max(stats.get('total_processing_time', 1), 0.1)
            logger.info(f"   Processing Rate: {fps:.2f} FPS")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for benchmarking."""
        return self.performance_stats.copy()
