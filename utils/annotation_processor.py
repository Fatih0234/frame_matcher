"""
Main annotation processor that handles the conversion pipeline.
"""

import json
import cv2
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from .video_matcher import VideoMatcher
from .frame_extractor import FrameExtractor
from .yolo_converter import YOLOConverter
from .coco_converter import COCOConverter


class AnnotationProcessor:
    def __init__(self, annotations_file: Path, video_files_dir: Path, class_mappings: Dict[str, int], use_exact_matching: bool = False):
        """
        Initialize the annotation processor.
        
        Args:
            annotations_file: Path to the JSON annotations file
            video_files_dir: Directory containing video files
            class_mappings: Dictionary mapping class names to their integer encodings
            use_exact_matching: If True, prefer exact filename matching for videos
        """
        self.annotations_file = annotations_file
        self.video_files_dir = video_files_dir
        self.class_mappings = class_mappings
        self.use_exact_matching = use_exact_matching
        
        # Load and validate annotations
        self.annotations = self._load_annotations()
        self._validate_class_mappings()
        
        # Initialize components
        self.video_matcher = VideoMatcher(video_files_dir)
        self.frame_extractor = FrameExtractor()
        
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from JSON file."""
        try:
            with open(self.annotations_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Error loading annotations: {e}")
    
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
        
        print(f"✅ Validated class mappings for classes: {sorted(annotation_classes)}")
    
    def _process_annotations(self) -> Dict[str, Any]:
        """
        Process all annotations and extract frame data.
        Collects all annotations for each unique frame to avoid duplicates.
        
        Returns:
            Dictionary containing processed data organized by video
        """
        processed_data = {}
        
        for annotation in self.annotations:
            video_path_str = annotation['video']
            video_file = self.video_matcher.find_matching_video(video_path_str, prefer_exact_match=self.use_exact_matching)
            
            if not video_file:
                print(f"⚠️  Warning: No matching video found for {video_path_str}")
                continue
            
            print(f"📹 Processing video: {video_file.name}")
            
            # Collect all frame data across all boxes/classes
            frame_annotations = defaultdict(list)
            
            # First pass: collect all annotations for all frames across all classes
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
                    
                    # Add to frame annotations (multiple classes can exist per frame)
                    frame_annotations[frame_num].append(bbox_data)
            
            # Store processed data for this video
            processed_data[str(video_file)] = {
                'video_file': video_file,
                'frame_annotations': dict(frame_annotations),
                'frames_count': annotation.get('box', [{}])[0].get('framesCount', 0),
                'duration': annotation.get('box', [{}])[0].get('duration', 0)
            }
            
            print(f"✅ Collected annotations for {len(frame_annotations)} unique frames from {video_file.name}")
        
        return processed_data
    
    def convert_to_yolo(self, output_path: Path):
        """Convert annotations to YOLO format."""
        print("🎯 Converting to YOLO format...")
        
        processed_data = self._process_annotations()
        yolo_converter = YOLOConverter(self.class_mappings)
        
        # Create YOLO directory structure
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        total_frames = 0
        
        for video_data in processed_data.values():
            video_file = video_data['video_file']
            frame_annotations = video_data['frame_annotations']
            
            print(f"🎬 Processing {len(frame_annotations)} frames from {video_file.name}...")
            
            # Process frames in smaller batches to avoid memory issues
            frame_numbers = list(frame_annotations.keys())
            batch_size = 500  # Process 500 frames at a time
            successful_extractions = 0
            
            for i in range(0, len(frame_numbers), batch_size):
                batch_frames = frame_numbers[i:i + batch_size]
                batch_end = min(i + batch_size, len(frame_numbers))
                print(f"⚡ Processing batch {i//batch_size + 1}: frames {i+1}-{batch_end} of {len(frame_numbers)}")
                
                # Extract frames in this batch
                extracted_frames = self.frame_extractor.extract_frames_batch(video_file, batch_frames)
                
                # Process each frame in the batch
                for frame_num in batch_frames:
                    frame_image = extracted_frames.get(frame_num)
                    if frame_image is None:
                        continue
                    
                    annotations = frame_annotations[frame_num]
                    
                    # Save frame image with unique name
                    image_filename = f"frame_{video_file.stem}_{frame_num:06d}.jpg"
                    image_path = images_dir / image_filename
                    cv2.imwrite(str(image_path), frame_image)
                    
                    # Create YOLO annotation with ALL classes for this frame
                    label_filename = f"frame_{video_file.stem}_{frame_num:06d}.txt"
                    label_path = labels_dir / label_filename
                    
                    yolo_converter.create_yolo_annotation(
                        annotations, frame_image.shape, label_path
                    )
                    
                    successful_extractions += 1
                    total_frames += 1
            
            print(f"✅ Successfully processed {successful_extractions}/{len(frame_numbers)} frames from {video_file.name}")
        
        # Create classes.txt and YAML config
        yolo_converter.create_classes_file(output_path / "classes.txt")
        yolo_converter.create_yaml_file(output_path / "data.yaml")
        
        print(f"🎉 YOLO conversion complete! Processed {total_frames} frames total")
    
    def convert_to_coco(self, output_path: Path):
        """Convert annotations to COCO format."""
        print("🎯 Converting to COCO format...")
        
        processed_data = self._process_annotations()
        coco_converter = COCOConverter(self.class_mappings)
        
        # Create images directory
        images_dir = output_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        total_frames = 0
        
        for video_data in processed_data.values():
            video_file = video_data['video_file']
            frame_annotations = video_data['frame_annotations']
            
            print(f"🎬 Processing {len(frame_annotations)} frames from {video_file.name}...")
            
            # Process frames in smaller batches to avoid memory issues
            frame_numbers = list(frame_annotations.keys())
            batch_size = 500  # Process 500 frames at a time
            successful_extractions = 0
            
            for i in range(0, len(frame_numbers), batch_size):
                batch_frames = frame_numbers[i:i + batch_size]
                batch_end = min(i + batch_size, len(frame_numbers))
                print(f"⚡ Processing batch {i//batch_size + 1}: frames {i+1}-{batch_end} of {len(frame_numbers)}")
                
                # Extract frames in this batch
                extracted_frames = self.frame_extractor.extract_frames_batch(video_file, batch_frames)
                
                # Process each frame in the batch
                for frame_num in batch_frames:
                    frame_image = extracted_frames.get(frame_num)
                    if frame_image is None:
                        continue
                    
                    annotations = frame_annotations[frame_num]
                    
                    # Save frame image with unique name
                    image_filename = f"frame_{video_file.stem}_{frame_num:06d}.jpg"
                    image_path = images_dir / image_filename
                    cv2.imwrite(str(image_path), frame_image)
                    
                    # Add to COCO dataset with ALL classes for this frame
                    coco_converter.add_image_with_annotations(
                        image_filename, frame_image.shape, annotations
                    )
                    
                    successful_extractions += 1
                    total_frames += 1
            
            print(f"✅ Successfully processed {successful_extractions}/{len(frame_numbers)} frames from {video_file.name}")
        
        # Save COCO JSON
        coco_converter.save_coco_file(output_path / "annotations.json")
        
        print(f"🎉 COCO conversion complete! Processed {total_frames} frames total")