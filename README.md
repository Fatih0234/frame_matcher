# Video Annotation Converter

Convert video annotations from LabelStudio JSON format to YOLO or COCO format for object detection training.

## Features

- ✅ Convert LabelStudio video annotations to YOLO format
- ✅ Convert LabelStudio video annotations to COCO format  
- ✅ Extract only annotated frames from videos
- ✅ Handle multiple sequences and classes per video
- ✅ Automatic video file matching
- ✅ Coordinate conversion from percentages to required formats
- ✅ Command-line interface with typer

## Installation

1. Clone or download the project files
2. Install dependencies:

```bash
pip install -r requirements.txt
```

**Note**: Make sure to install the updated requirements which now include PyYAML for YAML config generation.

## Project Structure

```
project/
├── main.py                 # CLI interface
├── utils/
│   ├── __init__.py
│   ├── annotation_processor.py
│   ├── video_matcher.py
│   ├── frame_extractor.py
│   ├── yolo_converter.py
│   └── coco_converter.py
├── json_file/
│   └── annotations.json    # Your LabelStudio annotations
├── video_files/
│   ├── video1.mp4
│   └── video2.mp4
└── requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py \
  --format yolo \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./dataset
```

The script will automatically use the directory where `main.py` is located as the project path.

### Parameters

- `--format` (`-f`): Output format - `yolo` or `coco` (default: yolo)
- `--classes` (`-c`): Class mappings as JSON string (required)
- `--output` (`-o`): Output directory path (required)
- `--project` (`-p`): Main project path (default: directory where main.py is located)

### Examples

#### Convert to YOLO format:
```bash
python main.py \
  --format yolo \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./yolo_dataset
```

#### Convert to COCO format:
```bash
python main.py \
  --format coco \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./coco_dataset
```

#### Custom project path (optional):
```bash
python main.py \
  --format yolo \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./dataset \
  --project "/path/to/your/project/"
```

**Note**: If `--project` is not specified, the script will use the directory where `main.py` is located as the project path.

## Output Formats

### YOLO Format
```
dataset/
├── images/
│   ├── frame_video1_000346.jpg
│   ├── frame_video1_000351.jpg
│   └── ...
├── labels/
│   ├── frame_video1_000346.txt
│   ├── frame_video1_000351.txt
│   └── ...
├── classes.txt
└── data.yaml
```

**YOLO annotation format**: `class_id center_x center_y width height` (normalized 0-1)

**YAML config example**:
```yaml
path: .
train: images
val: images
test: images
nc: 3
names:
  - cyclist
  - person
  - scooter-roller
```

### COCO Format
```
dataset/
├── images/
│   ├── frame_video1_000346.jpg
│   ├── frame_video1_000351.jpg
│   └── ...
└── annotations.json
```

**COCO annotation format**: Standard COCO JSON with bounding boxes in pixel coordinates

## Class Mapping

The `--classes` parameter expects a JSON string mapping class names to integer IDs:

```json
{
  "cyclist": 0,
  "person": 1, 
  "scooter-roller": 2
}
```

**Important**: The class names must exactly match those in your LabelStudio annotations. The tool will validate this and show an error if any classes are missing.

## Video File Matching

The tool automatically matches video files from your `video_files/` directory with the paths in the JSON annotations. It uses several strategies:

1. Direct filename match
2. Extract meaningful part after dash (e.g., `3b780495-video.mp4` → `video.mp4`)
3. Fuzzy matching based on filename similarity
4. Partial name matching

## Troubleshooting

### "Missing class mappings" error
Make sure all class names in your annotations are included in the `--classes` parameter.

### "No matching video found" warning  
Check that your video files are in the `video_files/` directory and have supported extensions (.mp4, .avi, .mov, .mkv, .wmv, .flv).

### "Cannot extract frame" error
This usually means the frame number in the annotation is outside the video's frame range. The tool will skip these frames and continue processing.

## Notes

- Only annotated frames are extracted from videos (not all frames)
- Multiple bounding boxes per frame are supported
- Coordinates are converted from percentages (0-100) to the required format
- The tool handles videos at 25 FPS as specified
- Frame filenames include the video name and frame number for uniqueness

## Dependencies

- `typer`: Command-line interface
- `opencv-python`: Video processing and frame extraction
- `numpy`: Numerical operations
- `pathlib`: Path handling
- `PyYAML`: YAML configuration file generation