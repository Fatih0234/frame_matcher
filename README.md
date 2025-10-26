# Video Annotation Processor

A tool for converting video annotations from Label Studio JSON format to YOLO format for object detection training.

## Features

- Convert Label Studio video annotations to YOLO format
- Interactive video selection from Label Studio projects
- Extract annotated frames from videos
- Handle multiple classes and video sequences
- Automatic video file matching and downloading
- Performance optimizations for large datasets
- Dataset analysis and statistics reporting

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Label Studio connection by creating a `.env` file:
```env
LABEL_STUDIO_URL=http://your-label-studio-url
LABEL_STUDIO_API_KEY=your-api-key
```

## Usage

```bash
python main.py \
  --classes '{"cyclist":0,"pedestrian":1,"scooter-roller":2}' \
  --output ./dataset \
  --project-id 5
```

The tool will:
1. Connect to Label Studio and list available videos
2. Allow interactive selection of videos to process
3. Download selected videos and annotations
4. Extract annotated frames and convert to YOLO format
5. Generate dataset analysis report

## Output Structure

```
dataset/
├── images/           # Extracted frame images
├── labels/           # YOLO format annotation files
├── classes.txt       # Class names mapping
└── data.yaml         # YOLO dataset configuration
```

## Configuration

### Class Mappings
Define object classes with their numeric IDs:
```json
{"cyclist": 0, "pedestrian": 1, "scooter-roller": 2}
```

### Performance Options
- `--max-workers`: Number of parallel processing workers (default: 4)
- `--memory-limit`: Memory limit in MB for batch processing (default: 2048)

## Dataset Analysis

The tool automatically generates comprehensive statistics after processing:
- Frame counts per video
- Object counts by class
- Average objects per frame
- Class distribution percentages

### Parameters

- `--classes` (`-c`): Class mappings as JSON string (required)
- `--output` (`-o`): Output directory path (required)
- `--project` (`-p`): Main project path (default: directory where main.py is located)
- `--project-id`: Label Studio project ID (required)
- `--workers` (`-w`): Maximum number of parallel workers (default: 4)
- `--memory` (`-m`): Memory limit in MB for batch processing (default: 2048)
- `--fps-limit`: Limit frames per second extraction (optional)
- `--benchmark`: Enable detailed performance benchmarking

### Examples

#### Basic usage:
```bash
python main.py \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./dataset \
  --project-id 5
```

#### With performance optimization:
```bash
python main.py \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./dataset \
  --project-id 5 \
  --workers 8 \
  --memory 4096 \
  --benchmark
```

## Output Format

The tool generates YOLO format datasets:
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
## Requirements

Create a `requirements.txt` file with:

```
typer
opencv-python
numpy
pathlib
PyYAML
label-studio-sdk
python-dotenv
```

## Notes

- Only annotated frames are extracted (not all video frames)
- Supports multi-annotator workflows with automatic merging
- Handles multiple bounding boxes per frame
- Frame filenames include video name and frame number for uniqueness
- Performance optimized for large datasets with batch processing