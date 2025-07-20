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
- ✅ **NEW**: Automatic download from Label Studio (no manual file handling)
- ✅ **NEW**: Smart folder detection and exact filename matching
- ✅ **OPTIMIZED**: High-performance batch frame extraction for large datasets
- ✅ **OPTIMIZED**: Memory-efficient processing with configurable batch sizes

## Performance Optimizations

This tool has been optimized for handling large datasets efficiently:

### Batch Frame Extraction
- **Problem**: Original implementation opened/closed video files for each individual frame
- **Solution**: New batch extraction opens each video once and extracts multiple frames in sequence
- **Result**: Significant reduction in I/O overhead for large datasets

### Memory Management
- Frames are processed in configurable batches (default: 500 frames)
- Prevents memory overflow when processing thousands of frames
- Maintains steady memory usage throughout processing

### Progress Tracking
- Clear batch-level progress indicators
- Reduced per-frame logging for better performance
- Summary statistics for completed processing

### Typical Performance
- **Small datasets** (< 1000 frames): Near-instant processing
- **Medium datasets** (1000-5000 frames): Processes in minutes
- **Large datasets** (> 5000 frames): Efficient batch processing with clear progress

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd frame_matcher/framer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
python main.py \
  --format yolo \
  --classes '{"cyclist":0,"person":1,"pedestrian":2}' \
  --output ./dataset \
  --auto-download \
  --project-id 5
```

**Note**: Make sure to install the updated requirements which now include PyYAML for YAML config generation, Label Studio SDK for automatic downloads, and python-dotenv for environment configuration.

## Setup for Auto-Download (Optional)

If you want to use the automatic download feature, create a `.env` file in the project root:

```bash
cp .env.example .env
# Edit .env with your Label Studio credentials
```

Example `.env` file:
```env
LABEL_STUDIO_URL=http://10.21.12.67
LABEL_STUDIO_API_KEY=519b2038c9c45a461bab720013d98373873a998a
PROJECT_ID=5
```

## Project Structure

```
project/
├── main.py                 # CLI interface
├── .env                    # Label Studio credentials (for auto-download)
├── .env.example           # Example environment file
├── utils/
│   ├── __init__.py
│   ├── annotation_processor.py
│   ├── video_matcher.py
│   ├── frame_extractor.py
│   ├── yolo_converter.py
│   ├── coco_converter.py
│   └── downloader.py      # NEW: Label Studio downloader
├── json_file/             # Manual annotations (legacy)
│   └── annotations.json
├── exported_json_annotation/  # NEW: Auto-downloaded annotations
│   └── annotations.json
├── video_files/           # Manual videos (legacy)
│   ├── video1.mp4
│   └── video2.mp4
├── exported_videos/       # NEW: Auto-downloaded videos
│   ├── video1.mp4
│   └── video2.mp4
└── requirements.txt
```

## Usage

### Auto-Download Mode (Recommended)

Automatically download annotations and videos from Label Studio:

```bash
python main.py \
  --format yolo \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./dataset \
  --auto-download \
  --project-id 5
```

**Benefits of Auto-Download:**
- ✅ No manual file handling required
- ✅ Always gets the latest annotations
- ✅ Exact filename matching between JSON and videos
- ✅ Automatic folder management
- ✅ Skips re-downloading existing files

### Manual Mode (Legacy)

Use manually placed files in `json_file/` and `video_files/`:

```bash
python main.py \
  --format yolo \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./dataset
```

### Smart Folder Detection

The system automatically detects which mode to use:
- If `exported_json_annotation/` exists → uses downloaded annotations
- If `exported_videos/` exists → uses downloaded videos  
- Falls back to `json_file/` and `video_files/` for manual mode

### Parameters

- `--format` (`-f`): Output format - `yolo` or `coco` (default: yolo)
- `--classes` (`-c`): Class mappings as JSON string (required)
- `--output` (`-o`): Output directory path (required)
- `--project` (`-p`): Main project path (default: directory where main.py is located)
- `--auto-download`: **NEW** - Automatically download from Label Studio
- `--project-id`: **NEW** - Label Studio project ID (required with --auto-download)

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