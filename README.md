# Video Annotation Processor

A tool for converting video annotations from Label Studio JSON format to YOLO format for object detection training.

## Features

- **Multi-Project Support**: Process videos from multiple Label Studio projects in a single run
- **Interactive Project Selection**: Browse and select projects interactively
- Convert Label Studio video annotations to YOLO format
- Interactive video selection from Label Studio projects
- Extract annotated frames from videos
- Handle multiple classes and video sequences
- Automatic video file matching and downloading
- Performance optimizations for large datasets
- Comprehensive dataset analysis and statistics reporting
- Project mapping for multi-project datasets

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
# PROJECT_ID is now optional - for backward compatibility only
# PROJECT_ID=5
```

## Usage

### Quick Start

#### List Available Projects
View all projects on your Label Studio server:
```bash
python main.py --list-projects
```

#### Interactive Mode (Recommended)
Select projects interactively when no project is specified:
```bash
python main.py \
  --classes '{"cyclist":0,"pedestrian":1,"scooter-roller":2}' \
  --output ./dataset
```

The tool will:
1. Display all available projects from Label Studio
2. Allow you to select one or multiple projects interactively
3. For each selected project, allow interactive video selection
4. Download videos and annotations
5. Extract annotated frames with project-specific naming
6. Generate combined YOLO dataset with project mapping

#### Process Multiple Projects
Specify multiple projects directly:
```bash
python main.py \
  --classes '{"cyclist":0,"pedestrian":1,"scooter-roller":2}' \
  --output ./dataset \
  --project-ids 5,7,12
```

#### Single Project Mode (Backward Compatible)
Process a single project using the traditional workflow:
```bash
python main.py \
  --classes '{"cyclist":0,"pedestrian":1,"scooter-roller":2}' \
  --output ./dataset \
  --project-id 5
```

## Output Structure

### Single Project Mode
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
├── classes.txt       # Class names mapping
└── data.yaml         # YOLO dataset configuration
```

### Multi-Project Mode
```
dataset/
├── images/
│   ├── project5_frame_video1_000346.jpg
│   ├── project5_frame_video2_000120.jpg
│   ├── project7_frame_video3_000050.jpg
│   ├── project12_frame_video4_000200.jpg
│   └── ...
├── labels/
│   ├── project5_frame_video1_000346.txt
│   ├── project5_frame_video2_000120.txt
│   ├── project7_frame_video3_000050.txt
│   ├── project12_frame_video4_000200.txt
│   └── ...
├── classes.txt           # Class names mapping (shared across projects)
├── data.yaml             # YOLO dataset configuration
└── project_mapping.json  # Project metadata and statistics
```

**project_mapping.json example**:
```json
{
  "projects": [
    {
      "id": 5,
      "title": "Cyclist Detection - Downtown",
      "frames_extracted": 456,
      "annotations_count": 456,
      "videos_processed": 3
    },
    {
      "id": 7,
      "title": "Pedestrian Tracking - Campus",
      "frames_extracted": 289,
      "annotations_count": 289,
      "videos_processed": 2
    }
  ],
  "total_frames": 745,
  "total_annotations": 745,
  "classes": {"cyclist": 0, "pedestrian": 1, "scooter-roller": 2}
}
```

**Note**: In multi-project mode, frame filenames are prefixed with `project{id}_` to prevent naming collisions between projects.

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

#### Required Parameters
- `--classes` (`-c`): Class mappings as JSON string (required for processing)
- `--output` (`-o`): Output directory path (required for processing)

#### Project Selection (choose one)
- `--list-projects`: List all available projects and exit (no other parameters needed)
- `--project-ids`: Comma-separated list of project IDs to process (e.g., "5,7,12")
- `--project-id`: Single project ID for backward compatibility
- _None_: Interactive project selection mode (default)

#### Optional Parameters
- `--project` (`-p`): Main project path (default: directory where main.py is located)
- `--workers` (`-w`): Maximum number of parallel workers (default: 4)
- `--memory` (`-m`): Memory limit in MB for batch processing (default: 2048)
- `--fps-limit`: Limit frames per second extraction (optional, e.g., 2.0)
- `--benchmark`: Enable detailed performance benchmarking

### Examples

#### List all projects:
```bash
python main.py --list-projects
```

#### Interactive mode (recommended for new users):
```bash
python main.py \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./dataset
```

#### Process multiple specific projects:
```bash
python main.py \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./dataset \
  --project-ids 5,7,12
```

#### Single project (backward compatible):
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
  --project-ids 5,7 \
  --workers 8 \
  --memory 4096 \
  --benchmark
```

#### With FPS limiting (extract every 0.5 seconds):
```bash
python main.py \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./dataset \
  --project-id 5 \
  --fps-limit 2.0
```

## YOLO Format Details

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

## Multi-Project Workflows

### When to Use Multi-Project Mode

Multi-project mode is useful when you want to:
- Combine datasets from multiple Label Studio projects
- Train a single model on data from different sources
- Maintain separation between different data collection campaigns
- Track which frames came from which project

### File Naming Convention

- **Single Project**: `frame_video1_000346.jpg`
- **Multi-Project**: `project5_frame_video1_000346.jpg`

The project ID prefix ensures no naming collisions when combining videos from different projects.

### Project Mapping File

When processing multiple projects, a `project_mapping.json` file is automatically generated with:
- List of processed projects with their IDs and titles
- Per-project statistics (frames, annotations, videos)
- Combined totals across all projects
- Class mappings used

This file helps you track the composition of your combined dataset.

## Backward Compatibility

The tool maintains full backward compatibility with existing workflows:
- Single `--project-id` argument still works as before
- `PROJECT_ID` in `.env` file is still supported
- Single-project mode uses the original file naming (without project prefix)
- All existing scripts and workflows continue to work unchanged

## Testing

The project includes a comprehensive test suite using pytest.

### Installing Test Dependencies

```bash
pip install -r requirements.txt
```

This will install pytest, pytest-cov, and pytest-mock along with the main dependencies.

### Running Tests

```bash
# Run all tests
pytest test_frame_matcher.py -v

# Run with coverage report
pytest test_frame_matcher.py --cov=. --cov-report=html

# Run specific test class
pytest test_frame_matcher.py::TestListAvailableProjects -v

# Run specific test
pytest test_frame_matcher.py::TestListAvailableProjects::test_list_projects_success -v

# Run tests and stop at first failure
pytest test_frame_matcher.py -x

# View coverage report
open htmlcov/index.html  # On macOS
```

### Test Coverage

The test suite covers:
- ✅ **API Integration**: Label Studio REST API calls and error handling
- ✅ **CLI Arguments**: Argument parsing and validation
- ✅ **Interactive Selection**: Project and video selection workflows
- ✅ **File Naming**: Single-project and multi-project file naming conventions
- ✅ **Error Handling**: Connection errors, invalid inputs, edge cases
- ✅ **Multi-Project**: Project mapping generation and statistics
- ✅ **Backward Compatibility**: Existing single-project workflows

### Writing New Tests

When adding new features, add corresponding tests to `test_frame_matcher.py`:

```python
class TestYourNewFeature:
    """Test your new feature."""

    def test_basic_functionality(self):
        """Test the basic functionality."""
        # Your test code here
        assert expected == actual
```

## Notes

- Only annotated frames are extracted (not all video frames)
- Supports multi-annotator workflows with automatic merging
- Handles multiple bounding boxes per frame
- Frame filenames include video name and frame number for uniqueness
- Performance optimized for large datasets with batch processing
- Multi-project support prevents filename collisions with project ID prefixes
- All projects in a multi-project run must use the same class schema