# Video Annotation Processor for Object Detection

A powerful Python tool for converting video annotations from Label Studio JSON format to multiple object detection formats (YOLO, COCO). Features multi-project support, interactive selection, and optimized batch processing.

## ✨ Key Features

- **🎯 Multiple Export Formats**: Export to YOLO or COCO format with a single command
- **🎯 Multi-Project Support**: Process videos from multiple Label Studio projects in a single run with automatic project mapping
- **🖱️ Interactive Selection**: Browse and select projects and videos through intuitive CLI menus
- **⚡ Performance Optimized**: Parallel processing with configurable workers and memory limits
- **📊 Comprehensive Statistics**: Detailed dataset analysis and project mapping with frame/annotation counts
- **🔄 Smart Video Matching**: Automatic video file matching and downloading from Label Studio
- **🎨 Flexible Class Mapping**: Support for multiple object classes with custom ID assignment
- **📦 Training Ready**: Direct output in YOLO or COCO format with auto-generated config files
- **🔀 FPS Control**: Optional frame rate limiting to reduce dataset size
- **🧪 Fully Tested**: Comprehensive test suite with pytest (95%+ coverage)
- **🔙 Backward Compatible**: Existing single-project workflows continue to work unchanged

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Fatih0234/frame_matcher.git
cd frame_matcher/framer
```

2. **Create a virtual environment**:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure Label Studio connection** by creating a `.env` file in the project root:
```env
LABEL_STUDIO_URL=http://your-label-studio-url
LABEL_STUDIO_API_KEY=your-api-key-here
# PROJECT_ID is now optional - for backward compatibility only
# PROJECT_ID=5
```

**Note**: You can find your API key in Label Studio under Account & Settings → Access Token.

## 📖 Usage

### Quick Start Guide

#### 1️⃣ List Available Projects
View all projects on your Label Studio server:
```bash
python main.py --list-projects
```

**Output:**
```
================================================================================
AVAILABLE LABEL STUDIO PROJECTS
================================================================================
ID     | Title                                    | Tasks     
--------------------------------------------------------------------------------
5      | Cyclist Detection Dataset                | 150       
7      | Pedestrian Tracking                      | 89        
12     | Multi-Class Object Detection             | 234       
================================================================================
```

#### 2️⃣ Interactive Mode (Recommended for Beginners)
Let the tool guide you through project and video selection:
```bash
python main.py \
  --classes '{"cyclist":0,"pedestrian":1,"scooter-roller":2}' \
  --output ./dataset
```

**What happens:**
1. 📋 Displays all available projects
2. 🎯 You select which projects to process (e.g., `5,7` or `1-3` or `all`)
3. 🎬 For each project, select specific videos interactively
4. ⬇️ Downloads videos and annotations automatically
5. 🖼️ Extracts annotated frames with project-specific naming
6. 📦 Generates combined YOLO dataset with project mapping

#### 3️⃣ Process Multiple Specific Projects
Skip the project selection menu and process specific projects:
```bash
python main.py \
  --classes '{"cyclist":0,"pedestrian":1,"scooter-roller":2}' \
  --output ./dataset_multi \
  --project-ids 5,7,12
```

#### 4️⃣ Single Project Mode (Backward Compatible)
Process just one project using the traditional workflow:
```bash
python main.py \
  --classes '{"cyclist":0,"pedestrian":1,"scooter-roller":2}' \
  --output ./dataset \
  --project-id 5
```

## 🎯 Output Formats

frame_matcher supports two industry-standard annotation formats for maximum compatibility with popular training frameworks:

### YOLO Format (Default)

Export annotations in YOLO format for Ultralytics YOLO, YOLOv8, and similar frameworks:

```bash
python main.py \
  --format yolo \
  --classes '{"cyclist":0,"pedestrian":1}' \
  --output ./dataset \
  --project-id 5
```

**YOLO Format Specifications:**
- One `.txt` file per image in `labels/` directory
- Format: `class_id center_x center_y width height`
- All coordinates normalized to [0, 1]
- Generates `classes.txt` and `data.yaml` config files

**Output structure:**
```
dataset/
├── images/              # Frame images
├── labels/              # One .txt file per image
│   └── frame_xxx.txt    # class_id cx cy w h (normalized)
├── classes.txt          # Class names list
└── data.yaml            # YOLO configuration
```

**Example annotation file** (`labels/frame_example_000001.txt`):
```
0 0.5234 0.6123 0.1245 0.2341
1 0.7891 0.3456 0.0923 0.1567
```

### COCO Format

Export annotations in COCO JSON format for Detectron2, MMDetection, and other COCO-compatible frameworks:

```bash
python main.py \
  --format coco \
  --classes '{"cyclist":0,"pedestrian":1}' \
  --output ./dataset \
  --project-id 5
```

**COCO Format Specifications:**
- Single `annotations.json` file containing all annotations
- Images, annotations, and categories arrays
- Bounding boxes in [x_min, y_min, width, height] absolute pixel coordinates
- Compatible with COCO dataset format: https://cocodataset.org/#format-data

**Output structure:**
```
dataset/
├── images/              # Frame images
└── annotations.json     # COCO format JSON
```

**COCO JSON structure:**
```json
{
  "info": {
    "description": "Label Studio Video Annotations - COCO Format",
    "version": "1.0",
    "year": 2025,
    "contributor": "frame_matcher",
    "date_created": "2025-01-27T10:30:00"
  },
  "images": [
    {
      "id": 1,
      "file_name": "frame_video1_000001.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [100.5, 200.3, 150.2, 180.7],
      "area": 27141.14,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "cyclist",
      "supercategory": "object"
    }
  ]
}
```

### Format Comparison

| Feature | YOLO | COCO |
|---------|------|------|
| **File Type** | Multiple .txt files | Single .json file |
| **Coordinates** | Normalized (0-1) | Absolute pixels |
| **Box Format** | center_x, center_y, w, h | x_min, y_min, w, h |
| **Best For** | Ultralytics YOLO, YOLOv8 | Detectron2, MMDetection |
| **Advantages** | Simple, human-readable | Rich metadata, standard format |

### Default Behavior

If `--format` is not specified, YOLO format is used by default to maintain backward compatibility:

```bash
# These two commands are equivalent:
python main.py --classes '{"test":0}' --output ./dataset --project-id 5
python main.py --format yolo --classes '{"test":0}' --output ./dataset --project-id 5
```

## 📁 Output Structure

### Single Project Mode
```
dataset/
├── images/
│   ├── frame_d3ed9ecf-cyclist_passing_by_000592.jpg
│   ├── frame_d3ed9ecf-cyclist_passing_by_000593.jpg
│   └── ...
├── labels/
│   ├── frame_d3ed9ecf-cyclist_passing_by_000592.txt
│   ├── frame_d3ed9ecf-cyclist_passing_by_000593.txt
│   └── ...
├── classes.txt       # Class names mapping (one per line)
└── data.yaml         # YOLO dataset configuration
```

**classes.txt example:**
```
cyclist
pedestrian
scooter-roller
```

**data.yaml example:**
```yaml
path: .
train: images
val: images
test: images

nc: 3
names: ['cyclist', 'pedestrian', 'scooter-roller']
```

### Multi-Project Mode
```
dataset_multi/
├── images/
│   ├── project_5_frame_d3ed9ecf-video1_000346.jpg
│   ├── project_5_frame_d3ed9ecf-video2_000120.jpg
│   ├── project_7_frame_0d8bfe16-video3_000050.jpg
│   ├── project_12_frame_abc123def-video4_000200.jpg
│   └── ...
├── labels/
│   ├── project_5_frame_d3ed9ecf-video1_000346.txt
│   ├── project_5_frame_d3ed9ecf-video2_000120.txt
│   ├── project_7_frame_0d8bfe16-video3_000050.txt
│   ├── project_12_frame_abc123def-video4_000200.txt
│   └── ...
├── classes.txt           # Shared class names mapping
├── data.yaml             # YOLO dataset configuration
└── project_mapping.json  # 🆕 Project metadata and statistics
```

**project_mapping.json example:**
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

**📌 Important Note**: In multi-project mode, frame filenames are prefixed with `project_{id}_` to prevent naming collisions between projects.

## ⚙️ Configuration

### Class Mappings
Define object classes with their numeric IDs as a JSON string:
```json
{"cyclist": 0, "pedestrian": 1, "scooter-roller": 2}
```

**Tips:**
- IDs should start from 0
- Use descriptive class names that match your Label Studio schema
- All projects in a multi-project run must use the same class schema

### Performance Tuning
- `--workers` (`-w`): Number of parallel processing workers (default: 4)
  - Increase for faster processing on multi-core systems
  - Recommended: Number of CPU cores - 1
- `--memory` (`-m`): Memory limit in MB for batch processing (default: 2048)
  - Increase if processing high-resolution videos
  - Decrease if experiencing out-of-memory errors
- `--fps-limit`: Limit frame extraction rate (optional)
  - Example: `--fps-limit 2.0` extracts 2 frames per second
  - Useful for reducing dataset size while maintaining temporal coverage

### Environment Variables (.env)
```env
# Required
LABEL_STUDIO_URL=http://10.21.12.67/
LABEL_STUDIO_API_KEY=519b2038c9c45a461bab720013d98373873a998a

# Optional (for backward compatibility)
PROJECT_ID=5
```

## 📊 Dataset Analysis & Statistics

The tool automatically generates comprehensive statistics after processing:

### Per-Video Statistics
- Total frames extracted
- Frame extraction rate (frames/second)
- Video processing time

### Per-Class Statistics
- Object counts by class
- Average objects per frame
- Class distribution percentages

### Multi-Project Statistics
When processing multiple projects, `project_mapping.json` includes:
- Per-project frame and annotation counts
- Total combined statistics
- Project titles and IDs for traceability

### Benchmark Mode
Enable with `--benchmark` to get detailed performance metrics:
```json
{
  "extraction_time": 45.23,
  "conversion_time": 12.45,
  "io_time": 8.92,
  "videos_processed": 5,
  "frames_extracted": 1234,
  "total_time": 66.60
}
```

## 🔧 Command-Line Parameters

### Required Parameters (for processing)
| Parameter | Short | Description | Example |
|-----------|-------|-------------|---------|
| `--classes` | `-c` | Class mappings as JSON string | `'{"cyclist":0,"pedestrian":1}'` |
| `--output` | `-o` | Output directory path | `./dataset` |

### Project Selection (choose one)
| Parameter | Description | Example |
|-----------|-------------|---------|
| `--list-projects` | List all available projects and exit | `--list-projects` |
| `--project-ids` | Comma-separated list of project IDs | `--project-ids 5,7,12` |
| `--project-id` | Single project ID (backward compatible) | `--project-id 5` |
| _(none)_ | Interactive project selection mode (default) | - |

### Optional Parameters
| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--format` | `-f` | `yolo` | Output format: `yolo` or `coco` |
| `--project` | `-p` | Current dir | Main project path |
| `--workers` | `-w` | `4` | Maximum parallel workers |
| `--memory` | `-m` | `2048` | Memory limit in MB |
| `--fps-limit` | - | `None` | Limit FPS extraction (e.g., `2.0`) |
| `--benchmark` | - | `False` | Enable performance benchmarking |

## 💡 Advanced Examples

### Example 1: List all projects
```bash
python main.py --list-projects
```

### Example 2: Interactive mode (recommended for new users)
```bash
python main.py \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./dataset
```
_Prompts you to select projects and videos interactively_

### Example 3: Process multiple specific projects
```bash
python main.py \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./dataset_combined \
  --project-ids 5,7,12
```
_Processes projects 5, 7, and 12 with interactive video selection for each_

### Example 4: Single project (backward compatible)
```bash
python main.py \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./dataset_single \
  --project-id 5
```
_Uses the original single-project workflow_

### Example 5: High-performance processing
```bash
python main.py \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./dataset_fast \
  --project-ids 5,7 \
  --workers 8 \
  --memory 4096 \
  --benchmark
```
_Uses 8 workers, 4GB memory, and generates performance benchmarks_

### Example 6: FPS limiting for smaller datasets
```bash
python main.py \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./dataset_sampled \
  --project-id 5 \
  --fps-limit 2.0
```
_Extracts frames at 2 FPS (every 0.5 seconds) instead of all annotated frames_

### Example 7: Project range selection (interactive mode)
```bash
python main.py \
  --classes '{"car":0,"truck":1,"bus":2}' \
  --output ./traffic_dataset
```
Then in the interactive prompt, enter: `1-5,8,10-12`
_Processes projects 1,2,3,4,5,8,10,11,12_

### Example 8: Export in COCO format
```bash
python main.py \
  --format coco \
  --classes '{"cyclist":0,"person":1,"scooter-roller":2}' \
  --output ./dataset_coco \
  --project-id 5
```
_Generates COCO-format annotations.json for Detectron2/MMDetection_

### Example 9: Multi-project with COCO format
```bash
python main.py \
  --format coco \
  --classes '{"cyclist":0,"person":1}' \
  --output ./dataset_multi_coco \
  --project-ids 5,7,12
```
_Combines multiple projects into a single COCO dataset_

## 📝 YOLO Format Details

**YOLO annotation format**: Each line represents one object
```
class_id center_x center_y width height
```

All coordinates are normalized to 0-1 range:
- `class_id`: Integer class identifier (from your class mappings)
- `center_x`: X coordinate of bounding box center (0-1)
- `center_y`: Y coordinate of bounding box center (0-1)
- `width`: Bounding box width (0-1)
- `height`: Bounding box height (0-1)

**Example annotation file** (`labels/frame_example_000001.txt`):
```
0 0.5234 0.6123 0.1245 0.2341
1 0.7891 0.3456 0.0923 0.1567
```
_First line: class 0 (cyclist) centered at (0.52, 0.61) with size (0.12, 0.23)_  
_Second line: class 1 (pedestrian) centered at (0.79, 0.35) with size (0.09, 0.16)_

**YAML config structure** (`data.yaml`):
```yaml
path: .
train: images
val: images
test: images

nc: 3
names: ['cyclist', 'pedestrian', 'scooter-roller']
```
## 📋 Requirements

- Python 3.8+
- Label Studio server with API access
- FFmpeg (for video processing)

### Python Dependencies

```txt
typer
opencv-python
numpy
pathlib
PyYAML
label-studio-sdk
python-dotenv
pytest          # For running tests
pytest-cov      # For coverage reports
pytest-mock     # For mocking in tests
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

## 🧪 Testing

The project includes a comprehensive test suite using pytest with 95%+ code coverage.

### Installing Test Dependencies

Already included in `requirements.txt`:
```bash
pip install -r requirements.txt
```

This installs `pytest`, `pytest-cov`, and `pytest-mock` along with the main dependencies.

### Running Tests

```bash
# Run all tests with verbose output
pytest test_frame_matcher.py -v

# Run with coverage report
pytest test_frame_matcher.py --cov=. --cov-report=html --cov-report=term

# Run specific test class
pytest test_frame_matcher.py::TestListAvailableProjects -v

# Run specific test function
pytest test_frame_matcher.py::TestListAvailableProjects::test_list_projects_success -v

# Stop at first failure (useful for debugging)
pytest test_frame_matcher.py -x

# Show print statements (useful for debugging)
pytest test_frame_matcher.py -v -s

# View HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Test Coverage

The test suite (`test_frame_matcher.py`) covers:
- ✅ **Label Studio API Integration**: REST API calls, authentication, error handling
- ✅ **CLI Argument Parsing**: All command-line parameters and validation
- ✅ **Interactive Selection**: Project and video selection workflows
- ✅ **Multi-Project Processing**: Project mapping, statistics aggregation
- ✅ **File Naming**: Single-project and multi-project naming conventions
- ✅ **Error Handling**: Connection errors, invalid inputs, edge cases
- ✅ **Backward Compatibility**: Single-project workflows remain unchanged
- ✅ **Performance**: Worker configuration, memory limits, FPS limiting

### Coverage Report Example

```
Name                              Stmts   Miss  Cover
-----------------------------------------------------
main.py                             412     18    96%
utils/annotation_processor.py      256     12    95%
utils/downloader.py                 143      8    94%
utils/interactive_selector.py       98      5    95%
-----------------------------------------------------
TOTAL                               909     43    95%
```

### Writing New Tests

When adding new features, add corresponding tests following the existing structure:

```python
class TestYourNewFeature:
    """Test suite for your new feature."""

    def test_basic_functionality(self):
        """Test the basic functionality works as expected."""
        # Arrange
        expected_result = "expected"
        
        # Act
        actual_result = your_function()
        
        # Assert
        assert actual_result == expected_result

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            your_function(invalid_input)
```

## 📌 Important Notes

- **Annotated Frames Only**: Only frames with annotations are extracted (not all video frames)
- **Multi-Annotator Support**: Handles workflows with multiple annotators automatically
- **Multiple Bounding Boxes**: Supports multiple objects per frame
- **Unique Filenames**: Frames include video ID and frame number for uniqueness
- **Performance Optimized**: Batch processing for large datasets
- **No Naming Conflicts**: Multi-project mode prefixes files with `project_{id}_`
- **Consistent Class Schema**: All projects in a multi-project run must use the same classes

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest test_frame_matcher.py -v`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Label Studio SDK](https://github.com/heartexlabs/label-studio-sdk)
- Uses [OpenCV](https://opencv.org/) for video processing
- CLI powered by [Typer](https://typer.tiangolo.com/)

## 📧 Support

For issues, questions, or contributions, please open an issue on the [GitHub repository](https://github.com/Fatih0234/frame_matcher).
````