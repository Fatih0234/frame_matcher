# Video Annotation Processor for Object Detection

A powerful Python tool for converting video annotations from Label Studio JSON format to multiple object detection formats (YOLO, COCO). Features multi-project support, interactive selection, and optimized batch processing.

## Installation

### Configure Environment Variables

Create a `.env` file in the project root:
```env
LABEL_STUDIO_URL=http://your-label-studio-url
LABEL_STUDIO_API_KEY=your-api-key-here
```

**Note**: You can find your API key in Label Studio under Account & Settings → Access Token.

## Usage

### Quick Start Guide

#### List Available Projects
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

#### Interactive Mode (Recommended)
Let the tool guide you through project and video selection:
```bash
python main.py \
  --classes '{"cyclist":0,"pedestrian":1,"scooter-roller":2}' \
  --output ./dataset \
  --cache-dir ./downloads
```

**What happens:**
- Displays all available projects
- You select which projects to process (e.g., `5,7` or `1-3` or `all`). This is also configurable with `--project-ids` parameter.
- For each project, select specific videos interactively
- Downloads videos and annotations automatically to the cache directory
- Extracts annotated frames with project-specific naming
- Generates combined dataset with project mapping (YOLO format by default, can be changed to COCO with `--format coco`)

#### Process Specific Projects
Skip the project selection menu and process specific projects:
```bash
python main.py \
  --classes '{"cyclist":0,"pedestrian":1,"scooter-roller":2}' \
  --output ./dataset_multi \
  --project-ids 5,7,12
```

## Directory Structure

### Cache Directory (`--cache-dir`)

Stores downloaded videos and annotations from Label Studio. Default location: `label_studio_data/`

### Output Directory (`--output`)

Final training dataset with extracted frames and labels.

**Example:**
```bash
# Use /tmp for automatic cleanup on reboot
python main.py \
  --cache-dir /tmp/label_studio_cache \
  --classes '{"cyclist":0}' \
  --output ./dataset \
  --project-ids 5
```

## Configuration

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

The tool includes automatic memory detection and will warn you if your configuration exceeds safe limits.

- `--workers` (`-w`): Number of parallel processing workers (default: 4)
- `--memory` (`-m`): Memory limit in MB per worker (default: 2048)
- `--fps-limit`: Limit frame extraction rate (optional, e.g., `--fps-limit 2.0` extracts 2 frames per second)

## Command-Line Parameters

### Required Parameters (for processing)
| Parameter | Short | Description | Example |
|-----------|-------|-------------|---------|
| `--classes` | `-c` | Class mappings as JSON string | `'{"cyclist":0,"pedestrian":1}'` |
| `--output` | `-o` | Output directory path | `./dataset` |

### Project Selection (choose one)
| Parameter | Description | Example |
|-----------|-------------|---------|
| `--list-projects` | List all available projects and exit | `--list-projects` |
| `--project-ids` | Single project ID or comma-separated list | `--project-ids 3` or `--project-ids 5,7,12` |
| _(none)_ | Interactive project selection mode (default) | - |

### Optional Parameters
| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--format` | `-f` | `yolo` | Output format: `yolo` or `coco` |
| `--project` | `-p` | Current dir | Main project path |
| `--cache-dir` | - | `label_studio_data` | Cache directory for downloads (safe to delete after processing) |
| `--workers` | `-w` | `4` | Maximum parallel workers |
| `--memory` | `-m` | `2048` | Memory limit in MB |
| `--fps-limit` | - | `None` | Limit FPS extraction (e.g., `2.0`) |
| `--benchmark` | - | `False` | Enable performance benchmarking |

## Requirements

- Python 3.8+
- Label Studio server with API access
- FFmpeg (for video processing)

## Important Notes

- Annotated Frames Only: Only frames with annotations are extracted (not all video frames)
- Multi-Annotator Support: Handles workflows with multiple annotators automatically
- Multiple Bounding Boxes: Supports multiple objects per frame
- Unique Filenames: Frames include video ID and frame number for uniqueness
- Performance Optimized: Batch processing for large datasets
- No Naming Conflicts: Multi-project mode prefixes files with `project_{id}_`
- Consistent Class Schema: All projects in a multi-project run must use the same classes

## License

This project is licensed under the MIT License - see the LICENSE file for details.
````