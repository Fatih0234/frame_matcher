"""
Unified CLI interface for converting video annotations to YOLO format.
This combines interactive selection, auto-download, and optimized processing.
Supports multi-project workflows for batch processing.
"""

import typer
from pathlib import Path
import json
import os
from dotenv import load_dotenv
import logging
import time
from typing import List, Dict, Any, Optional
from label_studio_sdk import Client

from utils.annotation_processor import AnnotationProcessor
from utils.downloader import LabelStudioDownloader
from utils.interactive_selector import create_interactive_selector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def list_available_projects(url: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Fetch and display all available projects from Label Studio server.

    Args:
        url: Label Studio server URL
        api_key: Label Studio API key

    Returns:
        List of project dictionaries with id, title, and task count
    """
    try:
        client = Client(url=url, api_key=api_key)
        projects = client.projects.list()

        if not projects:
            typer.echo("No projects found on Label Studio server.")
            return []

        # Display projects in a formatted table
        typer.echo("\n" + "="*80)
        typer.echo("AVAILABLE LABEL STUDIO PROJECTS")
        typer.echo("="*80)
        typer.echo(f"{'ID':<6} | {'Title':<40} | {'Tasks':<10}")
        typer.echo("-"*80)

        project_list = []
        for project in projects:
            project_id = project.get('id')
            title = project.get('title', 'Untitled')
            task_count = project.get('task_number', 0)

            # Truncate title if too long
            display_title = title[:37] + "..." if len(title) > 40 else title
            typer.echo(f"{project_id:<6} | {display_title:<40} | {task_count:<10}")

            project_list.append({
                'id': project_id,
                'title': title,
                'task_count': task_count
            })

        typer.echo("="*80 + "\n")
        return project_list

    except Exception as e:
        typer.echo(f"Error fetching projects: {e}", err=True)
        return []


def interactive_project_selection(projects: List[Dict[str, Any]]) -> List[int]:
    """
    Allow user to interactively select one or more projects.

    Args:
        projects: List of project dictionaries from list_available_projects()

    Returns:
        List of selected project IDs
    """
    if not projects:
        return []

    typer.echo("PROJECT SELECTION:")
    typer.echo("   • Enter project IDs separated by commas (e.g., '5,7,12')")
    typer.echo("   • Enter ranges with dash (e.g., '1-3,7-9')")
    typer.echo("   • Enter 'all' to select all projects")
    typer.echo("   • Enter 'quit' or 'q' to exit")
    typer.echo("-"*80)

    while True:
        try:
            user_input = input("\nSelect projects to process: ").strip().lower()

            if user_input in ['quit', 'q']:
                typer.echo("Selection cancelled.")
                return []

            if user_input == 'all':
                selected_ids = [p['id'] for p in projects]
                typer.echo(f"Selected all {len(selected_ids)} projects")
                return selected_ids

            # Parse comma-separated IDs and ranges
            selected_ids = set()
            parts = user_input.replace(',', ' ').split()

            for part in parts:
                if '-' in part:
                    # Handle ranges like "1-3"
                    try:
                        start, end = map(int, part.split('-'))
                        if start > end:
                            start, end = end, start
                        for pid in range(start, end + 1):
                            if any(p['id'] == pid for p in projects):
                                selected_ids.add(pid)
                    except ValueError:
                        typer.echo(f"Invalid range format: {part}", err=True)
                        continue
                else:
                    # Handle single IDs
                    try:
                        pid = int(part)
                        if any(p['id'] == pid for p in projects):
                            selected_ids.add(pid)
                        else:
                            typer.echo(f"Project ID {pid} not found", err=True)
                    except ValueError:
                        typer.echo(f"Invalid project ID: {part}", err=True)
                        continue

            if not selected_ids:
                typer.echo("No valid projects selected. Please try again.")
                continue

            # Confirm selection
            selected_projects = [p for p in projects if p['id'] in selected_ids]
            typer.echo(f"\nSelected {len(selected_projects)} projects:")
            for p in selected_projects:
                typer.echo(f"   • [ID: {p['id']}] {p['title']} ({p['task_count']} tasks)")

            confirm = input(f"\nProcess these {len(selected_projects)} projects? (y/N): ").strip().lower()

            if confirm in ['y', 'yes']:
                return sorted(list(selected_ids))
            else:
                typer.echo("Let's try again...")
                continue

        except KeyboardInterrupt:
            typer.echo("\n\nSelection cancelled by user.")
            return []
        except Exception as e:
            typer.echo(f"Error during selection: {e}", err=True)
            continue


def process_single_project(
    project_id: int,
    url: str,
    api_key: str,
    class_mappings: Dict[str, int],
    project_path: Path,
    output_path: Path,
    max_workers: int,
    memory_limit: int,
    fps_limit: Optional[float],
    benchmark: bool
) -> Dict[str, Any]:
    """
    Process a single project and return statistics.

    Returns:
        Dictionary with project statistics including frames_extracted, annotations_count, etc.
    """
    typer.echo(f"\n{'='*80}")
    typer.echo(f"PROCESSING PROJECT ID: {project_id}")
    typer.echo(f"{'='*80}")

    # Get project title
    try:
        client = Client(url=url, api_key=api_key)
        project = client.get_project(project_id)
        project_title = project.title if hasattr(project, 'title') else f"Project {project_id}"
    except:
        project_title = f"Project {project_id}"

    # Interactive video selection for this project
    typer.echo("Interactive video selection mode...")
    selector = create_interactive_selector(url, api_key, project_id)
    selected_videos = selector.select_videos_interactive()

    if not selected_videos:
        typer.echo(f"No videos selected for project {project_id}. Skipping.", err=True)
        return None

    selected_task_ids = selector.get_selected_video_tasks(selected_videos)
    typer.echo(f"Will process {len(selected_task_ids)} selected videos")

    # Download data from Label Studio
    typer.echo("Downloading data from Label Studio...")
    downloader = LabelStudioDownloader(url, api_key, project_id)

    success, annotations_file, video_files = downloader.download_all(
        video_dir=str(project_path / f"exported_videos/project_{project_id}"),
        json_dir=str(project_path / f"exported_json_annotation/project_{project_id}"),
        selected_task_ids=selected_task_ids
    )

    if not success:
        typer.echo(f"Failed to download data for project {project_id}", err=True)
        return None

    annotations_file = Path(annotations_file)
    video_files_dir = project_path / f"exported_videos/project_{project_id}"

    # Validate paths
    if not annotations_file.exists():
        typer.echo(f"Error: Annotations file not found at {annotations_file}", err=True)
        return None

    if not video_files_dir.exists():
        typer.echo(f"Error: Video files directory not found at {video_files_dir}", err=True)
        return None

    # Process annotations with project ID for file naming
    try:
        processor = AnnotationProcessor(
            annotations_file=annotations_file,
            video_files_dir=video_files_dir,
            class_mappings=class_mappings,
            use_exact_matching=True,
            max_workers=max_workers,
            memory_limit_mb=memory_limit,
            fps_limit=fps_limit,
            project_id=project_id  # Pass project_id for file naming
        )

        processor.convert_to_yolo(output_path)

        # Get statistics
        stats = processor.get_performance_stats()

        return {
            'id': project_id,
            'title': project_title,
            'frames_extracted': stats.get('frames_extracted', 0),
            'annotations_count': stats.get('frames_extracted', 0),  # Each frame has at least one annotation
            'videos_processed': stats.get('videos_processed', 0)
        }

    except Exception as e:
        typer.echo(f"Error processing project {project_id}: {e}", err=True)
        return None


def process_multiple_projects(
    project_ids: List[int],
    url: str,
    api_key: str,
    class_mappings: Dict[str, int],
    project_path: Path,
    output_path: Path,
    max_workers: int,
    memory_limit: int,
    fps_limit: Optional[float],
    benchmark: bool
) -> None:
    """
    Process multiple projects and generate combined dataset with project_mapping.json.

    Args:
        project_ids: List of Label Studio project IDs to process
        url: Label Studio server URL
        api_key: Label Studio API key
        class_mappings: Dictionary mapping class names to integer IDs
        project_path: Main project path
        output_path: Output directory for dataset
        max_workers: Number of parallel workers
        memory_limit: Memory limit in MB
        fps_limit: FPS limit for frame extraction
        benchmark: Whether to enable benchmarking
    """
    typer.echo(f"\n{'='*80}")
    typer.echo(f"MULTI-PROJECT PROCESSING MODE")
    typer.echo(f"Processing {len(project_ids)} projects: {project_ids}")
    typer.echo(f"{'='*80}\n")

    project_stats = []
    total_frames = 0
    total_annotations = 0

    for project_id in project_ids:
        stats = process_single_project(
            project_id=project_id,
            url=url,
            api_key=api_key,
            class_mappings=class_mappings,
            project_path=project_path,
            output_path=output_path,
            max_workers=max_workers,
            memory_limit=memory_limit,
            fps_limit=fps_limit,
            benchmark=benchmark
        )

        if stats:
            project_stats.append(stats)
            total_frames += stats['frames_extracted']
            total_annotations += stats['annotations_count']

    # Generate project_mapping.json
    if project_stats:
        mapping = {
            'projects': project_stats,
            'total_frames': total_frames,
            'total_annotations': total_annotations,
            'classes': class_mappings
        }

        mapping_file = output_path / "project_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)

        typer.echo(f"\n{'='*80}")
        typer.echo("MULTI-PROJECT SUMMARY")
        typer.echo(f"{'='*80}")
        typer.echo(f"Projects processed: {len(project_stats)}/{len(project_ids)}")
        typer.echo(f"Total frames extracted: {total_frames:,}")
        typer.echo(f"Total annotations: {total_annotations:,}")
        typer.echo(f"\nProject mapping saved to: {mapping_file}")
        typer.echo(f"{'='*80}\n")


def main(
    classes: str = typer.Option(None, "--classes", "-c", help="Class mappings as JSON string, e.g., '{\"cyclist\":0,\"person\":1,\"scooter-roller\":2}'"),
    output_path: str = typer.Option(None, "--output", "-o", help="Path where the dataset will be saved"),
    project_path: str = typer.Option(None, "--project", "-p", help="Main project path (default: directory where main.py is located)"),
    project_id: Optional[int] = typer.Option(None, "--project-id", help="Single Label Studio project ID (for backward compatibility)"),
    project_ids: Optional[str] = typer.Option(None, "--project-ids", help="Comma-separated list of project IDs (e.g., '5,7,12')"),
    list_projects: bool = typer.Option(False, "--list-projects", help="List all available projects and exit"),
    max_workers: int = typer.Option(4, "--workers", "-w", help="Maximum number of parallel workers (default: 4)"),
    memory_limit: int = typer.Option(2048, "--memory", "-m", help="Memory limit in MB for batch processing (default: 2048)"),
    fps_limit: Optional[float] = typer.Option(None, "--fps-limit", help="Limit frames per second extraction (e.g., 2.0 for 2 FPS, None for all frames)"),
    benchmark: bool = typer.Option(False, "--benchmark", help="Enable detailed performance benchmarking"),
):
    """
    Convert video annotations to YOLO format with optimized processing.
    Supports single-project and multi-project workflows.

    Features:
    - Multi-project support with automatic project selection
    - Interactive video selection from Label Studio
    - Automatic download from Label Studio
    - Optimized parallel processing
    - Vectorized coordinate conversion
    - Adaptive batch sizing
    - Performance benchmarking

    Example usage:

    # List all available projects
    python main.py --list-projects

    # Interactive mode (select projects interactively)
    python main.py --classes '{"cyclist":0,"person":1,"scooter-roller":2}' --output ./dataset

    # Process multiple projects
    python main.py --classes '{"cyclist":0,"person":1,"scooter-roller":2}' --output ./dataset --project-ids 5,7,12

    # Single project (backward compatible)
    python main.py --classes '{"cyclist":0,"person":1,"scooter-roller":2}' --output ./dataset --project-id 5

    # With performance optimization
    python main.py --classes '{"cyclist":0,"person":1,"scooter-roller":2}' --output ./dataset --project-ids 5,7 --workers 8 --memory 4096 --benchmark
    """

    total_start_time = time.time()

    # Setup paths
    if project_path is None:
        project_path = Path(__file__).parent
    else:
        project_path = Path(project_path)

    # Load environment variables
    load_dotenv(project_path / ".env")

    # Get credentials from environment
    url = os.getenv("LABEL_STUDIO_URL")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    env_project_id = os.getenv("PROJECT_ID")

    # Validate environment variables
    if not url or not api_key:
        typer.echo("Error: .env file must contain LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY", err=True)
        typer.echo("Please create a .env file with the required credentials.", err=True)
        raise typer.Exit(1)

    # Handle --list-projects mode (list and exit)
    if list_projects:
        list_available_projects(url, api_key)
        raise typer.Exit(0)

    # Determine which project(s) to process
    selected_project_ids: List[int] = []

    # Priority: --project-ids > --project-id > env PROJECT_ID > interactive mode
    if project_ids:
        # Parse comma-separated project IDs
        try:
            selected_project_ids = [int(pid.strip()) for pid in project_ids.split(',')]
            typer.echo(f"Processing projects from --project-ids: {selected_project_ids}")
        except ValueError as e:
            typer.echo(f"Error: Invalid format for --project-ids. Use comma-separated numbers (e.g., '5,7,12')", err=True)
            raise typer.Exit(1)
    elif project_id is not None:
        # Single project mode (backward compatibility)
        selected_project_ids = [project_id]
        typer.echo(f"Processing single project (--project-id): {project_id}")
    elif env_project_id:
        # Use environment variable (backward compatibility)
        selected_project_ids = [int(env_project_id)]
        typer.echo(f"Processing single project from .env (PROJECT_ID): {env_project_id}")
    else:
        # Interactive project selection mode
        typer.echo("No project specified. Entering interactive project selection mode...")
        available_projects = list_available_projects(url, api_key)

        if not available_projects:
            typer.echo("No projects available. Exiting.", err=True)
            raise typer.Exit(1)

        selected_project_ids = interactive_project_selection(available_projects)

        if not selected_project_ids:
            typer.echo("No projects selected. Exiting.")
            raise typer.Exit(0)

    # Validate required parameters for processing
    if not classes:
        typer.echo("Error: --classes is required for processing", err=True)
        raise typer.Exit(1)

    if not output_path:
        typer.echo("Error: --output is required for processing", err=True)
        raise typer.Exit(1)

    # Parse class mappings
    try:
        class_mappings = json.loads(classes)
        if not isinstance(class_mappings, dict):
            raise ValueError("Classes must be a dictionary")
    except (json.JSONDecodeError, ValueError) as e:
        typer.echo(f"Error parsing classes: {e}", err=True)
        raise typer.Exit(1)

    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process projects
    try:
        if len(selected_project_ids) == 1:
            # Single project mode (backward compatible workflow)
            single_project_id = selected_project_ids[0]
            typer.echo(f"\nProcessing single project: {single_project_id}")

            # Use the original single-project workflow
            selector = create_interactive_selector(url, api_key, single_project_id)
            selected_videos = selector.select_videos_interactive()

            if not selected_videos:
                typer.echo("No videos selected. Exiting.", err=True)
                raise typer.Exit(0)

            selected_task_ids = selector.get_selected_video_tasks(selected_videos)
            typer.echo(f"Will process {len(selected_task_ids)} selected videos")

            # Download data from Label Studio
            typer.echo("Downloading data from Label Studio...")
            downloader = LabelStudioDownloader(url, api_key, single_project_id)

            success, annotations_file, video_files = downloader.download_all(
                video_dir=str(project_path / "exported_videos"),
                json_dir=str(project_path / "exported_json_annotation"),
                selected_task_ids=selected_task_ids
            )

            if not success:
                typer.echo("Failed to download data from Label Studio", err=True)
                raise typer.Exit(1)

            # Update paths
            annotations_file = Path(annotations_file)
            video_files_dir = project_path / "exported_videos"

            # Validate paths
            if not annotations_file.exists():
                typer.echo(f"Error: Annotations file not found at {annotations_file}", err=True)
                raise typer.Exit(1)

            if not video_files_dir.exists():
                typer.echo(f"Error: Video files directory not found at {video_files_dir}", err=True)
                raise typer.Exit(1)

            # Process annotations
            typer.echo(f"Initializing processor with {max_workers} workers, {memory_limit}MB memory limit...")
            processor = AnnotationProcessor(
                annotations_file=annotations_file,
                video_files_dir=video_files_dir,
                class_mappings=class_mappings,
                use_exact_matching=True,
                max_workers=max_workers,
                memory_limit_mb=memory_limit,
                fps_limit=fps_limit
            )

            processor.convert_to_yolo(output_path)
            typer.echo(f"YOLO dataset created successfully at {output_path}")

            total_time = time.time() - total_start_time
            typer.echo(f"\nTotal Runtime: {total_time:.2f}s")

            if benchmark:
                stats = processor.get_performance_stats()
                typer.echo("\nDETAILED BENCHMARKS:")
                typer.echo(f"   Frame Extraction: {stats['extraction_time']:.2f}s")
                typer.echo(f"   Coordinate Conversion: {stats['conversion_time']:.2f}s")
                typer.echo(f"   File I/O: {stats['io_time']:.2f}s")
                typer.echo(f"   Videos Processed: {stats['videos_processed']}")
                typer.echo(f"   Frames Extracted: {stats['frames_extracted']}")

                benchmark_file = output_path / "benchmark_results.json"
                with open(benchmark_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                typer.echo(f"Benchmark results saved to: {benchmark_file}")

        else:
            # Multi-project mode
            process_multiple_projects(
                project_ids=selected_project_ids,
                url=url,
                api_key=api_key,
                class_mappings=class_mappings,
                project_path=project_path,
                output_path=output_path,
                max_workers=max_workers,
                memory_limit=memory_limit,
                fps_limit=fps_limit,
                benchmark=benchmark
            )

            total_time = time.time() - total_start_time
            typer.echo(f"\nTotal Runtime: {total_time:.2f}s")

    except Exception as e:
        typer.echo(f"Error during processing: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)

if __name__ == "__main__":
    typer.run(main)
