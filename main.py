"""
Unified CLI interface for converting video annotations to YOLO format.
This combines interactive selection, auto-download, and optimized processing.
"""

import typer
from pathlib import Path
import json
import os
from dotenv import load_dotenv
import logging
import time

from utils.annotation_processor import AnnotationProcessor
from utils.downloader import LabelStudioDownloader
from utils.interactive_selector import create_interactive_selector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(
    classes: str = typer.Option(..., "--classes", "-c", help="Class mappings as JSON string, e.g., '{\"cyclist\":0,\"person\":1,\"scooter-roller\":2}'"),
    output_path: str = typer.Option(..., "--output", "-o", help="Path where the dataset will be saved"),
    project_path: str = typer.Option(None, "--project", "-p", help="Main project path (default: directory where main.py is located)"),
    project_id: int = typer.Option(..., "--project-id", help="Label Studio project ID (required)"),
    max_workers: int = typer.Option(4, "--workers", "-w", help="Maximum number of parallel workers (default: 4)"),
    memory_limit: int = typer.Option(2048, "--memory", "-m", help="Memory limit in MB for batch processing (default: 2048)"),
    fps_limit: float = typer.Option(None, "--fps-limit", help="Limit frames per second extraction (e.g., 2.0 for 2 FPS, None for all frames)"),
    benchmark: bool = typer.Option(False, "--benchmark", help="Enable detailed performance benchmarking"),
):
    """
    Convert video annotations to YOLO format with optimized processing.
    
    Features:
    - Interactive video selection from Label Studio
    - Automatic download from Label Studio
    - Optimized parallel processing
    - Vectorized coordinate conversion
    - Adaptive batch sizing
    - Performance benchmarking
    
    Example usage:
    python main.py --classes '{"cyclist":0,"person":1,"scooter-roller":2}' --output ./dataset --project-id 5
    
    With performance optimization:
    python main.py --classes '{"cyclist":0,"person":1,"scooter-roller":2}' --output ./dataset --workers 8 --memory 4096 --benchmark
    """
    
    total_start_time = time.time()
    
    # Setup paths
    if project_path is None:
        project_path = Path(__file__).parent
    else:
        project_path = Path(project_path)
    output_path = Path(output_path)
    
    # Load environment variables
    load_dotenv(project_path / ".env")
    
    # Get credentials from environment
    url = os.getenv("LABEL_STUDIO_URL")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    env_project_id = os.getenv("PROJECT_ID")
    
    # Use project_id from CLI or environment
    if project_id is None and env_project_id:
        project_id = int(env_project_id)
    
    # Validate environment variables and project_id
    if not url or not api_key:
        typer.echo("Error: .env file must contain LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY", err=True)
        typer.echo("Please create a .env file with the required credentials.", err=True)
        raise typer.Exit(1)
    
    if project_id is None:
        typer.echo("Error: --project-id is required", err=True)
        raise typer.Exit(1)
    
    # Interactive video selection
    typer.echo("Interactive video selection mode...")
    selector = create_interactive_selector(url, api_key, project_id)
    selected_videos = selector.select_videos_interactive()
    
    if not selected_videos:
        typer.echo("No videos selected. Exiting.", err=True)
        raise typer.Exit(0)
    
    selected_task_ids = selector.get_selected_video_tasks(selected_videos)
    typer.echo(f"Will process {len(selected_task_ids)} selected videos")
    
    # Download data from Label Studio with smart selective download
    typer.echo("Downloading data from Label Studio...")
    downloader = LabelStudioDownloader(url, api_key, project_id)
    
    success, annotations_file, video_files = downloader.download_all(
        video_dir=str(project_path / "exported_videos"),
        json_dir=str(project_path / "exported_json_annotation"),
        selected_task_ids=selected_task_ids
    )
    
    if not success:
        typer.echo("Failed to download data from Label Studio", err=True)
        raise typer.Exit(1)
    
    # Update paths to use downloaded data
    annotations_file = Path(annotations_file)
    video_files_dir = project_path / "exported_videos"
    
    # Use exact matching when using Label Studio
    use_exact_matching = True
    
    # Validate paths exist
    if not annotations_file.exists():
        typer.echo(f"Error: Annotations file not found at {annotations_file}", err=True)
        raise typer.Exit(1)
    
    if not video_files_dir.exists():
        typer.echo(f"Error: Video files directory not found at {video_files_dir}", err=True)
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
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize processor with optimization settings
        typer.echo(f"Initializing processor with {max_workers} workers, {memory_limit}MB memory limit...")
        init_start = time.time()
        
        processor = AnnotationProcessor(
            annotations_file=annotations_file,
            video_files_dir=video_files_dir,
            class_mappings=class_mappings,
            use_exact_matching=use_exact_matching,
            max_workers=max_workers,
            memory_limit_mb=memory_limit,
            fps_limit=fps_limit
        )
        
        init_time = time.time() - init_start
        typer.echo(f"Initialization completed in {init_time:.2f}s")
        
        # Process annotations
        typer.echo("Starting annotation processing...")
        processing_start = time.time()
        
        processor.convert_to_yolo(output_path)
        typer.echo(f"YOLO dataset created successfully at {output_path}")
        
        processing_time = time.time() - processing_start
        total_time = time.time() - total_start_time
        
        # Performance summary
        typer.echo("\\nPERFORMACE SUMMARY:")
        typer.echo(f"   Total Runtime: {total_time:.2f}s")
        typer.echo(f"   Processing Time: {processing_time:.2f}s")
        
        if benchmark:
            # Get detailed performance stats
            stats = processor.get_performance_stats()
            typer.echo("\\nDETAILED BENCHMARKS:")
            typer.echo(f"   Frame Extraction: {stats['extraction_time']:.2f}s")
            typer.echo(f"   Coordinate Conversion: {stats['conversion_time']:.2f}s")
            typer.echo(f"   File I/O: {stats['io_time']:.2f}s")
            typer.echo(f"   Videos Processed: {stats['videos_processed']}")
            typer.echo(f"   Frames Extracted: {stats['frames_extracted']}")
            
            if stats['frames_extracted'] > 0:
                fps = stats['frames_extracted'] / max(stats['total_processing_time'], 0.1)
                typer.echo(f"   Processing Rate: {fps:.2f} FPS")
            
            # Save benchmark results
            benchmark_file = output_path / "benchmark_results.json"
            with open(benchmark_file, 'w') as f:
                json.dump(stats, f, indent=2)
            typer.echo(f"Benchmark results saved to: {benchmark_file}")
            
    except Exception as e:
        typer.echo(f"Error during processing: {e}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    typer.run(main)
