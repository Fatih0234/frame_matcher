"""
Main CLI interface for converting video annotations to YOLO or COCO format.
"""

import typer
from pathlib import Path
import json
import os
from dotenv import load_dotenv
import logging

from utils.annotation_processor import AnnotationProcessor
from utils.downloader import LabelStudioDownloader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(
    format_type: str = typer.Option("yolo", "--format", "-f", help="Output format: yolo or coco"),
    classes: str = typer.Option(..., "--classes", "-c", help="Class mappings as JSON string, e.g., '{\"cyclist\":0,\"person\":1,\"scooter-roller\":2}'"),
    output_path: str = typer.Option(..., "--output", "-o", help="Path where the dataset will be saved"),
    project_path: str = typer.Option(None, "--project", "-p", help="Main project path (default: directory where main.py is located)"),
    auto_download: bool = typer.Option(False, "--auto-download", help="Automatically download annotations and videos from Label Studio"),
    project_id: int = typer.Option(None, "--project-id", help="Label Studio project ID (required if auto-download is used)"),
):
    """
    Convert video annotations to specified format.
    
    Example usage:
    python main.py --format yolo --classes '{"cyclist":0,"person":1,"scooter-roller":2}' --output ./dataset
    
    With auto-download:
    python main.py --format yolo --classes '{"cyclist":0,"person":1,"scooter-roller":2}' --output ./dataset --auto-download --project-id 5
    """
    
    # Setup paths - use directory where main.py is located as default
    if project_path is None:
        project_path = Path(__file__).parent
    else:
        project_path = Path(project_path)
    output_path = Path(output_path)
    
    # Handle auto-download
    if auto_download:
        # Load environment variables
        load_dotenv(project_path / ".env")
        
        # Get credentials from environment
        url = os.getenv("LABEL_STUDIO_URL")
        api_key = os.getenv("LABEL_STUDIO_API_KEY")
        env_project_id = os.getenv("PROJECT_ID")
        
        # Use project_id from CLI or environment
        if project_id is None and env_project_id:
            project_id = int(env_project_id)
        elif project_id is None:
            typer.echo("Error: --project-id is required when using --auto-download", err=True)
            raise typer.Exit(1)
        
        # Validate environment variables
        if not url or not api_key:
            typer.echo("Error: .env file must contain LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY", err=True)
            typer.echo("Please create a .env file with the required credentials.", err=True)
            raise typer.Exit(1)
        
        # Download data from Label Studio
        typer.echo("🔽 Downloading data from Label Studio...")
        downloader = LabelStudioDownloader(url, api_key, project_id)
        
        success, annotations_file, video_files = downloader.download_all(
            video_dir=str(project_path / "exported_videos"),
            json_dir=str(project_path / "exported_json_annotation")
        )
        
        if not success:
            typer.echo("❌ Failed to download data from Label Studio", err=True)
            raise typer.Exit(1)
        
        # Update paths to use downloaded data
        annotations_file = Path(annotations_file)
        video_files_dir = project_path / "exported_videos"
        
    else:
        # Use traditional file paths
        annotations_file = project_path / "json_file" / "annotations.json"
        video_files_dir = project_path / "video_files"
        
        # Check if we should look in exported folders instead
        exported_json_dir = project_path / "exported_json_annotation"
        exported_videos_dir = project_path / "exported_videos"
        
        # If exported_json_annotation exists, look for any JSON file there
        if exported_json_dir.exists():
            json_files = list(exported_json_dir.glob("*.json"))
            if json_files:
                annotations_file = json_files[0]  # Use the first JSON file found
                typer.echo(f"📂 Using annotations from: {annotations_file}")
        
        # If exported_videos exists, use it instead of video_files
        if exported_videos_dir.exists():
            video_files_dir = exported_videos_dir
            typer.echo(f"📂 Using videos from: {video_files_dir}")
        
        # Validate paths exist
        if not annotations_file.exists():
            typer.echo(f"Error: Annotations file not found at {annotations_file}", err=True)
            raise typer.Exit(1)
        
        if not video_files_dir.exists():
            typer.echo(f"Error: Video files directory not found at {video_files_dir}", err=True)
            raise typer.Exit(1)
    # Validate format type
    if format_type.lower() not in ["yolo", "coco"]:
        typer.echo("Error: Format must be either 'yolo' or 'coco'", err=True)
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
        # Initialize processor
        processor = AnnotationProcessor(
            annotations_file=annotations_file,
            video_files_dir=video_files_dir,
            class_mappings=class_mappings,
            use_exact_matching=auto_download  # Use exact matching when auto-downloading
        )
        
        # Process annotations
        typer.echo("🚀 Starting annotation processing...")
        
        if format_type.lower() == "yolo":
            processor.convert_to_yolo(output_path)
            typer.echo(f"✅ YOLO dataset created successfully at {output_path}")
        else:
            processor.convert_to_coco(output_path)
            typer.echo(f"✅ COCO dataset created successfully at {output_path}")
            
    except Exception as e:
        typer.echo(f"❌ Error during processing: {e}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    typer.run(main)