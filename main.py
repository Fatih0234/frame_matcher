"""
Main CLI interface for converting video annotations to YOLO or COCO format.
"""

import typer
from pathlib import Path
from typing import Dict, Optional
import json
import sys

from utils.annotation_processor import AnnotationProcessor

def main(
    format_type: str = typer.Option("yolo", "--format", "-f", help="Output format: yolo or coco"),
    classes: str = typer.Option(..., "--classes", "-c", help="Class mappings as JSON string, e.g., '{\"cyclist\":0,\"person\":1,\"scooter-roller\":2}'"),
    output_path: str = typer.Option(..., "--output", "-o", help="Path where the dataset will be saved"),
    project_path: str = typer.Option(None, "--project", "-p", help="Main project path (default: directory where main.py is located)"),
):
    """
    Convert video annotations to specified format.
    
    Example usage:
    python main.py --format yolo --classes '{"cyclist":0,"person":1,"scooter-roller":2}' --output ./dataset
    """
    
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
    
    # Setup paths - use directory where main.py is located as default
    if project_path is None:
        project_path = Path(__file__).parent
    else:
        project_path = Path(project_path)
    output_path = Path(output_path)
    annotations_file = project_path / "json_file" / "annotations.json"
    video_files_dir = project_path / "video_files"
    
    # Validate paths exist
    if not annotations_file.exists():
        typer.echo(f"Error: Annotations file not found at {annotations_file}", err=True)
        raise typer.Exit(1)
    
    if not video_files_dir.exists():
        typer.echo(f"Error: Video files directory not found at {video_files_dir}", err=True)
        raise typer.Exit(1)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize processor
        processor = AnnotationProcessor(
            annotations_file=annotations_file,
            video_files_dir=video_files_dir,
            class_mappings=class_mappings
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