from abc import ABC, abstractmethod
from typing import Dict, List, Any
from pathlib import Path


class BaseExporter(ABC):
    """
    Abstract base class for annotation format exporters.

    This class defines the interface that all exporters must implement,
    enabling easy addition of new formats in the future.
    """

    def __init__(self, output_dir: Path, class_mapping: Dict[str, int]):
        """
        Initialize the exporter.

        Args:
            output_dir: Directory where output files will be saved
            class_mapping: Dictionary mapping class names to integer IDs
        """
        self.output_dir = Path(output_dir)
        self.class_mapping = class_mapping
        self.setup_directories()

    @abstractmethod
    def setup_directories(self) -> None:
        """Create necessary output directories for this format."""
        pass

    @abstractmethod
    def export_frame(
        self,
        frame_data: Dict[str, Any],
        image_filename: str,
        frame_width: int,
        frame_height: int
    ) -> None:
        """
        Export annotations for a single frame.

        Args:
            frame_data: Dictionary containing frame annotations
            image_filename: Name of the image file
            frame_width: Width of the frame in pixels
            frame_height: Height of the frame in pixels
        """
        pass

    @abstractmethod
    def finalize(self) -> None:
        """
        Finalize the export process.

        This method is called after all frames have been exported.
        Use it to write summary files, generate configs, etc.
        """
        pass
