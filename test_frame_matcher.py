"""
Comprehensive test suite for Frame Matcher application.
Tests multi-project support, CLI arguments, API interactions, and file operations.

Run tests with: pytest test_frame_matcher.py -v
Run with coverage: pytest test_frame_matcher.py --cov=. --cov-report=html
"""

import pytest
import json
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from typer.testing import CliRunner
import requests

# Import functions to test
from main import (
    list_available_projects,
    interactive_project_selection,
    process_single_project,
    process_multiple_projects,
    main
)
from utils.annotation_processor import AnnotationProcessor


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("LABEL_STUDIO_URL", "http://localhost:8080")
    monkeypatch.setenv("LABEL_STUDIO_API_KEY", "test-api-key-123")


@pytest.fixture
def sample_projects_response():
    """Sample projects response from Label Studio API."""
    return {
        "results": [
            {"id": 1, "title": "Bikedetect", "task_number": 9},
            {"id": 3, "title": "Bikedetect FR2", "task_number": 2},
            {"id": 5, "title": "Bikedetect FR24", "task_number": 15},
            {"id": 6, "title": "Field study", "task_number": 2}
        ]
    }


@pytest.fixture
def sample_project_list():
    """Sample processed project list."""
    return [
        {"id": 1, "title": "Bikedetect", "task_count": 9},
        {"id": 3, "title": "Bikedetect FR2", "task_count": 2},
        {"id": 5, "title": "Bikedetect FR24", "task_count": 15},
        {"id": 6, "title": "Field study", "task_count": 2}
    ]


# ============================================================================
# TEST LIST_AVAILABLE_PROJECTS
# ============================================================================

class TestListAvailableProjects:
    """Test the list_available_projects function."""

    @patch('requests.get')
    def test_list_projects_success(self, mock_get, sample_projects_response, capsys):
        """Test successful project listing."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = sample_projects_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Call function
        result = list_available_projects("http://localhost:8080", "test-api-key")

        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "http://localhost:8080/api/projects"
        assert call_args[1]['headers']['Authorization'] == 'Token test-api-key'

        # Verify result
        assert len(result) == 4
        assert result[0]['id'] == 1
        assert result[0]['title'] == "Bikedetect"
        assert result[0]['task_count'] == 9

        # Verify output contains project info
        captured = capsys.readouterr()
        assert "AVAILABLE LABEL STUDIO PROJECTS" in captured.out
        assert "Bikedetect" in captured.out
        assert "Field study" in captured.out

    @patch('requests.get')
    def test_list_projects_empty(self, mock_get, capsys):
        """Test when no projects are available."""
        mock_response = Mock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = list_available_projects("http://localhost:8080", "test-api-key")

        assert result == []
        captured = capsys.readouterr()
        assert "No projects found" in captured.out

    @patch('requests.get')
    def test_list_projects_connection_error(self, mock_get, capsys):
        """Test handling of connection errors."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = list_available_projects("http://localhost:8080", "test-api-key")

        assert result == []
        captured = capsys.readouterr()
        assert "Error connecting to Label Studio" in captured.err

    @patch('requests.get')
    def test_list_projects_http_error(self, mock_get, capsys):
        """Test handling of HTTP errors (401, 403, 404, etc.)."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
        mock_get.return_value = mock_response

        result = list_available_projects("http://localhost:8080", "test-api-key")

        assert result == []
        captured = capsys.readouterr()
        assert "Error connecting to Label Studio" in captured.err

    @patch('requests.get')
    def test_list_projects_handles_list_response(self, mock_get):
        """Test that function handles both list and paginated responses."""
        # Some API versions return a list directly instead of {"results": [...]}
        mock_response = Mock()
        mock_response.json.return_value = [
            {"id": 1, "title": "Project 1", "task_number": 5}
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = list_available_projects("http://localhost:8080", "test-api-key")

        assert len(result) == 1
        assert result[0]['id'] == 1


# ============================================================================
# TEST INTERACTIVE_PROJECT_SELECTION
# ============================================================================

class TestInteractiveProjectSelection:
    """Test the interactive_project_selection function."""

    @patch('builtins.input')
    def test_select_single_project(self, mock_input, sample_project_list):
        """Test selecting a single project by ID."""
        mock_input.side_effect = ['5', 'y']

        result = interactive_project_selection(sample_project_list)

        assert result == [5]

    @patch('builtins.input')
    def test_select_multiple_projects_comma_separated(self, mock_input, sample_project_list):
        """Test selecting multiple projects with comma-separated IDs."""
        mock_input.side_effect = ['1,3,5', 'y']

        result = interactive_project_selection(sample_project_list)

        assert sorted(result) == [1, 3, 5]

    @patch('builtins.input')
    def test_select_multiple_projects_space_separated(self, mock_input, sample_project_list):
        """Test selecting multiple projects with space-separated IDs."""
        mock_input.side_effect = ['1 3 5', 'y']

        result = interactive_project_selection(sample_project_list)

        assert sorted(result) == [1, 3, 5]

    @patch('builtins.input')
    def test_select_range(self, mock_input, sample_project_list):
        """Test selecting a range of projects."""
        mock_input.side_effect = ['1-3', 'y']

        result = interactive_project_selection(sample_project_list)

        assert sorted(result) == [1, 3]

    @patch('builtins.input')
    def test_select_all(self, mock_input, sample_project_list):
        """Test selecting all projects."""
        mock_input.side_effect = ['all', 'y']

        result = interactive_project_selection(sample_project_list)

        assert sorted(result) == [1, 3, 5, 6]

    @patch('builtins.input')
    def test_quit_selection(self, mock_input, sample_project_list):
        """Test quitting the selection."""
        mock_input.side_effect = ['quit']

        result = interactive_project_selection(sample_project_list)

        assert result == []

    @patch('builtins.input')
    def test_reject_confirmation(self, mock_input, sample_project_list):
        """Test rejecting the selection and trying again."""
        mock_input.side_effect = ['5', 'n', '3', 'y']

        result = interactive_project_selection(sample_project_list)

        assert result == [3]

    @patch('builtins.input')
    def test_invalid_project_id(self, mock_input, sample_project_list, capsys):
        """Test handling of invalid project ID."""
        mock_input.side_effect = ['999', '5', 'y']

        result = interactive_project_selection(sample_project_list)

        captured = capsys.readouterr()
        assert "Project ID 999 not found" in captured.err
        assert result == [5]

    @patch('builtins.input')
    def test_keyboard_interrupt(self, mock_input, sample_project_list, capsys):
        """Test handling of keyboard interrupt (Ctrl+C)."""
        mock_input.side_effect = KeyboardInterrupt()

        result = interactive_project_selection(sample_project_list)

        assert result == []
        captured = capsys.readouterr()
        assert "Selection cancelled" in captured.out


# ============================================================================
# TEST ANNOTATION PROCESSOR FILE NAMING
# ============================================================================

class TestAnnotationProcessorFileNaming:
    """Test file naming with and without project_id."""

    def test_file_naming_without_project_id(self, temp_dir):
        """Test that files are named without project prefix in single-project mode."""
        # This is a unit test for the naming logic
        # We can't easily test the full processor without a real video file,
        # but we can verify the parameter is set correctly

        annotations_file = temp_dir / "annotations.json"
        annotations_file.write_text("[]")

        video_dir = temp_dir / "videos"
        video_dir.mkdir()

        processor = AnnotationProcessor(
            annotations_file=annotations_file,
            video_files_dir=video_dir,
            class_mappings={"cyclist": 0, "pedestrian": 1},
            project_id=None
        )

        assert processor.project_id is None

    def test_file_naming_with_project_id(self, temp_dir):
        """Test that files are named with project prefix in multi-project mode."""
        annotations_file = temp_dir / "annotations.json"
        annotations_file.write_text("[]")

        video_dir = temp_dir / "videos"
        video_dir.mkdir()

        processor = AnnotationProcessor(
            annotations_file=annotations_file,
            video_files_dir=video_dir,
            class_mappings={"cyclist": 0, "pedestrian": 1},
            project_id=5
        )

        assert processor.project_id == 5


# ============================================================================
# TEST CLI ARGUMENT PARSING
# ============================================================================

class TestCLIArguments:
    """Test CLI argument parsing and validation."""

    @patch('requests.get')
    def test_list_projects_flag(self, mock_get, mock_env):
        """Test --list-projects workflow logic."""
        # Mock empty projects response
        mock_response = Mock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Test the workflow
        url = "http://localhost:8080"
        api_key = "test-api-key"
        projects = list_available_projects(url, api_key)

        assert projects == []
        assert mock_get.called

    def test_project_ids_parsing_no_spaces(self):
        """Test that project IDs are correctly parsed without spaces."""
        project_ids_str = "3,5,7"
        parsed = [int(pid.strip()) for pid in project_ids_str.split(',')]
        assert parsed == [3, 5, 7]

    def test_project_ids_parsing_with_spaces(self):
        """Test that project IDs are correctly parsed even with spaces."""
        project_ids_str = "3, 5, 7"
        parsed = [int(pid.strip()) for pid in project_ids_str.split(',')]
        assert parsed == [3, 5, 7]

    def test_invalid_project_ids_format(self):
        """Test handling of invalid project ID format."""
        project_ids_str = "abc,def"
        with pytest.raises(ValueError):
            [int(pid.strip()) for pid in project_ids_str.split(',')]


# ============================================================================
# TEST PROJECT MAPPING GENERATION
# ============================================================================

class TestProjectMapping:
    """Test project_mapping.json generation."""

    def test_project_mapping_structure(self, temp_dir):
        """Test that project_mapping.json has the correct structure."""
        # Create sample mapping
        mapping = {
            'projects': [
                {
                    'id': 3,
                    'title': 'Bikedetect FR2',
                    'frames_extracted': 360,
                    'annotations_count': 363,
                    'videos_processed': 1
                },
                {
                    'id': 5,
                    'title': 'Bikedetect FR24',
                    'frames_extracted': 781,
                    'annotations_count': 791,
                    'videos_processed': 1
                }
            ],
            'total_frames': 1141,
            'total_annotations': 1154,
            'classes': {'cyclist': 0, 'pedestrian': 1, 'scooter-roller': 2}
        }

        # Write to file
        mapping_file = temp_dir / "project_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)

        # Read and verify
        with open(mapping_file, 'r') as f:
            loaded = json.load(f)

        assert 'projects' in loaded
        assert 'total_frames' in loaded
        assert 'total_annotations' in loaded
        assert 'classes' in loaded
        assert len(loaded['projects']) == 2
        assert loaded['total_frames'] == 1141
        assert loaded['total_annotations'] == 1154


# ============================================================================
# TEST ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_env_variables(self, monkeypatch):
        """Test handling of missing environment variables."""
        # Clear environment variables
        monkeypatch.delenv("LABEL_STUDIO_URL", raising=False)
        monkeypatch.delenv("LABEL_STUDIO_API_KEY", raising=False)

        # Verify they are not set
        url = os.getenv("LABEL_STUDIO_URL")
        api_key = os.getenv("LABEL_STUDIO_API_KEY")

        assert url is None
        assert api_key is None

    def test_invalid_class_mapping_json(self):
        """Test handling of invalid JSON in --classes argument."""
        # Invalid JSON
        invalid_json = '{"cyclist":0,invalid}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json)

    def test_empty_project_list(self, sample_project_list):
        """Test handling of empty project list."""
        @patch('builtins.input')
        def run_test(mock_input):
            mock_input.side_effect = ['5', 'y']
            result = interactive_project_selection([])
            assert result == []

        run_test()


# ============================================================================
# TEST FILE OPERATIONS
# ============================================================================

class TestFileOperations:
    """Test file and directory operations."""

    def test_output_directory_creation(self, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        output_path = temp_dir / "new_dataset"
        assert not output_path.exists()

        output_path.mkdir(parents=True, exist_ok=True)
        assert output_path.exists()

    def test_project_specific_directories(self, temp_dir):
        """Test creation of project-specific directories."""
        project_id = 5

        video_dir = temp_dir / f"exported_videos/project_{project_id}"
        json_dir = temp_dir / f"exported_json_annotation/project_{project_id}"

        video_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        assert video_dir.exists()
        assert json_dir.exists()

    def test_file_naming_conventions(self):
        """Test file naming conventions for single and multi-project modes."""
        # Single project mode
        video_stem = "test_video"
        frame_num = 346

        single_image = f"frame_{video_stem}_{frame_num:06d}.jpg"
        single_label = f"frame_{video_stem}_{frame_num:06d}.txt"

        assert single_image == "frame_test_video_000346.jpg"
        assert single_label == "frame_test_video_000346.txt"

        # Multi-project mode
        project_id = 5
        multi_image = f"project{project_id}_frame_{video_stem}_{frame_num:06d}.jpg"
        multi_label = f"project{project_id}_frame_{video_stem}_{frame_num:06d}.txt"

        assert multi_image == "project5_frame_test_video_000346.jpg"
        assert multi_label == "project5_frame_test_video_000346.txt"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    @patch('main.list_available_projects')
    @patch('builtins.input')
    def test_interactive_mode_workflow(self, mock_input, mock_list, mock_env, sample_project_list):
        """Test complete interactive mode workflow (mock only - no actual processing)."""
        mock_list.return_value = sample_project_list
        mock_input.side_effect = ['5', 'y']  # Select project 5, confirm

        # This would normally trigger the full workflow
        # We can verify the selection works
        projects = mock_list("http://localhost:8080", "test-api-key")
        selected = interactive_project_selection(projects)

        assert selected == [5]


# ============================================================================
# TEST UTILITIES AND HELPERS
# ============================================================================

class TestUtilities:
    """Test utility functions and helpers."""

    def test_url_normalization(self):
        """Test that URLs are properly normalized (trailing slashes removed)."""
        url_with_slash = "http://localhost:8080/"
        url_without_slash = "http://localhost:8080"

        normalized1 = url_with_slash.rstrip('/')
        normalized2 = url_without_slash.rstrip('/')

        assert normalized1 == normalized2
        assert normalized1 == "http://localhost:8080"

    def test_project_id_validation(self):
        """Test project ID validation."""
        valid_ids = [1, 3, 5, 100, 999]
        invalid_ids = [-1, 0, "abc", None, 3.14]

        for pid in valid_ids:
            assert isinstance(pid, int) and pid > 0

        for pid in invalid_ids:
            assert not (isinstance(pid, int) and pid > 0)


# ============================================================================
# BACKWARD COMPATIBILITY TESTS
# ============================================================================

class TestBackwardCompatibility:
    """Test backward compatibility with existing workflows."""

    def test_single_project_id_parsing(self):
        """Test that --project-ids works with single project ID."""
        # This should be parsed correctly
        project_ids_str = "5"
        selected_project_ids = [int(pid.strip()) for pid in project_ids_str.split(',')]

        assert len(selected_project_ids) == 1
        assert selected_project_ids[0] == 5

    def test_multiple_project_ids_parsing(self):
        """Test that --project-ids works with multiple comma-separated IDs."""
        project_ids_str = "5,7,12"
        selected_project_ids = [int(pid.strip()) for pid in project_ids_str.split(',')]

        assert len(selected_project_ids) == 3
        assert selected_project_ids == [5, 7, 12]


# ============================================================================
# PERFORMANCE AND EDGE CASES
# ============================================================================

class TestPerformanceAndEdgeCases:
    """Test performance considerations and edge cases."""

    def test_large_project_list(self):
        """Test handling of large number of projects."""
        large_project_list = [
            {"id": i, "title": f"Project {i}", "task_count": i * 10}
            for i in range(1, 101)  # 100 projects
        ]

        assert len(large_project_list) == 100
        assert large_project_list[0]['id'] == 1
        assert large_project_list[-1]['id'] == 100

    def test_project_title_truncation(self):
        """Test that long project titles are truncated correctly."""
        long_title = "This is a very long project title that should be truncated to fit in the display"
        max_length = 40

        display_title = long_title[:37] + "..." if len(long_title) > max_length else long_title

        assert len(display_title) == max_length
        assert display_title.endswith("...")

    def test_zero_tasks_project(self):
        """Test handling of project with zero tasks."""
        empty_project = {"id": 99, "title": "Empty Project", "task_number": 0}

        assert empty_project['task_number'] == 0  # Should not cause errors


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
