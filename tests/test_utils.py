"""
Test module for utility functions.
Author: Sharma Saurabh (250863131007)
Course: BE05000231 - Python for Data Science (GTU Semester-5)
"""
import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    print_section_header,
    print_subsection_header,
    display_dataframe_info,
    validate_dataframe,
    ensure_directory,
    get_file_size,
    get_timestamp,
    save_text_report,
    load_text_report,
    format_dict_for_display,
    create_project_summary,
)


class TestPrintSectionHeader:
    """Test cases for print_section_header function."""

    def test_print_section_header(self, capsys):
        """Test printing section header."""
        print_section_header("TEST HEADER")
        captured = capsys.readouterr()
        assert "TEST HEADER" in captured.out


class TestPrintSubsectionHeader:
    """Test cases for print_subsection_header function."""

    def test_print_subsection_header(self, capsys):
        """Test printing subsection header."""
        print_subsection_header("TEST SUBHEADER")
        captured = capsys.readouterr()
        assert "TEST SUBHEADER" in captured.out


class TestDisplayDataframeInfo:
    """Test cases for display_dataframe_info function."""

    def test_display_dataframe_info(self, capsys):
        """Test displaying DataFrame info."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        display_dataframe_info(df, "Test")
        captured = capsys.readouterr()
        assert "Test" in captured.out
        assert "2 rows" in captured.out


class TestValidateDataframe:
    """Test cases for validate_dataframe function."""

    def test_validate_valid_dataframe(self):
        """Test validating valid DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert validate_dataframe(df) is True

    def test_validate_empty_dataframe(self):
        """Test validating empty DataFrame."""
        assert validate_dataframe(pd.DataFrame()) is False

    def test_validate_none(self):
        """Test validating None."""
        assert validate_dataframe(None) is False

    def test_validate_missing_columns(self):
        """Test validating DataFrame with missing columns."""
        df = pd.DataFrame({"a": [1, 2]})
        assert validate_dataframe(df, ["a", "b"]) is False


class TestEnsureDirectory:
    """Test cases for ensure_directory function."""

    def test_create_directory(self, tmp_path):
        """Test creating directory."""
        dir_path = str(tmp_path / "test_dir")
        ensure_directory(dir_path)
        assert os.path.exists(dir_path)

    def test_existing_directory(self, tmp_path):
        """Test with existing directory."""
        dir_path = str(tmp_path / "existing_dir")
        os.makedirs(dir_path)
        ensure_directory(dir_path)  # Should not raise error
        assert os.path.exists(dir_path)


class TestGetFileSize:
    """Test cases for get_file_size function."""

    def test_get_file_size(self, tmp_path):
        """Test getting file size."""
        file_path = str(tmp_path / "test.txt")
        with open(file_path, "w") as f:
            f.write("test content")
        size = get_file_size(file_path)
        assert "B" in size or "KB" in size

    def test_nonexistent_file(self):
        """Test with non-existent file."""
        size = get_file_size("nonexistent.txt")
        assert "not found" in size.lower() or "error" in size.lower()


class TestGetTimestamp:
    """Test cases for get_timestamp function."""

    def test_get_timestamp(self):
        """Test getting timestamp."""
        timestamp = get_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0


class TestSaveTextReport:
    """Test cases for save_text_report function."""

    def test_save_text_report(self, tmp_path):
        """Test saving text report."""
        file_path = str(tmp_path / "report.txt")
        save_text_report("Test content", file_path)
        assert os.path.exists(file_path)

    def test_save_creates_directory(self, tmp_path):
        """Test that save creates directory."""
        file_path = str(tmp_path / "subdir" / "report.txt")
        save_text_report("Test content", file_path)
        assert os.path.exists(file_path)


class TestLoadTextReport:
    """Test cases for load_text_report function."""

    def test_load_text_report(self, tmp_path):
        """Test loading text report."""
        file_path = str(tmp_path / "report.txt")
        with open(file_path, "w") as f:
            f.write("Test content")
        content = load_text_report(file_path)
        assert content == "Test content"

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        content = load_text_report("nonexistent.txt")
        assert content is None


class TestFormatDictForDisplay:
    """Test cases for format_dict_for_display function."""

    def test_format_simple_dict(self):
        """Test formatting simple dictionary."""
        data = {"key1": "value1", "key2": "value2"}
        result = format_dict_for_display(data)
        assert "key1" in result
        assert "value1" in result

    def test_format_nested_dict(self):
        """Test formatting nested dictionary."""
        data = {"outer": {"inner": "value"}}
        result = format_dict_for_display(data)
        assert "outer" in result
        assert "inner" in result


class TestCreateProjectSummary:
    """Test cases for create_project_summary function."""

    def test_create_project_summary(self):
        """Test creating project summary."""
        summary = create_project_summary()
        assert isinstance(summary, dict)
        assert "Project Title" in summary
        assert "Student" in summary
        assert "Enrollment" in summary
        assert "Phases" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
