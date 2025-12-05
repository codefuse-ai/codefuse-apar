"""
Read Tracker - Tracks which files have been read in the current session

This module provides file read tracking for edit tool validation.
The edit_file tool requires files to be read before editing to prevent
accidental modifications to files the agent hasn't seen.
"""

from pathlib import Path
from typing import Set

from codefuse.observability import mainLogger


class ReadTracker:
    """
    File read tracker for edit tool validation
    
    Tracks which files have been read in the current session.
    Used by EditFileTool to ensure files are read before editing.
    """
    
    def __init__(self):
        """Initialize empty read tracker"""
        self._read_files: Set[str] = set()
    
    def mark_as_read(self, file_path: str) -> None:
        """
        Mark a file as having been read
        
        Args:
            file_path: Path to the file that was read
        """
        resolved_path = str(Path(file_path).resolve())
        self._read_files.add(resolved_path)
        mainLogger.debug("Marked file as read", file_path=resolved_path)
    
    def is_read(self, file_path: str) -> bool:
        """
        Check if a file has been read
        
        Args:
            file_path: Path to check
            
        Returns:
            True if the file has been read, False otherwise
        """
        resolved_path = str(Path(file_path).resolve())
        return resolved_path in self._read_files
    
    def clear(self) -> None:
        """
        Clear the read file tracking
        
        This can be used to reset the tracking state, for example
        when starting a new user query.
        """
        self._read_files.clear()
        mainLogger.debug("Cleared read file tracking")

