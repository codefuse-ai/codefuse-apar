"""
Trajectory Writer - Records agent execution events to JSONL format
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any


class TrajectoryWriter:
    """
    Writes agent execution trajectory events to a JSONL file
    
    Each event is a JSON object on a separate line, enabling:
    - Streaming writes (append-only)
    - Real-time monitoring (tail -f)
    - Easy parsing (line-by-line)
    """
    
    def __init__(self, file_path: Path):
        """
        Initialize trajectory writer
        
        Args:
            file_path: Path to the trajectory JSONL file
        """
        self.file_path = Path(file_path)
        self._file_handle: Optional[Any] = None
        self._opened = False
    
    def _ensure_open(self):
        """Ensure file handle is open"""
        if not self._opened:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self.file_path, 'a', encoding='utf-8')
            self._opened = True
    
    def write(self, event_data: Dict[str, Any]):
        """
        Write a single event to the trajectory file
        
        Automatically adds timestamp if not present.
        
        Args:
            event_data: Event data dictionary
        """
        self._ensure_open()
        
        # Add timestamp if not present
        if 'timestamp' not in event_data:
            event_data['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Write as single line JSON
        json_line = json.dumps(event_data, ensure_ascii=False)
        self._file_handle.write(json_line + '\n')
        self._file_handle.flush()
    
    def write_summary(self, summary_data: Dict[str, Any]):
        """
        Write session summary event
        
        This is typically called at the end of a session.
        
        Args:
            summary_data: Summary data from MetricsCollector
        """
        event = {
            'event_type': 'session_summary',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **summary_data
        }
        self.write(event)
    
    def close(self):
        """Close the file handle"""
        if self._opened and self._file_handle:
            self._file_handle.close()
            self._opened = False
            self._file_handle = None
    
    def __enter__(self):
        """Context manager entry"""
        self._ensure_open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False

