"""
LLM Messages Writer - Records latest LLM messages snapshot
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any


class LLMMessagesWriter:
    """
    Writes the latest LLM messages snapshot to a JSON file
    
    This writer uses overwrite mode, keeping only the most recent state.
    Useful for debugging and inspecting the current conversation context.
    """
    
    def __init__(self, file_path: Path):
        """
        Initialize LLM messages writer
        
        Args:
            file_path: Path to the LLM messages JSON file
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def write(self, formatted_data: Dict[str, Any]):
        """
        Write LLM messages snapshot (overwrites existing file)
        
        Args:
            formatted_data: Formatted data containing messages and tools
                Expected keys: 'messages', 'tools', and optionally 'session_id'
        """
        # Build complete snapshot
        snapshot = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **formatted_data
        }
        
        # Write to file (overwrite mode)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
    
    def close(self):
        """Close method for consistency (no-op for this writer)"""
        pass

