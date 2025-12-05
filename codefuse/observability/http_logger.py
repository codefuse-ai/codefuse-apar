"""
HTTP Server Log Management - File-based logging with rotation and cleanup

Features:
- Dual format logging: text (access.log) and JSON (access-YYYYMMDD.json)
- Daily log rotation
- Automatic cleanup of old logs (default: 7 days retention)
- Thread-safe for Gunicorn multi-worker setup
"""

import json
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import atexit


class HTTPLogger:
    """Thread-safe HTTP request logger with rotation and cleanup"""
    
    def __init__(self, log_dir: str, retention_days: int = 7, cleanup_interval: int = 3600):
        """
        Initialize HTTP logger
        
        Args:
            log_dir: Base directory for log files
            retention_days: Number of days to retain logs (default: 7)
            cleanup_interval: Cleanup check interval in seconds (default: 3600)
        """
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.retention_days = retention_days
        self.cleanup_interval = cleanup_interval
        
        # File paths
        self.access_log_path = self.log_dir / "access.log"
        self.error_log_path = self.log_dir / "error.log"
        
        # Thread safety
        self._write_lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        # Current date for rotation check
        self._current_date = datetime.now().date()
        
        # Register cleanup on exit
        atexit.register(self.stop_cleanup_thread)
    
    def _get_json_log_path(self, date: Optional[datetime] = None) -> Path:
        """Get JSON log file path for a specific date"""
        if date is None:
            date = datetime.now()
        date_str = date.strftime("%Y%m%d")
        return self.log_dir / f"access-{date_str}.json"
    
    def _format_text_log(
        self,
        request_id: str,
        method: str,
        path: str,
        status: int,
        duration: float,
        tool_name: Optional[str] = None,
        workdir: Optional[str] = None,
    ) -> str:
        """Format log entry as human-readable text"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tool_info = f" | tool:{tool_name}" if tool_name else ""
        workdir_info = f" | wd:{workdir}" if workdir else ""
        return f"{timestamp} | {method} {path} | {status} | {duration:.3f}s | {request_id}{tool_info}{workdir_info}\n"
    
    def _format_json_log(
        self,
        request_id: str,
        method: str,
        path: str,
        status: int,
        duration: float,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        workdir: Optional[str] = None,
        success: Optional[bool] = None,
        error: Optional[str] = None,
    ) -> str:
        """Format log entry as JSON"""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "method": method,
            "path": path,
            "status": status,
            "duration": round(duration, 3),
        }
        
        if tool_name:
            log_data["tool_name"] = tool_name
        if tool_args:
            log_data["tool_args"] = tool_args
        if workdir:
            log_data["workdir"] = workdir
        if success is not None:
            log_data["success"] = success
        if error:
            log_data["error"] = error
        
        return json.dumps(log_data, ensure_ascii=False) + "\n"
    
    def log_request(
        self,
        request_id: str,
        method: str,
        path: str,
        status: int,
        duration: float,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        workdir: Optional[str] = None,
        success: Optional[bool] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Log HTTP request to both text and JSON files
        
        Thread-safe for concurrent writes from multiple workers.
        """
        with self._write_lock:
            try:
                # Check if date has changed (rotation needed)
                current_date = datetime.now().date()
                if current_date != self._current_date:
                    self._current_date = current_date
                
                # Write text log
                text_entry = self._format_text_log(
                    request_id, method, path, status, duration, tool_name, workdir
                )
                with open(self.access_log_path, 'a', encoding='utf-8') as f:
                    f.write(text_entry)
                
                # Write JSON log
                json_entry = self._format_json_log(
                    request_id, method, path, status, duration,
                    tool_name, tool_args, workdir, success, error
                )
                json_log_path = self._get_json_log_path()
                with open(json_log_path, 'a', encoding='utf-8') as f:
                    f.write(json_entry)
                
            except Exception as e:
                # Avoid blocking the request if logging fails
                print(f"[HTTPLogger] Failed to write log: {e}", flush=True)
    
    def log_error(
        self,
        request_id: str,
        error: str,
        traceback: Optional[str] = None,
        method: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        """Log error to error log file"""
        with self._write_lock:
            try:
                error_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "request_id": request_id,
                    "error": error,
                }
                
                if method:
                    error_data["method"] = method
                if path:
                    error_data["path"] = path
                if traceback:
                    error_data["traceback"] = traceback
                
                error_entry = json.dumps(error_data, ensure_ascii=False) + "\n"
                with open(self.error_log_path, 'a', encoding='utf-8') as f:
                    f.write(error_entry)
                
            except Exception as e:
                print(f"[HTTPLogger] Failed to write error log: {e}", flush=True)
    
    def _cleanup_old_logs(self) -> None:
        """Delete log files older than retention_days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            # Find and delete old JSON log files
            pattern = "access-*.json"
            for log_file in self.log_dir.glob(pattern):
                try:
                    # Extract date from filename: access-20251120.json
                    date_str = log_file.stem.split('-', 1)[1]  # "20251120"
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    
                    if file_date < cutoff_date:
                        log_file.unlink()
                        print(f"[HTTPLogger] Deleted old log: {log_file.name}", flush=True)
                
                except (ValueError, IndexError) as e:
                    # Skip files with invalid date format
                    print(f"[HTTPLogger] Skipping invalid log file: {log_file.name} ({e})", flush=True)
        
        except Exception as e:
            print(f"[HTTPLogger] Cleanup failed: {e}", flush=True)
    
    def _cleanup_worker(self) -> None:
        """Background thread worker for periodic cleanup"""
        print(f"[HTTPLogger] Cleanup thread started (interval: {self.cleanup_interval}s, retention: {self.retention_days} days)", flush=True)
        
        while not self._stop_cleanup.wait(timeout=self.cleanup_interval):
            self._cleanup_old_logs()
        
        print("[HTTPLogger] Cleanup thread stopped", flush=True)
    
    def start_cleanup_thread(self) -> None:
        """Start background cleanup thread"""
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            print("[HTTPLogger] Cleanup thread already running", flush=True)
            return
        
        # Run initial cleanup
        self._cleanup_old_logs()
        
        # Start background thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="HTTPLoggerCleanup"
        )
        self._cleanup_thread.start()
    
    def stop_cleanup_thread(self) -> None:
        """Stop background cleanup thread gracefully"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            return
        
        print("[HTTPLogger] Stopping cleanup thread...", flush=True)
        self._stop_cleanup.set()
        self._cleanup_thread.join(timeout=5)


def create_http_logger(
    log_dir: Optional[str] = None,
    retention_days: Optional[int] = None,
    cleanup_interval: Optional[int] = None,
) -> HTTPLogger:
    """
    Create and configure HTTP logger from environment variables
    
    Environment variables:
        CFUSE_HTTP_LOG_DIR: Log directory (default: ~/.cfuse/logs/http_server)
        CFUSE_HTTP_LOG_RETENTION_DAYS: Retention in days (default: 7)
        CFUSE_HTTP_LOG_CLEANUP_INTERVAL: Cleanup interval in seconds (default: 3600)
    
    Args:
        log_dir: Override log directory
        retention_days: Override retention days
        cleanup_interval: Override cleanup interval
    
    Returns:
        Configured HTTPLogger instance
    """
    if log_dir is None:
        log_dir = os.getenv("CFUSE_HTTP_LOG_DIR", "~/.cfuse/logs/http_server")
    
    if retention_days is None:
        retention_days = int(os.getenv("CFUSE_HTTP_LOG_RETENTION_DAYS", "7"))
    
    if cleanup_interval is None:
        cleanup_interval = int(os.getenv("CFUSE_HTTP_LOG_CLEANUP_INTERVAL", "3600"))
    
    return HTTPLogger(log_dir, retention_days, cleanup_interval)

