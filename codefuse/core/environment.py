"""
Environment Information Collection
"""

import os
import platform
import sys
import subprocess
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from codefuse.observability import mainLogger


@dataclass
class EnvironmentInfo:
    """
    Information about the current environment
    
    This information is used to provide context to the agent about
    the system it's running on.
    """
    os_type: str  # "darwin", "linux", "windows"
    os_version: str
    python_version: str
    cwd: str
    git_branch: Optional[str] = None
    git_status: Optional[str] = None
    
    def to_context_string(self) -> str:
        """
        Convert environment info to a formatted string for system prompt
        
        Returns:
            Formatted string describing the environment
        """
        lines = [
            "# Environment Information",
            f"- OS: {self.os_type} {self.os_version}",
            f"- Python: {self.python_version}",
            f"- Working Directory: {self.cwd}",
        ]
        
        if self.git_branch:
            lines.append(f"- Git Branch: {self.git_branch}")
        
        if self.git_status:
            lines.append(f"- Git Status:\n{self.git_status}")
        
        return "\n".join(lines)
    
    @classmethod
    def collect(cls, cwd: Optional[str] = None) -> "EnvironmentInfo":
        """
        Collect current environment information
        
        Args:
            cwd: Working directory (defaults to current directory)
            
        Returns:
            EnvironmentInfo instance
        """
        if cwd is None:
            cwd = os.getcwd()
        
        cwd_path = Path(cwd).resolve()
        
        # Collect OS information
        os_type = platform.system().lower()
        os_version = platform.release()
        
        # Collect Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Try to collect Git information
        git_branch = cls._get_git_branch(cwd_path)
        git_status = cls._get_git_status(cwd_path)
        
        mainLogger.info(
            "Collected environment info",
            os_type=os_type,
            os_version=os_version,
            python_version=python_version,
        )
        
        return cls(
            os_type=os_type,
            os_version=os_version,
            python_version=python_version,
            cwd=str(cwd_path),
            git_branch=git_branch,
            git_status=git_status,
        )
    
    @staticmethod
    def _get_git_branch(cwd: Path) -> Optional[str]:
        """
        Get current git branch if in a git repository
        
        Args:
            cwd: Working directory
            
        Returns:
            Branch name or None
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            mainLogger.debug("Failed to get git branch", error=str(e))
        
        return None
    
    @staticmethod
    def _get_git_status(cwd: Path) -> Optional[str]:
        """
        Get git status if in a git repository
        
        Args:
            cwd: Working directory
            
        Returns:
            Git status output or None
        """
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                status = result.stdout.strip()
                if status:
                    return status
                else:
                    return "Clean (no changes)"
        except Exception as e:
            mainLogger.debug("Failed to get git status", error=str(e))
        
        return None
    
    @staticmethod
    def _get_git_diff_stats(cwd: Path) -> Optional[dict]:
        """
        Get git diff statistics using git add -A and git diff --cached --numstat
        
        This captures all changes including untracked files.
        
        Args:
            cwd: Working directory
            
        Returns:
            Dict with stats and file-level changes, or None
        """
        try:
            # Stage all changes
            add_result = subprocess.run(
                ["git", "add", "-A"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if add_result.returncode != 0:
                mainLogger.debug("Failed to stage changes", error=add_result.stderr)
                return None
            
            # Get numstat for staged changes
            numstat_result = subprocess.run(
                ["git", "diff", "--cached", "--numstat"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if numstat_result.returncode != 0:
                return None
            
            numstat_output = numstat_result.stdout.strip()
            if not numstat_output:
                return None
            
            # Parse numstat output
            files = []
            total_insertions = 0
            total_deletions = 0
            
            for line in numstat_output.split('\n'):
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 3:
                    insertions = int(parts[0]) if parts[0] != '-' else 0
                    deletions = int(parts[1]) if parts[1] != '-' else 0
                    path = parts[2]
                    
                    files.append({
                        "path": path,
                        "insertions": insertions,
                        "deletions": deletions,
                    })
                    
                    total_insertions += insertions
                    total_deletions += deletions
            
            return {
                "stats": {
                    "files_changed": len(files),
                    "insertions": total_insertions,
                    "deletions": total_deletions,
                },
                "files": files,
            }
        except Exception as e:
            mainLogger.debug("Failed to get git diff stats", error=str(e))
            return None
    
    @staticmethod
    def _get_git_diff_text(cwd: Path) -> Optional[str]:
        """
        Get full git diff text for staged changes
        
        Note: This assumes git add -A has already been called by _get_git_diff_stats()
        
        Args:
            cwd: Working directory
            
        Returns:
            Full diff text or None
        """
        try:
            # Get diff for staged changes
            diff_result = subprocess.run(
                ["git", "diff", "--cached"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if diff_result.returncode == 0 and diff_result.stdout.strip():
                return diff_result.stdout.strip()
            
            return None
        except Exception as e:
            mainLogger.debug("Failed to get git diff text", error=str(e))
            return None
    
    @staticmethod
    def get_git_diff_info(cwd: Optional[str] = None) -> Optional[dict]:
        """
        Get complete git diff information using git add -A and git diff --cached
        
        Note: This will stage all changes in the repository.
        
        Args:
            cwd: Working directory (defaults to current directory)
            
        Returns:
            Dictionary with stats, file list, and full diff text, or None if not a git repo
            {
                "stats": {
                    "files_changed": 3,
                    "insertions": 45,
                    "deletions": 12
                },
                "files": [
                    {
                        "path": "file.py",
                        "insertions": 30,
                        "deletions": 5
                    }
                ],
                "diff_text": "diff --git ..."
            }
        """
        if cwd is None:
            cwd = os.getcwd()
        
        cwd_path = Path(cwd).resolve()
        
        # Get stats (this will also run git add -A)
        diff_info = EnvironmentInfo._get_git_diff_stats(cwd_path)
        if diff_info is None:
            return None
        
        # Get full diff text (uses git diff --cached)
        diff_text = EnvironmentInfo._get_git_diff_text(cwd_path)
        if diff_text:
            diff_info["diff_text"] = diff_text
        
        return diff_info

