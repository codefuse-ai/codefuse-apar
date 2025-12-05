"""
Ripgrep utility - Find and execute ripgrep commands
"""

import subprocess
import shutil
import platform
import os
from typing import List, Optional
from pathlib import Path

from codefuse.observability import mainLogger


# Cache for ripgrep path
_ripgrep_path: Optional[str] = None
_ripgrep_type: Optional[str] = None  # 'system', 'python', or 'bundled'


def _get_bundled_ripgrep_path() -> Optional[Path]:
    """
    Get path to bundled ripgrep binary for current platform
    
    Directory structure: codefuse/tools/utils/ripgrep/{arch}-{platform}/rg
    Example: x64-darwin, arm64-darwin, x64-linux, arm64-linux, x64-win32
    
    Returns:
        Path to bundled ripgrep binary, or None if not available
    """
    try:
        # Detect platform
        system = platform.system().lower()  # 'linux', 'darwin', 'windows'
        machine = platform.machine().lower()  # 'x86_64', 'arm64', 'amd64', etc.
        
        # Normalize machine architecture
        if machine in ('amd64', 'x86_64', 'x64'):
            arch = 'x64'
        elif machine in ('arm64', 'aarch64'):
            arch = 'arm64'
        else:
            mainLogger.warning(f"Unsupported architecture for bundled ripgrep: {machine}")
            return None
        
        # Normalize platform name
        if system == 'darwin':
            platform_name = 'darwin'
        elif system == 'linux':
            platform_name = 'linux'
        elif system == 'windows':
            platform_name = 'win32'
        else:
            mainLogger.warning(f"Unsupported platform for bundled ripgrep: {system}")
            return None
        
        # Build directory name: {arch}-{platform}
        dir_name = f"{arch}-{platform_name}"
        
        # Get the package directory (where this file is located)
        utils_dir = Path(__file__).parent
        ripgrep_dir = utils_dir / 'ripgrep' / dir_name
        
        # Determine binary name
        binary_name = 'rg.exe' if system == 'windows' else 'rg'
        rg_binary = ripgrep_dir / binary_name
        
        # Check if binary exists
        if rg_binary.exists():
            # Make sure it's executable on Unix-like systems
            if system != 'windows':
                try:
                    os.chmod(rg_binary, 0o755)
                except Exception as e:
                    mainLogger.warning(f"Failed to set executable permission on {rg_binary}: {e}")
            
            mainLogger.info(f"Found bundled ripgrep at: {rg_binary}")
            return rg_binary
        else:
            mainLogger.debug(f"Bundled ripgrep not found at: {rg_binary}")
            return None
            
    except Exception as e:
        mainLogger.warning(f"Error locating bundled ripgrep: {e}")
        return None


def find_ripgrep() -> "tuple[Optional[str], Optional[str]]":
    """
    Find available ripgrep executable
    
    Priority:
    1. System-installed ripgrep (rg command)
    2. Python ripgrep-python package
    3. Bundled ripgrep binary (platform-specific)
    
    Returns:
        Tuple of (ripgrep_path, ripgrep_type) where type is 'system', 'python', or 'bundled'
        Returns (None, None) if ripgrep is not found
    """
    global _ripgrep_path, _ripgrep_type
    
    # Return cached result if available
    if _ripgrep_path is not None:
        return _ripgrep_path, _ripgrep_type
    
    # 1. Try system ripgrep
    rg_path = shutil.which('rg')
    if rg_path:
        _ripgrep_path = rg_path
        _ripgrep_type = 'system'
        mainLogger.info(f"Found system ripgrep at: {rg_path}")
        return _ripgrep_path, _ripgrep_type
    
    # 2. Try Python ripgrep-python package
    try:
        import ripgrep
        # ripgrep-python provides a 'rg' function or path
        if hasattr(ripgrep, 'rg'):
            _ripgrep_path = 'ripgrep-python'
            _ripgrep_type = 'python'
            mainLogger.info("Found Python ripgrep-python package")
            return _ripgrep_path, _ripgrep_type
    except ImportError:
        pass
    
    # 3. Try bundled ripgrep binary
    bundled_path = _get_bundled_ripgrep_path()
    if bundled_path:
        _ripgrep_path = str(bundled_path)
        _ripgrep_type = 'bundled'
        mainLogger.info(f"Using bundled ripgrep: {bundled_path}")
        return _ripgrep_path, _ripgrep_type
    
    # Not found
    mainLogger.warning(
        "Ripgrep not found. Please install ripgrep:\n"
        "  - macOS: brew install ripgrep\n"
        "  - Ubuntu/Debian: apt install ripgrep\n"
        "  - Or: pip install ripgrep-python"
    )
    return None, None


def execute_ripgrep(
    args: List[str],
    search_path: str,
    timeout: Optional[float] = 30.0
) -> List[str]:
    """
    Execute ripgrep command and return output lines
    
    Args:
        args: List of ripgrep arguments (without the 'rg' command itself)
        search_path: Path to search in
        timeout: Command timeout in seconds (default: 30.0)
    
    Returns:
        List of output lines (stdout)
        
    Raises:
        RuntimeError: If ripgrep is not found
        subprocess.TimeoutExpired: If command times out
        subprocess.CalledProcessError: If ripgrep returns non-zero exit code (except 1 for no matches)
    """
    rg_path, rg_type = find_ripgrep()
    
    if rg_path is None:
        raise RuntimeError(
            "Ripgrep is not available. Please install ripgrep:\n"
            "  - macOS: brew install ripgrep\n"
            "  - Ubuntu/Debian: apt install ripgrep\n"
            "  - Or: pip install ripgrep-python"
        )
    
    # Build command
    if rg_type == 'system':
        cmd = [rg_path] + args + ['--', search_path]
    elif rg_type == 'bundled':
        # Use the bundled binary path directly
        cmd = [rg_path] + args + ['--', search_path]
    elif rg_type == 'python':
        # For ripgrep-python, we still use subprocess but with 'rg' command
        # The package should have made 'rg' available
        cmd = ['rg'] + args + ['--', search_path]
    else:
        raise RuntimeError(f"Unknown ripgrep type: {rg_type}")
    
    mainLogger.debug(f"Executing ripgrep: {' '.join(cmd)}")
    
    try:
        # Run ripgrep command
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,  # We'll handle exit codes manually
        )
        
        # Exit code 0: matches found
        # Exit code 1: no matches found (not an error)
        # Exit code 2+: actual error
        if result.returncode == 0:
            # Matches found
            lines = result.stdout.splitlines()
            mainLogger.debug(f"Ripgrep found {len(lines)} result lines")
            return lines
        elif result.returncode == 1:
            # No matches found (not an error for ripgrep)
            mainLogger.debug("Ripgrep found no matches")
            return []
        else:
            # Actual error
            error_msg = result.stderr.strip() or f"Ripgrep exited with code {result.returncode}"
            mainLogger.error(f"Ripgrep error: {error_msg}")
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr
            )
    
    except subprocess.TimeoutExpired as e:
        mainLogger.error(f"Ripgrep command timed out after {timeout}s")
        raise
    except FileNotFoundError as e:
        # This shouldn't happen if find_ripgrep worked, but handle it
        raise RuntimeError(f"Ripgrep executable not found: {e}")

