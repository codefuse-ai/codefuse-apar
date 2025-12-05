"""
Production HTTP Server - Flask-based tool execution service

Features:
- Flask + Gunicorn for production stability
- Automatic BrokenPipeError handling
- Request ID tracking
- File-based logging with rotation and cleanup
- Log-based metrics endpoint
- Health check endpoint
- Graceful shutdown support

Error Handling Strategy:
- 400: Only for basic parameter parsing errors (invalid JSON, missing required fields)
- 200 with success=false: All other errors (tool not found, workspace doesn't exist, execution failures)
"""

import json
import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List

from flask import Flask, request, jsonify, Response
from werkzeug.exceptions import BadRequest
from rich.console import Console

from codefuse.config import Config
from codefuse.tools.registry import create_default_registry
from codefuse.observability import mainLogger
from codefuse.observability.http_logger import HTTPLogger, create_http_logger

console = Console()

# Global config and logger (set by run_http_server)
_global_config: Optional[Config] = None
_http_logger: Optional[HTTPLogger] = None


def create_app() -> Flask:
    """Create and configure Flask application"""
    # Set template folder to codefuse/cli/templates
    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    
    # Disable Flask's default logger in favor of structlog
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    @app.before_request
    def before_request():
        """Add request ID and start time to request context"""
        request.request_id = str(uuid.uuid4())
        request.start_time = time.time()
    
    @app.after_request
    def after_request(response: Response) -> Response:
        """Log request completion to file"""
        duration = time.time() - request.start_time
        
        # Extract tool info from request context if available
        tool_name = getattr(request, 'tool_name', None)
        tool_args = getattr(request, 'tool_args', None)
        workdir = getattr(request, 'workdir', None)
        tool_success = getattr(request, 'tool_success', None)
        tool_error = getattr(request, 'tool_error', None)
        
        # Write to log files
        if _http_logger:
            _http_logger.log_request(
                request_id=request.request_id,
                method=request.method,
                path=request.path,
                status=response.status_code,
                duration=duration,
                tool_name=tool_name,
                tool_args=tool_args,
                workdir=workdir,
                success=tool_success,
                error=tool_error,
            )
        
        # Add request ID to response headers
        response.headers['X-Request-ID'] = request.request_id
        
        return response
    
    @app.errorhandler(Exception)
    def handle_exception(e: Exception) -> tuple:
        """Global exception handler"""
        request_id = getattr(request, 'request_id', 'unknown')
        
        # Handle BrokenPipeError and ConnectionError silently
        if isinstance(e, (BrokenPipeError, ConnectionError, ConnectionResetError)):
            if _http_logger:
                _http_logger.log_error(
                    request_id=request_id,
                    error=f"Client disconnected: {str(e)}",
                    method=getattr(request, 'method', None),
                    path=getattr(request, 'path', None),
                )
            # Return empty response, but it won't reach client anyway
            return '', 500
        
        # Log unexpected errors
        import traceback
        error_traceback = traceback.format_exc()
        
        if _http_logger:
            _http_logger.log_error(
                request_id=request_id,
                error=str(e),
                traceback=error_traceback,
                method=getattr(request, 'method', None),
                path=getattr(request, 'path', None),
            )
        
        # Return generic error response
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint for load balancers"""
        return jsonify({
            "status": "healthy",
            "service": "codefuse-http-server"
        }), 200
    
    @app.route('/metrics', methods=['GET'])
    def metrics():
        """Log-based metrics endpoint (Prometheus format)"""
        try:
            stats = _compute_metrics_from_logs()
            metrics_text = _format_prometheus_metrics(stats)
            return Response(metrics_text, mimetype='text/plain; version=0.0.4; charset=utf-8')
        except Exception as e:
            console.print(f"[red]Error computing metrics:[/red] {e}")
            return Response(f"# Error computing metrics: {e}\n", mimetype='text/plain'), 500
    
    @app.route('/api/metrics', methods=['GET'])
    def api_metrics():
        """API metrics endpoint (JSON format for dashboard)"""
        try:
            # Get days parameter from query string (default: 2)
            days = request.args.get('days', default=2, type=int)
            days = max(1, min(days, 30))  # Limit to 1-30 days
            
            metrics_data = _compute_dashboard_metrics(days=days)
            return jsonify(metrics_data), 200
        except Exception as e:
            console.print(f"[red]Error computing dashboard metrics:[/red] {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "error": f"Failed to compute metrics: {str(e)}"
            }), 500
    
    @app.route('/dashboard', methods=['GET'])
    def dashboard():
        """Dashboard endpoint - HTML visualization"""
        try:
            from flask import render_template
            
            # Debug: print template folder info
            template_dir = Path(__file__).parent / "templates"
            template_file = template_dir / "dashboard.html"
            
            mainLogger.info(
                "Dashboard access",
                template_dir=str(template_dir),
                template_exists=template_file.exists(),
                template_dir_exists=template_dir.exists()
            )
            
            if not template_file.exists():
                error_msg = f"Template file not found: {template_file}"
                mainLogger.error(error_msg)
                return f"<html><body><h1>Error</h1><p>{error_msg}</p><p>Template dir: {template_dir}</p><p>Dir exists: {template_dir.exists()}</p></body></html>", 500
            
            return render_template('dashboard.html')
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            
            mainLogger.error(
                "Error rendering dashboard",
                error=str(e),
                traceback=error_trace,
                exc_info=True
            )
            console.print(f"[red]Error rendering dashboard:[/red] {e}")
            console.print(error_trace)
            
            return f"<html><body><h1>Error</h1><p>Failed to render dashboard: {str(e)}</p><pre>{error_trace}</pre></body></html>", 500
    
    @app.route('/', methods=['OPTIONS'])
    def handle_options():
        """Handle CORS preflight requests"""
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response, 200
    
    @app.route('/', methods=['POST'])
    @app.route('/execute', methods=['POST'])
    def execute_tool():
        """Execute tool endpoint - main API"""
        try:
            # Parse request body
            try:
                request_data = request.get_json(force=True)
            except BadRequest as e:
                mainLogger.warning(
                    "Invalid JSON in request",
                    request_id=request.request_id,
                    error=str(e)
                )
                return jsonify({
                    "success": False,
                    "error": f"Invalid JSON: {str(e)}"
                }), 400
            
            # Validate request data
            if not isinstance(request_data, dict):
                return jsonify({
                    "success": False,
                    "error": "Request must be a JSON object"
                }), 400
            
            workdir = request_data.get("workdir")
            tool_name = request_data.get("toolName")
            tool_args = request_data.get("toolArgs")
            
            if not workdir:
                return jsonify({
                    "success": False,
                    "error": "Missing required field: workdir"
                }), 400
            
            if not tool_name:
                return jsonify({
                    "success": False,
                    "error": "Missing required field: toolName"
                }), 400
            
            if not isinstance(tool_args, dict):
                return jsonify({
                    "success": False,
                    "error": "toolArgs must be a JSON object"
                }), 400
            
            # Store tool info in request context for logging
            request.tool_name = tool_name
            request.tool_args = tool_args
            request.workdir = workdir
            
            # Execute tool
            result = _execute_tool(workdir, tool_name, tool_args)
            
            # Store execution result for logging
            request.tool_success = result["data"]["response"].get("success")
            if not request.tool_success:
                request.tool_error = result["data"]["response"].get("result", "Unknown error")
            
            # Add CORS headers
            response = jsonify(result["data"])
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response, result["status_code"]
        
        except Exception as e:
            mainLogger.error(
                "Tool execution request failed",
                request_id=request.request_id,
                error=str(e),
                exc_info=True
            )
            return jsonify({
                "success": False,
                "error": f"Internal server error: {str(e)}"
            }), 500
    
    return app


def _execute_tool(
    workdir: str,
    tool_name: str,
    tool_args: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute a tool and return result
    
    Returns:
        Dictionary with 'status_code' and 'data' keys
    """
    start_time = time.time()
    success = False
    
    try:
        # Resolve workspace path
        workspace_path = Path(workdir).expanduser().resolve()
        if not workspace_path.exists():
            return {
                "status_code": 200,
                "data": {
                    "response": {
                        "success": False,
                        "result": f"Error: Workspace directory does not exist: {workdir}",
                        "toolName": tool_name,
                        "toolArgs": tool_args,
                    }
                }
            }
        
        # Initialize tool registry
        tool_registry = create_default_registry(
            workspace_root=workspace_path,
            read_tracker=None,
            config=_global_config,
        )
        
        # Check if tool exists
        tool = tool_registry.get_tool(tool_name)
        if tool is None:
            available_tools = tool_registry.list_tool_names()
            return {
                "status_code": 200,
                "data": {
                    "response": {
                        "success": False,
                        "result": f"Error: Tool not found: {tool_name}. Available tools: {', '.join(available_tools)}",
                        "availableTools": available_tools,
                        "toolName": tool_name,
                        "toolArgs": tool_args,
                    }
                }
            }
        
        # Execute the tool
        tool_result = tool.execute(**tool_args)
        
        # Handle both ToolResult and string returns
        if isinstance(tool_result, str):
            result_str = tool_result
        else:
            result_str = tool_result.content
        
        # Check if tool execution was successful
        is_error = result_str.startswith("Error:")
        
        if is_error:
            error_msg = result_str.replace("Error: ", "", 1)
            return {
                "status_code": 200,
                "data": {
                    "response": {
                        "success": False,
                        "result": f"Error: {error_msg}",
                        "toolName": tool_name,
                        "toolArgs": tool_args,
                    }
                }
            }
        else:
            success = True
            return {
                "status_code": 200,
                "data": {
                    "response": {
                        "success": True,
                        "result": result_str,
                        "toolName": tool_name,
                        "toolArgs": tool_args,
                    }
                }
            }
    
    except Exception as e:
        return {
            "status_code": 200,
            "data": {
                "response": {
                    "success": False,
                    "result": f"Error: {str(e)}",
                    "toolName": tool_name,
                    "toolArgs": tool_args,
                }
            }
        }


def _compute_metrics_from_logs(days: int = 2) -> Dict[str, Any]:
    """
    Compute metrics by parsing JSON log files
    
    Args:
        days: Number of days to include (default: 2 for today and yesterday)
    
    Returns:
        Dictionary with computed metrics
    """
    if not _http_logger:
        return {
            "http_requests_total": {},
            "tool_executions_total": {},
            "request_durations": [],
            "tool_durations": {},
        }
    
    log_dir = _http_logger.log_dir
    
    # Collect stats
    http_requests = defaultdict(int)  # (method, path, status) -> count
    tool_executions = defaultdict(int)  # (tool_name, status) -> count
    request_durations = []  # All request durations
    tool_durations = defaultdict(list)  # tool_name -> [durations]
    
    # Parse logs from recent days
    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        log_file = _http_logger._get_json_log_path(date)
        
        if not log_file.exists():
            continue
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        # HTTP request metrics
                        method = entry.get('method', 'UNKNOWN')
                        path = entry.get('path', 'UNKNOWN')
                        status = entry.get('status', 0)
                        duration = entry.get('duration', 0)
                        
                        http_requests[(method, path, status)] += 1
                        request_durations.append(duration)
                        
                        # Tool execution metrics
                        tool_name = entry.get('tool_name')
                        if tool_name:
                            success = entry.get('success', False)
                            status_label = 'success' if success else 'error'
                            tool_executions[(tool_name, status_label)] += 1
                            tool_durations[tool_name].append(duration)
                    
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to read {log_file.name}: {e}")
    
    return {
        "http_requests_total": dict(http_requests),
        "tool_executions_total": dict(tool_executions),
        "request_durations": request_durations,
        "tool_durations": {k: list(v) for k, v in tool_durations.items()},
    }


def _compute_dashboard_metrics(days: int = 2) -> Dict[str, Any]:
    """
    Compute detailed metrics for dashboard visualization
    
    Args:
        days: Number of days to include (default: 2)
    
    Returns:
        Dictionary with detailed metrics for dashboard
    """
    if not _http_logger:
        return {
            "overview": {
                "total_requests": 0,
                "success_rate": 0,
                "tool_success_rate": 0,
                "avg_response_time": 0,
            },
            "requests": {
                "by_status": {},
                "by_endpoint": {},
                "timeline": [],
            },
            "tools": {
                "by_name": {},
                "success_rate_by_tool": {},
                "avg_duration_by_tool": {},
            },
            "performance": {
                "percentiles": {},
                "duration_histogram": [],
            },
            "errors": [],
        }
    
    log_dir = _http_logger.log_dir
    
    # Collect detailed stats - ONLY for tool calls
    all_tool_entries = []  # Only entries with tool_name
    http_by_status = defaultdict(int)
    http_by_endpoint = defaultdict(int)
    tool_by_name = defaultdict(lambda: {"total": 0, "success": 0, "failed": 0, "durations": []})
    request_durations = []
    errors = []
    
    # Parse logs from recent days
    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        log_file = _http_logger._get_json_log_path(date)
        
        if not log_file.exists():
            continue
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        # Only process entries with tool_name (actual tool calls)
                        tool_name = entry.get('tool_name')
                        if not tool_name:
                            continue
                        
                        # This is a tool call, add it to our stats
                        all_tool_entries.append(entry)
                        
                        # HTTP request metrics (only for tool calls)
                        status = entry.get('status', 0)
                        path = entry.get('path', 'UNKNOWN')
                        duration = entry.get('duration', 0)
                        
                        http_by_status[status] += 1
                        http_by_endpoint[path] += 1
                        request_durations.append(duration)
                        
                        # Tool execution metrics
                        success = entry.get('success', False)
                        tool_by_name[tool_name]["total"] += 1
                        if success:
                            tool_by_name[tool_name]["success"] += 1
                        else:
                            tool_by_name[tool_name]["failed"] += 1
                        tool_by_name[tool_name]["durations"].append(duration)
                        
                        # Collect errors
                        if not success:
                            error_msg = entry.get('error', 'Unknown error')
                            errors.append({
                                "timestamp": entry.get('timestamp'),
                                "tool_name": tool_name,
                                "error": error_msg[:100],  # Truncate long errors
                            })
                    
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to read {log_file.name}: {e}")
    
    # Compute overview metrics
    total_requests = sum(http_by_status.values())
    success_requests = http_by_status.get(200, 0)
    success_rate = (success_requests / total_requests * 100) if total_requests > 0 else 0
    
    total_tool_calls = sum(t["total"] for t in tool_by_name.values())
    successful_tool_calls = sum(t["success"] for t in tool_by_name.values())
    tool_success_rate = (successful_tool_calls / total_tool_calls * 100) if total_tool_calls > 0 else 0
    
    avg_response_time = (sum(request_durations) / len(request_durations)) if request_durations else 0
    
    # Compute percentiles
    sorted_durations = sorted(request_durations)
    percentiles = {}
    if sorted_durations:
        count = len(sorted_durations)
        percentiles = {
            "p50": sorted_durations[int(count * 0.5)],
            "p95": sorted_durations[int(count * 0.95)],
            "p99": sorted_durations[int(count * 0.99)],
            "min": sorted_durations[0],
            "max": sorted_durations[-1],
        }
    
    # Create duration histogram (bins: 0-0.1s, 0.1-0.5s, 0.5-1s, 1-2s, 2-5s, 5s+)
    histogram_bins = [0, 0.1, 0.5, 1, 2, 5, float('inf')]
    histogram_labels = ["<100ms", "100ms-500ms", "500ms-1s", "1s-2s", "2s-5s", ">5s"]
    histogram_counts = [0] * len(histogram_labels)
    
    for duration in request_durations:
        for i in range(len(histogram_bins) - 1):
            if histogram_bins[i] <= duration < histogram_bins[i + 1]:
                histogram_counts[i] += 1
                break
    
    duration_histogram = [
        {"label": label, "count": count}
        for label, count in zip(histogram_labels, histogram_counts)
    ]
    
    # Compute per-tool metrics
    success_rate_by_tool = {}
    avg_duration_by_tool = {}
    for tool_name, stats in tool_by_name.items():
        total = stats["total"]
        success = stats["success"]
        success_rate_by_tool[tool_name] = (success / total * 100) if total > 0 else 0
        
        durations = stats["durations"]
        avg_duration_by_tool[tool_name] = (sum(durations) / len(durations)) if durations else 0
    
    # Create timeline (group by minute) - only for tool calls
    timeline = defaultdict(lambda: {"timestamp": "", "total": 0, "success": 0, "failed": 0})
    for entry in all_tool_entries:
        timestamp_str = entry.get('timestamp', '')
        if timestamp_str:
            try:
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                # Group by minute
                minute_key = dt.strftime('%Y-%m-%d %H:%M')
                timeline[minute_key]["timestamp"] = minute_key
                timeline[minute_key]["total"] += 1
                
                status = entry.get('status', 0)
                if status == 200:
                    timeline[minute_key]["success"] += 1
                else:
                    timeline[minute_key]["failed"] += 1
            except:
                pass
    
    # Sort timeline by timestamp
    timeline_data = sorted(timeline.values(), key=lambda x: x["timestamp"])
    
    return {
        "overview": {
            "total_requests": total_requests,
            "success_rate": round(success_rate, 2),
            "tool_success_rate": round(tool_success_rate, 2),
            "avg_response_time": round(avg_response_time, 3),
        },
        "requests": {
            "by_status": dict(http_by_status),
            "by_endpoint": dict(http_by_endpoint),
            "timeline": timeline_data,
        },
        "tools": {
            "by_name": {
                name: {
                    "total": stats["total"],
                    "success": stats["success"],
                    "failed": stats["failed"],
                }
                for name, stats in tool_by_name.items()
            },
            "success_rate_by_tool": {k: round(v, 2) for k, v in success_rate_by_tool.items()},
            "avg_duration_by_tool": {k: round(v, 3) for k, v in avg_duration_by_tool.items()},
        },
        "performance": {
            "percentiles": {k: round(v, 3) for k, v in percentiles.items()},
            "duration_histogram": duration_histogram,
        },
        "errors": {
            "total": len(errors),
            "items": list(reversed(errors)),  # Most recent first
        },
    }


def _format_prometheus_metrics(stats: Dict[str, Any]) -> str:
    """Format metrics as Prometheus text format"""
    lines = []
    
    # HTTP requests total
    lines.append("# HELP http_requests_total Total HTTP requests")
    lines.append("# TYPE http_requests_total counter")
    for (method, path, status), count in stats["http_requests_total"].items():
        lines.append(f'http_requests_total{{method="{method}",endpoint="{path}",status="{status}"}} {count}')
    
    # Tool executions total
    lines.append("")
    lines.append("# HELP tool_executions_total Total tool executions")
    lines.append("# TYPE tool_executions_total counter")
    for (tool_name, status), count in stats["tool_executions_total"].items():
        lines.append(f'tool_executions_total{{tool_name="{tool_name}",status="{status}"}} {count}')
    
    # Request duration percentiles
    durations = sorted(stats["request_durations"])
    if durations:
        lines.append("")
        lines.append("# HELP http_request_duration_seconds HTTP request duration in seconds")
        lines.append("# TYPE http_request_duration_seconds summary")
        
        # Calculate percentiles
        count = len(durations)
        total = sum(durations)
        p50 = durations[int(count * 0.5)] if count > 0 else 0
        p95 = durations[int(count * 0.95)] if count > 0 else 0
        p99 = durations[int(count * 0.99)] if count > 0 else 0
        
        lines.append(f'http_request_duration_seconds{{quantile="0.5"}} {p50:.3f}')
        lines.append(f'http_request_duration_seconds{{quantile="0.95"}} {p95:.3f}')
        lines.append(f'http_request_duration_seconds{{quantile="0.99"}} {p99:.3f}')
        lines.append(f'http_request_duration_seconds_sum {total:.3f}')
        lines.append(f'http_request_duration_seconds_count {count}')
    
    # Tool duration percentiles
    lines.append("")
    lines.append("# HELP tool_execution_duration_seconds Tool execution duration in seconds")
    lines.append("# TYPE tool_execution_duration_seconds summary")
    for tool_name, durations in stats["tool_durations"].items():
        if durations:
            sorted_durations = sorted(durations)
            count = len(sorted_durations)
            total = sum(sorted_durations)
            p50 = sorted_durations[int(count * 0.5)] if count > 0 else 0
            p95 = sorted_durations[int(count * 0.95)] if count > 0 else 0
            p99 = sorted_durations[int(count * 0.99)] if count > 0 else 0
            
            lines.append(f'tool_execution_duration_seconds{{tool_name="{tool_name}",quantile="0.5"}} {p50:.3f}')
            lines.append(f'tool_execution_duration_seconds{{tool_name="{tool_name}",quantile="0.95"}} {p95:.3f}')
            lines.append(f'tool_execution_duration_seconds{{tool_name="{tool_name}",quantile="0.99"}} {p99:.3f}')
            lines.append(f'tool_execution_duration_seconds_sum{{tool_name="{tool_name}"}} {total:.3f}')
            lines.append(f'tool_execution_duration_seconds_count{{tool_name="{tool_name}"}} {count}')
    
    return '\n'.join(lines) + '\n'


def run_http_server(config: Config, host: str, port: int) -> None:
    """
    Run HTTP server using Gunicorn for production
    
    Args:
        config: Configuration object
        host: Host address to bind to
        port: Port to listen on
    """
    global _global_config, _http_logger
    _global_config = config
    
    # Initialize HTTP logger
    _http_logger = create_http_logger()
    _http_logger.start_cleanup_thread()
    
    console.print(f"[cyan]Logs directory:[/cyan] {_http_logger.log_dir}")
    console.print(f"[cyan]Log retention:[/cyan] {_http_logger.retention_days} days")
    
    # Create Flask app
    app = create_app()
    
    # Check if we should use Gunicorn or development server
    use_gunicorn = os.getenv("CFUSE_USE_DEV_SERVER") != "1"
    
    if use_gunicorn:
        # Use Gunicorn for production
        try:
            import gunicorn.app.base
        except ImportError:
            console.print("[red]Error:[/red] Gunicorn not installed. Install with: pip install gunicorn")
            console.print("[yellow]Tip:[/yellow] Or set CFUSE_USE_DEV_SERVER=1 to use development server")
            sys.exit(1)
        
        class StandaloneApplication(gunicorn.app.base.BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()
            
            def load_config(self):
                for key, value in self.options.items():
                    if key in self.cfg.settings and value is not None:
                        self.cfg.set(key.lower(), value)
            
            def load(self):
                return self.application
        
        # Determine number of workers (2-4 × CPU cores, max 16)
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        workers = min(cpu_count * 2, 16)
        
        # Gunicorn options
        options = {
            'bind': f'{host}:{port}',
            'workers': workers,
            'worker_class': 'sync',
            'timeout': 300,  # 5 minutes for long-running tools
            'graceful_timeout': 30,
            'keepalive': 5,
            'accesslog': '-',  # Log to stdout
            'errorlog': '-',
            'access_log_format': '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s',
            'preload_app': False,  # Load app in each worker (better isolation)
        }
        
        console.print(f"\n[green]✓ CFuse HTTP Server started (Production Mode)[/green]")
        console.print(f"[cyan]Workers:[/cyan] {workers}")
        if host == "0.0.0.0":
            console.print(f"[cyan]Listening on:[/cyan] http://0.0.0.0:{port} (all interfaces)")
            console.print(f"[cyan]Access via:[/cyan] http://localhost:{port} or http://<your-ip>:{port}")
        else:
            console.print(f"[cyan]Listening on:[/cyan] http://{host}:{port}")
        console.print(f"[cyan]Health check:[/cyan] http://{host}:{port}/health")
        console.print(f"[cyan]Metrics:[/cyan] http://{host}:{port}/metrics (log-based)")
        console.print(f"[cyan]Press Ctrl+C to stop[/cyan]\n")
        
        try:
            StandaloneApplication(app, options).run()
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Server stopped[/yellow]")
            if _http_logger:
                _http_logger.stop_cleanup_thread()
            sys.exit(0)
    
    else:
        # Use Flask development server (NOT for production)
        console.print(f"\n[yellow]⚠ CFuse HTTP Server started (Development Mode)[/yellow]")
        console.print(f"[yellow]Warning:[/yellow] Development server is not suitable for production!")
        if host == "0.0.0.0":
            console.print(f"[cyan]Listening on:[/cyan] http://0.0.0.0:{port} (all interfaces)")
        else:
            console.print(f"[cyan]Listening on:[/cyan] http://{host}:{port}")
        console.print(f"[cyan]Press Ctrl+C to stop[/cyan]\n")
        
        try:
            app.run(host=host, port=port, debug=False, threaded=True)
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Server stopped[/yellow]")
            if _http_logger:
                _http_logger.stop_cleanup_thread()
            sys.exit(0)
