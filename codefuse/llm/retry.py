"""
Retry Logic for LLM Requests
"""

import time
from functools import wraps
from typing import Callable, Tuple, Type, Optional

from codefuse.llm.exceptions import RetryableError, RateLimitError, TimeoutError
from codefuse.observability import mainLogger


def retry_on_failure(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (RetryableError, RateLimitError, TimeoutError)
):
    """
    Decorator to retry function calls on specific exceptions
    
    Retry Strategy:
    - Timeout errors: Retry with exponential backoff
    - Rate limit errors (429): Retry with exponential backoff or Retry-After header
    - Other errors: Raise immediately
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        retryable_exceptions: Tuple of exception types that should trigger retry
        
    Returns:
        Decorated function that will retry on retryable errors
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                    
                except retryable_exceptions as e:
                    last_exception = e
                    
                    # If this was the last attempt, raise the exception
                    if attempt == max_retries - 1:
                        mainLogger.error(
                            f"Failed after {max_retries} attempts: {type(e).__name__}: {e}"
                        )
                        raise
                    
                    # Calculate wait time
                    if isinstance(e, RateLimitError) and e.retry_after:
                        # Use the Retry-After value from the API response
                        wait_time = e.retry_after
                        mainLogger.warning(
                            f"Rate limit hit. Waiting {wait_time:.1f}s as specified by API."
                        )
                    else:
                        # Exponential backoff: 1s, 2s, 4s, 8s, etc.
                        wait_time = initial_delay * (exponential_base ** attempt)
                        mainLogger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}"
                        )
                    
                    mainLogger.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    
                except Exception as e:
                    # Non-retryable error - raise immediately
                    mainLogger.error(f"Non-retryable error occurred: {type(e).__name__}: {e}")
                    raise
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


def should_retry(exception: Exception) -> bool:
    """
    Determine if an exception should trigger a retry
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the exception is retryable, False otherwise
    """
    return isinstance(exception, (RetryableError, RateLimitError, TimeoutError))


def get_retry_delay(
    attempt: int,
    exception: Optional[Exception] = None,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0
) -> float:
    """
    Calculate retry delay based on attempt number and exception type
    
    Args:
        attempt: Current attempt number (0-indexed)
        exception: The exception that triggered the retry
        initial_delay: Initial delay in seconds
        exponential_base: Base for exponential backoff
        
    Returns:
        Delay in seconds before next retry
    """
    # Check if exception has a retry_after attribute (e.g., RateLimitError)
    if isinstance(exception, RateLimitError) and exception.retry_after:
        return exception.retry_after
    
    # Default exponential backoff
    return initial_delay * (exponential_base ** attempt)

