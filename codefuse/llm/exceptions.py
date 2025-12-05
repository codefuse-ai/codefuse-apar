"""
LLM Exception Classes
"""

from typing import Optional


class LLMError(Exception):
    """Base exception for all LLM-related errors"""
    pass


class RetryableError(LLMError):
    """Base class for errors that can be retried"""
    pass


class TimeoutError(RetryableError):
    """Request timeout error - will be retried"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class RateLimitError(RetryableError):
    """Rate limit exceeded error - will be retried with backoff"""
    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.retry_after = retry_after  # Seconds to wait before retry
        self.original_error = original_error


class APIError(LLMError):
    """General API error - not retryable"""
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.original_error = original_error


class AuthenticationError(LLMError):
    """Authentication failed - invalid API key or credentials"""
    pass


class ContextLengthExceededError(LLMError):
    """Context length exceeded the model's maximum"""
    def __init__(self, message: str, max_tokens: Optional[int] = None):
        super().__init__(message)
        self.max_tokens = max_tokens


class InvalidRequestError(LLMError):
    """Invalid request parameters"""
    pass


class ModelNotFoundError(LLMError):
    """Requested model not found or not available"""
    pass

