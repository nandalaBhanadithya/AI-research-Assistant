from typing import Optional


class AppError(Exception):
    """Base class for application errors that should map to a clean HTTP response."""

    status_code = 500
    error_code = "internal_error"

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    status_code = 404
    error_code = "not_found"


class ValidationFailedError(AppError):
    status_code = 422
    error_code = "validation_failed"


class ProcessingError(AppError):
    status_code = 422
    error_code = "processing_error"


class GuardrailRefusal(AppError):
    """Raised internally to short-circuit generation; caught and turned into a normal 200 response."""

    status_code = 200
    error_code = "guardrail_refusal"


class LLMProviderError(AppError):
    status_code = 502
    error_code = "llm_provider_error"
