"""Custom exceptions for the Autonomous Document Agent."""


class AgentError(Exception):
    """Base exception for all agent errors."""
    pass


class LLMUnavailableError(AgentError):
    """Raised when the LLM service is unreachable, authentication fails, or retries/rate limits are exhausted."""
    pass


class LLMOutputError(AgentError):
    """Raised when LLM output cannot be parsed or validated after retries."""

    def __init__(self, step: str, reason: str):
        self.step = step
        self.reason = reason
        super().__init__(f"[{step}] LLM output invalid: {reason}")


class ToolError(AgentError):
    """Raised when tool arguments are invalid or tool execution fails. Message must be readable by an LLM."""
    pass
