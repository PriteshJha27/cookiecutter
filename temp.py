# Custom exception hierarchy
class LlamaAPIError(Exception):
    """Base exception for Llama API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

class LlamaAuthError(LlamaAPIError):
    """Authentication related errors."""
    pass

class LlamaResponseError(LlamaAPIError):
    """Response parsing and validation errors."""
    pass

class LlamaStreamError(LlamaAPIError):
    """Streaming specific errors."""
    pass

# Retry decorator for API calls
def retry_api_call(func):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((TransportError, LlamaAPIError))
    )
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
    return wrapper

class BaseChatAmexLlama(BaseChatModel):
    @retry_api_call
    async def _get_auth_token_async(self) -> str:
        """Get authentication token with proper error handling and retries."""
        if self._auth_token:
            return self._auth_token

        try:
            response = await self._async_client.post(
                self.auth_url,
                data={
                    "userId": self.user_id,
                    "pwd": self.pwd
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "*/*"
                }
            )

            if response.status_code != 200:
                raise LlamaAuthError(
                    f"Authentication failed: {response.status_code} - {response.text}",
                    status_code=response.status_code
                )

            auth_token = response.headers.get("Set-Cookie")
            if not auth_token:
                raise LlamaAuthError("No authentication token received")

            self._auth_token = auth_token
            return auth_token

        except Exception as e:
            raise LlamaAuthError(f"Authentication failed: {str(e)}")
