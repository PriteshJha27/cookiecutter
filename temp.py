from typing import Any, List, Mapping, Optional, Dict, Iterator
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import Field, root_validator
import httpx
import os
from datetime import datetime, timedelta

class ChatAmexLlama(LLM):
    """Enhanced LLM wrapper for Llama API with improved auth and streaming."""
    
    # API Configuration
    auth_uri: str = Field(
        default="https://authbluescoa-vip.phx.aexp.com/ssoi/signin",
        description="Authentication endpoint URL"
    )
    llama_uri: str = Field(
        default="https://eidgenesiservices-oa.aexp.com/app/v1/opensource/models/llama3-70b-instruct/chat/completions",
        description="Llama API endpoint URL"
    )
    cert_path: str = Field(
        default_factory=lambda: os.getenv("cert_path"),
        description="Path to certificate file"
    )
    user_id: str = Field(
        default_factory=lambda: os.getenv("userid_for_llama"),
        description="User ID for authentication"
    )
    pwd: str = Field(
        default_factory=lambda: os.getenv("pwd_for_llama"),
        description="Password for authentication"
    )

    # Session Management
    _auth_token: Optional[str] = None
    _token_expiry: Optional[datetime] = None
    _http_client: Optional[httpx.AsyncClient] = None

    class Config:
        """Configuration for this pydantic object."""
        underscore_attrs_are_private = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that required fields are present."""
        required_fields = ["auth_uri", "llama_uri", "user_id", "pwd"]
        for field in required_fields:
            if not values.get(field):
                raise ValueError(f"{field} must be provided")
        return values

    async def _init_client(self) -> None:
        """Initialize HTTP client with certificate."""
        if not self._http_client:
            self._http_client = httpx.AsyncClient(
                verify=self.cert_path,
                timeout=60.0
            )

    async def _get_auth_token(self) -> str:
        """Get or refresh authentication token."""
        now = datetime.now()
        
        # Return existing token if still valid
        if self._auth_token and self._token_expiry and now < self._token_expiry:
            return self._auth_token

        await self._init_client()
        
        # Auth request
        payload = {
            "userid": self.user_id,
            "pwd": self.pwd
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "*/*"
        }

        response = await self._http_client.post(
            self.auth_uri,
            data=payload,
            headers=headers
        )

        if response.status_code != 200:
            raise Exception(f"Authentication failed: {response.status_code} {response.text}")

        auth_cookie = response.headers.get("Set-Cookie")
        if not auth_cookie:
            raise Exception("No authentication token received")

        # Set token and expiry (8 hours from now)
        self._auth_token = auth_cookie
        self._token_expiry = now + timedelta(hours=8)
        
        return self._auth_token

    async def _generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Generate LLM response with streaming support."""
        auth_token = await self._get_auth_token()
        await self._init_client()

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "llama3-70b-instruct",
            "stream": True
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Cookie": auth_token
        }

        async with self._http_client.stream(
            "POST",
            self.llama_uri,
            json=payload,
            headers=headers
        ) as response:
            if response.status_code != 200:
                raise Exception(f"Llama API call failed: {response.status_code} {response.text}")

            async for line in response.aiter_lines():
                if line.strip():
                    content = self._parse_chunk(line)
                    if content and run_manager:
                        await run_manager.on_llm_new_token(content)
                    yield content

    def _parse_chunk(self, chunk: str) -> str:
        """Parse streaming response chunk."""
        try:
            # Add your specific chunk parsing logic here
            # This is a placeholder - adjust based on your API's response format
            return chunk.strip()
        except Exception as e:
            raise Exception(f"Error parsing chunk: {str(e)}")

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "llama_authenticated"

    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "llama_uri": self.llama_uri,
            "model": "llama3-70b-instruct"
        }

    async def _aclose(self) -> None:
        """Cleanup resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
