from typing import Any, List, Mapping, Optional, Dict, AsyncGenerator, AsyncIterator
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import Field, root_validator
import httpx
import os
import json
from datetime import datetime, timedelta

class ChatAmexLlama(LLM):
    """Enhanced LLM wrapper for Llama API with improved auth and streaming."""
    
    # Base URLs and configurations remain the same...
    base_auth_url: str = Field(
        default_factory=lambda: os.getenv("LLAMA_AUTH_URL", "http://localhost:8000/auth"),
        description="Base authentication service URL"
    )
    base_api_url: str = Field(
        default_factory=lambda: os.getenv("LLAMA_API_URL", "http://localhost:8000/api"),
        description="Base Llama API service URL"
    )
    auth_endpoint: str = Field(default="/signin")
    completion_endpoint: str = Field(default="/chat/completions")
    cert_path: str = Field(default_factory=lambda: os.getenv("CERT_PATH"))
    user_id: str = Field(default_factory=lambda: os.getenv("LLAMA_USER_ID"))
    pwd: str = Field(default_factory=lambda: os.getenv("LLAMA_PASSWORD"))
    model_name: str = Field(default="llama3-70b-instruct")

    _auth_token: Optional[str] = None
    _token_expiry: Optional[datetime] = None
    _http_client: Optional[httpx.AsyncClient] = None

    class Config:
        underscore_attrs_are_private = True

    async def _generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator for streaming responses from the LLM.
        
        Args:
            prompt: The input prompt
            stop: Optional stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments
            
        Yields:
            Generated text chunks
        """
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
            "model": self.model_name,
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
                    try:
                        content = self._parse_chunk(line)
                        if content:
                            if run_manager:
                                await run_manager.on_llm_new_token(content)
                            yield content
                    except Exception as e:
                        print(f"Error parsing chunk: {str(e)}")

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Async call to generate complete response.
        
        Args:
            prompt: The input prompt
            stop: Optional stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments
            
        Returns:
            Complete generated text
        """
        chunks = []
        async for chunk in self._generate(prompt, stop, run_manager, **kwargs):
            chunks.append(chunk)
        return "".join(chunks)

    def _parse_chunk(self, chunk: str) -> Optional[str]:
        """
        Parse a chunk from the streaming response.
        
        Args:
            chunk: Raw chunk from the API
            
        Returns:
            Parsed content or None if invalid
        """
        try:
            # Handle different response formats
            if chunk.startswith("data: "):
                chunk = chunk[6:]
            
            if chunk == "[DONE]":
                return None
                
            data = json.loads(chunk)
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("delta", {}).get("content")
                return content if content else None
            return None
        except json.JSONDecodeError:
            return None
        except Exception as e:
            raise Exception(f"Error parsing chunk: {str(e)}")

    async def __aiter__(self) -> AsyncIterator[str]:
        """
        Make the class async iterable.
        Required for proper async iteration.
        """
        raise NotImplementedError("Direct iteration not supported. Use _generate() method.")

    async def __anext__(self) -> str:
        """
        Get next value in async iteration.
        Required for proper async iteration.
        """
        raise NotImplementedError("Direct iteration not supported. Use _generate() method.")

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "llama_authenticated"

    async def _aclose(self) -> None:
        """Cleanup resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # Auth methods and URL properties remain the same...
    @property
    def auth_uri(self) -> str:
        return f"{self.base_auth_url.rstrip('/')}{self.auth_endpoint}"

    @property
    def llama_uri(self) -> str:
        return f"{self.base_api_url.rstrip('/')}{self.completion_endpoint}"

    async def _get_auth_token(self) -> str:
        """Get or refresh authentication token."""
        now = datetime.now()
        
        if self._auth_token and self._token_expiry and now < self._token_expiry:
            return self._auth_token

        await self._init_client()
        
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

        self._auth_token = auth_cookie
        self._token_expiry = now + timedelta(hours=8)
        
        return self._auth_token

    async def _init_client(self) -> None:
        """Initialize HTTP client with certificate."""
        if not self._http_client:
            verify = self.cert_path if self.cert_path else True
            self._http_client = httpx.AsyncClient(
                verify=verify,
                timeout=60.0
            )
