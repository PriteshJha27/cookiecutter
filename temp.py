from typing import Any, List, Mapping, Optional, Dict
from langchain.llms.base import LLM
from langchain.pydantic_v1 import Field, root_validator
import httpx
import os
import json

class ChatAmexLlama(LLM):
    """Synchronous LLM wrapper for Llama API."""
    
    # Configuration fields
    base_url: str = Field(
        default_factory=lambda: os.getenv("LLAMA_API_URL", "http://localhost:8000")
    )
    auth_url: str = Field(
        default_factory=lambda: os.getenv("LLAMA_AUTH_URL", "http://localhost:8000/auth")
    )
    cert_path: str = Field(
        default_factory=lambda: os.getenv("CERT_PATH")
    )
    user_id: str = Field(
        default_factory=lambda: os.getenv("LLAMA_USER_ID")
    )
    pwd: str = Field(
        default_factory=lambda: os.getenv("LLAMA_PASSWORD")
    )
    model_name: str = Field(default="llama3-70b-instruct")

    # Internal state
    _auth_token: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""
        underscore_attrs_are_private = True

    def _get_auth_token(self) -> str:
        """Get authentication token."""
        if self._auth_token:
            return self._auth_token

        # Create HTTP client with certificate if provided
        verify = self.cert_path if self.cert_path else True
        client = httpx.Client(verify=verify)

        try:
            response = client.post(
                self.auth_url,
                data={
                    "userid": self.user_id,
                    "pwd": self.pwd
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "*/*"
                }
            )

            if response.status_code != 200:
                raise ValueError(f"Authentication failed: {response.status_code} {response.text}")

            auth_cookie = response.headers.get("Set-Cookie")
            if not auth_cookie:
                raise ValueError("No authentication token received")

            self._auth_token = auth_cookie
            return self._auth_token

        finally:
            client.close()

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the LLM call."""
        
        auth_token = self._get_auth_token()
        verify = self.cert_path if self.cert_path else True
        
        # Create a new client for this call
        with httpx.Client(verify=verify) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                json={
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
                    "stream": False
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Cookie": auth_token
                }
            )

            if response.status_code != 200:
                raise ValueError(f"API call failed: {response.status_code} {response.text}")

            try:
                return response.json()['choices'][0]['message']['content']
            except Exception as e:
                raise ValueError(f"Error parsing response: {str(e)}")

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "llama_custom"

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that required fields are present."""
        required_fields = ["base_url", "auth_url", "user_id", "pwd"]
        for field in required_fields:
            if not values.get(field):
                raise ValueError(f"{field} must be provided")
        return values
