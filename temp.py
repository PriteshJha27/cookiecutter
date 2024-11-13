from typing import Any, List, Optional, Dict
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field, root_validator
import httpx
import os
import json

class ChatAmexLlama(LLM):
    """LLM wrapper for internal Llama API."""
    
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
    
    _auth_token: Optional[str] = None
    
    class Config:
        underscore_attrs_are_private = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        required = ["base_url", "auth_url", "user_id", "pwd"]
        for field in required:
            if not values.get(field):
                raise ValueError(f"`{field}` is required")
        return values

    def _get_auth_token(self) -> str:
        """Get authentication token."""
        if self._auth_token:
            return self._auth_token

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

            self._auth_token = response.headers.get("Set-Cookie")
            if not self._auth_token:
                raise ValueError("No authentication token received")

            return self._auth_token
        finally:
            client.close()

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the LLM call."""
        auth_token = self._get_auth_token()
        verify = self.cert_path if self.cert_path else True

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
        return "llama_custom"

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in a text."""
        # Implement token counting if needed
        return len(text.split())
