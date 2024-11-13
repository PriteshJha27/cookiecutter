from typing import Any, List, Optional, Dict
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import BaseModel, Field, validator
import httpx
import os
import json

class ChatAmexLlama(LLM, BaseModel):
    """LLM wrapper for internal Llama API."""
    
    base_url: str = Field(default="")
    auth_url: str = Field(default="")
    cert_path: Optional[str] = Field(default=None)
    user_id: str = Field(default="")
    pwd: str = Field(default="")
    model_name: str = Field(default="llama3-70b-instruct")
    
    _auth_token: Optional[str] = None

    @validator("base_url", "auth_url", "user_id", "pwd", pre=True)
    def validate_string_fields(cls, v, field):
        if not v:
            env_value = os.getenv(f"LLAMA_{field.name.upper()}")
            if not env_value:
                raise ValueError(f"{field.name} must be provided either directly or through environment variable")
            return env_value
        return v

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
