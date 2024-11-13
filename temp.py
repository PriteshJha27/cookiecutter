from typing import Any, List, Optional, Dict
from langchain_core.language_models import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
import httpx
from pydantic import BaseModel, ConfigDict
import os

class ChatAmexLlama(BaseLLM, BaseModel):
    """Base LLM wrapper for Llama API."""
    
    base_url: str
    auth_url: str
    user_id: str
    pwd: str
    cert_path: Optional[str] = None
    model_name: str = "llama3-70b-instruct"
    
    _auth_token: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_auth_token(self) -> str:
        """Get authentication token."""
        if self._auth_token:
            return self._auth_token

        verify = self.cert_path if self.cert_path else True
        with httpx.Client(verify=verify) as client:
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
                raise ValueError(f"Authentication failed: {response.status_code}")

            self._auth_token = response.headers.get("Set-Cookie")
            if not self._auth_token:
                raise ValueError("No authentication token received")

            return self._auth_token

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
                raise ValueError(f"API call failed: {response.status_code}")

            try:
                return response.json()['choices'][0]['message']['content']
            except Exception as e:
                raise ValueError(f"Error parsing response: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "llama_custom"
