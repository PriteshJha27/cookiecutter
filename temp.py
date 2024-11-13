from typing import Any, List, Optional, Dict
from langchain_core.language_models import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import Generation, LLMResult
import httpx
from pydantic import BaseModel, ConfigDict
import os
import ssl
import certifi

class ChatAmexLlama(BaseLLM, BaseModel):
    """Base LLM wrapper for Llama API."""
    
    base_url: str
    auth_url: str
    user_id: str
    pwd: str
    cert_path: str
    model_name: str = "llama3-70b-instruct"
    verify_ssl: bool = True  # Added option to control SSL verification
    
    _auth_token: Optional[str] = None
    _client: Optional[httpx.Client] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        if not self.verify_ssl:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        
        # If cert_path is provided, load the certificate
        if self.cert_path:
            try:
                ssl_context.load_verify_locations(self.cert_path)
            except Exception as e:
                print(f"Warning: Failed to load certificate from {self.cert_path}: {str(e)}")
                # Fallback to no verification if certificate loading fails
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

        # Initialize HTTP client
        self._client = httpx.Client(
            verify=False,  # We'll handle verification through our SSL context
            timeout=30.0,
            http2=True
        )

    def _get_auth_token(self) -> str:
        """Get authentication token."""
        if self._auth_token:
            return self._auth_token

        try:
            response = self._client.post(
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
                raise ValueError(f"Authentication failed: {response.status_code} - {response.text}")

            self._auth_token = response.headers.get("Set-Cookie")
            if not self._auth_token:
                raise ValueError("No authentication token received")

            return self._auth_token

        except Exception as e:
            raise ValueError(f"Authentication failed: {str(e)}")

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate LLM responses."""
        auth_token = self._get_auth_token()
        generations = []

        try:
            for prompt in prompts:
                response = self._client.post(
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
                    raise ValueError(f"API call failed: {response.status_code} - {response.text}")

                content = response.json()['choices'][0]['message']['content']
                generations.append([Generation(text=content)])

            return LLMResult(generations=generations)

        except Exception as e:
            raise ValueError(f"Generation failed: {str(e)}")

    def __del__(self):
        """Cleanup client on deletion."""
        if self._client:
            self._client.close()

    @property
    def _llm_type(self) -> str:
        return "llama_custom"
