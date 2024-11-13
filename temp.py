from typing import Any, List, Optional, Dict
from langchain_core.language_models import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import Generation, LLMResult
import httpx
from pydantic import BaseModel, ConfigDict
import os

class ChatAmexLlama(BaseLLM, BaseModel):
    """Base LLM wrapper for Llama API."""
    
    base_url: str
    auth_url: str
    user_id: str
    pwd: str
    cert_path: str
    model_name: str = "llama3-70b-instruct"
    
    _auth_token: Optional[str] = None
    _client: Optional[httpx.Client] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Print URLs for debugging
        print(f"Auth URL: {self.auth_url}")
        print(f"Base URL: {self.base_url}")
        
        self._client = httpx.Client(
            verify=False,  # Disable SSL verification
            timeout=30.0
        )

    def _get_auth_token(self) -> str:
        """Get authentication token."""
        if self._auth_token:
            return self._auth_token

        try:
            # Print request details
            print("Sending auth request...")
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
            print(f"Auth Response Status: {response.status_code}")

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
                # Construct complete URL
                api_url = f"{self.base_url.rstrip('/')}/chat/completions"
                print(f"Making API call to: {api_url}")  # Debug print
                
                # Prepare request payload
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
                    "stream": False
                }
                
                # Debug print payload
                print(f"Request Payload: {payload}")

                response = self._client.post(
                    api_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Cookie": auth_token
                    }
                )
                
                # Debug print response
                print(f"Response Status: {response.status_code}")
                print(f"Response Headers: {response.headers}")
                
                if response.status_code != 200:
                    print(f"Error Response Body: {response.text}")  # Debug print
                    raise ValueError(f"API call failed: {response.status_code} - {response.text}")

                try:
                    response_data = response.json()
                    content = response_data['choices'][0]['message']['content']
                    generations.append([Generation(text=content)])
                except Exception as e:
                    print(f"Error parsing response: {str(e)}")
                    print(f"Raw Response: {response.text}")
                    raise

            return LLMResult(generations=generations)

        except Exception as e:
            print(f"Error in _generate: {str(e)}")
            raise ValueError(f"Generation failed: {str(e)}")

    def __del__(self):
        if self._client:
            self._client.close()

    @property
    def _llm_type(self) -> str:
        return "llama_custom"
