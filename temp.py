from typing import Any, Dict, Iterator, List, Optional, Union
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_core.outputs import Generation, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import ConfigDict
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

class ChatAmexLlama(BaseLLM, BaseChatModel):
    """Base LLM wrapper for Llama API."""
    
    base_url: str = "https://sidegenieservices-qa.aexp.com/app/v1/opensource/models/llama3-70b-instruct/"
    auth_url: str = "https://antiblauevcqa-v1p.phx.amex.com/fcol/signin/"
    cert_path: str = os.getenv('CERT_PATH')
    user_id: str = os.getenv('LLAMA_USER_ID')
    pwd: str = os.getenv('LLAMA_PASSWORD')
    model_name: str = "llama3-70b-instruct"
    
    _auth_token: Optional[str] = None
    _client: Optional[httpx.Client] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"Auth URL: {self.auth_url}")
        print(f"Base URL: {self.base_url}")
        
        self._client = httpx.Client(
            verify=False,
            timeout=30.0
        )

    def get_auth_token(self) -> str:
        """Get authentication token."""
        if self._auth_token:
            return self._auth_token
            
        try:
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

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[Generation]:
        """Stream response from LLM."""
        auth_token = self.get_auth_token()
        api_url = f"{self.base_url.rstrip('/')}/chat/completions"
        
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
        
        print(f"Making streaming request to: {api_url}")
        print(f"With payload: {json.dumps(payload, indent=2)}")
        
        with self._client.stream(
            "POST",
            api_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Cookie": auth_token
            }
        ) as response:
            if response.status_code != 200:
                error_msg = f"API call failed: {response.status_code}"
                try:
                    error_msg += f" - {response.json()}"
                except:
                    error_msg += f" - {response.text}"
                raise ValueError(error_msg)
            
            # Process the stream
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                
                try:
                    chunk_str = chunk.decode('utf-8')
                    # Handle SSE format
                    if chunk_str.startswith('data: '):
                        chunk_str = chunk_str[6:]
                    
                    chunk_data = json.loads(chunk_str)
                    if chunk_data.get("choices"):
                        content = chunk_data["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            yield Generation(text=content)
                            if run_manager:
                                run_manager.on_llm_new_token(content)
                
                except json.JSONDecodeError as e:
                    print(f"Error parsing chunk: {chunk_str}")
                    continue
                except Exception as e:
                    print(f"Error processing chunk: {str(e)}")
                    continue

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate LLM result."""
        auth_token = self.get_auth_token()
        generations = []
        
        for prompt in prompts:
            # Handle streaming if requested
            if kwargs.get("stream", False):
                tokens = []
                try:
                    for gen in self._stream(prompt, stop, run_manager, **kwargs):
                        tokens.append(gen.text)
                    generations.append([Generation(text="".join(tokens))])
                except Exception as e:
                    raise ValueError(f"Streaming generation failed: {str(e)}")
                continue
            
            # Non-streaming generation
            api_url = f"{self.base_url.rstrip('/')}/chat/completions"
            
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
            
            try:
                response = self._client.post(
                    api_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Cookie": auth_token
                    }
                )
                
                if response.status_code != 200:
                    raise ValueError(f"API call failed: {response.status_code} - {response.text}")
                
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"]
                generations.append([Generation(text=content)])
                
            except Exception as e:
                raise ValueError(f"Generation failed: {str(e)}")
        
        return LLMResult(generations=generations)

    def __del__(self):
        """Cleanup client on deletion."""
        if self._client:
            self
