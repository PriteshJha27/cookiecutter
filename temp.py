
from typing import Any, Dict, Iterator, List, Optional, Union
from langchain_core.language_models import BaseLLM, BaseModel
from langchain_core.outputs import Generation, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun
import httpx
import json

class ChatAmexLlama(BaseLLM, BaseModel):
    """Base LLM wrapper for Llama API."""
    
    base_url: str = "https://sidegenieservices-qa.amex.com/app/v1/opensource/models/llama3-70b-instruct/"
    auth_url: str = "https://antiblauncvae-v1p.phx.amex.com/fcol/signin/"
    cert_path: str = os.getenv('CERT_PATH')
    user_id: str = os.getenv('LLAMA_USER_ID')
    pwd: str = os.getenv('LLAMA_PASSWORD')
    model_name: str = "llama3-70b-instruct"
    
    _auth_token: Optional[str] = None
    _client: Optional[httpx.Client] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Print URLs for debugging. Delete later
        print(f"Auth URL: {self.auth_url}")
        print(f"Base URL: {self.base_url}")
        
        self._client = httpx.Client(
            verify=False,  # Disable SSL verification. Gave error earlier with cert path. recheck in v2
            timeout=30.0
        )
        
    def get_auth_token(self) -> str:
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

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[Generation]:
        """Stream response from LLM."""
        auth_token = self.get_auth_token()
        
        # Construct API URL
        api_url = f"{self.base_url.rstrip('/')}/chat/completions"
        print(f"Making streaming API call to: {api_url}")  # Debug print
        
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
            "stream": True  # Enable streaming
        }
        
        # Debug print payload
        print(f"Request Payload: {payload}")
        
        try:
            # Make streaming request
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
                    print(f"Error Response Body: {response.text}")  # Debug print
                    raise ValueError(f"API call failed: {response.status_code} - {response.text}")
                
                # Process streaming response
                for line in response.iter_lines():
                    if not line:
                        continue
                        
                    if line.startswith(b"data: "):
                        line = line[6:]  # Remove "data: " prefix
                    
                    try:
                        chunk = json.loads(line)
                        if chunk.get("choices"):
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                # Yield the token
                                yield Generation(text=delta["content"])
                                
                                if run_manager:
                                    run_manager.on_llm_new_token(delta["content"])
                    except json.JSONDecodeError as e:
                        print(f"Error parsing chunk: {str(e)}")
                        print(f"Raw chunk: {line}")
                        continue
                        
        except Exception as e:
            print(f"Error in stream: {str(e)}")
            raise ValueError(f"Streaming failed: {str(e)}")

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate method remains same but adds streaming support check"""
        auth_token = self.get_auth_token()
        generations: List[List[Generation]] = []
        
        for prompt in prompts:
            # Check if streaming is requested
            if kwargs.get("stream", False):
                # Collect streamed tokens
                generation_tokens = []
                for gen in self._stream(prompt, stop, run_manager, **kwargs):
                    generation_tokens.append(gen.text)
                # Combine tokens into single generation
                generations.append([Generation(text="".join(generation_tokens))])
                continue
            
            # Non-streaming logic (your existing code)
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
                    
                try:
                    response_data = response.json()
                    content = response_data["choices"][0]["message"]["content"]
                    generations.append([Generation(text=content)])
                except Exception as e:
                    print(f"Error parsing response: {str(e)}")
                    raise
                    
            except Exception as e:
                print(f"Error in generate: {str(e)}")
                raise ValueError(f"Generation failed: {str(e)}")
                
        return LLMResult(generations=generations)

    def __del__(self):
        if self._client:
            self._client.close()

    @property
    def _llm_type(self) -> str:
        return "llama_custom"
