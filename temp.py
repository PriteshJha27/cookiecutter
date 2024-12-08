from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain.chat_models import ChatOllama
import os
from dotenv import load_dotenv
import httpx
import json
from typing import Any, Dict, List, Optional, Iterator, AsyncIterator, Union, Mapping
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage
)

class AmexLlamaChatModel(ChatOllama):
    """Custom ChatOllama for AEXP's hosted Llama model."""
    
    def __init__(self, **kwargs):
        # Load environment variables
        load_dotenv()
        
        # Set base configuration
        base_url = "https://aidageniservices-qa.aexp.com/app/v1/opensource/models/llama3-70b-instruct/chat/completions"
        auth_url = "https://authbluesvcqa-vip.phx.aexp.com/sso1/signin/"
        model_name = "llama3-70b-instruct"
        
        # Get credentials from environment
        cert_path = os.getenv('CERT_PATH')
        user_id = os.getenv('LLAMA_USER_ID')
        pwd = os.getenv('LLAMA_PASSWORD')
        
        # Get authentication token
        self.auth_token = self._get_auth_token(auth_url, user_id, pwd)
        
        # Initialize parent class
        super().__init__(
            base_url=base_url,
            model=model_name,
            **kwargs
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers including auth token."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Cookie": self.auth_token
        }

    def _get_auth_token(self, auth_url: str, user_id: str, pwd: str) -> str:
        """Get authentication token from the auth service."""
        try:
            client = httpx.Client(
                verify=os.getenv('CERT_PATH'),
                timeout=30.0
            )
            
            print("Sending auth request...")
            response = client.post(
                auth_url,
                data={
                    "userid": user_id,
                    "pwd": pwd
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "*/*"
                }
            )
            
            print(f"Auth Response Status: {response.status_code}")
            
            if response.status_code != 200:
                raise ValueError(f"Authentication failed: {response.status_code} - {response.text}")
            
            auth_token = response.headers.get("Set-Cookie")
            if not auth_token:
                raise ValueError("No authentication token received")
            
            return auth_token
            
        except Exception as e:
            raise ValueError(f"Authentication failed: {str(e)}")
        finally:
            client.close()

    def _create_chat_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[Union[Mapping[str, Any], str]]:
        chat_params = self._chat_params(messages, stop, **kwargs)
        
        client = httpx.Client(
            verify=os.getenv('CERT_PATH'),
            timeout=30.0
        )
        
        try:
            if chat_params["stream"]:
                response = client.post(
                    self.base_url,
                    json=chat_params,
                    headers=self._get_headers()
                )
                yield from response.iter_lines()
            else:
                response = client.post(
                    self.base_url,
                    json=chat_params,
                    headers=self._get_headers()
                )
                yield response.json()
        finally:
            client.close()

    async def _acreate_chat_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[Mapping[str, Any], str]]:
        chat_params = self._chat_params(messages, stop, **kwargs)
        
        async with httpx.AsyncClient(
            verify=os.getenv('CERT_PATH'),
            timeout=30.0
        ) as client:
            if chat_params["stream"]:
                async with client.stream(
                    'POST',
                    self.base_url,
                    json=chat_params,
                    headers=self._get_headers()
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            yield json.loads(line)
            else:
                response = await client.post(
                    self.base_url,
                    json=chat_params,
                    headers=self._get_headers()
                )
                yield response.json()

def main():
    # Initialize the model
    llm = AmexLlamaChatModel(
        temperature=0.7,
    )
    
    # Example chat
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="What is 2+2?")
    ]
    
    response = llm.invoke(messages)
    print(response)

if __name__ == "__main__":
    main()
