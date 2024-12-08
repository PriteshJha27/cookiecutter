
#### ollama_baseclient.py --------------------------------------------------------------------------

import httpx
from typing import Any, Mapping, Union, Iterator, Optional
from collections.abc import AsyncIterator
from httpx import Response

class BaseClient:
    def __init__(self, base_url: str, headers: dict, **kwargs) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self._client = httpx.Client(base_url=self.base_url, headers=self.headers, **kwargs)
    
    def post(self, endpoint: str, json: dict) -> Response:
        response = self._client.post(endpoint, json=json)
        response.raise_for_status()
        return response

    def stream(self, endpoint: str, json: dict) -> Iterator[Mapping[str, Any]]:
        with self._client.stream("POST", endpoint, json=json) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                yield httpx.Response.json(line)

class AsyncBaseClient:
    def __init__(self, base_url: str, headers: dict, **kwargs) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=self.headers, **kwargs)
    
    async def post(self, endpoint: str, json: dict) -> Response:
        response = await self._client.post(endpoint, json=json)
        response.raise_for_status()
        return response

    async def stream(self, endpoint: str, json: dict) -> AsyncIterator[Mapping[str, Any]]:
        async with self._client.stream("POST", endpoint, json=json) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield httpx.Response.json(line)



### ollama_ChatOllama.py ----------------------------------------------------------------------------------------------

from ollama_baseclient import BaseClient, AsyncBaseClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from typing import List, Optional, Union, Any
import os

class ChatOllama(BaseChatModel):
    def __init__(self, base_url: str, auth_url: str, user_id: str, password: str) -> None:
        self.base_url = base_url
        self.auth_url = auth_url
        self.user_id = user_id
        self.password = password
        self.token = self._authenticate()
        self.client = BaseClient(base_url=self.base_url, headers={"Authorization": f"Bearer {self.token}"})
        self.async_client = AsyncBaseClient(base_url=self.base_url, headers={"Authorization": f"Bearer {self.token}"})

    def _authenticate(self) -> str:
        response = httpx.post(self.auth_url, data={"userid": self.user_id, "pwd": self.password})
        response.raise_for_status()
        token = response.headers.get("Set-Cookie")
        if not token:
            raise ValueError("Authentication failed: Token not received")
        return token

    def invoke(self, messages: List[BaseMessage]) -> str:
        payload = {"messages": [{"role": msg.role, "content": msg.content} for msg in messages]}
        response = self.client.post("/chat/completions", json=payload)
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

    def bind_tools(self, tools: List[Any]) -> None:
        # Placeholder: Allow binding of tools
        self.tools = tools
      
#### Implementation -----------------------------------------------------------------------------------------------

from ollama_ChatOllama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage

# Initialize ChatOllama with environment variables
import os
from dotenv import load_dotenv

load_dotenv()
base_url = os.getenv("BASE_URL")
auth_url = os.getenv("AUTH_URL")
user_id = os.getenv("USER_ID")
password = os.getenv("PASSWORD")

llm = ChatOllama(base_url=base_url, auth_url=auth_url, user_id=user_id, password=password)

# Test invoke
messages = [
    HumanMessage(content="What is 2 + 2?"),
]
response = llm.invoke(messages)
print(f"Response: {response}")

# Test bind_tools
def custom_tool(input_text: str) -> str:
    return f"Custom tool processed: {input_text}"

llm.bind_tools([custom_tool])
print("Tools bound successfully!")
