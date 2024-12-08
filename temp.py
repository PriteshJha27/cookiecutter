from ollama_baseclient import BaseClient, AsyncBaseClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, ChatGeneration, ChatResult, AIMessage
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

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        """Generate a single response for the given messages."""
        payload = {"messages": [{"role": msg.role, "content": msg.content} for msg in messages]}
        response = self.client.post("/chat/completions", json=payload)
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        chat_message = AIMessage(content=content)
        chat_generation = ChatGeneration(message=chat_message)
        return ChatResult(generations=[chat_generation])

    @property
    def _llm_type(self) -> str:
        """Return the type of the LLM."""
        return "chat-ollama"
