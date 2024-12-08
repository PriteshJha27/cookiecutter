from ollama_baseclient import BaseClient, AsyncBaseClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, ChatGeneration, ChatResult, AIMessage
from typing import List, Optional, Union, Any
from pydantic import Field, PrivateAttr
import httpx


class ChatOllama(BaseChatModel):
    base_url: str = Field(..., description="The base URL for the Llama API.")
    auth_url: str = Field(..., description="The URL for authentication.")
    user_id: str = Field(..., description="The user ID for authentication.")
    password: str = Field(..., description="The password for authentication.")
    _token: str = PrivateAttr(default=None)
    _client: BaseClient = PrivateAttr(default=None)
    _async_client: AsyncBaseClient = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._token = self._authenticate()
        self._client = BaseClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self._token}"},
        )
        self._async_client = AsyncBaseClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self._token}"},
        )

    def _authenticate(self) -> str:
        """Authenticate with the API and retrieve a token."""
        response = httpx.post(self.auth_url, data={"userid": self.user_id, "pwd": self.password})
        response.raise_for_status()
        token = response.headers.get("Set-Cookie")
        if not token:
            raise ValueError("Authentication failed: Token not received")
        return token

    def invoke(self, messages: List[BaseMessage]) -> str:
        """Invoke the LLM with the given messages."""
        payload = {"messages": [{"role": msg.role, "content": msg.content} for msg in messages]}
        response = self._client.post("/chat/completions", json=payload)
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

    def bind_tools(self, tools: List[Any]) -> None:
        """Bind custom tools to the LLM."""
        self.tools = tools

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        """Generate a single response for the given messages."""
        payload = {"messages": [{"role": msg.role, "content": msg.content} for msg in messages]}
        response = self._client.post("/chat/completions", json=payload)
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        chat_message = AIMessage(content=content)
        chat_generation = ChatGeneration(message=chat_message)
        return ChatResult(generations=[chat_generation])

    @property
    def _llm_type(self) -> str:
        """Return the type of the LLM."""
        return "chat-ollama"




from ollama_ChatOllama import ChatOllama
from langchain_core.messages import HumanMessage

# Initialize ChatOllama
llm = ChatOllama(
    base_url="https://your-llama-base-url.com",
    auth_url="https://your-authentication-url.com",
    user_id="your-user-id",
    password="your-password",
)

# Test invoke
messages = [HumanMessage(content="What is 2 + 2?")]
response = llm.invoke(messages)
print(f"Response: {response}")

# Test bind_tools
def custom_tool(input_text: str) -> str:
    return f"Processed by tool: {input_text}"

llm.bind_tools([custom_tool])
print("Tools bound successfully!")
