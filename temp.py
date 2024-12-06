#### v2
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, Self
from pydantic import BaseModel, Field, ConfigDict, SecretStr
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
import os
import warnings
import json
import httpx
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import model_validator
from __future__ import annotations

import base64
import json
import logging
import os
import sys
import warnings
from io import BytesIO
from math import ceil
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)
from urllib.parse import urlparse

import openai
import tiktoken
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough, chain
from langchain_core.runnables.config import run_in_executor
from langchain_core.tools import BaseTool
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import (
    PydanticBaseModel,
    TypeBaseModel,
    is_basemodel_subclass,
)
from langchain_core.utils.utils import build_extra_kwargs, from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)



class ChatLlama(BaseChatModel):
    client: Any = Field(default=None, exclude=True)
    async_client: Any = Field(default=None, exclude=True)
    
    model_name: str = Field(default="llama3-70b-instruct")
    temperature: float = 0.7
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    base_url: Optional[str] = Field(default=None)
    user_id: Optional[str] = Field(default=None)
    pwd: Optional[str] = Field(default=None)
    auth_token: Optional[str] = Field(default=None)
    
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(default=None)
    max_retries: int = 2
    streaming: bool = False
    n: int = 1
    max_tokens: Optional[int] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate environment and setup client."""
        if self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        if not self.client:
            self.client = httpx.Client(
                base_url=self.base_url,
                timeout=self.request_timeout,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
        if not self.async_client:
            self.async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.request_timeout,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

        return self
    
    def _get_auth_token(self) -> str:
        """Get authentication token."""
        if self.auth_token:
            return self.auth_token
            
        try:
            response = self.client.post(
                self.auth_url,
                data={
                    "userid": self.user_id,
                    "pwd": self.pwd
                }
            )
            if response.status_code != 200:
                raise ValueError(f"Authentication failed: {response.status_code} - {response.text}")
                
            self.auth_token = response.headers.get("Set-Cookie")
            if not self.auth_token:
                raise ValueError("No authentication token received")
                
            return self.auth_token
            
        except Exception as e:
            raise ValueError(f"Authentication failed: {str(e)}")

    def _get_request_payload(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """Get the request payload for the Llama API."""
        if stop is not None:
            kwargs["stop"] = stop
            
        return {
            "messages": [self._convert_message_to_dict(m) for m in messages],
            "model": self.model_name,
            "stream": self.streaming,
            **self._default_params,
            **kwargs,
        }

    def _convert_message_to_dict(self, message: BaseMessage) -> dict:
        """Convert a LangChain message to a dictionary."""
        message_dict = {
            "content": message.content
        }
        
        if isinstance(message, SystemMessage):
            message_dict["role"] = "system"
        elif isinstance(message, HumanMessage):
            message_dict["role"] = "user"
        elif isinstance(message, AIMessage):
            message_dict["role"] = "assistant"
            if message.function_call:
                message_dict["function_call"] = message.function_call
            if message.tool_calls:
                message_dict["tool_calls"] = [
                    self._convert_tool_call_to_dict(tc) for tc in message.tool_calls
                ]
        else:
            message_dict["role"] = message.type
            
        return message_dict

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response."""
        if self.streaming:
            raise NotImplementedError("Streaming not implemented for Llama API")
            
        auth_token = self._get_auth_token()
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        
        response = self.client.post(
            f"{self.base_url}/chat/completions",
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
        return self._create_chat_result(response_data)

    def _create_chat_result(self, response: dict) -> ChatResult:
        """Create ChatResult from API response."""
        generations = []
        
        if response.get("error"):
            raise ValueError(response.get("error"))
            
        for choice in response["choices"]:
            message = self._convert_dict_to_message(choice["message"])
            generation_info = {
                "finish_reason": choice.get("finish_reason"),
            }
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info
            )
            generations.append(gen)
            
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
        }
        
        return ChatResult(generations=generations, llm_output=llm_output)

    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        tool_choice: Optional[Union[dict, str, Literal["auto", "none"]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to chat model."""
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        
        if tool_choice:
            if isinstance(tool_choice, str):
                if tool_choice not in ("auto", "none"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
            elif isinstance(tool_choice, dict):
                tool_names = [
                    formatted_tool["function"]["name"]
                    for formatted_tool in formatted_tools
                ]
                if tool_choice["function"]["name"] not in tool_names:
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            kwargs["tool_choice"] = tool_choice
            
        return super().bind(tools=formatted_tools, **kwargs)

    def _convert_tool_call_to_dict(self, tool_call: ToolCall) -> dict:
        """Convert tool call to API format."""
        return {
            "type": "function",
            "id": tool_call.get("id", ""),
            "function": {
                "name": tool_call["name"],
                "arguments": json.dumps(tool_call["args"]),
            },
        }

    def _parse_tool_calls(self, message_dict: dict) -> Tuple[List[ToolCall], List[InvalidToolCall]]:
        """Parse tool calls from message."""
        tool_calls = []
        invalid_tool_calls = []
        
        if raw_tool_calls := message_dict.get("tool_calls"):
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append({
                        "id": raw_tool_call.get("id", ""),
                        "name": raw_tool_call["function"]["name"],
                        "args": json.loads(raw_tool_call["function"]["arguments"]),
                    })
                except Exception as e:
                    invalid_tool_calls.append({
                        "id": raw_tool_call.get("id", ""),
                        "name": raw_tool_call["function"].get("name", ""),
                        "args": raw_tool_call["function"].get("arguments", ""),
                        "error": str(e),
                    })
                    
        return tool_calls, invalid_tool_calls

    def _convert_dict_to_message(self, message_dict: dict) -> BaseMessage:
        """Convert API message to LangChain message."""
        role = message_dict.get("role", "")
        content = message_dict.get("content", "")
        name = message_dict.get("name")
        
        if role == "user":
            return HumanMessage(content=content, name=name)
        elif role == "assistant":
            tool_calls, invalid_tool_calls = self._parse_tool_calls(message_dict)
            additional_kwargs = {}
            
            if function_call := message_dict.get("function_call"):
                additional_kwargs["function_call"] = dict(function_call)
                
            if tool_calls:
                additional_kwargs["tool_calls"] = tool_calls
                
            return AIMessage(
                content=content,
                additional_kwargs=additional_kwargs,
                name=name,
                tool_calls=tool_calls,
                invalid_tool_calls=invalid_tool_calls,
            )
        elif role == "system":
            return SystemMessage(content=content, name=name)
        elif role == "function":
            return FunctionMessage(content=content, name=name)
        elif role == "tool":
            return ToolMessage(
                content=content,
                tool_call_id=message_dict.get("tool_call_id", ""),
                name=name,
            )
        else:
            return ChatMessage(content=content, role=role)
        
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for API call."""
        params = {
            "model": self.model_name,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
        }
        
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
            
        return params

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name, **self._default_params}
    
    
    
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, ChatMessage,
    FunctionMessage, HumanMessage, SystemMessage,
    ToolMessage
)
from langchain_core.tools import BaseTool

llm = ChatLlama(
    base_url="your-api-base-url",
    user_id="your-user-id",
    pwd="your-password"
)

def test_tool_calling():
    # Define multiple tools
    class Calculator(BaseModel):
        """Perform basic math operations."""
        operation: str = Field(description="Operation to perform (add/subtract/multiply/divide)")
        x: float = Field(description="First number")
        y: float = Field(description="Second number")

    class Translator(BaseModel):
        """Translate text to specified language."""
        text: str = Field(description="Text to translate")
        target_language: str = Field(description="Target language")

    # Initialize with multiple tools
    llm_with_tools = llm.bind_tools([Calculator, Translator])
    
    # Test message
    response = llm_with_tools.invoke(
        "Calculate 25 divided by 5 and translate the result to Spanish"
    )
    
    # Process tool calls
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"Tool: {tool_call.name}")
            print(f"Arguments: {tool_call.args}")


test_tool_calling()
