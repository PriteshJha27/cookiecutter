
from typing import Any, Dict, Iterator, List, Optional, Union, Sequence, Literal, Type, AsyncIterator
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_core.messages import (
    BaseMessage, AIMessage, HumanMessage, SystemMessage,
    AIMessageChunk, BaseMessageChunk
)
from langchain_core.outputs import Generation, LLMResult, ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.tools import BaseTool
from pydantic import ConfigDict
import httpx
import json
import os
from dotenv import load_dotenv
import asyncio
import warnings

load_dotenv()

class BaseChatAmexLlama(BaseChatModel):
    """Base Chat wrapper for Llama API with async support."""
    
    base_url: str = "https://sidegenieservices-qa.aexp.com/app/v1/opensource/models/llama3-70b-instruct/"
    auth_url: str = "https://antiblauevcqa-v1p.phx.amex.com/fcol/signin/"
    cert_path: str = os.getenv('CERT_PATH')
    user_id: str = os.getenv('LLAMA_USER_ID')
    pwd: str = os.getenv('LLAMA_PASSWORD')
    model_name: str = "llama3-70b-instruct"
    
    # Model Parameters
    temperature: float = 0.7
    streaming: bool = False
    response_format: Optional[dict] = None
    
    _auth_token: Optional[str] = None
    _client: Optional[httpx.Client] = None
    _async_client: Optional[httpx.AsyncClient] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = httpx.Client(
            verify=False,
            timeout=30.0
        )
        self._async_client = httpx.AsyncClient(
            verify=False,
            timeout=30.0
        )

    async def _get_auth_token_async(self) -> str:
        """Get authentication token asynchronously."""
        if self._auth_token:
            return self._auth_token
            
        try:
            print("Sending async auth request...")
            response = await self._async_client.post(
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
            print(f"Async Auth Response Status: {response.status_code}")
            
            if response.status_code != 200:
                raise ValueError(f"Async authentication failed: {response.status_code} - {response.text}")
                
            self._auth_token = response.headers.get("Set-Cookie")
            if not self._auth_token:
                raise ValueError("No authentication token received in async call")
                
            return self._auth_token
            
        except Exception as e:
            raise ValueError(f"Async authentication failed: {str(e)}")

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream chat completion results."""
        auth_token = await self._get_auth_token_async()
        api_url = f"{self.base_url.rstrip('/')}/chat/completions"
        
        messages_dict = [self._convert_message_to_dict(m) for m in messages]
        
        payload = {
            "messages": messages_dict,
            "model": self.model_name,
            "stream": True,
            "temperature": self.temperature,
            **kwargs
        }
        
        if self.response_format:
            payload["response_format"] = self.response_format
        
        async with self._async_client.stream(
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
                raise ValueError(f"Async API call failed: {response.status_code}")
            
            async for line in response.aiter_lines():
                if not line:
                    continue
                    
                if line.startswith('data: '):
                    line = line[6:]
                
                try:
                    chunk_data = json.loads(line)
                    if chunk_data.get("choices"):
                        delta = chunk_data["choices"][0].get("delta", {})
                        if content := delta.get("content"):
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(content=content),
                                generation_info={"finish_reason": None}
                            )
                            yield chunk
                            
                            if run_manager:
                                await run_manager.on_llm_new_token(content)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing async chunk: {str(e)}")
                    continue

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion results asynchronously."""
        if self.streaming:
            chunks: List[ChatGenerationChunk] = []
            async for chunk in self._astream(messages, stop, run_manager, **kwargs):
                chunks.append(chunk)
            return self._combine_chat_chunks(chunks)
            
        auth_token = await self._get_auth_token_async()
        api_url = f"{self.base_url.rstrip('/')}/chat/completions"
        
        messages_dict = [self._convert_message_to_dict(m) for m in messages]
        payload = {
            "messages": messages_dict,
            "model": self.model_name,
            "stream": False,
            "temperature": self.temperature,
            **kwargs
        }
        
        if self.response_format:
            payload["response_format"] = self.response_format
        
        response = await self._async_client.post(
            api_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Cookie": auth_token
            }
        )
        
        if response.status_code != 200:
            raise ValueError(f"Async API call failed: {response.status_code} - {response.text}")
        
        response_data = response.json()
        message = AIMessage(content=response_data["choices"][0]["message"]["content"])
        generations = [ChatGeneration(message=message)]
        
        return ChatResult(generations=generations)

    def _combine_chat_chunks(self, chunks: List[ChatGenerationChunk]) -> ChatResult:
        """Combine chat chunks into a single result."""
        if not chunks:
            return ChatResult(generations=[])
            
        messages: List[ChatGeneration] = []
        current_message = ""
        
        for chunk in chunks:
            current_message += chunk.message.content
            
        messages.append(
            ChatGeneration(
                message=AIMessage(content=current_message),
                generation_info=chunks[-1].generation_info
            )
        )
        
        return ChatResult(generations=messages)

    
    def format_tool_call(
        self,
        name: str,
        arguments: str,
        tool_call_id: Optional[str] = None,
    ) -> dict:
        """Format tool call for Llama API."""
        return {
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments
            },
            "id": tool_call_id or f"call_{name}"
        }

    def with_structured_output(
        self,
        schema: Union[Dict[str, Any], Type[BaseModel]],
        *,
        method: Literal["json", "function"] = "json",
        include_raw: bool = False,
        **kwargs: Any,
    ):
        """Configure the model for structured output."""
        if method == "json":
            if isinstance(schema, dict):
                response_format = {"type": "json_object", "schema": schema}
            else:
                response_format = {
                    "type": "json_object",
                    "schema": self._pydantic_to_json_schema(schema)
                }
            llm = self.bind(response_format=response_format)
        else:
            if isinstance(schema, dict):
                tool = schema
            else:
                tool = self._pydantic_to_tool(schema)
            llm = self.bind_tools([tool], tool_choice={"type": "function", "function": {"name": tool["name"]}})
            
        return llm

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], BaseTool]],
        *,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs: Any,
    ):
        """Bind tools to the model."""
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                formatted_tools.append(tool)
            elif isinstance(tool, type) and issubclass(tool, BaseModel):
                formatted_tools.append(self._pydantic_to_tool(tool))
            elif isinstance(tool, BaseTool):
                formatted_tools.append(self._langchain_tool_to_dict(tool))
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")
                
        tool_args = {"tools": formatted_tools}
        if tool_choice:
            if isinstance(tool_choice, str):
                tool_args["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice}
                }
            else:
                tool_args["tool_choice"] = tool_choice
                
        return self.bind(**tool_args, **kwargs)

    def _pydantic_to_json_schema(self, model: Type[BaseModel]) -> dict:
        """Convert Pydantic model to JSON schema."""
        schema = model.schema()
        if "title" in schema:
            del schema["title"]
        return schema

    def _pydantic_to_tool(self, model: Type[BaseModel]) -> dict:
        """Convert Pydantic model to tool format."""
        schema = self._pydantic_to_json_schema(model)
        return {
            "type": "function",
            "function": {
                "name": model.__name__,
                "description": model.__doc__ or "",
                "parameters": schema
            }
        }

    def _langchain_tool_to_dict(self, tool: BaseTool) -> dict:
        """Convert LangChain tool to dictionary format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema.schema() if tool.args_schema else {"type": "object", "properties": {}}
            }
        }

    def _create_chat_result(
        self, 
        response_data: dict,
        tool_calls: Optional[List[dict]] = None
    ) -> ChatResult:
        """Create chat result with tool calls if present."""
        content = response_data["choices"][0]["message"].get("content", "")
        additional_kwargs = {}
        
        if tool_calls:
            additional_kwargs["tool_calls"] = tool_calls
            
        message = AIMessage(
            content=content,
            additional_kwargs=additional_kwargs
        )
        
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _aprocess_tool_calls(
        self,
        messages: List[BaseMessage],
        tools: List[dict],
        **kwargs: Any
    ) -> ChatResult:
        """Process tool calls asynchronously."""
        response = await self._agenerate(
            messages=messages,
            tools=tools,
            **kwargs
        )
        
        if not response.generations:
            return response
            
        message = response.generations[0].message
        if not message.additional_kwargs.get("tool_calls"):
            return response
            
        # Process tool calls and create new response
        tool_calls = message.additional_kwargs["tool_calls"]
        return self._create_chat_result(
            {"choices": [{"message": {"content": message.content}}]},
            tool_calls
        )

    def _combine_tools_with_messages(
        self,
        messages: List[BaseMessage],
        tools: List[dict]
    ) -> List[BaseMessage]:
        """Combine tools with messages for context."""
        tool_system_message = SystemMessage(
            content=f"You have access to the following tools: {json.dumps(tools, indent=2)}"
        )
        return [tool_system_message] + messages

    
    stream_usage: bool = False
    max_retries: int = 2
    
    def bind(
        self,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> "ChatAmexLlama":
        """Create a new instance with updated parameters."""
        config = self._default_config.copy()
        config.update(kwargs)
        
        return self.__class__(**{
            **config,
            "streaming": self.streaming,
            "temperature": self.temperature,
            **kwargs
        })

    def invoke(
        self,
        input: Union[str, List[BaseMessage]],
        config: Optional[dict] = None,
        **kwargs: Any
    ) -> Union[BaseMessage, ChatResult]:
        """Invoke the chat model."""
        messages = self._convert_input_to_messages(input)
        result = self._generate(messages, **kwargs)
        return result.generations[0].message if len(result.generations) == 1 else result

    async def ainvoke(
        self,
        input: Union[str, List[BaseMessage]],
        config: Optional[dict] = None,
        **kwargs: Any
    ) -> Union[BaseMessage, ChatResult]:
        """Invoke the chat model asynchronously."""
        messages = self._convert_input_to_messages(input)
        result = await self._agenerate(messages, **kwargs)
        return result.generations[0].message if len(result.generations) == 1 else result

    def stream(
        self,
        input: Union[str, List[BaseMessage]],
        config: Optional[dict] = None,
        **kwargs: Any
    ) -> Iterator[BaseMessageChunk]:
        """Stream the chat model response."""
        messages = self._convert_input_to_messages(input)
        for chunk in self._stream(messages, **kwargs):
            yield chunk.message

    async def astream(
        self,
        input: Union[str, List[BaseMessage]],
        config: Optional[dict] = None,
        **kwargs: Any
    ) -> AsyncIterator[BaseMessageChunk]:
        """Stream the chat model response asynchronously."""
        messages = self._convert_input_to_messages(input)
        async for chunk in self._astream(messages, **kwargs):
            yield chunk.message

    def _convert_input_to_messages(
        self,
        input: Union[str, List[BaseMessage]]
    ) -> List[BaseMessage]:
        """Convert input to messages format."""
        if isinstance(input, str):
            return [HumanMessage(content=input)]
        return input

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "streaming": self.streaming,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {
            **super()._identifying_params,
            "model_name": self.model_name,
            "temperature": self.temperature,
        }

    def get_num_tokens(self, text: str) -> int:
        """Get number of tokens in text."""
        # Implement token counting logic here
        # This is a simplified version
        return len(text.split())

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Get number of tokens in messages."""
        return sum(self.get_num_tokens(str(m.content)) for m in messages)

    def _validate_response(self, response: dict) -> None:
        """Validate API response."""
        if "error" in response:
            raise ValueError(f"API Error: {response['error']}")
        if "choices" not in response:
            raise ValueError("Invalid response format: missing 'choices'")

    def _handle_response_format(self, response: dict) -> dict:
        """Handle different response formats."""
        if self.response_format and self.response_format.get("type") == "json_object":
            try:
                content = response["choices"][0]["message"]["content"]
                json_content = json.loads(content)
                response["choices"][0]["message"]["content"] = json_content
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON response")
        return response

    async def __aenter__(self):
        """Async context manager entry."""
        if not self._async_client:
            self._async_client = httpx.AsyncClient(
                verify=False,
                timeout=30.0
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._client:
            self._client.close()
            self._client = None

# Optional: Add error classes for better error handling
class LlamaAPIError(Exception):
    """Base exception for Llama API errors."""
    pass

class LlamaAuthError(LlamaAPIError):
    """Exception for authentication errors."""
    pass

class LlamaResponseError(LlamaAPIError):
    """Exception for response parsing errors."""
    pass

# Example usage:
"""
llm = ChatAmexLlama(
    temperature=0.7,
    streaming=True
)

# Basic usage
response = llm.invoke("Tell me a joke")

# Streaming
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="")

# With tools
from pydantic import BaseModel, Field

class WeatherTool(BaseModel):
    location: str = Field(description="The city and state")
    date: str = Field(description="The date to get weather for")

llm_with_tools = llm.bind_tools([WeatherTool])
response = llm_with_tools.invoke("What's the weather in NYC today?")

# Structured output
class ResponseFormat(BaseModel):
    answer: str
    confidence: float

structured_llm = llm.with_structured_output(ResponseFormat)
response = structured_llm.invoke("What is 2+2?")
"""


