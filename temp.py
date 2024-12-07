import requests
from typing import List, Optional, Iterable, Union
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    LLMResult,
)
from langchain.chat_models.base import BaseChatModel


class ChatOllama(BaseChatModel):
    """
    A LangChain-compatible Chat Model wrapper for a custom LLaMA endpoint.
    """

    def __init__(
        self,
        endpoint_url: str,
        auth_token: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.endpoint_url = endpoint_url
        self.auth_token = auth_token
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def _llm_type(self) -> str:
        return "ollama_custom"

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        # Convert LangChain messages to a prompt format expected by your API.
        # Adjust this function based on your model’s prompt format requirements.
        # For example, if your endpoint expects:
        # [ { "role": "system", "content": "..." }, { "role": "user", "content": "..." } ]
        # you might serialize them as JSON or just concatenate them.
        
        # Here we assume the API accepts a simple concatenation of system and user messages:
        prompt_parts = []
        for m in messages:
            if m.type == "system":
                prompt_parts.append(f"[SYSTEM]: {m.content}\n")
            elif m.type == "human":
                prompt_parts.append(f"[USER]: {m.content}\n")
            elif m.type == "ai":
                prompt_parts.append(f"[ASSISTANT]: {m.content}\n")
            else:
                prompt_parts.append(m.content + "\n")
        return "".join(prompt_parts)

    def _call_api(self, prompt: str, stream: bool = False) -> Union[str, Iterable[str]]:
        """
        Call the custom LLaMA endpoint. If stream=True, return an iterator for streaming responses.
        Otherwise, return the full string response.
        """
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # Additional parameters as required by your endpoint
        }

        response = requests.post(
            self.endpoint_url,
            headers=headers,
            json=payload,
            stream=stream
        )

        response.raise_for_status()

        if stream:
            # Assuming the endpoint sends partial responses line-by-line or in chunks
            for chunk in response.iter_lines(decode_unicode=True):
                if chunk.strip():
                    yield chunk
        else:
            # Full response
            return response.json().get("completion", "")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        """
        Synchronous call for a one-shot generation.
        """
        prompt = self._convert_messages_to_prompt(messages)
        response_text = self._call_api(prompt, stream=False)

        # If stop sequences are provided, handle them:
        if stop:
            for s in stop:
                if s in response_text:
                    response_text = response_text.split(s)[0]

        # Return a ChatResult with one ChatGeneration
        generation = ChatGeneration(message=AIMessage(content=response_text))
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Iterable[AIMessage]:
        """
        Streaming generation. Yields AIMessage chunks as they come.
        """
        prompt = self._convert_messages_to_prompt(messages)

        current_text = ""
        for chunk in self._call_api(prompt, stream=True):
            current_text += chunk
            # If we detect stop sequences, break early
            if stop and any(s in current_text for s in stop):
                # Cut off at the stop sequence
                for s in stop:
                    if s in current_text:
                        current_text = current_text.split(s)[0]
                yield AIMessage(content=current_text)
                return
            # Yield the incremental updates
            yield AIMessage(content=chunk)

    def invoke(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> AIMessage:
        """
        A convenient wrapper to just get a single AIMessage result synchronously.
        """
        result = self._generate(messages, stop=stop)
        return result.generations[0].message

    def stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> Iterable[AIMessage]:
        """
        A convenient wrapper to get an iterator over AIMessage results.
        """
        return self._stream(messages, stop=stop)


from langchain.schema import SystemMessage, HumanMessage

# Initialize your custom Chat Model
chat = ChatOllama(
    endpoint_url="https://your-gcp-llama-endpoint.com/v1/generate",
    auth_token="your_blue_auth_token_here",
    model_name="llama-3.1-70b"
)

# Invoke a synchronous single-turn completion
response = chat.invoke([SystemMessage(content="You are a helpful assistant."), 
                        HumanMessage(content="Tell me something interesting about black holes.")])
print(response.content)

# Streaming responses
for msg in chat.stream([SystemMessage(content="You are a helpful assistant."), 
                        HumanMessage(content="Explain the concept of quantum entanglement")]):
    print(msg.content, end="", flush=True)
