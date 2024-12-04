def _stream(
    self,
    messages: List[BaseMessage],
    stop: Optional[List[str]] = None,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Iterator[ChatGenerationChunk]:
    """Stream chat completion results."""
    try:
        auth_token = self.get_auth_token()
        api_url = f"{self.base_url.rstrip('/')}/chat/completions"
        
        messages_dict = [self._convert_message_to_dict(m) for m in messages]
        
        payload = {
            "messages": messages_dict,
            "model": self.model_name,
            "stream": True,
            "temperature": self.temperature,
            **kwargs
        }
        
        # Important: Use requests directly for streaming
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Cookie": auth_token
        }
        
        # Make streaming request
        response = self._client.request(
            method="POST",
            url=api_url,
            headers=headers,
            json=payload,
            stream=True  # This is the key part
        )
        
        # Check response status immediately
        if response.status_code != 200:
            raise ValueError(f"API call failed: {response.status_code} - {response.text}")

        # Process the streaming response
        for raw_chunk in response.iter_raw():
            if not raw_chunk:
                continue
            
            try:
                # Decode chunk
                chunk_text = raw_chunk.decode('utf-8')
                if chunk_text.startswith('data: '):
                    chunk_text = chunk_text[6:]
                
                # Skip empty chunks
                if not chunk_text.strip():
                    continue
                    
                # Parse JSON
                chunk_data = json.loads(chunk_text)
                if chunk_data.get("choices"):
                    delta = chunk_data["choices"][0].get("delta", {})
                    if content := delta.get("content"):
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=content),
                            generation_info={"finish_reason": None}
                        )
                        
                        if run_manager:
                            run_manager.on_llm_new_token(content)
                            
                        yield chunk
                        
            except json.JSONDecodeError:
                print(f"Failed to decode JSON from chunk: {chunk_text}")
                continue
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
                
    except Exception as e:
        raise ValueError(f"Streaming failed: {str(e)}")


def stream(
    self,
    input: Union[str, List[BaseMessage]],
    config: Optional[dict] = None,
    **kwargs: Any
) -> Iterator[BaseMessageChunk]:
    """Stream the chat model response."""
    try:
        messages = self._convert_input_to_messages(input)
        for chunk in self._stream(messages, **kwargs):
            if chunk.message:
                yield chunk.message
    except Exception as e:
        raise ValueError(f"Streaming failed: {str(e)}")
