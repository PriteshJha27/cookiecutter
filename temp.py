
def _stream(
    self,
    messages: List[BaseMessage],
    stop: Optional[List[str]] = None,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Iterator[ChatGenerationChunk]:
    """Stream chat completion results."""
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
    
    if self.response_format:
        payload["response_format"] = self.response_format

    # Changed this part to handle streaming correctly
    with self._client.stream('POST',  # Use stream method directly
        api_url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Cookie": auth_token
        }
    ) as response:
        if response.status_code != 200:
            raise ValueError(f"API call failed: {response.status_code} - {response.text}")
        
        for line in response.iter_lines():
            if not line:
                continue
            
            try:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    line_text = line_text[6:]
                
                if line_text.strip():
                    chunk_data = json.loads(line_text)
                    if chunk_data.get("choices"):
                        delta = chunk_data["choices"][0].get("delta", {})
                        if content := delta.get("content"):
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(content=content),
                                generation_info={"finish_reason": None}
                            )
                            yield chunk
                            
                            if run_manager:
                                run_manager.on_llm_new_token(content)
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
