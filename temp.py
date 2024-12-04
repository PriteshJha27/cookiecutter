def _stream(
    self,
    messages: List[BaseMessage],
    stop: Optional[List[str]] = None,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Iterator[ChatGenerationChunk]:
    """Stream chat completion results."""
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
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
            
            # Debug information
            print(f"Attempt {attempt + 1}/{max_retries}")
            print(f"URL: {api_url}")
            print(f"Payload: {json.dumps(payload, indent=2)}")
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Cookie": auth_token
            }

            # Try non-streaming first to validate endpoint
            test_response = self._client.post(
                url=api_url,
                headers=headers,
                json=payload,
                timeout=30.0
            )

            # If we get here and status is not 200, raise error with details
            if test_response.status_code != 200:
                error_detail = ""
                try:
                    error_detail = test_response.json()
                except:
                    error_detail = test_response.text
                raise ValueError(
                    f"API call failed (Status: {test_response.status_code}): {error_detail}"
                )

            # If we get here, endpoint is working, now try streaming
            with self._client.stream(
                "POST",
                url=api_url,
                headers=headers,
                json=payload,
                timeout=30.0
            ) as response:
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            line_text = line_text[6:]
                        
                        if not line_text.strip():
                            continue
                            
                        chunk_data = json.loads(line_text)
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
                                
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error on line: {line_text}")
                        print(f"Error: {str(e)}")
                        continue
                    except Exception as e:
                        print(f"Error processing line: {str(e)}")
                        continue

            # If we get here successfully, break the retry loop
            break
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise ValueError(f"Streaming failed after {max_retries} attempts: {str(e)}")
            else:
                import time
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
