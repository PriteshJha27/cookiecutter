def _generate(
    self,
    messages: List[BaseMessage],
    stop: Optional[List[str]] = None,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> ChatResult:
    """Generate chat completion results."""
    auth_token = self.get_auth_token()
    api_url = f"{self.base_url.rstrip('/')}/chat/completions"
    
    messages_dict = [self._convert_message_to_dict(m) for m in messages]
    payload = {
        "messages": messages_dict,
        "model": self.model_name,
        "stream": False,  # Ensure streaming is False for regular generation
        "temperature": self.temperature,
        **kwargs
    }
    
    try:
        response = self._client.post(
            api_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Cookie": auth_token
            }
        )
        
        if response.status_code != 200:
            error_msg = f"API call failed: {response.status_code}"
            try:
                error_msg += f" - {response.json()}"
            except:
                error_msg += f" - {response.text}"
            raise ValueError(error_msg)
        
        response_data = response.json()
        
        # Validate response structure
        if "choices" not in response_data or not response_data["choices"]:
            raise ValueError("Invalid response format: missing or empty choices")
            
        if "message" not in response_data["choices"][0]:
            raise ValueError("Invalid response format: missing message in choice")
            
        content = response_data["choices"][0]["message"].get("content", "")
        message = AIMessage(content=content)
        generations = [ChatGeneration(message=message)]
        
        return ChatResult(generations=generations)
        
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode API response: {response.text}")
    except Exception as e:
        raise ValueError(f"Error during API call: {str(e)}")


def invoke(
    self,
    input: Union[str, List[BaseMessage]],
    config: Optional[dict] = None,
    **kwargs: Any
) -> Union[BaseMessage, ChatResult]:
    """Invoke the chat model."""
    try:
        messages = self._convert_input_to_messages(input)
        result = self._generate(messages, **kwargs)
        
        if not result.generations:
            raise ValueError("No generations returned from model")
            
        # Return just the message if single generation
        return result.generations[0].message if len(result.generations) == 1 else result
        
    except Exception as e:
        raise ValueError(f"Error during invocation: {str(e)}")
