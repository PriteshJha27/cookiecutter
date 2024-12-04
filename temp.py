import logging
from typing import AsyncIterator, Dict, Any
from httpx import AsyncClient, Response, TransportError
import json
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class BaseChatAmexLlama(BaseChatModel):
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Enhanced streaming implementation with proper error handling."""
        try:
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

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Cookie": auth_token
            }

            async with self._async_client.stream(
                method="POST",
                url=api_url,
                json=payload,
                headers=headers,
                timeout=30.0
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    logger.error(f"Streaming request failed: {response.status_code} - {error_body}")
                    raise LlamaResponseError(
                        f"Streaming request failed: {response.status_code} - {error_body.decode()}"
                    )

                # Process the streaming response
                async for line in response.aiter_lines():
                    if not line or line.isspace():
                        continue

                    if line.startswith('data: '):
                        line = line[6:]  # Remove 'data: ' prefix
                        
                    try:
                        chunk_data = json.loads(line)
                        if chunk_data.get("choices"):
                            delta = chunk_data["choices"][0].get("delta", {})
                            if content := delta.get("content"):
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(content=content),
                                    generation_info={"finish_reason": None}
                                )
                                
                                if run_manager:
                                    await run_manager.on_llm_new_token(content)
                                    
                                yield chunk
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to decode chunk: {line} - Error: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}")
                        continue

        except Exception as e:
            logger.error(f"Streaming failed: {str(e)}")
            raise LlamaResponseError(f"Streaming failed: {str(e)}")




