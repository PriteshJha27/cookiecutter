import logging
from typing import AsyncIterator, Dict, Any
import json
import httpx

logger = logging.getLogger(__name__)
# Set logging level to DEBUG for development
logger.setLevel(logging.DEBUG)

class BaseChatAmexLlama(BaseChatModel):
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Enhanced streaming implementation with detailed debugging."""
        try:
            # Log authentication status
            auth_token = await self._get_auth_token_async()
            logger.debug(f"Authentication successful, token received")
            
            api_url = f"{self.base_url.rstrip('/')}/chat/completions"
            logger.debug(f"Making streaming request to: {api_url}")
            
            messages_dict = [self._convert_message_to_dict(m) for m in messages]
            
            # Construct payload with all required parameters
            payload = {
                "messages": messages_dict,
                "model": self.model_name,
                "stream": True,
                "temperature": self.temperature,
                **kwargs
            }
            logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")

            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",  # Changed to handle SSE properly
                "Cookie": auth_token
            }
            logger.debug(f"Request headers: {headers}")

            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                try:
                    async with client.stream(
                        "POST",
                        api_url,
                        json=payload,
                        headers=headers
                    ) as response:
                        logger.debug(f"Initial response status: {response.status_code}")
                        
                        if response.status_code != 200:
                            error_body = await response.aread()
                            error_text = error_body.decode('utf-8')
                            logger.error(f"Streaming request failed - Status: {response.status_code}, Response: {error_text}")
                            raise ValueError(f"Streaming request failed: {response.status_code} - {error_text}")

                        async for line in response.aiter_lines():
                            logger.debug(f"Received line: {line}")
                            
                            if not line or line.isspace():
                                continue

                            # Handle SSE format
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
                                        
                                        if run_manager:
                                            await run_manager.on_llm_new_token(content)
                                        
                                        yield chunk
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to decode chunk: {line} - Error: {str(e)}")
                                continue
                            
                except httpx.RequestError as e:
                    logger.error(f"Request failed: {str(e)}")
                    raise

        except Exception as e:
            logger.error(f"Streaming failed: {str(e)}", exc_info=True)
            raise ValueError(f"Streaming failed: {str(e)}")
