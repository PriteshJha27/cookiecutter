from typing import Any, List, Mapping, Optional, Dict, AsyncGenerator
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import Field, root_validator
import httpx
import os
import json
from datetime import datetime, timedelta
import asyncio

class ChatAmexLlama(LLM):
    """Enhanced LLM wrapper for Llama API with improved auth and streaming."""
    
    # ... [previous attributes remain the same] ...

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Synchronous call to the LLM.
        
        Args:
            prompt: The input prompt
            stop: Optional stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments
            
        Returns:
            Generated text
        """
        # Use asyncio to run the async call in a sync context
        return asyncio.run(self._acall(prompt, stop, run_manager, **kwargs))

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Async call to generate complete response.
        """
        auth_token = await self._get_auth_token()
        await self._init_client()

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": self.model_name,
            "stream": False  # Set to False for non-streaming response
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Cookie": auth_token
        }

        response = await self._http_client.post(
            self.llama_uri,
            json=payload,
            headers=headers
        )

        if response.status_code != 200:
            raise Exception(f"Llama API call failed: {response.status_code} {response.text}")

        try:
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        except Exception as e:
            raise Exception(f"Error parsing response: {str(e)}")

    # ... [rest of the previous methods remain the same] ...
