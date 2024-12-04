async def _validate_streaming_support(self) -> bool:
    """Validate if the API endpoint supports streaming."""
    try:
        # Make a small test request
        test_messages = [{"role": "user", "content": "test"}]
        payload = {
            "messages": test_messages,
            "model": self.model_name,
            "stream": True,
            "temperature": 0.1,
            "max_tokens": 5  # Minimal response for testing
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cookie": await self._get_auth_token_async()
        }
        
        logger.debug("Making test request to validate streaming support")
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers
            )
            
            logger.debug(f"Validation response status: {response.status_code}")
            logger.debug(f"Validation response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                return True
            else:
                logger.error(f"API does not support streaming. Status: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"Failed to validate streaming support: {str(e)}")
        return False
