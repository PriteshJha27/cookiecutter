
def _create_chat_stream(
    self,
    messages: List[BaseMessage],
    stop: Optional[List[str]] = None,
    **kwargs: Any,
) -> Iterator[Union[Mapping[str, Any], str]]:
    chat_params = self._chat_params(messages, stop, **kwargs)
    
    # Create direct HTTP client instead of using ollama Client
    client = httpx.Client(
        verify=os.getenv('CERT_PATH'),
        timeout=30.0
    )
    
    if chat_params["stream"]:
        response = client.post(
            self.base_url,
            json=chat_params,
            headers=self._get_headers()
        )
        yield from response.iter_lines()
    else:
        response = client.post(
            self.base_url,
            json=chat_params,
            headers=self._get_headers()
        )
        yield response.json()

async def _acreate_chat_stream(
    self,
    messages: List[BaseMessage],
    stop: Optional[List[str]] = None,
    **kwargs: Any,
) -> AsyncIterator[Union[Mapping[str, Any], str]]:
    chat_params = self._chat_params(messages, stop, **kwargs)
    
    # Create direct async HTTP client
    async with httpx.AsyncClient(
        verify=os.getenv('CERT_PATH'),
        timeout=30.0
    ) as client:
        if chat_params["stream"]:
            async with client.stream(
                'POST',
                self.base_url,
                json=chat_params,
                headers=self._get_headers()
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        yield json.loads(line)
        else:
            response = await client.post(
                self.base_url,
                json=chat_params,
                headers=self._get_headers()
            )
            yield response.json()

  def _get_headers(self):
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Cookie": self.auth_token
    }
