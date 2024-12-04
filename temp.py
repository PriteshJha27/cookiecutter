# test.py
from pydantic import BaseModel, Field

class WeatherTool(BaseTool):
    name = "weather"
    description = "Get weather information"
    
    def _run(self, location: str, date: str) -> str:
        return f"Weather for {location} on {date}: Sunny"
        
    async def _arun(self, location: str, date: str) -> str:
        return self._run(location=location, date=date)

# Use the tool
llm_with_tools = llm.bind_tools([WeatherTool()])
response = await llm_with_tools.invoke("What's the weather in NYC today?")
print(response)
