
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type

# First, create the schema for the tool's input
class WeatherInput(BaseModel):
    location: str = Field(description="The city and state")
    date: str = Field(description="The date to get weather for")

class WeatherTool(BaseTool):
    name: str = "weather"
    description: str = "Get weather information for a specific location and date"
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, location: str, date: str) -> str:
        """Execute the weather lookup"""
        # Mock implementation
        return f"Weather forecast for {location} on {date}: Sunny, 72°F"
        
    async def _arun(self, location: str, date: str) -> str:
        """Execute the weather lookup asynchronously"""
        return self._run(location=location, date=date)

# Response format for structured output
class WeatherResponse(BaseModel):
    temperature: float = Field(description="Temperature in Fahrenheit")
    condition: str = Field(description="Weather condition description")
    location: str = Field(description="Location of the weather report")
    date: str = Field(description="Date of the weather report")

# Test script
async def test_weather_tool():
    # Initialize the LLM
    llm = ChatAmexLlama(
        base_url="your_base_url",
        auth_url="your_auth_url",
        cert_path="your_cert_path",
        user_id="your_user_id",
        pwd="your_password",
        model_name="llama3-70b-instruct"
    )
    
    # Create and bind the weather tool
    weather_tool = WeatherTool()
    
    try:
        # Bind the tool
        llm_with_tools = llm.bind_tools(
            tools=[weather_tool],
            tool_choice={"type": "function", "function": {"name": "weather"}}
        )
        
        # Test the tool
        response = await llm_with_tools.ainvoke(
            messages=[
                HumanMessage(content="What's the weather in New York today?")
            ]
        )
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_weather_tool())





from typing import Sequence, Union, Optional, Type, Dict, Any
from pydantic import BaseModel
from langchain.tools import BaseTool

class ChatAmexLlama(BaseChatModel):
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], BaseTool]],
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs: Any,
    ) -> "ChatAmexLlama":
        """Bind tools to the model with Pydantic v2 support."""
        formatted_tools = []
        
        for tool in tools:
            if isinstance(tool, dict):
                formatted_tools.append(tool)
            elif isinstance(tool, BaseTool):
                # Handle BaseTool instances
                tool_dict = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                    }
                }
                
                # Handle args schema if present
                if hasattr(tool, 'args_schema'):
                    schema = tool.args_schema.model_json_schema()
                    # Remove title and extra Pydantic metadata
                    if "title" in schema:
                        del schema["title"]
                    tool_dict["function"]["parameters"] = schema
                
                formatted_tools.append(tool_dict)
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")
        
        config = self._default_config.copy()
        config.update(kwargs)
        
        if tool_choice:
            if isinstance(tool_choice, str):
                config["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice}
                }
            else:
                config["tool_choice"] = tool_choice
        
        return self.__class__(
            **config,
            tools=formatted_tools,
            streaming=self.streaming,
            temperature=self.temperature,
            **kwargs
        )
