from langchain.tools import Tool
from dotenv import load_dotenv
from chat_amex_llama import ChatAmexLlama
from chat_amex_llama_agent import create_amex_agent
import os

# Load environment variables
load_dotenv()

def get_weather(location: str) -> str:
    """Get weather information for a location."""
    # Dummy implementation
    return f"The weather in {location} is sunny with a temperature of 22°C"

def search_database(query: str) -> str:
    """Search the database for information."""
    # Dummy implementation
    return f"Found relevant information for: {query}"

def test_agent():
    """Test the Amex agent with tools."""
    
    # Initialize LLM
    llm = ChatAmexLlama(
        base_url=os.getenv("LLAMA_API_URL"),
        auth_url=os.getenv("LLAMA_AUTH_URL"),
        user_id=os.getenv("LLAMA_USER_ID"),
        pwd=os.getenv("LLAMA_PASSWORD"),
        cert_path=os.getenv("CERT_PATH")
    )
    
    # Create tools
    tools = [
        Tool(
            name="WeatherInfo",
            func=get_weather,
            description="Get current weather information for a specific location"
        ),
        Tool(
            name="DatabaseSearch",
            func=search_database,
            description="Search the database for specific information"
        )
    ]
    
    # Create agent
    agent_executor = create_amex_agent(
        llm=llm,
        tools=tools,
        verbose=True  # Show the agent's thought process
    )
    
    # Test single tool
    try:
        print("\nTesting weather tool...")
        result1 = agent_executor.invoke({
            "input": "What's the weather like in New York?"
        })
        print("Result:", result1["output"])
        
        # Test multiple tools
        print("\nTesting multiple tools...")
        result2 = agent_executor.invoke({
            "input": "What's the weather in London and find information about climate change?"
        })
        print("Result:", result2["output"])
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent()
