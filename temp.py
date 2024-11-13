from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import Tool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def get_weather(location: str) -> str:
    """Dummy weather tool."""
    return f"The weather in {location} is sunny and 22°C"

def main():
    # Initialize LLM
    llm = ChatAmexLlama(
        base_url=os.getenv("LLAMA_API_URL"),
        auth_url=os.getenv("LLAMA_AUTH_URL"),
        user_id=os.getenv("LLAMA_USER_ID"),
        pwd=os.getenv("LLAMA_PASSWORD")
    )

    # Basic LCEL chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # Test basic chain
    result = chain.invoke({"input": "What is machine learning?"})
    print("Basic Chain Result:", result)

    # Test with tools
    tools = [
        Tool(
            name="WeatherInfo",
            func=get_weather,
            description="Get weather information for a location"
        )
    ]

    llm_with_tools = ChatAmexLlamaWithTools(
        base_url=os.getenv("LLAMA_API_URL"),
        auth_url=os.getenv("LLAMA_AUTH_URL"),
        user_id=os.getenv("LLAMA_USER_ID"),
        pwd=os.getenv("LLAMA_PASSWORD")
    )
    llm_with_tools.bind_tools(tools)

    tool_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the available tools to answer the question."),
        ("human", "{input}")
    ])
    
    tool_chain = tool_prompt | llm_with_tools | StrOutputParser()
    
    # Test tool chain
    result = tool_chain.invoke({"input": "What's the weather in New York?"})
    print("\nTool Chain Result:", result)

if __name__ == "__main__":
    main()
