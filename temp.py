from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser
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
    llm = ChatAmexLlama()
    
    # Create a simple prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ])
    
    # Create chain
    chain = prompt | llm | StrOutputParser()
    
    # Test chain
    try:
        result = chain.invoke({
            "input": "Tell me a short joke."
        })
        print("Basic Chain Result:", result)
        
        # Test with weather tool
        tools = [
            Tool(
                name="WeatherInfo",
                func=get_weather,
                description="Get weather information for a location"
            )
        ]
        
        # Create and test tool chain
        from chat_amex_llama_with_bindtools import ChatAmexLlamaWithTools
        
        llm_with_tools = ChatAmexLlamaWithTools()
        llm_with_tools.bind_tools(tools)
        
        tool_prompt = ChatPromptTemplate.from_messages([
            ("system", "Use the available tools to answer the question."),
            ("human", "{input}")
        ])
        
        tool_chain = tool_prompt | llm_with_tools | StrOutputParser()
        
        tool_result = tool_chain.invoke({
            "input": "What's the weather in New York?"
        })
        print("\nTool Chain Result:", tool_result)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
