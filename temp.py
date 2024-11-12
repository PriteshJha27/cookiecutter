from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

async def test_llama_chain():
    """Test the basic LCEL chain with ChatAmexLlama."""
    
    # Initialize LLM
    llm = ChatAmexLlama()
    
    # Create a simple prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{text}")
    ])
    
    # Create chain
    chain = prompt | llm | StrOutputParser()
    
    # Test synchronous call
    print("Testing synchronous call...")
    result_sync = chain.invoke({"text": "Tell me a short joke."})
    print(f"Sync Result: {result_sync}\n")
    
    # Test asynchronous call
    print("Testing asynchronous call...")
    result_async = await chain.ainvoke({"text": "Tell me a short joke."})
    print(f"Async Result: {result_async}")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_llama_chain())
