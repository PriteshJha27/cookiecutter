from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from chat_amex_llama import ChatAmexLlama

# Load environment variables
load_dotenv()

def test_basic_chain():
    """Test basic LLM functionality."""
    
    # Initialize LLM
    llm = ChatAmexLlama(
        base_url=os.getenv("LLAMA_API_URL", ""),
        auth_url=os.getenv("LLAMA_AUTH_URL", ""),
        user_id=os.getenv("LLAMA_USER_ID", ""),
        pwd=os.getenv("LLAMA_PASSWORD", "")
    )
    
    # Create a simple prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ])
    
    # Create chain
    chain = prompt | llm | StrOutputParser()
    
    # Test
    try:
        result = chain.invoke({"input": "What is machine learning?"})
        print("Success! Result:", result)
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_basic_chain()
