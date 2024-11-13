from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# First, ensure httpx is installed
try:
    import httpx
except ImportError:
    print("Installing httpx...")
    import subprocess
    subprocess.check_call(["pip", "install", "httpx"])
    import httpx

from chat_amex_llama import ChatAmexLlama

# Load environment variables
load_dotenv()

def test_basic_chain():
    """Test basic LLM functionality."""
    
    # Get and verify environment variables
    base_url = os.getenv("LLAMA_API_URL")
    auth_url = os.getenv("LLAMA_AUTH_URL")
    user_id = os.getenv("LLAMA_USER_ID")
    pwd = os.getenv("LLAMA_PASSWORD")
    cert_path = os.getenv("CERT_PATH")

    print("Environment Variables:")
    print(f"Base URL: {base_url}")
    print(f"Auth URL: {auth_url}")
    print(f"Cert Path: {cert_path}")
    print(f"User ID: {user_id}")
    print("Password: [HIDDEN]")

    # Initialize LLM
    llm = ChatAmexLlama(
        base_url=base_url,
        auth_url=auth_url,
        user_id=user_id,
        pwd=pwd,
        cert_path=cert_path
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
        print("\nTesting chain...")
        result = chain.invoke({"input": "What is machine learning?"})
        print("\nSuccess! Result:", result)
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_chain()
