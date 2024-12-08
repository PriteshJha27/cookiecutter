
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models import ChatOllama
import os
from dotenv import load_dotenv
import httpx
import urllib3
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class AmexLlamaChatModel(ChatOllama):
    """Custom ChatOllama for AEXP's hosted Llama model."""
    
    def __init__(self, **kwargs):
        # Load environment variables
        load_dotenv()
        
        # Set base configuration
        base_url = "https://aidageniservices-qa.aexp.com/app/v1/opensource/models/llama3-70b-instruct/"
        auth_url = "https://authbluesvcqa-vip.phx.aexp.com/sso1/signin/"
        model_name = "llama3-70b-instruct"
        
        # Get credentials from environment
        cert_path = os.getenv('CERT_PATH')
        user_id = os.getenv('LLAMA_USER_ID')
        pwd = os.getenv('LLAMA_PASSWORD')
        
        # Create SSL context
        verify = False  # Disable SSL verification
        
        # Get authentication token
        auth_token = self._get_auth_token(auth_url, user_id, pwd)
        
        # Set up headers for the client
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Cookie": auth_token
        }
        
        # Initialize parent class with our custom configuration
        super().__init__(
            base_url=base_url,
            model=model_name,
            client_kwargs={
                "headers": headers,
                "verify": verify,
                "timeout": 30.0
            },
            **kwargs
        )
    
    def _get_auth_token(self, auth_url: str, user_id: str, pwd: str) -> str:
        """Get authentication token from the auth service."""
        try:
            # Configure client with SSL verification disabled
            transport = httpx.HTTPTransport(verify=False)
            client = httpx.Client(transport=transport, timeout=30.0)
            
            print("Sending auth request...")
            response = client.post(
                auth_url,
                data={
                    "userid": user_id,
                    "pwd": pwd
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "*/*"
                }
            )
            
            print(f"Auth Response Status: {response.status_code}")
            
            if response.status_code != 200:
                raise ValueError(f"Authentication failed: {response.status_code} - {response.text}")
            
            auth_token = response.headers.get("Set-Cookie")
            if not auth_token:
                raise ValueError("No authentication token received")
            
            return auth_token
            
        except Exception as e:
            raise ValueError(f"Authentication failed: {str(e)}")
        finally:
            client.close()

def main():
    try:
        # Initialize the model
        llm = AmexLlamaChatModel(
            temperature=0.7,
        )
        
        # Example chat
        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="What is 2+2?")
        ]
        
        response = llm.invoke(messages)
        print(response)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
