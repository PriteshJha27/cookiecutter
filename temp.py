from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Step 1: Define your custom LLM
class CustomGCPModel(BaseLLM):
    def __init__(self, auth_token):
        self.auth_token = auth_token
        self.api_url = "YOUR_MODEL_API_URL"

    def generate(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }
        data = {
            "prompt": prompt,
            "max_tokens": 100
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        return response.json().get("generated_text")

# Step 2: Define custom tools
@tool
def add(a: int, b: int) -> int:
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    return a * b

# Step 3: Create the agent
auth_token = "YOUR_GENERATED_AUTH_TOKEN"
custom_llm = CustomGCPModel(auth_token)

# Create a list of tools
tools = [add, multiply]

# Create the agent
agent = create_tool_calling_agent(custom_llm, tools)

# Step 4: Execute the agent
agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.invoke({"input": "What is 2 + 3?"})
print(result)
