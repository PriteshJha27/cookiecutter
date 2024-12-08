
from langchain.agents import initialize agent, Tool
from langchain.chains import LLMMathChain

from Langchain.ttms import OpenAI

from langchain.tools import tool

import os

os.environ[“OPENAI_API_KEY"] = “xyz" # Set your OpenAI API key
lm = OpenAI (temperature=0)

@tool # Define the tools the agent can use

def sample_tool(input_text: str) -> str:
""“This is a sample custom tool."""
return f"Tool received: {input_text}"

math_tool = LLMMathChain(llm=l1m)

tools = [
Tool (
name="Calculator",
func=math_tool.run,
description="Performs mathematical calculations for numerical queries.”

),
Tool (

name="SampleTool",

func=sample tool,
"A simple example tool for processing text input."

# Initialize the Tools-Enabled Agent
agent = initialize_agent(
tools=tools,
Unelin,
agent="zero-shot-react-description", # Agent with tool integration
verbose=True
)

# Provide a query for the agent
query = "What is 25 times 3? Also, use the sample tool to process ‘Hello LangChain!'"

# Run the agent
response = agent. run(query)

# Print the response
print("Response:", response)

# Define a math tool the agent can use
math tool = LLMMathChain(Ulm=ULm)
tools = [
Tool (
name="Calculator",
func=math_tool.run,
description="Useful for performing mathematical calculations or solving equations."

]

# Initialize the Zero-Shot ReAct agent
agent = initialize_agent(
tools=tools,
UmeUn,
agent="zero-shot-react-description", # Specifies the ReAct-based agent
verbose=True
)

# Example query
query = “If I have 20 apples and give half to my friend, then add 10 more, how many do I have?"

# Run the agent
response = agent. run(query)

# Print the response
print("Response:", response)













  
