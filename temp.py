
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel
import functools
import operator
from langgraph.graph import END, StateGraph, START
from chat_amex_llama import ChatAmexLlama
from chat_amex_llama_agent import create_amex_agent

# Load environment variables
load_dotenv()

# Initialize ChatAmexLlama
llm = ChatAmexLlama(
    base_url=os.getenv("LLAMA_API_URL"),
    auth_url=os.getenv("LLAMA_AUTH_URL"),
    user_id=os.getenv("LLAMA_USER_ID"),
    pwd=os.getenv("LLAMA_PASSWORD"),
    cert_path=os.getenv("CERT_PATH")
)

# Tools setup
search_tool = DuckDuckGoSearchRun()
python_repl_tool = PythonREPLTool()

# Agent node function
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # Extract final answer from intermediate steps
    if result['intermediate_steps']:
        agent_action = result['intermediate_steps'][0][0]
        final_answer = agent_action.log.split("Final Answer:")[-1].strip()
        return {
            "messages": [HumanMessage(content=final_answer, name=name)]
        }
    return {
        "messages": [HumanMessage(content="No result found", name=name)]
    }

# Supervisor setup
members = ["Researcher", "Coder"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

options = ["FINISH"] + members

class routeResponse(BaseModel):
    next: Literal[tuple(options)]  # Convert list to tuple for Literal

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

# Supervisor agent function
def supervisor_agent(state):
    # Create a tool for the supervisor to make decisions
    supervisor_tool = Tool(
        name="make_decision",
        func=lambda x: x,  # Simple passthrough
        description="Decide which worker should act next or if we should FINISH"
    )
    
    supervisor = create_amex_agent(
        llm=llm,
        tools=[supervisor_tool],
        verbose=False
    )
    
    result = supervisor.invoke({
        "input": prompt.format_messages(messages=state["messages"])[-1].content
    })
    
    if result['intermediate_steps']:
        agent_action = result['intermediate_steps'][0][0]
        decision = agent_action.log.split("Final Answer:")[-1].strip()
        return {"next": decision.upper()}  # Ensure consistent casing
    
    return {"next": "FINISH"}  # Default to FINISH if no clear decision

# State definition for the graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Create specialized agents
research_agent = create_amex_agent(
    llm=llm,
    tools=[search_tool],
    verbose=False
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

code_agent = create_amex_agent(
    llm=llm,
    tools=[python_repl_tool],
    verbose=False
)
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

# Create and configure the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_agent)

# Add edges
for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.add_edge(START, "supervisor")

# Compile the graph
graph = workflow.compile()

# Example usage function
def run_workflow(query: str):
    """Run the workflow with a given query."""
    for s in graph.stream(
        {
            "messages": [
                HumanMessage(content=query)
            ]
        },
        {"recursion_limit": 100},
    ):
        if "__end__" not in s:
            print(s)
            print("----")

if __name__ == "__main__":
    # Test the workflow with examples
    print("Testing Hello World example:")
    run_workflow("Code hello world and print it to the terminal")
    
    print("\nTesting Research example:")
    run_workflow("Write a brief research report on pikas.")
