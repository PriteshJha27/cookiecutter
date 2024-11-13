from typing import List
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from chat_amex_llama import ChatAmexLlama

def create_amex_agent(
    llm: ChatAmexLlama,
    tools: List[BaseTool],
    verbose: bool = True
) -> AgentExecutor:
    """Create an agent using ChatAmexLlama."""
    
    # Create the agent prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a helpful AI assistant. "
            "Use the following tools to help answer questions:\n\n"
            "{tools}\n\n"
            "To use a tool, use the following format:\n"
            "Thought: Consider what to do\n"
            "Action: Tool name\n"
            "Action Input: Tool input\n"
            "Observation: Tool output\n"
            "... (repeat if needed)\n"
            "Thought: I know what to answer\n"
            "Final Answer: The final answer\n"
        )),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create the agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True
    )
    
    return agent_executor
