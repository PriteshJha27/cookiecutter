from typing import List, Optional, Dict, Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from chat_amex_llama import ChatAmexLlama

def create_amex_agent(
    llm: ChatAmexLlama,
    tools: List[BaseTool],
    verbose: bool = False
) -> AgentExecutor:
    """Create a ReAct agent using ChatAmexLlama."""
    
    # Create the prompt template with tool descriptions
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to tools. Answer the questions using the appropriate tools when needed.

Available Tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Format tool descriptions
    tools_str = "\n".join(f"{tool.name}: {tool.description}" for tool in tools)
    tool_names = ", ".join(tool.name for tool in tools)

    # Create the agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Create the executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=5,  # Limit maximum tool usage iterations
        early_stopping_method="generate",  # Stop if answer is found before max_iterations
        handle_parsing_errors=True  # Gracefully handle parsing errors
    )

    return agent_executor
