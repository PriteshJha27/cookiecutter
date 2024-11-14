from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain.callbacks import BaseCallbackManager
from langchain.schema import AgentAction, AgentFinish

class AmexAgentExecutor:
    """Wrapper class for AgentExecutor that handles output processing."""
    
    def __init__(self, agent_executor: AgentExecutor):
        self.agent_executor = agent_executor
        
    def invoke(self, input_dict: Dict[str, Any]) -> str:
        """
        Invoke the agent and return only the final answer.
        Args:
            input_dict: Dictionary containing the input query
        Returns:
            str: Final answer from the agent
        """
        # Run the agent executor
        result = self.agent_executor.invoke(input_dict)
        
        # Get the agent's steps from the callbacks
        steps = self.agent_executor.agent.intermediate_steps
        
        # Extract final answer from the last step
        if steps and isinstance(steps[-1], AgentFinish):
            return steps[-1].return_values.get('output', '')
        
        # If no clear final answer, return the last observation
        if steps:
            last_step = steps[-1]
            if isinstance(last_step, tuple) and len(last_step) > 1:
                return last_step[1]  # Return the observation
                
        # Fallback to raw output if no structured data available
        return result.get('output', '')

def create_amex_agent(
    llm: ChatAmexLlama,
    tools: List[BaseTool],
    verbose: bool = True
) -> AmexAgentExecutor:
    """
    Create an agent using ChatAmexLlama that returns clean final answers.
    Args:
        llm: ChatAmexLlama instance
        tools: List of tools available to the agent
        verbose: Whether to print verbose output
    Returns:
        AmexAgentExecutor: Wrapped agent executor that handles output processing
    """
    # Create the template (keeping your existing template)
    template = """Answer the following questions as best you can using the provided tools.

Available Tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""

    # Create prompt
    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            "tool_names": ", ".join([tool.name for tool in tools])
        }
    )

    # Create the agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Create the executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=1
    )

    # Return wrapped executor
    return AmexAgentExecutor(executor)
