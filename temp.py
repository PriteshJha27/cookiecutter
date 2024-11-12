from typing import List, Optional, Dict, Any, Sequence
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.tools import BaseTool
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools import DuckDuckGoSearchTool

# Custom prompt template for the React agent
REACT_CUSTOM_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{scratchpad}"""

class ChatAmexLlamaWithTools(ChatAmexLlama):
    tools: List[BaseTool] = []
    
    def bind_tools(self, tools: Sequence[BaseTool]) -> 'ChatAmexLlamaWithTools':
        """Bind tools to the LLM for agent use."""
        self.tools = list(tools)
        return self
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Enhanced _acall to handle tool descriptions and function calling."""
        if self.tools:
            response = await super()._acall(prompt, stop, run_manager, **kwargs)
            
            # Parse potential tool calls from response
            try:
                import json
                tool_call = json.loads(response)
                if "tool" in tool_call and "input" in tool_call:
                    # Find and execute the appropriate tool
                    for tool in self.tools:
                        if tool.name == tool_call["tool"]:
                            return await tool.arun(tool_call["input"])
            except:
                pass
            
            return response
        else:
            return await super()._acall(prompt, stop, run_manager, **kwargs)
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous version of _acall with tool support."""
        import asyncio
        return asyncio.run(self._acall(prompt, stop, run_manager, **kwargs))

# Function to create and setup the agent
def setup_react_agent(llm: ChatAmexLlamaWithTools, tools: List[BaseTool]):
    """
    Setup a REACT agent with the custom LLM and tools
    """
    # Create prompt template
    prompt = PromptTemplate(
        template=REACT_CUSTOM_TEMPLATE,
        input_variables=["input", "scratchpad", "tools", "tool_names"]
    )
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create the agent
    agent = create_react_agent(
        llm=llm_with_tools,
        tools=tools,
        prompt=prompt
    )
    
    # Create the agent executor
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )
