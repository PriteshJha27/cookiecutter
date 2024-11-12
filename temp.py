from typing import List, Optional, Dict, Any, Sequence
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.tools import BaseTool
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools import DuckDuckGoSearchTool

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
    """Enhanced ChatAmexLlama with proper tool binding and agent support."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tools: List[BaseTool] = []
        self._tool_descriptions: Optional[str] = None
        self._tool_names: Optional[str] = None
    
    @property
    def tools(self) -> List[BaseTool]:
        """Get the list of bound tools."""
        return self._tools
    
    def bind_tools(self, tools: Sequence[BaseTool]) -> 'ChatAmexLlamaWithTools':
        """
        Bind tools to the LLM and prepare tool descriptions for agent use.
        Returns a new instance with bound tools.
        """
        self._tools = list(tools)
        
        # Prepare tool descriptions and names
        self._tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}" 
            for tool in self._tools
        )
        self._tool_names = ", ".join(tool.name for tool in self._tools)
        
        return self
    
    def get_tool_prompt_variables(self) -> Dict[str, str]:
        """Get the tool-related variables needed for the prompt template."""
        if not self._tools:
            return {"tools": "", "tool_names": ""}
        return {
            "tools": self._tool_descriptions,
            "tool_names": self._tool_names
        }
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Enhanced _acall with tool handling."""
        response = await super()._acall(prompt, stop, run_manager, **kwargs)
        
        # Handle tool execution if tools are bound
        if self._tools:
            try:
                # Look for tool calls in the format specified in the prompt
                if "Action:" in response and "Action Input:" in response:
                    # Extract action and input
                    action_line = [line for line in response.split('\n') if line.startswith('Action:')][0]
                    input_line = [line for line in response.split('\n') if line.startswith('Action Input:')][0]
                    
                    tool_name = action_line.replace('Action:', '').strip()
                    tool_input = input_line.replace('Action Input:', '').strip()
                    
                    # Find and execute the appropriate tool
                    for tool in self._tools:
                        if tool.name == tool_name:
                            return await tool.arun(tool_input)
            except Exception as e:
                print(f"Tool execution error: {e}")
                
        return response
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous version of _acall."""
        import asyncio
        return asyncio.run(self._acall(prompt, stop, run_manager, **kwargs))

def create_amex_agent(llm: ChatAmexLlamaWithTools, tools: List[BaseTool], verbose: bool = True):
    """
    Create an agent using the ChatAmexLlamaWithTools LLM.
    """
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create prompt template with tool variables
    prompt = PromptTemplate(
        template=REACT_CUSTOM_TEMPLATE,
        input_variables=["input", "scratchpad", "tools", "tool_names"]
    )
    
    # Create the agent with bound tools
    agent = create_react_agent(
        llm=llm_with_tools,
        tools=tools,
        prompt=prompt
    )
    
    # Create and return the agent executor
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose
    )
