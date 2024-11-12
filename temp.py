from typing import List, Optional, Dict, Any, Sequence
from langchain.tools import BaseTool
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManagerForLLMRun

class ChatAmexLlamaWithTools(ChatAmexLlama):
    """Enhanced ChatAmexLlama with proper tool binding and agent support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tools: List[BaseTool] = []
        self._tool_descriptions: Dict[str, str] = {}
        self._tool_names: List[str] = []
        self._last_tool_output: Optional[str] = None

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
        self._tool_descriptions = {
            tool.name: f"{tool.name}: {tool.description}"
            for tool in self._tools
        }
        self._tool_names = [tool.name for tool in self._tools]
        
        return self

    def get_tool_prompt_variables(self) -> Dict[str, Any]:
        """Get tool-related variables needed for the prompt template."""
        if not self._tools:
            return {"tools": "", "tool_names": ""}
        
        return {
            "tools": "\n".join(self._tool_descriptions.values()),
            "tool_names": ", ".join(self._tool_names)
        }

    async def _process_tool_response(self, response: str) -> Dict[str, Any]:
        """Process tool execution response and extract relevant information."""
        try:
            # Split response into observation and action parts
            parts = response.split("Observation:", 1)
            if len(parts) > 1:
                action_part = parts[0]
                observation = parts[1].strip()
            else:
                action_part = response
                observation = ""

            # Extract action and input
            action_lines = [line for line in action_part.split("\n") if line.strip()]
            action = None
            action_input = None

            for line in action_lines:
                if line.startswith("Action:"):
                    action = line.replace("Action:", "").strip()
                elif line.startswith("Action Input:"):
                    action_input = line.replace("Action Input:", "").strip()

            return {
                "action": action,
                "action_input": action_input,
                "observation": observation
            }
        except Exception as e:
            raise ValueError(f"Error processing tool response: {str(e)}")

    async def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and return its result."""
        for tool in self._tools:
            if tool.name == tool_name:
                try:
                    result = await tool.arun(tool_input)
                    self._last_tool_output = result
                    return result
                except Exception as e:
                    return f"Tool execution error: {str(e)}"
        return f"Tool '{tool_name}' not found"

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Enhanced _acall with tool handling."""
        # Get initial response
        response = await super()._acall(prompt, stop, run_manager, **kwargs)
        
        # Check for tool execution
        if self._tools:
            try:
                tool_info = await self._process_tool_response(response)
                
                if tool_info["action"] and tool_info["action_input"]:
                    # Execute tool
                    tool_result = await self._execute_tool(
                        tool_info["action"],
                        tool_info["action_input"]
                    )
                    
                    # Format final response
                    final_response = (
                        f"{response}\n"
                        f"Observation: {tool_result}\n"
                        f"Final Answer: Based on the tool execution, {tool_result}"
                    )
                    return final_response
            except Exception as e:
                print(f"Tool execution error: {str(e)}")
        
        return response

    def create_agent_prompt(self) -> PromptTemplate:
        """Create a prompt template for the agent."""
        template = """Answer the following questions as best you can. You have access to the following tools:

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
{agent_scratchpad}"""

        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables=self.get_tool_prompt_variables()
        )
