from typing import List, Optional, Dict, Any, Sequence
from langchain.tools import BaseTool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field
import re

class ChatAmexLlamaWithTools(ChatAmexLlama):
    """ChatAmexLlama with tool binding capabilities."""

    tools: List[BaseTool] = Field(default_factory=list)
    agent_template: str = Field(default="""Answer the following questions as best you can using the provided tools.

Available Tools:
{tool_descriptions}

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
Thought: Let's solve this step by step
{agent_scratchpad}""")

    def bind_tools(self, tools: Sequence[BaseTool]) -> 'ChatAmexLlamaWithTools':
        """Bind tools to the LLM."""
        self.tools = list(tools)
        return self

    def get_tool_prompts(self) -> Dict[str, Any]:
        """Get tool-related prompt variables."""
        tool_descriptions = "\n".join(
            f"{tool.name}: {tool.description}" for tool in self.tools
        )
        tool_names = ", ".join(tool.name for tool in self.tools)
        
        return {
            "tool_descriptions": tool_descriptions,
            "tool_names": tool_names
        }

    def parse_output(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM output to extract action and input."""
        if "Final Answer:" in text:
            answer = text.split("Final Answer:")[-1].strip()
            return {"type": "finish", "output": answer}

        action_match = re.search(r"Action: (.*?)\nAction Input: (.*?)(?:\n|$)", text, re.DOTALL)
        if action_match:
            return {
                "type": "action",
                "action": action_match.group(1).strip(),
                "action_input": action_match.group(2).strip()
            }
        
        return None

    def execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a specific tool."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.run(tool_input)
        raise ValueError(f"Tool '{tool_name}' not found")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Override _call to handle tool execution."""
        response = super()._call(prompt, stop, **kwargs)
        
        if not self.tools:
            return response

        parsed = self.parse_output(response)
        if parsed:
            if parsed["type"] == "finish":
                return parsed["output"]
            elif parsed["type"] == "action":
                try:
                    result = self.execute_tool(parsed["action"], parsed["action_input"])
                    return f"{response}\nObservation: {result}"
                except Exception as e:
                    return f"{response}\nObservation: Tool execution failed: {str(e)}"
        
        return response
