from typing import List, Optional, Dict, Any, Sequence
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.tools import BaseTool
from langchain.schema import HumanMessage, SystemMessage, AIMessage

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
            # Format tool descriptions for the system prompt
            tool_descriptions = "\n".join(
                f"- {tool.name}: {tool.description}" 
                for tool in self.tools
            )
            
            enhanced_prompt = (
                f"You have access to the following tools:\n{tool_descriptions}\n\n"
                f"To use a tool, respond with:\n"
                f"```json\n{{\n  \"tool\": \"tool_name\",\n  \"input\": \"tool input\"\n}}\n```\n\n"
                f"Original prompt: {prompt}"
            )
            response = await super()._acall(enhanced_prompt, stop, run_manager, **kwargs)
            
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
