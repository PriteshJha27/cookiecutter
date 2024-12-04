
import unittest
from langchain.tools import BaseTool
from typing import Optional

# Sample Tool for Testing
class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Useful for performing mathematical calculations"
    
    def _run(self, query: str) -> str:
        try:
            return str(eval(query))
        except Exception as e:
            return f"Error: {str(e)}"
            
    async def _arun(self, query: str) -> str:
        return self._run(query)

class TestLlamaToolBinding(unittest.TestCase):
    def setUp(self):
        self.llm = ChatAmexLlama(
            base_url="your_base_url",
            auth_url="your_auth_url",
            cert_path="your_cert_path",
            user_id="your_user_id",
            pwd="your_password",
            model_name="llama3-70b-instruct"
        )
        self.calculator_tool = CalculatorTool()
    
    def test_tool_formatting(self):
        """Test if tools are formatted correctly for the API"""
        formatted_tool = self.llm.format_tool_call(
            name="calculator",
            arguments="2 + 2",
            tool_call_id="test_call_1"
        )
        
        self.assertEqual(formatted_tool["type"], "function")
        self.assertEqual(formatted_tool["function"]["name"], "calculator")
        self.assertEqual(formatted_tool["id"], "test_call_1")
    
    def test_bind_tools(self):
        """Test binding tools to the model"""
        bound_llm = self.llm.bind_tools(
            tools=[self.calculator_tool],
            tool_choice={"type": "function", "function": {"name": "calculator"}}
        )
        
        self.assertIsNotNone(bound_llm)
        # Verify the tool is properly bound
        self.assertTrue(hasattr(bound_llm, "_tools"))
        
    async def test_tool_execution(self):
        """Test actual execution of bound tools"""
        bound_llm = self.llm.bind_tools([self.calculator_tool])
        
        messages = [
            HumanMessage(content="Calculate 15 + 27")
        ]
        
        response = await bound_llm.ainvoke(messages)
        self.assertIsNotNone(response)
        # Add more specific assertions based on your expected response format

if __name__ == '__main__':
    unittest.main()
