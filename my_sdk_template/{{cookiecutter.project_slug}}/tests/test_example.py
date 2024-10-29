import pytest
from langchain_project import process_chain  # replace with your project name

@pytest.fixture
def setup_environment(monkeypatch):
    """Setup test environment"""
    monkeypatch.setenv("OPENAI_API_KEY", "your-test-key")

def test_list_output(setup_environment):
    result = process_chain({
        "prompt_template": "List 3 colors: {format_instructions}",
        "output_format": "list"
    })
    assert isinstance(result, list)
    assert len(result) == 3

def test_json_output(setup_environment):
    result = process_chain({
        "prompt_template": "Create a profile with name and age",
        "output_format": "json"
    })
    assert isinstance(result, dict)
    assert "name" in result
    assert "age" in result

def test_custom_system_prompt(setup_environment):
    result = process_chain({
        "prompt_template": "What is 2+2?",
        "system_prompt": "You are a math teacher"
    })
    assert result is not None

def test_error_handling(setup_environment):
    with pytest.raises(Exception):
        process_chain({
            "prompt_template": "",  # Empty prompt should raise error
            "output_format": "invalid_format"
        })