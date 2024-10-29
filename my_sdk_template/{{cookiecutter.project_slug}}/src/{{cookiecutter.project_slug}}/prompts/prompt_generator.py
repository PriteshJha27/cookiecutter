# {{cookiecutter.project_slug}}/src/{{cookiecutter.project_slug}}/prompts/prompt_generator.py
from langchain.prompts import ChatPromptTemplate
from ..utils.helpers import load_config

def create_prompt_template(template: str) -> ChatPromptTemplate:
    """
    Create a ChatPromptTemplate with system message and user template
    """
    config = load_config()
    system_prompt = config["prompts"]["default_system_prompt"]
    
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", template)
    ])